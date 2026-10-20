"""
Reward table: every action evaluated on every (image, sensor) context.

Because all arms can be run offline against the ground truth, the full
reward matrix is computed once. Bandit experiments then *replay* it:
at each step the agent only sees the reward of the arm it chose, but the
other values are available to measure regret and the oracle. This is the
standard offline evaluation protocol for contextual bandits and makes
every experiment cheap and exactly reproducible.

Outputs (CSV):
    contexts.csv        one row per (image, sensor): site, n_true, features
    rewards.csv         one row per (image, sensor, arm): metrics, seconds, reward
    hsi_validation.csv  (with --hsi) synthetic vs real NDVI per (image, sensor)
    hsi_pixels.npz      (with --hsi) the 1 m pixel pairs behind that validation
"""

from __future__ import annotations

import csv
import logging
import multiprocessing
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..indices import NDVI
from ..pipeline import read_rgb
from ..sensors.registry import load_sensor
from ..spectral.engine import SpectralEngine
from .actions import arms_for, detect, make_signals, run_arm
from .evaluation import DEFAULT_IOU, RewardWeights, reward, score_detections
from .features import FEATURE_NAMES, scene_features
from .groundtruth import GroundTruth, load_annotations, site_of

logger = logging.getLogger("agrispectralsynth.agent")

DRONES = ["dji_mavic3m", "dji_phantom4m", "micasense_rededge_mx", "parrot_sequoia_plus"]
IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
SCORE_FIELDS = ["precision", "recall", "f1", "n_pred", "n_true", "tp", "count_error", "rel_count_error", "mean_iou"]


HSI_FIELDS = ["image", "site", "sensor", "n_px", "real_ndvi_mean", "syn_ndvi_mean", "bias", "rmse", "r",
              "real_red", "syn_red", "real_nir", "syn_nir"]


@dataclass
class RewardJob:
    sensors: List[str] = field(default_factory=lambda: list(DRONES))
    gsd_m: float = 0.1
    iou: float = DEFAULT_IOU
    weights: RewardWeights = RewardWeights()
    seed: int = 42
    deepforest: bool = False                  # add the DeepForest arms (needs torch + weights)
    deepforest_weights: Optional[str] = None
    hsi_dir: Optional[str] = None             # NEON hyperspectral tiles -> real-NDVI arms + validation
    hsi_bands_csv: Optional[str] = None       # neon_aop_bands.csv (default: two levels above hsi_dir)

    @property
    def arms(self):
        return arms_for(self.deepforest, self.hsi_dir is not None)

    def bands_csv(self) -> Path:
        return Path(self.hsi_bands_csv) if self.hsi_bands_csv else Path(self.hsi_dir).parents[1] / "neon_aop_bands.csv"


def _find_image(folder: Path, stem: str) -> Optional[Path]:
    for suf in IMAGE_SUFFIXES:
        p = folder / f"{stem}{suf}"
        if p.exists():
            return p
    return None


def evaluate_image(path: Path, gt: GroundTruth, job: RewardJob):
    """All arms x sensors for one image. Returns (context rows, reward rows, hsi rows, hsi pixel pairs)."""
    rgb, _ = read_rgb(path)
    if rgb.dtype != np.uint8:
        rgb = (rgb.astype(np.float32) / np.iinfo(rgb.dtype).max * 255 + 0.5).astype(np.uint8)
    stem, site = path.stem, site_of(path.stem)

    cube = None
    if job.hsi_dir:
        from .hyperspectral import hsi_path, read_cube

        hp = hsi_path(Path(job.hsi_dir), stem)
        if hp is None:
            raise FileNotFoundError(f"no hyperspectral tile for {stem}")
        cube, wl = read_cube(hp, job.bands_csv())

    engine = SpectralEngine()
    scene = engine.prepare(rgb)
    shared: Dict[str, dict] = {}        # sensor-independent arms are run once per image
    contexts, rewards, hsi_rows, pairs = [], [], [], {}
    det = None
    for sid in job.sensors:
        sensor = load_sensor(sid)
        rng = np.random.default_rng(job.seed + zlib.crc32(f"{stem}|{sid}".encode()))
        bands = engine.render(scene, sensor, rng=rng)
        red, nir = bands[sensor.roles["red"]], bands[sensor.roles["nir"]]
        ndvi = NDVI().compute(red, nir)
        sig = make_signals(rgb, ndvi, nir)
        if det is not None:
            sig.detections = det
        feats = scene_features(sig, scene.veg_fraction, sid, job.gsd_m)
        contexts.append({"image": stem, "site": site, "sensor": sid, "n_true": gt.count, **feats})

        if cube is not None:
            from .hyperspectral import RealSpectra, aggregate

            real = RealSpectra(cube, wl, sensor, rgb.shape[:2])
            sig.index["hsi"], sig.peaks["hsi"] = real.ndvi, real.nir
            shape_1m = real.ndvi_1m.shape
            syn_ndvi = aggregate(ndvi, shape_1m)
            ok = np.isfinite(real.bands_1m[sensor.roles["nir"]])
            rn, sn = real.ndvi_1m[ok], syn_ndvi[ok]
            hsi_rows.append({
                "image": stem, "site": site, "sensor": sid, "n_px": int(ok.sum()),
                "real_ndvi_mean": round(float(rn.mean()), 4), "syn_ndvi_mean": round(float(sn.mean()), 4),
                "bias": round(float((sn - rn).mean()), 4), "rmse": round(float(np.sqrt(((sn - rn) ** 2).mean())), 4),
                "r": round(float(np.corrcoef(sn, rn)[0, 1]), 4) if rn.std() > 1e-6 and sn.std() > 1e-6 else "",
                "real_red": round(float(np.nanmean(real.bands_1m[sensor.roles["red"]])), 4),
                "syn_red": round(float(aggregate(red, shape_1m).mean()), 4),
                "real_nir": round(float(np.nanmean(real.bands_1m[sensor.roles["nir"]])), 4),
                "syn_nir": round(float(aggregate(nir, shape_1m).mean()), 4),
            })
            pairs[sid] = np.stack([rn, sn]).astype(np.float32)

        for arm in job.arms:
            if not arm.sensor_dependent and arm.name in shared:
                row = dict(shared[arm.name])
            else:
                t0 = time.perf_counter()
                if arm.is_detector and sig.detections is None:
                    det = detect(sig, job.deepforest_weights)
                t1 = time.perf_counter()
                boxes = run_arm(arm, sig, job.gsd_m)
                secs = time.perf_counter() - t1
                if arm.is_detector:
                    secs += sig.detections[2]          # every detector arm pays the inference
                sc = score_detections(boxes, gt.boxes, job.iou)
                row = {k: getattr(sc, k) for k in SCORE_FIELDS}
                row["seconds"] = round(secs, 4)
                row["reward"] = reward(sc, secs, job.weights)
                if not arm.sensor_dependent:
                    shared[arm.name] = row
            rewards.append({"image": stem, "site": site, "sensor": sid, "arm": arm.name, **row})
    return contexts, rewards, hsi_rows, pairs


def _worker(args):
    path, gt, job = args
    try:
        return evaluate_image(path, gt, job), None
    except Exception as exc:  # report and continue
        return ([], [], [], {}), f"{path.name}: {type(exc).__name__}: {exc}"


def _init():
    import cv2

    cv2.setNumThreads(1)
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def build_reward_table(
    images_dir: Path,
    annotations_dir: Path,
    out_dir: Path,
    job: Optional[RewardJob] = None,
    workers: int = 0,
    limit: Optional[int] = None,
    progress: bool = True,
) -> dict:
    job = job or RewardJob()
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gts = load_annotations(annotations_dir)
    tasks = []
    for stem in sorted(gts):
        p = _find_image(images_dir, stem)
        if p is not None:
            tasks.append((p, gts[stem], job))
    if job.hsi_dir:
        from .hyperspectral import hsi_path

        missing = [t[0].stem for t in tasks if hsi_path(Path(job.hsi_dir), t[0].stem) is None]
        if missing:
            logger.warning("%d images without hyperspectral tile are left out: %s", len(missing), ", ".join(missing))
        tasks = [t for t in tasks if t[0].stem not in missing]
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        raise FileNotFoundError(f"No annotated images found in {images_dir}")
    workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    logger.info("%d annotated images, %d sensors, %d arms, %d workers", len(tasks), len(job.sensors), len(job.arms), workers)

    t0 = time.perf_counter()
    if workers == 1:
        results = map(_worker, tasks)
    else:
        pool = ProcessPoolExecutor(workers, initializer=_init, mp_context=multiprocessing.get_context("spawn"))
        results = pool.map(_worker, tasks, chunksize=2)
    if progress:
        try:
            from tqdm import tqdm

            results = tqdm(results, total=len(tasks), unit="img", desc="rewards")
        except ImportError:
            pass
    contexts, rewards, hsi_rows, pairs, errors = [], [], [], {}, []
    try:
        for (c, r, h, px), err in results:
            contexts += c
            rewards += r
            hsi_rows += h
            for sid, arr in px.items():
                pairs.setdefault(sid, []).append(arr)
            if err:
                errors.append(err)
    finally:
        if workers != 1:
            pool.shutdown()

    ctx_fields = ["image", "site", "sensor", "n_true"] + FEATURE_NAMES
    rew_fields = ["image", "site", "sensor", "arm"] + SCORE_FIELDS + ["seconds", "reward"]
    for name, rows, fields in (("contexts.csv", contexts, ctx_fields), ("rewards.csv", rewards, rew_fields)):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    if hsi_rows:
        with open(out_dir / "hsi_validation.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HSI_FIELDS)
            w.writeheader()
            w.writerows(hsi_rows)
        np.savez_compressed(out_dir / "hsi_pixels.npz",
                            **{sid: np.concatenate(v, axis=1) for sid, v in pairs.items()})
    for e in errors:
        logger.error(e)
    return {
        "images": len(tasks) - len(errors),
        "contexts": len(contexts),
        "rewards": len(rewards),
        "errors": errors,
        "seconds": round(time.perf_counter() - t0, 1),
        "out_dir": str(out_dir),
    }


def _context_worker(args):
    path, gt, job = args
    rgb, _ = read_rgb(path)
    if rgb.dtype != np.uint8:
        rgb = (rgb.astype(np.float32) / np.iinfo(rgb.dtype).max * 255 + 0.5).astype(np.uint8)
    engine = SpectralEngine()
    scene = engine.prepare(rgb)
    rows = []
    for sid in job.sensors:
        sensor = load_sensor(sid)
        rng = np.random.default_rng(job.seed + zlib.crc32(f"{path.stem}|{sid}".encode()))
        bands = engine.render(scene, sensor, rng=rng)
        ndvi = NDVI().compute(bands[sensor.roles["red"]], bands[sensor.roles["nir"]])
        sig = make_signals(rgb, ndvi, bands[sensor.roles["nir"]])
        feats = scene_features(sig, scene.veg_fraction, sid, job.gsd_m)
        rows.append({"image": path.stem, "site": site_of(path.stem), "sensor": sid, "n_true": gt.count, **feats})
    return rows


def recompute_contexts(images_dir: Path, annotations_dir: Path, out_dir: Path,
                       job: Optional[RewardJob] = None, workers: int = 0) -> int:
    """Rewrite contexts.csv with the current feature set (rewards.csv is untouched)."""
    job = job or RewardJob()
    gts = load_annotations(annotations_dir)
    tasks = [(p, gts[s], job) for s in sorted(gts) if (p := _find_image(Path(images_dir), s)) is not None]
    workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers, initializer=_init, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = [r for rs in pool.map(_context_worker, tasks, chunksize=4) for r in rs]
    with open(Path(out_dir) / "contexts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "site", "sensor", "n_true"] + FEATURE_NAMES)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Loading for experiments
# ---------------------------------------------------------------------------

@dataclass
class RewardTable:
    """Dense view: X (n_ctx, n_feat), R (n_ctx, n_arms) plus metadata columns."""

    X: np.ndarray
    R: np.ndarray
    F1: np.ndarray
    count_err: np.ndarray        # rel_count_error per (ctx, arm)
    n_pred: np.ndarray
    seconds: np.ndarray
    n_true: np.ndarray
    images: np.ndarray
    sites: np.ndarray
    sensors: np.ndarray
    arms: List[str]
    features: List[str]


def load_reward_table(folder: Path, weights: Optional[RewardWeights] = None, arms=None) -> RewardTable:
    """
    weights: recompute the reward from the stored metrics with other weights
             (e.g. a time-cost sweep) without re-running any method.
    arms:    keep only these arms (e.g. to compare action sets).
    """
    import pandas as pd

    folder = Path(folder)
    ctx = pd.read_csv(folder / "contexts.csv")
    rew = pd.read_csv(folder / "rewards.csv")
    if arms is not None:
        rew = rew[rew["arm"].isin(list(arms))]
    if weights is not None:
        rew = rew.copy()
        rew["reward"] = (rew["f1"] - weights.count_weight * rew["rel_count_error"].clip(upper=1.0)
                         - weights.time_weight * rew["seconds"])
    arms = list(dict.fromkeys(rew["arm"]))
    key = ["image", "sensor"]
    ctx = ctx.sort_values(key).reset_index(drop=True)

    def pivot(col):
        p = rew.pivot_table(index=key, columns="arm", values=col, aggfunc="first")
        p = p.reindex(pd.MultiIndex.from_frame(ctx[key]))[arms]
        return p.to_numpy(dtype=np.float64)

    feats = [c for c in FEATURE_NAMES if c in ctx.columns]
    return RewardTable(
        X=ctx[feats].to_numpy(dtype=np.float64),
        R=pivot("reward"),
        F1=pivot("f1"),
        count_err=pivot("rel_count_error"),
        n_pred=pivot("n_pred"),
        seconds=pivot("seconds"),
        n_true=ctx["n_true"].to_numpy(dtype=np.float64),
        images=ctx["image"].to_numpy(),
        sites=ctx["site"].to_numpy(),
        sensors=ctx["sensor"].to_numpy(),
        arms=arms,
        features=feats,
    )
