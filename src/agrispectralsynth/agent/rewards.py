"""
Reward table: every action evaluated on every (image, sensor) context.

Because all arms can be run offline against the ground truth, the full
reward matrix is computed once. Bandit experiments then *replay* it:
at each step the agent only sees the reward of the arm it chose, but the
other values are available to measure regret and the oracle. This is the
standard offline evaluation protocol for contextual bandits and makes
every experiment cheap and exactly reproducible.

Outputs (CSV):
    contexts.csv  one row per (image, sensor): site, n_true, features
    rewards.csv   one row per (image, sensor, arm): metrics, seconds, reward
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
from .actions import ARMS, make_signals, run_arm
from .evaluation import DEFAULT_IOU, RewardWeights, reward, score_detections
from .features import FEATURE_NAMES, scene_features
from .groundtruth import GroundTruth, load_annotations, site_of

logger = logging.getLogger("agrispectralsynth.agent")

DRONES = ["dji_mavic3m", "dji_phantom4m", "micasense_rededge_mx", "parrot_sequoia_plus"]
IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
SCORE_FIELDS = ["precision", "recall", "f1", "n_pred", "n_true", "tp", "count_error", "rel_count_error", "mean_iou"]


@dataclass
class RewardJob:
    sensors: List[str] = field(default_factory=lambda: list(DRONES))
    gsd_m: float = 0.1
    iou: float = DEFAULT_IOU
    weights: RewardWeights = RewardWeights()
    seed: int = 42


def _find_image(folder: Path, stem: str) -> Optional[Path]:
    for suf in IMAGE_SUFFIXES:
        p = folder / f"{stem}{suf}"
        if p.exists():
            return p
    return None


def evaluate_image(path: Path, gt: GroundTruth, job: RewardJob) -> Tuple[List[dict], List[dict]]:
    """All arms x sensors for one image. Returns (context rows, reward rows)."""
    rgb, _ = read_rgb(path)
    if rgb.dtype != np.uint8:
        rgb = (rgb.astype(np.float32) / np.iinfo(rgb.dtype).max * 255 + 0.5).astype(np.uint8)
    stem, site = path.stem, site_of(path.stem)
    engine = SpectralEngine()
    scene = engine.prepare(rgb)
    shared: Dict[str, dict] = {}        # sensor-independent arms are run once per image
    contexts, rewards = [], []
    for sid in job.sensors:
        sensor = load_sensor(sid)
        rng = np.random.default_rng(job.seed + zlib.crc32(f"{stem}|{sid}".encode()))
        bands = engine.render(scene, sensor, rng=rng)
        ndvi = NDVI().compute(bands[sensor.roles["red"]], bands[sensor.roles["nir"]])
        sig = make_signals(rgb, ndvi, bands[sensor.roles["nir"]])
        feats = scene_features(sig, scene.veg_fraction, sid, job.gsd_m)
        contexts.append({"image": stem, "site": site, "sensor": sid, "n_true": gt.count, **feats})
        for arm in ARMS:
            if not arm.sensor_dependent and arm.name in shared:
                row = dict(shared[arm.name])
            else:
                t0 = time.perf_counter()
                boxes = run_arm(arm, sig, job.gsd_m)
                secs = time.perf_counter() - t0
                sc = score_detections(boxes, gt.boxes, job.iou)
                row = {k: getattr(sc, k) for k in SCORE_FIELDS}
                row["seconds"] = round(secs, 4)
                row["reward"] = reward(sc, secs, job.weights)
                if not arm.sensor_dependent:
                    shared[arm.name] = row
            rewards.append({"image": stem, "site": site, "sensor": sid, "arm": arm.name, **row})
    return contexts, rewards


def _worker(args):
    path, gt, job = args
    try:
        return evaluate_image(path, gt, job), None
    except Exception as exc:  # report and continue
        return ([], []), f"{path.name}: {type(exc).__name__}: {exc}"


def _init():
    import cv2

    cv2.setNumThreads(1)


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
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        raise FileNotFoundError(f"No annotated images found in {images_dir}")
    workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    logger.info("%d annotated images, %d sensors, %d arms, %d workers", len(tasks), len(job.sensors), len(ARMS), workers)

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
    contexts, rewards, errors = [], [], []
    try:
        for (c, r), err in results:
            contexts += c
            rewards += r
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
    n_true: np.ndarray
    images: np.ndarray
    sites: np.ndarray
    sensors: np.ndarray
    arms: List[str]
    features: List[str]


def load_reward_table(folder: Path) -> RewardTable:
    import pandas as pd

    folder = Path(folder)
    ctx = pd.read_csv(folder / "contexts.csv")
    rew = pd.read_csv(folder / "rewards.csv")
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
        n_true=ctx["n_true"].to_numpy(dtype=np.float64),
        images=ctx["image"].to_numpy(),
        sites=ctx["site"].to_numpy(),
        sensors=ctx["sensor"].to_numpy(),
        arms=arms,
        features=feats,
    )
