#!/usr/bin/env python3
"""
SpecTree S01 — regenerate and SAVE every intermediate signal the experiments need.

For each of the 189 NEON plots with a valid hyperspectral tile:
  * DeepForest raw detections (boxes + scores, before any filter) and their time
  * boxes of every classical action (RGB once; synthetic- and real-NDVI per sensor) and their time
  * per-box medians of synthetic and real NDVI under every raw DeepForest box (for E02)
  * context attributes: the 25 v0.5 descriptors + the E01 extensions (C2 synthetic extras,
    C3 RGB transforms, C4 real-NDVI statistics) + E06 synthesis-variant statistics
  * 1 m pixel arrays (real/synthetic NDVI, red, NIR for each variant, RGB-derived features)
  * component timings (synthesis, features, actions)

Outputs (in --out):
  plots/<stem>.npz        boxes, scores, medians, 1 m arrays
  contexts_ext.csv        one row per (plot, sensor)
  action_times.csv        one row per (plot, sensor, action) with seconds
  component_times.csv     one row per (plot, sensor, component)
  data_manifest.csv       plot provenance
  run_notes_s01.md        failures and exclusions
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FEATURE_GROUPS = {
    # C0: RGB-only descriptors of v0.5 (+ nominal sensor one-hot)
    "C0": ["exg_mean", "exg_std", "lum_mean", "lum_std", "dark_fraction", "green_dark_corr", "texture",
           "gran_2m", "gran_4m", "gran_7m", "otsu_exg", "green_blobs_ha", "green_blob_area_m2", "green_blob_fill",
           "dark_blobs_ha", "dark_blob_area_m2", "dark_blob_fill",
           "sensor_dji_mavic3m", "sensor_dji_phantom4m", "sensor_micasense_rededge_mx", "sensor_parrot_sequoia_plus"],
    "C1_extra": ["ndvi_mean", "ndvi_std", "ndvi_p90"],
    "C2_extra": ["ndvi_mean", "ndvi_std", "ndvi_p90", "veg_fraction", "nir_mean", "nir_std",
                 "ndre_mean", "ndre_std", "ndvi_frac05"],
    "C3_extra": ["vari_mean", "vari_std", "vari_p90", "gli_mean", "gli_std",
                 "chrom_r", "chrom_g", "chrom_b", "exg_p90"],
    "C4_extra": ["real_ndvi_mean", "real_ndvi_std", "real_ndvi_p90"],
}
VARIANTS = ["V0", "V1", "V2", "V3", "V4", "V5"]   # V6 (supervised regression) is fitted in the analysis
DRONES = ["dji_mavic3m", "dji_phantom4m", "micasense_rededge_mx", "parrot_sequoia_plus"]


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def box_medians(boxes: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Median of `index` inside each box, with the same clipping as actions.filter_by_index."""
    h, w = index.shape
    out = np.zeros(len(boxes), np.float32)
    for i, (x0, y0, x1, y1) in enumerate(np.round(boxes).astype(int)):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))
        out[i] = np.median(index[y0:y1, x0:x1])
    return out


def ndvi_from_f(f, plan, sensor):
    """Proposition 1: synthetic NDVI as a function of the vegetation fraction only."""
    names = [b.name for b in sensor.band_list]
    r, n = names.index(sensor.roles["red"]), names.index(sensor.roles["nir"])
    vr, vn, sr, sn = plan.veg[r], plan.veg[n], plan.soil[r], plan.soil[n]
    red = f * (vr - sr) + sr
    nir = f * (vn - sn) + sn
    return ((nir - red) / (nir + red)).astype(np.float32)


def stats3(x):
    return float(np.mean(x)), float(np.std(x)), float(np.percentile(x, 90))


def process(args):
    stem, cfg = args
    try:
        return _process(stem, cfg), None
    except Exception:
        return None, f"{stem}: {traceback.format_exc()}"


def _process(stem, cfg):
    import cv2

    from agrispectralsynth.agent.actions import (ARMS, exg_chromatic, luminance, make_signals, run_arm)
    from agrispectralsynth.agent.deepforest import get_detector
    from agrispectralsynth.agent.features import scene_features
    from agrispectralsynth.agent.groundtruth import load_annotations, site_of
    from agrispectralsynth.agent.hyperspectral import RealSpectra, aggregate, hsi_path, read_cube
    from agrispectralsynth.indices import NDVI
    from agrispectralsynth.pipeline import read_rgb
    from agrispectralsynth.sensors.registry import load_sensor
    from agrispectralsynth.spectral.engine import SpectralEngine
    from agrispectralsynth.spectral.reflectance import ReflectanceModel, ReflectanceParams

    bench = Path(cfg["bench"])
    gt = load_annotations(bench / "annotations")[stem]
    rgb, _ = read_rgb(bench / "evaluation" / "RGB" / f"{stem}.tif")
    if rgb.dtype != np.uint8:
        rgb = (rgb.astype(np.float32) / np.iinfo(rgb.dtype).max * 255 + 0.5).astype(np.uint8)
    H, W = rgb.shape[:2]
    site = site_of(stem)
    store, ctx_rows, act_rows, comp_rows = {}, [], [], []

    # ---------------- DeepForest (raw, before any filter) ----------------
    det = get_detector(cfg["weights"])
    df_boxes, df_scores, df_secs = det.predict(rgb)
    store["df/boxes"], store["df/scores"] = df_boxes, df_scores
    comp_rows.append({"image": stem, "sensor": "*", "component": "deepforest", "seconds": df_secs})

    # ---------------- synthesis (sensor independent part) ----------------
    t0 = time.perf_counter()
    eng = SpectralEngine()
    scene = eng.prepare(rgb)
    comp_rows.append({"image": stem, "sensor": "*", "component": "synth_prepare", "seconds": time.perf_counter() - t0})

    # alternative vegetation fractions for E06
    disp = ReflectanceModel.normalize(rgb)
    s_int = disp.sum(axis=2) / 3.0
    p = ReflectanceParams()
    dark_gate = smoothstep(s_int, p.dark_low, p.dark_high)
    exg = exg_chromatic(rgb)
    rr, gg, bb = disp[..., 0], disp[..., 1], disp[..., 2]
    vari = np.clip((gg - rr) / (gg + rr - bb + 1e-6), -1, 1)
    gli = (2 * gg - rr - bb) / (2 * gg + rr + bb + 1e-6)
    f_v4 = smoothstep(exg, -0.02, 0.06) * dark_gate          # more permissive ExG thresholds
    f_v5 = smoothstep(vari, 0.0, 0.15) * dark_gate           # fraction from VARI instead of ExG
    lin_exg = np.clip((exg - p.exg_low) / (p.exg_high - p.exg_low), 0, 1)   # V1: linear map of ExG, no gating

    # RGB-only (sensor independent) C3 transforms
    ssum = disp.sum(axis=2) + 1e-6
    rgb_extra = {
        "vari_mean": float(vari.mean()), "vari_std": float(vari.std()), "vari_p90": float(np.percentile(vari, 90)),
        "gli_mean": float(gli.mean()), "gli_std": float(gli.std()),
        "chrom_r": float((rr / ssum).mean()), "chrom_g": float((gg / ssum).mean()), "chrom_b": float((bb / ssum).mean()),
        "exg_p90": float(np.percentile(exg, 90)),
    }

    # hyperspectral cube
    cube, wl = read_cube(hsi_path(bench / "evaluation" / "Hyperspectral", stem), bench / "neon_aop_bands.csv")
    shape_1m = cube.shape[1:]
    # RGB-derived 1 m features (for the supervised RGB->NDVI regression of E06/V6)
    lum = luminance(rgb)
    store["px/rgbfeat"] = np.stack([aggregate(x, shape_1m) for x in
                                    (disp[..., 0], disp[..., 1], disp[..., 2], exg, lum, vari)]).astype(np.float32)

    rgb_arms_done = False
    for sid in DRONES:
        sensor = load_sensor(sid)
        t0 = time.perf_counter()
        bands = eng.render(scene, sensor, noise_std=0)
        red, nir = bands[sensor.roles["red"]], bands[sensor.roles["nir"]]
        ndvi = NDVI().compute(red, nir)
        comp_rows.append({"image": stem, "sensor": sid, "component": "synth_render", "seconds": time.perf_counter() - t0})
        ndre = NDVI().compute(bands[sensor.roles["red_edge"]], nir)

        sig = make_signals(rgb, ndvi, nir)
        sig.detections = (df_boxes, df_scores, df_secs)
        t0 = time.perf_counter()
        feats = scene_features(sig, scene.veg_fraction, sid, cfg["gsd"])
        comp_rows.append({"image": stem, "sensor": sid, "component": "features_v05", "seconds": time.perf_counter() - t0})

        real = RealSpectra(cube, wl, sensor, (H, W))
        sig.index["hsi"], sig.peaks["hsi"] = real.ndvi, real.nir
        valid = np.isfinite(real.bands_1m[sensor.roles["nir"]])

        # ---- E06 synthesis variants (NDVI maps at 10 cm) ----
        plan = eng.plan(sensor)
        var_maps = {
            "V0": ndvi,
            "V1": (ndvi_from_f(np.zeros(1, np.float32), plan, sensor)[0]
                   + (ndvi_from_f(np.ones(1, np.float32), plan, sensor)[0]
                      - ndvi_from_f(np.zeros(1, np.float32), plan, sensor)[0]) * lin_exg).astype(np.float32),
            "V2": np.full((H, W), ndvi_from_f(np.array([scene.veg_fraction.mean()], np.float32), plan, sensor)[0], np.float32),
            "V4": ndvi_from_f(f_v4.astype(np.float32), plan, sensor),
            "V5": ndvi_from_f(f_v5.astype(np.float32), plan, sensor),
        }
        # V3: no chromatic modulation (c = 1). By Proposition 1 its NDVI equals V0; red/NIR levels change.
        k = scene.brightness
        red_v3 = np.clip((scene.veg_fraction * (plan.veg - plan.soil)[[b.name for b in sensor.band_list].index(sensor.roles["red"])]
                          + plan.soil[[b.name for b in sensor.band_list].index(sensor.roles["red"])]) * k, 0, 1)
        nir_v3 = np.clip((scene.veg_fraction * (plan.veg - plan.soil)[[b.name for b in sensor.band_list].index(sensor.roles["nir"])]
                          + plan.soil[[b.name for b in sensor.band_list].index(sensor.roles["nir"])]) * k, 0, 1)
        var_maps["V3"] = NDVI().compute(red_v3.astype(np.float32), nir_v3.astype(np.float32))
        for v in VARIANTS:
            store[f"px/{sid}/{v}"] = aggregate(var_maps[v], shape_1m).astype(np.float32)
        store[f"px/{sid}/real_ndvi"] = real.ndvi_1m.astype(np.float32)
        store[f"px/{sid}/valid"] = valid
        store[f"px/{sid}/real_red"] = np.nan_to_num(real.bands_1m[sensor.roles["red"]]).astype(np.float32)
        store[f"px/{sid}/real_nir"] = np.nan_to_num(real.bands_1m[sensor.roles["nir"]]).astype(np.float32)
        store[f"px/{sid}/V0_red"] = aggregate(red, shape_1m).astype(np.float32)
        store[f"px/{sid}/V0_nir"] = aggregate(nir, shape_1m).astype(np.float32)
        store[f"px/{sid}/V3_red"] = aggregate(red_v3.astype(np.float32), shape_1m)
        store[f"px/{sid}/V3_nir"] = aggregate(nir_v3.astype(np.float32), shape_1m)

        # ---- context row ----
        row = {"image": stem, "site": site, "sensor": sid, "n_true": gt.count, **feats, **rgb_extra}
        row["nir_mean"], row["nir_std"] = float(nir.mean()), float(nir.std())
        row["ndre_mean"], row["ndre_std"] = float(ndre.mean()), float(ndre.std())
        row["ndvi_frac05"] = float((ndvi > 0.5).mean())
        rn = real.ndvi_1m[valid]
        row["real_ndvi_mean"], row["real_ndvi_std"], row["real_ndvi_p90"] = stats3(rn)
        for v in VARIANTS:
            m, s_, p90 = stats3(var_maps[v])
            row[f"{v}_ndvi_mean"], row[f"{v}_ndvi_std"], row[f"{v}_ndvi_p90"] = m, s_, p90
        ctx_rows.append(row)

        # ---- DeepForest per-box medians (synthetic V0 and real NDVI), for E02 ----
        store[f"dfmed/{sid}/syn"] = box_medians(df_boxes, ndvi)
        store[f"dfmed/{sid}/real"] = box_medians(df_boxes, real.ndvi)

        # ---- classical actions ----
        for arm in ARMS:
            if arm.is_detector:
                continue
            if not arm.sensor_dependent:
                if rgb_arms_done:
                    continue
                key = f"box/*/{arm.name}"
            else:
                key = f"box/{sid}/{arm.name}"
            t0 = time.perf_counter()
            b = run_arm(arm, sig, cfg["gsd"])
            secs = time.perf_counter() - t0
            store[key] = b.astype(np.float32)
            act_rows.append({"image": stem, "sensor": "*" if not arm.sensor_dependent else sid,
                             "action": arm.name, "seconds": round(secs, 5), "n_pred": len(b)})
        rgb_arms_done = True

    store["gt/boxes"] = gt.boxes.astype(np.float32)
    np.savez_compressed(Path(cfg["out"]) / "plots" / f"{stem}.npz", **store)
    manifest = {"site": site, "plot": stem, "year": next((t for t in stem.split("_") if t.isdigit() and len(t) == 4), ""),
                "area_m2": round(H * W * cfg["gsd"] ** 2, 1), "width_px": W, "height_px": H, "gsd_m": cfg["gsd"],
                "modalities": "RGB;HSI(426 bands,1 m)", "n_true": gt.count,
                "provenance": "NeonTreeEvaluation (Weinstein et al. 2021), CC0; evaluation/RGB + evaluation/Hyperspectral"}
    return ctx_rows, act_rows, comp_rows, manifest


def _init():
    import cv2

    cv2.setNumThreads(1)
    import torch

    torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    from agrispectralsynth.agent.groundtruth import load_annotations
    from agrispectralsynth.agent.hyperspectral import hsi_path, read_band_table

    bench = Path(a.bench)
    gts = load_annotations(bench / "annotations")
    stems = sorted(s for s in gts if (bench / "evaluation" / "RGB" / f"{s}.tif").exists())
    notes = []
    no_hsi = [s for s in stems if hsi_path(bench / "evaluation" / "Hyperspectral", s) is None]
    notes.append(f"Sin cubo hiperespectral (excluidas): {', '.join(no_hsi)}")
    stems = [s for s in stems if s not in no_hsi]
    # tiles whose band count does not match the band table are excluded (known: ONAQ_021_2019 has 425 bands)
    import rasterio

    nb = len(read_band_table(str(bench / "neon_aop_bands.csv"))[0])
    bad = []
    for s in stems:
        with rasterio.open(hsi_path(bench / "evaluation" / "Hyperspectral", s)) as src:
            if src.count != nb:
                bad.append(f"{s} ({src.count} bandas)")
    notes.append(f"Cubo con número de bandas inválido (excluidas): {', '.join(bad)}")
    stems = [s for s in stems if not any(b.startswith(s + " ") for b in bad)]
    if a.limit:
        stems = stems[: a.limit]
    cfg = {"bench": str(bench), "weights": a.weights, "out": str(out), "gsd": 0.1}
    t0 = time.perf_counter()
    ctx, act, comp, man, errors = [], [], [], [], []
    with ProcessPoolExecutor(a.workers, initializer=_init, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, (res, err) in enumerate(pool.map(process, [(s, cfg) for s in stems])):
            if err:
                errors.append(err)
                continue
            c, ac, co, m = res
            ctx += c; act += ac; comp += co; man.append(m)
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(stems)} parcelas, {time.perf_counter() - t0:.0f} s", flush=True)

    def write(name, rows):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write("contexts_ext.csv", ctx)
    write("action_times.csv", act)
    write("component_times.csv", comp)
    write("data_manifest.csv", man)
    notes.append(f"Parcelas procesadas: {len(man)}; errores: {len(errors)}; tiempo total {time.perf_counter() - t0:.0f} s")
    notes += errors
    (out / "run_notes_s01.md").write_text("# S01 — notas de ejecución\n\n" + "\n".join(f"- {n}" for n in notes) + "\n",
                                          encoding="utf-8")
    json.dump(FEATURE_GROUPS, open(out / "feature_groups.json", "w"), indent=1)
    print("\n".join(notes[:4]))


if __name__ == "__main__":
    main()
