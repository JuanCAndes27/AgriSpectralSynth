#!/usr/bin/env python3
"""
SpecTree S02 — score every stored prediction (action_metrics.csv).

For each plot x sensor x action (58) x IoU threshold {0.3, 0.4, 0.5, 0.6} x matching rule:
  hung  : Hungarian assignment maximising total IoU, then pairs with IoU >= tau are TP (v0.5 rule)
  card  : maximum number of pairs with IoU >= tau (ties broken by IoU)
DeepForest actions are rebuilt from the raw (pre-filter) detections + per-box NDVI medians.
"""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

IOUS = (0.3, 0.4, 0.5, 0.6)
DF_FILTER = 0.3
DRONES = ["dji_mavic3m", "dji_phantom4m", "micasense_rededge_mx", "parrot_sequoia_plus"]


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def tp_counts(iou):
    """{(rule, tau): TP}."""
    out = {}
    if iou.size == 0:
        return {(r, t): 0 for r in ("hung", "card") for t in IOUS}
    r, c = linear_sum_assignment(-iou)
    m = iou[r, c]
    for t in IOUS:
        out[("hung", t)] = int((m >= t).sum())
        w = np.where(iou >= t, 1000.0 + iou, 0.0)
        r2, c2 = linear_sum_assignment(-w)
        out[("card", t)] = int((iou[r2, c2] >= t).sum())
    return out


def action_boxes(z, sid, arm):
    if arm.is_detector:
        b, s = z["df/boxes"], z["df/scores"]
        keep = s >= arm.score
        if arm.signal == "ndvi":
            keep &= z[f"dfmed/{sid}/syn"] >= DF_FILTER
        elif arm.signal == "hsi":
            keep &= z[f"dfmed/{sid}/real"] >= DF_FILTER
        return b[keep]
    key = f"box/{sid}/{arm.name}" if arm.sensor_dependent else f"box/*/{arm.name}"
    return z[key]


def score_plot(args):
    path, = args
    from agrispectralsynth.agent.actions import ARMS

    z = np.load(path)
    gt = z["gt/boxes"]
    stem = Path(path).stem
    rows = []
    cache = {}
    for sid in DRONES:
        for arm in ARMS:
            pred = action_boxes(z, sid, arm)
            key = (arm.name, "*" if not arm.sensor_dependent else sid)
            if key not in cache:
                cache[key] = (len(pred), tp_counts(box_iou(pred, gt)))
            n_pred, tps = cache[key]
            for (rule, t), tp in tps.items():
                rows.append((stem, sid, arm.name, rule, t, tp, n_pred, len(gt)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s01", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    s01 = Path(a.s01)
    paths = sorted((s01 / "plots").glob("*.npz"))
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = [r for rs in pool.map(score_plot, [(str(p),) for p in paths], chunksize=4) for r in rs]
    df = pd.DataFrame(rows, columns=["image", "sensor", "action", "matching", "iou_thr", "tp", "n_pred", "n_true"])
    df["fp"] = df.n_pred - df.tp
    df["fn"] = df.n_true - df.tp
    df["precision"] = np.where(df.n_pred > 0, df.tp / df.n_pred.clip(lower=1), 0.0)
    df["recall"] = df.tp / df.n_true.clip(lower=1)
    df["f1"] = 2 * df.tp / (df.n_pred + df.n_true).clip(lower=1)
    df["count_err"] = df.n_pred - df.n_true
    df["rel_count_error"] = (df.count_err.abs() / df.n_true.clip(lower=1)).round(6)

    # action time: classical from the measured run; DeepForest = one inference per plot
    at = pd.read_csv(s01 / "action_times.csv")
    ct = pd.read_csv(s01 / "component_times.csv")
    from agrispectralsynth.agent.actions import ARMS

    sens_dep = {a.name: a.sensor_dependent for a in ARMS}
    det = {a.name for a in ARMS if a.is_detector}
    t_cls = at.set_index(["image", "sensor", "action"]).seconds
    t_df = ct[ct.component == "deepforest"].set_index("image").seconds
    secs = []
    for im, sid, act in zip(df.image, df.sensor, df.action):
        if act in det:
            secs.append(t_df[im])
        else:
            secs.append(t_cls[(im, sid if sens_dep[act] else "*", act)])
    df["seconds"] = np.round(secs, 5)
    from agrispectralsynth.agent.groundtruth import site_of

    df.insert(1, "site", df.image.map(site_of))
    df.to_csv(s01.parent / "action_metrics.csv", index=False)
    print(len(df), "filas ->", s01.parent / "action_metrics.csv")


if __name__ == "__main__":
    main()
