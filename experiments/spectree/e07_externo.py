#!/usr/bin/env python3
"""
E07 — External transfer (protocol frozen in e07_protocolo.md, commit 3f31fab).

External set: Zamboni et al. (2021), urban tree crowns, Campo Grande (Brazil), 220 RGB patches of
512 x 512 px at 10 cm, 3382 boxes. Data are read from --ext and never copied into the repository.

Stage 1 (--stage predict): per patch, DeepForest raw detections, Mavic 3M synthesis, v0.5 context,
  the 32 classical RGB-portfolio actions; component timings. -> results/spectree/e07/patches/*.npz + csv
Stage 2 (--stage analyse): score the 40 actions, train every policy on the 189 NEON plots only and
  apply it unchanged to the external patches. -> results/spectree/out/e07_*.csv
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

SENSOR = "dji_mavic3m"
GSD = 0.1
DF_FILTER = 0.3
E07 = Path(__file__).resolve().parents[2] / "results" / "spectree" / "e07"


def read_gt(path: Path) -> tuple[np.ndarray, int]:
    """Boxes xmin ymin xmax ymax (checked against the authors' rendered gt/*.png).
    Degenerate boxes (width or height < 1 px, all on the image border) are dropped and counted."""
    txt = path.read_text().strip()
    if not txt:
        return np.zeros((0, 4), np.float32), 0
    b = np.loadtxt(path, ndmin=2).astype(np.float32)
    ok = ((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1)
    return b[ok], int((~ok).sum())


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def _init():
    import cv2

    cv2.setNumThreads(1)
    import torch

    torch.set_num_threads(1)


def predict_patch(args):
    stem, cfg = args
    from PIL import Image

    from agrispectralsynth.agent.actions import ARMS, make_signals, run_arm
    from agrispectralsynth.agent.deepforest import get_detector
    from agrispectralsynth.agent.features import scene_features
    from agrispectralsynth.indices import NDVI
    from agrispectralsynth.sensors.registry import load_sensor
    from agrispectralsynth.spectral.engine import SpectralEngine
    from s01_predicciones import box_medians

    ext = Path(cfg["ext"])
    rgb = np.asarray(Image.open(ext / "rgb" / f"{stem}.png").convert("RGB"), dtype=np.uint8)
    gt, n_degenerate = read_gt(ext / "bbox_txt" / f"{stem}.txt")
    store, comp, act = {}, [], []

    det = get_detector(cfg["weights"])
    b, s, secs = det.predict(rgb)
    store["df/boxes"], store["df/scores"] = b, s
    comp.append((stem, "deepforest", secs))

    t0 = time.perf_counter()
    eng = SpectralEngine()
    scene = eng.prepare(rgb)
    sensor = load_sensor(SENSOR)
    bands = eng.render(scene, sensor, noise_std=0)
    red, nir = bands[sensor.roles["red"]], bands[sensor.roles["nir"]]
    ndvi = NDVI().compute(red, nir)
    comp.append((stem, "synth", time.perf_counter() - t0))

    sig = make_signals(rgb, ndvi, nir)
    sig.detections = (b, s, secs)
    t0 = time.perf_counter()
    feats = scene_features(sig, scene.veg_fraction, SENSOR, GSD)
    comp.append((stem, "features", time.perf_counter() - t0))
    store["dfmed/syn"] = box_medians(b, ndvi)

    for arm in ARMS:
        if arm.is_detector or arm.signal == "hsi":
            continue
        t0 = time.perf_counter()
        boxes = run_arm(arm, sig, GSD)
        act.append((stem, arm.name, time.perf_counter() - t0, len(boxes)))
        store[f"box/{arm.name}"] = boxes.astype(np.float32)
    store["gt/boxes"] = gt
    np.savez_compressed(Path(cfg["out"]) / "patches" / f"{stem}.npz", **store)
    ctx = {"image": f"ZAMB_{stem}", "site": "ZAMB", "sensor": SENSOR, "n_true": len(gt),
           "n_degenerate_dropped": n_degenerate, "mean_rgb": float(rgb.mean()), **feats}
    return ctx, comp, act


def stage_predict(a):
    out = Path(a.out)
    (out / "patches").mkdir(parents=True, exist_ok=True)
    ext = Path(a.ext)
    stems = sorted((p.stem for p in (ext / "rgb").glob("*.png")), key=int)
    if a.limit:
        stems = stems[: a.limit]
    cfg = {"ext": str(ext), "weights": a.weights, "out": str(out)}
    ctx, comp, act = [], [], []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers, initializer=_init, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, (c, co, ac) in enumerate(pool.map(predict_patch, [(s, cfg) for s in stems])):
            ctx.append(c); comp += co; act += ac
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(stems)} parches, {time.perf_counter() - t0:.0f} s", flush=True)
    pd.DataFrame(ctx).to_csv(out / "contexts.csv", index=False)
    pd.DataFrame(comp, columns=["stem", "component", "seconds"]).to_csv(out / "component_times.csv", index=False)
    pd.DataFrame(act, columns=["stem", "action", "seconds", "n_pred"]).to_csv(out / "action_times.csv", index=False)
    print(f"{len(ctx)} parches en {time.perf_counter() - t0:.0f} s")


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def score_external(out: Path) -> pd.DataFrame:
    from agrispectralsynth.agent.actions import ARMS
    from s02_metricas import box_iou, tp_counts

    at = pd.read_csv(out / "action_times.csv").set_index(["stem", "action"]).seconds
    ct = pd.read_csv(out / "component_times.csv").pivot(index="stem", columns="component", values="seconds")
    rows = []
    for p in sorted((out / "patches").glob("*.npz"), key=lambda q: int(q.stem)):
        z = np.load(p)
        stem, gt = int(p.stem), z["gt/boxes"]
        for arm in ARMS:
            if arm.signal == "hsi":
                continue
            if arm.is_detector:
                keep = z["df/scores"] >= arm.score
                if arm.signal == "ndvi":
                    keep &= z["dfmed/syn"] >= DF_FILTER
                pred, secs = z["df/boxes"][keep], ct.loc[stem, "deepforest"]
            else:
                pred, secs = z[f"box/{arm.name}"], at[(stem, arm.name)]
            for (rule, t), tp in tp_counts(box_iou(pred, gt)).items():
                rows.append((f"ZAMB_{stem}", SENSOR, arm.name, rule, t, tp, len(pred), len(gt), secs))
    df = pd.DataFrame(rows, columns=["image", "sensor", "action", "matching", "iou_thr", "tp", "n_pred", "n_true",
                                     "seconds"])
    df.insert(1, "site", "ZAMB")
    df["fp"] = df.n_pred - df.tp
    df["fn"] = df.n_true - df.tp
    df["f1"] = 2 * df.tp / (df.n_pred + df.n_true).clip(lower=1)
    df["count_err"] = df.n_pred - df.n_true
    df["rel_count_error"] = (df.count_err.abs() / df.n_true.clip(lower=1)).round(6)
    return df


def combined_table(ext_ctx, ext_metrics, features, iou=0.4, matching="hung", kappa=0.25, lam=0.0):
    """NEON rows (training) stacked with external rows (test) in one common.Table."""
    import common as C

    Tn = C.build_table(C.RGB40, features, kappa=kappa, lam=lam, iou=iou, matching=matching)
    m = ext_metrics[(ext_metrics.iou_thr == iou) & (ext_metrics.matching == matching)]
    ctx = ext_ctx[ext_ctx.n_true > 0].reset_index(drop=True)

    def piv(col):
        p = m.pivot_table(index="image", columns="action", values=col, aggfunc="first")
        return p.reindex(ctx.image)[C.RGB40].to_numpy(float)

    F1, CE, NP, TP, SEC = piv("f1"), piv("rel_count_error"), piv("n_pred"), piv("tp"), piv("seconds")
    NT = ctx.n_true.to_numpy(float)
    R = F1 - kappa * np.minimum(1.0, CE) - lam * SEC
    Te = C.Table(ctx[Tn.features].to_numpy(float), R, F1, CE, np.abs(NP - NT[:, None]), NP, TP, SEC, NT,
                 ctx.image.to_numpy(), ctx.site.to_numpy(), ctx.sensor.to_numpy(), Tn.arms, Tn.features)
    T = C.Table(*[np.concatenate([getattr(Tn, k), getattr(Te, k)]) for k in
                  ("X", "R", "F1", "CE", "AE", "NP", "TP", "SEC", "NT", "images", "sites", "sensors")],
                Tn.arms, Tn.features)
    train = np.arange(len(Tn.R))
    test = np.arange(len(Tn.R), len(T.R))
    return T, train, test


def apply_policies(T, train, test, tag):
    import common as C

    rgb_feats = [f for f in C.context_features("C0") if not f.startswith("sensor_")]
    out = {}
    out["SBS"] = (np.full(len(test), int(np.argmax(T.R[train].mean(axis=0)))), {})
    out["VBS-ext"] = (T.R[test].argmax(axis=1), {})
    out["DF-defecto"] = (np.full(len(test), T.arms.index("df_rgb_s0.1")), {})
    out["Ridge"] = (C._ridge(T, train, test), {})
    out["Trees"] = C._trees(T, train, test)
    out["CheapRule"] = C._cheap_rule(T, train, test, rgb_feats)
    for seed in C.SEEDS:
        out[f"LinUCB-F|{seed}"] = C._bandit(T, train, test, "LinUCB-F", seed)
    # Selector-VI: the learned selector with the best site-grouped inner validation on NEON (as in E03/P2)
    scores = {}
    for pol in ["CheapRule", "LinUCB", "LinUCB-F", "LinTS", "Ridge", "Trees"]:
        vals = []
        for inner in C._inner_folds(T, train):
            tr = np.setdiff1d(train, inner)
            if pol == "Ridge":
                ch = C._ridge(T, tr, inner)
            elif pol == "Trees":
                ch, _ = C._trees(T, tr, inner, inner=False)
            elif pol == "CheapRule":
                ch, _ = C._cheap_rule(T, tr, inner, rgb_feats)
            else:
                ch, _ = C._bandit(T, tr, inner, pol, 0)
            vals.append(T.R[inner, ch].mean())
        scores[pol] = float(np.mean(vals))
    best = max(scores, key=scores.get)
    if best in ("Ridge", "Trees", "CheapRule"):
        ch = out[best][0]
        for seed in C.SEEDS:
            out[f"Selector-VI|{seed}"] = (ch, {"elegido": best})
    else:
        for seed in C.SEEDS:
            out[f"Selector-VI|{seed}"] = (C._bandit(T, train, test, best, seed)[0], {"elegido": best})
    rows = []
    for key, (ch, meta) in out.items():
        pol, seed = (key.split("|") + ["-1"])[:2]
        for i, a in zip(test, ch):
            rows.append((tag, pol if tag == "V05" or pol in ("VBS-ext", "DF-defecto", "SBS") else f"{pol}·{tag}",
                         int(seed), T.images[i], T.arms[a], T.R[i, a], T.F1[i, a], T.CE[i, a], T.AE[i, a],
                         T.NP[i, a], T.TP[i, a], T.NT[i], T.SEC[i, a], json.dumps(meta, default=str)))
    cols = ["contexto", "policy", "seed", "image", "action", "reward", "f1", "rel_count_error", "abs_count_error",
            "n_pred", "tp", "n_true", "seconds", "meta"]
    return pd.DataFrame(rows, columns=cols), scores


def boot_diff(d: np.ndarray, B=5000, seed=12345, block=None):
    rng = np.random.default_rng(seed)
    if block is None:
        reps = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    else:
        blocks = [d[i:i + block] for i in range(0, len(d), block)]
        reps = np.array([np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]).mean()
                         for _ in range(B)])
    return float(d.mean()), float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def stage_analyse(a):
    import common as C
    from agrispectralsynth.agent.actions import ARMS

    out = Path(a.out)
    ext_ctx = pd.read_csv(out / "contexts.csv")
    met = score_external(out)
    met.to_csv(out / "action_metrics_ext.csv", index=False)
    ct = pd.read_csv(out / "component_times.csv").pivot(index="stem", columns="component", values="seconds")
    overhead = (ct.synth + ct.features).rename(lambda s: f"ZAMB_{s}")

    decs, iv = [], {}
    for feats in ("V05", "C0"):
        T, tr, te = combined_table(ext_ctx, met, feats)
        d, sc = apply_policies(T, tr, te, feats)
        iv[feats] = sc
        if feats == "C0":
            d = d[d.policy.isin(["Ridge·C0", "Trees·C0", "LinUCB-F·C0"])]
        decs.append(d)
    dec = pd.concat(decs, ignore_index=True)
    dec["seconds_total"] = dec.seconds + dec.image.map(overhead).to_numpy()
    dec.to_csv(C.OUT / "e07_decisiones.csv", index=False)

    # ---- per-policy summary (mean over seeds within patch, then over patches) ----
    pp = dec.groupby(["policy", "image"], as_index=False)[["reward", "f1", "abs_count_error", "rel_count_error",
                                                           "seconds_total", "n_pred", "n_true", "tp"]].mean()
    neon = pd.read_csv(C.OUT / "e03_resumen.csv").set_index("policy")
    neon_iv = pd.read_csv(C.OUT / "e03_resumen.csv")
    rows = []
    sbs = pp[pp.policy == "SBS"].set_index("image").reward
    for pol, g in pp.groupby("policy"):
        g = g.set_index("image")
        est, lo, hi = boot_diff((g.reward - sbs.reindex(g.index)).to_numpy()) if pol != "SBS" else (np.nan,) * 3
        _, blo, bhi = boot_diff((g.reward - sbs.reindex(g.index)).to_numpy(), block=10) if pol != "SBS" else (np.nan,) * 3
        sums = dec[dec.policy == pol][["tp", "n_pred", "n_true"]].sum()
        base = pol.split("·")[0]
        rows.append({"policy": pol, "reward": g.reward.mean(), "f1_media": g.f1.mean(),
                     "f1_micro": 2 * sums.tp / (sums.n_pred + sums.n_true),
                     "count_mae": g.abs_count_error.mean(), "count_rmse": float(np.sqrt((g.abs_count_error ** 2).mean())),
                     "inventory_err": (g.n_pred.sum() - g.n_true.sum()) / g.n_true.sum(),
                     "seconds": g.seconds_total.mean(), "d_vs_SBS": est, "ci_lo": lo, "ci_hi": hi,
                     "ci_lo_bloques": blo, "ci_hi_bloques": bhi,
                     "neon_reward_macro": neon.reward_macro.get(base, np.nan) if pol == base else np.nan,
                     "neon_reward_micro": neon.reward_micro.get(base, np.nan) if pol == base else np.nan})
    s = pd.DataFrame(rows).sort_values("reward", ascending=False)
    vbs, sb = s.set_index("policy").reward["VBS-ext"], s.set_index("policy").reward["SBS"]
    s["gap_closed"] = (s.reward - sb) / (vbs - sb)
    s.to_csv(C.OUT / "e07_resumen.csv", index=False)

    # ---- primary comparison verdict ----
    r = s.set_index("policy").loc["Selector-VI"]
    verdict = ("superior" if r.ci_lo > 0.02 else "no inferior" if r.ci_lo > -0.02 else
               "inferior" if r.ci_hi < -0.02 else "inconcluso")
    elegido = json.loads(dec[dec.policy == "Selector-VI"].meta.iloc[0])["elegido"]
    pd.DataFrame([{"comparacion": "Selector-VI − SBS (externo)", "elegido_en_VI": elegido, "estimacion": r.d_vs_SBS,
                   "ic95_lo": r.ci_lo, "ic95_hi": r.ci_hi, "ic95_bloques_lo": r.ci_lo_bloques,
                   "ic95_bloques_hi": r.ci_hi_bloques, "margen": 0.02, "veredicto": verdict}]
                 ).to_csv(C.OUT / "e07_primaria.csv", index=False)
    pd.DataFrame([{"contexto": k, **v} for k, v in iv.items()]).to_csv(C.OUT / "e07_validacion_interna.csv", index=False)

    # ---- every fixed action on the external set vs NEON (Mavic 3M, IoU 0.4, hung) ----
    fam = {x.name: C.family(x.name) for x in ARMS}
    m4 = met[(met.iou_thr == 0.4) & (met.matching == "hung") & (met.n_true > 0)].copy()
    m4["reward"] = m4.f1 - 0.25 * np.minimum(1, m4.rel_count_error)
    ext_a = m4.groupby("action").agg(reward_ext=("reward", "mean"), f1_ext=("f1", "mean"),
                                     sesgo_ext=("count_err", "mean"))
    nm = C.metrics(0.4, "hung")
    nm = nm[(nm.sensor == SENSOR) & nm.action.isin(C.RGB40)].copy()
    nm["reward"] = nm.f1 - 0.25 * np.minimum(1, nm.rel_count_error)
    neon_a = nm.groupby(["action", "site"]).reward.mean().groupby("action").mean().rename("reward_neon_macro")
    acc = ext_a.join(neon_a)
    acc["familia"] = acc.index.map(fam)
    acc["rank_ext"] = acc.reward_ext.rank(ascending=False)
    acc["rank_neon"] = acc.reward_neon_macro.rank(ascending=False)
    acc.sort_values("reward_ext", ascending=False).to_csv(C.OUT / "e07_acciones.csv")

    # ---- sensitivity to IoU / matching for the fixed and deterministic policies ----
    sens = []
    for iou in (0.3, 0.4, 0.5):
        for rule in ("hung", "card"):
            T, tr, te = combined_table(ext_ctx, met, "V05", iou=iou, matching=rule)
            for name, ch in (("SBS", np.full(len(te), int(np.argmax(T.R[tr].mean(axis=0))))),
                             ("DF-defecto", np.full(len(te), T.arms.index("df_rgb_s0.1"))),
                             ("Ridge", C._ridge(T, tr, te)), ("VBS-ext", T.R[te].argmax(axis=1))):
                sens.append({"iou": iou, "matching": rule, "policy": name,
                             "reward": float(T.R[te, ch].mean()), "f1": float(T.F1[te, ch].mean())})
    pd.DataFrame(sens).to_csv(C.OUT / "e07_sensibilidad_iou.csv", index=False)

    # ---- density terciles, empty patch, authors' split 0 ----
    pp2 = pp.merge(ext_ctx[["image", "n_true"]].rename(columns={"n_true": "nt"}), on="image")
    pp2["tercil"] = pd.qcut(pp2.nt, 3, labels=["baja", "media", "alta"])
    pp2.groupby(["tercil", "policy"], observed=True).reward.mean().unstack().round(4).to_csv(C.OUT / "e07_estratos.csv")
    ext_root = Path(a.ext)
    test0 = {f"ZAMB_{l.strip()[:-4]}" for l in (ext_root / "img_list" / "0" / "test.txt").read_text().split() if l.strip()}
    pp[pp.image.isin(test0)].groupby("policy")[["reward", "f1", "abs_count_error"]].mean().to_csv(
        C.OUT / "e07_particion_autores0.csv")
    empty = ext_ctx[ext_ctx.n_true == 0].image.tolist()
    fp_empty = []
    for im in empty:
        g = met[(met.image == im) & (met.iou_thr == 0.4) & (met.matching == "hung")].set_index("action").n_pred
        sbs_a = dec[dec.policy == "SBS"].action.iloc[0]
        fp_empty.append({"image": im, "fp_SBS": int(g[sbs_a]), "fp_DF-defecto": int(g["df_rgb_s0.1"]),
                         "fp_min_accion": int(g[C.RGB40].min()), "fp_max_accion": int(g[C.RGB40].max())})
    pd.DataFrame(fp_empty).to_csv(C.OUT / "e07_parches_vacios.csv", index=False)
    pd.DataFrame([{"n_parches": len(ext_ctx), "n_vacios": len(empty), "n_cajas": int(ext_ctx.n_true.sum()),
                   "cajas_degeneradas_descartadas": int(ext_ctx.n_degenerate_dropped.sum()),
                   "n_particion_autores0": len(test0)}]).to_csv(C.OUT / "e07_datos.csv", index=False)
    print(s.round(3).to_string(index=False))
    print(verdict, elegido)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["predict", "analyse"], required=True)
    ap.add_argument("--ext", required=True, help="checkout of pedrozamboni/individual_urban_tree_crown_detection")
    ap.add_argument("--weights")
    ap.add_argument("--out", default=str(E07))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    stage_predict(a) if a.stage == "predict" else stage_analyse(a)


if __name__ == "__main__":
    main()
