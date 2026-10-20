#!/usr/bin/env python3
"""
E02 — Separate radiometric calibration from the usefulness of the NDVI filter.

The same DeepForest boxes and scores (stored before any filter) are used for every variant:
  F0  no filter                                  (score threshold s tuned on training)
  F1  synthetic NDVI >= 0.3 (original)           (s tuned)
  F2  synthetic NDVI >= t, t tuned on training   (s, t tuned)
  F3a calibrated synthetic NDVI >= 0.3           (linear map fitted on TRAINING-site 1 m pixel pairs; s tuned)
  F3b calibrated synthetic NDVI >= t, t tuned    (s, t tuned)
  F4a real NDVI >= 0.3   (reference, extra modality)
  F4b real NDVI >= t, t tuned
Tuning criterion: mean F1 on the training plots (site-grouped outer folds). Grid t in [-0.2, 0.8] step 0.05
plus "no filter". Nothing is tuned on the test sites.
Also: share of original TP / FP discarded (pre-filter matching), box-level discrimination (AUC),
radiometric error of red / NIR / NDVI (bias, MAE, RMSE) before and after calibration.
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from s02_metricas import box_iou  # noqa: E402

SCORES = (0.1, 0.2, 0.3, 0.4)
GRID = [-np.inf] + list(np.round(np.arange(-0.2, 0.8001, 0.05), 2))
TAU = 0.4
DRONES = ["dji_mavic3m", "dji_phantom4m", "micasense_rededge_mx", "parrot_sequoia_plus"]


class Plot:
    def __init__(self, path):
        z = np.load(path)
        self.stem = Path(path).stem
        self.boxes, self.scores, gt = z["df/boxes"], z["df/scores"], z["gt/boxes"]
        self.n_true = len(gt)
        self.iou = box_iou(self.boxes, gt)
        self.med = {(s, k): z[f"dfmed/{s}/{k}"] for s in DRONES for k in ("syn", "real")}
        self.px = {s: {k: z[f"px/{s}/{k}"] for k in ("V0", "real_ndvi", "valid", "V0_red", "V0_nir", "real_red",
                                                      "real_nir")} for s in DRONES}

    def score(self, keep):
        n = int(keep.sum())
        if n == 0:
            return 0, 0
        sub = self.iou[keep]
        r, c = linear_sum_assignment(-sub)
        return int((sub[r, c] >= TAU).sum()), n

    def tp_flags(self, s):
        keep = self.scores >= s
        idx = np.flatnonzero(keep)
        flags = np.zeros(len(self.scores), bool)
        if len(idx):
            sub = self.iou[idx]
            r, c = linear_sum_assignment(-sub)
            flags[idx[r[sub[r, c] >= TAU]]] = True
        return keep, flags


def main():
    plots = {Path(p).stem: Plot(p) for p in sorted((C.S01 / "plots").glob("*.npz"))}
    ctx = C.contexts()
    sites = ctx.groupby("image").site.first()
    images = np.array(sorted(plots))
    site_of = sites.reindex(images).to_numpy()
    folds = C.site_folds(site_of, C.K)

    @lru_cache(maxsize=None)
    def metrics_for(stem, sid, s, idx, thr):
        P = plots[stem]
        keep = P.scores >= s
        if idx != "none" and np.isfinite(thr):
            keep &= P.med[(sid, idx)] >= thr
        tp, n = P.score(keep)
        f1 = 2 * tp / max(n + P.n_true, 1)
        return f1, tp, n, abs(n - P.n_true)

    def mean_f1(stems, s, idx, thr):
        return np.mean([metrics_for(st, sid, s, idx, thr)[0] for st in stems for sid in DRONES])

    def calib(stems, sid):
        x = np.concatenate([plots[st].px[sid]["V0"][plots[st].px[sid]["valid"]] for st in stems])
        y = np.concatenate([plots[st].px[sid]["real_ndvi"][plots[st].px[sid]["valid"]] for st in stems])
        b, a = np.polyfit(x, y, 1)
        return a, b

    rows, cal_rows, chosen = [], [], []
    for f, test_idx in enumerate(folds):
        test = list(images[test_idx])
        train = [s for s in images if s not in set(test)]
        cal = {sid: calib(train, sid) for sid in DRONES}
        for sid in DRONES:
            cal_rows.append({"fold": f, "sensor": sid, "a": cal[sid][0], "b": cal[sid][1]})
        variants = {}
        variants["F0 sin filtro"] = max(((s, "none", -np.inf) for s in SCORES), key=lambda v: mean_f1(train, *v))
        variants["F1 sintético 0.3"] = max(((s, "syn", 0.3) for s in SCORES), key=lambda v: mean_f1(train, *v))
        variants["F2 sintético umbral ajustado"] = max(((s, "syn", t) for s in SCORES for t in GRID),
                                                       key=lambda v: mean_f1(train, *v))
        variants["F4a real 0.3"] = max(((s, "real", 0.3) for s in SCORES), key=lambda v: mean_f1(train, *v))
        variants["F4b real umbral ajustado"] = max(((s, "real", t) for s in SCORES for t in GRID),
                                                   key=lambda v: mean_f1(train, *v))
        for name, (s, idx, thr) in variants.items():
            chosen.append({"fold": f, "variante": name, "s": s, "indice": idx, "umbral": thr})
            for st in test:
                for sid in DRONES:
                    rows.append((f, name, st, sid, s, thr, *metrics_for(st, sid, s, idx, thr), plots[st].n_true))
        # calibrated variants: threshold on the calibrated scale t_c <=> syn threshold (t_c - a) / b (b > 0)
        for name, tune in (("F3a calibrado 0.3", False), ("F3b calibrado umbral ajustado", True)):
            grid_c = GRID if tune else [0.3]
            best, bestv = None, -1
            for s in SCORES:
                for tc in grid_c:
                    v = np.mean([metrics_for(st, sid, s, "syn", (tc - cal[sid][0]) / cal[sid][1] if np.isfinite(tc) else tc)[0]
                                 for st in train for sid in DRONES])
                    if v > bestv:
                        best, bestv = (s, tc), v
            s, tc = best
            chosen.append({"fold": f, "variante": name, "s": s, "indice": "syn calibrado", "umbral": tc})
            for st in test:
                for sid in DRONES:
                    t_syn = (tc - cal[sid][0]) / cal[sid][1] if np.isfinite(tc) else tc
                    rows.append((f, name, st, sid, s, tc, *metrics_for(st, sid, s, "syn", t_syn), plots[st].n_true))
        print("fold", f, flush=True)

    d = pd.DataFrame(rows, columns=["fold", "variante", "image", "sensor", "s", "umbral", "f1", "tp", "n_pred",
                                    "abs_count_error", "n_true"])
    d["site"] = d.image.map(sites)
    d["recall"] = d.tp / d.n_true
    d["precision"] = d.tp / d.n_pred.clip(lower=1)
    d.to_csv(C.OUT / "e02_detalle.csv", index=False)
    pd.DataFrame(chosen).to_csv(C.OUT / "e02_umbrales_elegidos.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(C.OUT / "e02_calibracion.csv", index=False)

    # summary with paired site bootstrap vs F0 (P3: F3b vs F0 on F1, Bonferroni level)
    dd = d.rename(columns={"variante": "policy"})
    dd["reward"] = dd.f1
    res = []
    for v in dd.policy.unique():
        pp = dd[dd.policy == v].groupby(["image", "site"])[["f1", "recall", "precision", "abs_count_error"]].mean()
        mac = pp.groupby("site").mean().mean()
        est = lo = hi = np.nan
        if v != "F0 sin filtro":
            lvl = 1 - 0.05 / 4 if v.startswith("F3b") else 0.95
            est, lo, hi, _ = C.paired_boot(dd, v, "F0 sin filtro", col="f1", level=lvl)
        g = dd[dd.policy == v]
        res.append({"variante": v, "f1_macro": mac.f1, "f1_micro": 2 * g.tp.sum() / (g.n_pred.sum() + g.n_true.sum()),
                    "recall": mac.recall, "precision": mac.precision, "mae_conteo": pp.abs_count_error.mean(),
                    "dF1_vs_F0": est, "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(res)
    res.to_csv(C.OUT / "e02_resumen.csv", index=False)
    print(C.fmt(res).to_string(index=False))
    print(pd.DataFrame(chosen).to_string(index=False))

    # attribution of rejections: share of original TP and FP discarded (pre-filter matching), s = 0.3
    att = []
    for idx in ("syn", "real"):
        for thr in (0.0, 0.2, 0.3, 0.4, 0.5):
            tp_rm = tp_all = fp_rm = fp_all = 0
            for st, P in plots.items():
                keep, flags = P.tp_flags(0.3)
                for sid in DRONES:
                    rem = keep & (P.med[(sid, idx)] < thr)
                    tp_rm += (rem & flags).sum(); tp_all += (keep & flags).sum()
                    fp_rm += (rem & ~flags).sum(); fp_all += (keep & ~flags).sum()
            att.append({"indice": idx, "umbral": thr, "frac_TP_descartados": tp_rm / tp_all,
                        "frac_FP_descartados": fp_rm / fp_all})
    att = pd.DataFrame(att)
    # box-level discrimination: AUC of the median index separating TP from FP boxes (s = 0.1, all boxes)
    from sklearn.metrics import roc_auc_score

    auc = []
    for idx in ("syn", "real"):
        y, x = [], []
        for st, P in plots.items():
            keep, flags = P.tp_flags(0.1)
            for sid in DRONES:
                y.append(flags[keep]); x.append(P.med[(sid, idx)][keep])
        y, x = np.concatenate(y), np.concatenate(x)
        auc.append({"indice": idx, "auc_TP_vs_FP": roc_auc_score(y, x), "n_cajas": len(y), "frac_TP": y.mean()})
    att.to_csv(C.OUT / "e02_atribucion.csv", index=False)
    pd.DataFrame(auc).to_csv(C.OUT / "e02_auc.csv", index=False)
    print(C.fmt(att).to_string(index=False)); print(pd.DataFrame(auc).round(3).to_string(index=False))

    # curve TP retained vs FP rejected (s = 0.3) for synthetic and real NDVI
    curve = []
    for idx in ("syn", "real"):
        for thr in np.round(np.arange(-0.2, 0.95, 0.05), 2):
            tp_k = tp_all = fp_r = fp_all = 0
            for st, P in plots.items():
                keep, flags = P.tp_flags(0.3)
                for sid in DRONES:
                    k2 = keep & (P.med[(sid, idx)] >= thr)
                    tp_k += (k2 & flags).sum(); tp_all += (keep & flags).sum()
                    fp_r += (keep & ~k2 & ~flags).sum(); fp_all += (keep & ~flags).sum()
            curve.append({"indice": idx, "umbral": thr, "TP_retenidos": tp_k / tp_all, "FP_rechazados": fp_r / fp_all})
    pd.DataFrame(curve).to_csv(C.OUT / "e02_curva.csv", index=False)

    # radiometric error per plot (red, NIR, NDVI) before / after out-of-fold calibration; common valid mask
    err = []
    cal_df = pd.DataFrame(cal_rows)
    for f, test_idx in enumerate(folds):
        for st in images[test_idx]:
            P = plots[st]
            for sid in DRONES:
                px = P.px[sid]
                v = px["valid"]
                a, b = cal_df[(cal_df.fold == f) & (cal_df.sensor == sid)][["a", "b"]].iloc[0]
                pairs = {"rojo": (px["V0_red"], px["real_red"]), "NIR": (px["V0_nir"], px["real_nir"]),
                         "NDVI": (px["V0"], px["real_ndvi"]), "NDVI calibrado": (a + b * px["V0"], px["real_ndvi"])}
                for k, (sv, rv) in pairs.items():
                    e = sv[v] - rv[v]
                    err.append({"image": st, "site": sites[st], "sensor": sid, "magnitud": k, "sesgo": e.mean(),
                                "mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean())})
    err = pd.DataFrame(err)
    agg = err.groupby(["magnitud", "image", "site"])[["sesgo", "mae", "rmse"]].mean().groupby(["magnitud", "site"]).mean()
    out = agg.groupby("magnitud").mean()
    err.to_csv(C.OUT / "e02_error_radiometrico_detalle.csv", index=False)
    out.to_csv(C.OUT / "e02_error_radiometrico.csv")
    agg.to_csv(C.OUT / "e02_error_radiometrico_por_sitio.csv")
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
