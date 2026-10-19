#!/usr/bin/env python3
"""
E05 — Does the system change behaviour in the conditions that motivate the paper?

Difficulty descriptors per plot, computed from the ANNOTATIONS (analysis only; never fed to a policy):
  densidad     annotated trees per hectare
  tamano       median annotated box area (m2)
  proximidad   median nearest-neighbour centre distance / mean diameter of the two boxes (low = crowded)
  solapamiento share of annotated boxes overlapping another annotated box (IoU > 0)  [geometric proxy,
               not a measurement of crown interlocking]
  sombra*      share of pixels inside annotated boxes below the plot's Otsu luminance threshold
               [UNVALIDATED proxy: low luminance is not always shadow; no shadow annotation exists]
Strata = terciles computed on the TRAINING plots of each outer fold and applied to its test plots.
Uses the E03 decisions (same folds, context v0.5, RGB 40 actions).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

BENCH = Path("/tmp/claude-0/-home-claude/e24bc419-cf00-53d3-96c8-274fdd70d0b3/scratchpad/nte")
POLS = ["SBS", "VBS", "LinUCB-F", "Ridge", "Trees", "CheapRule", "Selector-VI"]
DESC = ["densidad", "tamano", "proximidad", "solapamiento", "sombra_proxy"]


def descriptors():
    from agrispectralsynth.agent.actions import luminance, otsu_threshold
    from agrispectralsynth.pipeline import read_rgb
    from s02_metricas import box_iou

    man = pd.read_csv(C.S01 / "data_manifest.csv").set_index("plot")
    rows = []
    for p in sorted((C.S01 / "plots").glob("*.npz")):
        g = np.load(p)["gt/boxes"]
        stem = p.stem
        gsd = 0.1
        area = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) * gsd ** 2
        c = np.c_[(g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2] * gsd
        diam = np.sqrt(area)
        if len(g) > 1:
            dist = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
            np.fill_diagonal(dist, np.inf)
            j = dist.argmin(axis=1)
            prox = float(np.median(dist[np.arange(len(g)), j] / ((diam + diam[j]) / 2)))
            iou = box_iou(g, g)
            np.fill_diagonal(iou, 0)
            ov = float((iou.max(axis=1) > 0).mean())
        else:
            prox, ov = np.nan, 0.0
        rgb, _ = read_rgb(BENCH / "evaluation" / "RGB" / f"{stem}.tif")
        lum = luminance(rgb)
        thr = otsu_threshold(lum)
        mask = np.zeros(lum.shape, bool)
        for x0, y0, x1, y1 in np.round(g).astype(int):
            mask[max(0, y0):y1, max(0, x0):x1] = True
        sh = float((lum[mask] < thr).mean()) if mask.any() else np.nan
        rows.append({"image": stem, "densidad": len(g) / (man.loc[stem, "area_m2"] / 1e4),
                     "tamano": float(np.median(area)), "proximidad": prox, "solapamiento": ov, "sombra_proxy": sh})
    return pd.DataFrame(rows).set_index("image")


def main():
    D = descriptors()
    D.to_csv(C.OUT / "e05_descriptores.csv")
    dec = pd.read_csv(C.OUT / "e03_decisiones.csv")
    dec = dec[dec.policy.isin(POLS)]
    dec["fp"] = dec.n_pred - dec.tp
    dec["fn"] = dec.n_true - dec.tp
    folds = C.site_folds(C.contexts().groupby("image").site.first().reindex(sorted(D.index)).to_numpy(), C.K)
    images = np.array(sorted(D.index))
    lab = {}
    for f, test in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(images)), test)
        for d in DESC:
            q = np.nanquantile(D.loc[images[tr], d], [1 / 3, 2 / 3])
            for im in images[test]:
                v = D.loc[im, d]
                lab[(im, d)] = "sin dato" if np.isnan(v) else ("bajo" if v <= q[0] else ("medio" if v <= q[1] else "alto"))
    rows = []
    for d in DESC:
        dec["estrato"] = [lab[(im, d)] for im in dec.image]
        for est, g in dec.groupby("estrato"):
            mac = C.macro(g)
            f1 = C.macro(g, "f1")
            pp = g.groupby(["policy", "image"])[["fp", "fn", "abs_count_error"]].mean().groupby("policy").mean()
            inv = g.groupby("policy").apply(lambda x: (x.n_pred.sum() - x.n_true.sum()) / x.n_true.sum(), include_groups=False)
            fam = g.assign(df=g.action.str.startswith("df_")).groupby("policy").df.mean()
            for pol in mac.index:
                rows.append({"descriptor": d, "estrato": est, "policy": pol, "sitios": g.site.nunique(),
                             "parcelas": g.image.nunique(), "recompensa": mac[pol], "f1": f1[pol],
                             "perdida_vs_SBS": mac[pol] - mac["SBS"], "omisiones": pp.loc[pol, "fn"],
                             "falsas": pp.loc[pol, "fp"], "mae_conteo": pp.loc[pol, "abs_count_error"],
                             "inventario": inv[pol], "frac_DeepForest": fam[pol]})
    s = pd.DataFrame(rows)
    s.to_csv(C.OUT / "e05_estratos.csv", index=False)
    print(C.fmt(s[s.policy.isin(["SBS", "LinUCB-F", "Ridge", "VBS"])]).to_string(index=False))

    # within-site (SJER, TEAK): density terciles inside the site, to avoid confounding density with site identity
    w = []
    for site in ("SJER", "TEAK"):
        g = dec[dec.site == site].copy()
        q = D.loc[g.image.unique(), "densidad"].quantile([1 / 3, 2 / 3]).to_numpy()
        g["estrato"] = pd.cut(g.image.map(D.densidad), [-np.inf, q[0], q[1], np.inf], labels=["bajo", "medio", "alto"])
        for est, gg in g.groupby("estrato", observed=True):
            pp = gg.groupby(["policy", "image"]).reward.mean().groupby("policy").mean()
            f1 = gg.groupby(["policy", "image"]).f1.mean().groupby("policy").mean()
            for pol in pp.index:
                w.append({"sitio": site, "estrato_densidad": est, "policy": pol, "parcelas": gg.image.nunique(),
                          "recompensa": pp[pol], "f1": f1[pol], "perdida_vs_SBS": pp[pol] - pp["SBS"]})
    w = pd.DataFrame(w)
    w.to_csv(C.OUT / "e05_dentro_de_sitio.csv", index=False)
    print(C.fmt(w[w.policy.isin(["SBS", "LinUCB-F", "VBS"])]).to_string(index=False))

    # examples chosen by an explicit rule: median-loss and largest-loss plot of LinUCB-F vs SBS
    pp = dec[dec.policy.isin(["LinUCB-F", "SBS"])].groupby(["policy", "image", "site"]).reward.mean().unstack("policy")
    pp["perdida"] = pp["LinUCB-F"] - pp["SBS"]
    pp = pp.sort_values("perdida")
    ex = pd.concat([pp.head(3), pp.iloc[[len(pp) // 2]]])
    ex.to_csv(C.OUT / "e05_ejemplos.csv")
    print(ex.round(3).to_string())


if __name__ == "__main__":
    main()
