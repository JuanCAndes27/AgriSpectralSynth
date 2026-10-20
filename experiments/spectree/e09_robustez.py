#!/usr/bin/env python3
"""
E09 — Robustness of detection/count metrics and empty scenes.

  * IoU in {0.3, 0.4, 0.5, 0.6} x matching in {hung (v0.5: Hungarian on total IoU, then threshold),
    card (maximum number of pairs with IoU >= tau)}
  * for fixed methods (DeepForest s = 0.1 / 0.3, best classical RGB, best classical synthetic NDVI)
    and for the decisions of the E00 selectors (made at IoU 0.4) re-scored at every threshold
  * count error per plot: MAE and RMSE (trees), aggregated bias, false detections per ha
  * empty scenes: reported only if the benchmark has them (it does not invent cases)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

FIXED = ["df_rgb_s0.3", "df_rgb_s0.1", "wsd_exg-otsu_4m", "wsd_ndvi-otsu_4m", "lmw_exg-otsu_4m"]


def main():
    allm = pd.read_csv(C.ROOT / "action_metrics.csv")
    man = pd.read_csv(C.S01 / "data_manifest.csv").set_index("plot")
    area_ha = allm.image.map(man.area_m2) / 1e4
    allm["fp_ha"] = allm.fp / area_ha
    rows = []
    for (rule, iou), g in allm.groupby(["matching", "iou_thr"]):
        for a in FIXED:
            x = g[g.action == a]
            pp = x.groupby(["image", "site"]).agg(f1=("f1", "mean"), ae=("count_err", lambda v: np.abs(v).mean()),
                                                  ce=("count_err", "mean"), fp_ha=("fp_ha", "mean")).reset_index()
            rows.append({"emparejamiento": rule, "iou": iou, "metodo": a,
                         "f1_macro_sitio": pp.groupby("site").f1.mean().mean(),
                         "f1_micro": 2 * x.tp.sum() / (x.n_pred.sum() + x.n_true.sum()),
                         "mae_conteo": pp.ae.mean(), "rmse_conteo": float(np.sqrt((pp.ce ** 2).mean())),
                         "sesgo_inventario": (x.n_pred.sum() - x.n_true.sum()) / x.n_true.sum(),
                         "fp_por_ha": pp.fp_ha.mean()})
    fx = pd.DataFrame(rows)
    fx.to_csv(C.OUT / "e09_metodos_fijos.csv", index=False)
    print(C.fmt(fx).to_string(index=False))

    # selector decisions (E00, made at IoU 0.4 / hung) re-scored at other thresholds
    dec = pd.read_csv(C.ROOT / "policy_decisions.csv")
    dec = dec[dec.tag == "E00-rgb40"]
    rows = []
    for (rule, iou), g in allm.groupby(["matching", "iou_thr"]):
        key = g.set_index(["image", "sensor", "action"])
        m = dec.join(key[["f1", "tp", "n_pred"]], on=["image", "sensor", "action"], rsuffix="_new")
        m["f1"] = m["f1_new"]
        for pol, gg in m.groupby("policy"):
            pp = gg.groupby(["image", "site"]).f1.mean().reset_index()
            rows.append({"emparejamiento": rule, "iou": iou, "policy": pol,
                         "f1_macro_sitio": pp.groupby("site").f1.mean().mean()})
        # SBS recomputed AT this threshold (best fixed action on training, by F1 - 0.25 min(1,e))
        T = C.build_table(C.RGB40, "V05", iou=iou, matching=rule)
        d = C.cross_validate(T, ["SBS", "VBS"], seeds=[0])
        for pol in ("SBS", "VBS"):
            rows.append({"emparejamiento": rule, "iou": iou, "policy": f"{pol} recalculado",
                         "f1_macro_sitio": C.macro(d, "f1")[pol], "reward_macro": C.macro(d)[pol],
                         "acciones": ",".join(sorted(set(d[d.policy == pol].action))) if pol == "SBS" else ""})
    sel = pd.DataFrame(rows)
    sel.to_csv(C.OUT / "e09_selectores.csv", index=False)
    print(C.fmt(sel).to_string(index=False))

    nt = man.n_true
    empty = {"parcelas_sin_arboles": int((nt == 0).sum()), "min_arboles": int(nt.min()),
             "parcelas_con_<=3_arboles": int((nt <= 3).sum()), "mediana": float(nt.median())}
    pd.Series(empty).to_csv(C.OUT / "e09_escenas_vacias.csv")
    print(empty)


if __name__ == "__main__":
    main()
