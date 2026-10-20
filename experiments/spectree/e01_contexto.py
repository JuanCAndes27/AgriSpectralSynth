#!/usr/bin/env python3
"""
E01 — Does the synthesis improve the CONTEXT, or does it only add variables?

Same 40 RGB actions, rewards and folds; only the context changes:
  C0  RGB descriptors (21, incl. nominal sensor)
  C1  C0 + synthetic NDVI statistics (3)
  C2  C0 + all implemented synthetic attributes (9)
  C3  C0 + simple RGB transforms (9; same dimension as C2)
  C4  C0 + REAL NDVI statistics (3; diagnostic, needs hyperspectral)
Selectors: LinUCB-F, Ridge (full information) and LinUCB (disjoint).
Primary comparison P1: C2 vs C0 with LinUCB-F (Bonferroni 98.75 % interval).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

CTX = ["C0", "C1", "C2", "C3", "C4"]
POLS = ["LinUCB-F", "Ridge", "LinUCB"]


def main():
    decs = []
    for g in CTX:
        T = C.build_table(C.RGB40, g)
        d = C.cross_validate(T, ["SBS", "VBS"] + POLS, tag=f"E01-{g}")
        d["policy"] = np.where(d.policy.isin(["SBS", "VBS"]), d.policy, d.policy + "|" + g)
        decs.append(d)
        print(g, len(T.features), "atributos", flush=True)
    dec = pd.concat(decs)
    dec = dec[~(dec.policy.isin(["SBS", "VBS"]) & (dec.tag != "E01-C0"))]     # SBS/VBS do not depend on context
    dec.to_csv(C.OUT / "e01_decisiones.csv", index=False)

    s = C.summary(dec)
    s.to_csv(C.OUT / "e01_resumen.csv", index=False)
    print(C.fmt(s).to_string(index=False))

    rows = []
    comps = [("C1", "C0"), ("C2", "C0"), ("C3", "C0"), ("C2", "C3"), ("C1", "C3"), ("C4", "C0"), ("C4", "C1")]
    for pol in POLS:
        for a, b in comps:
            primary = pol == "LinUCB-F" and (a, b) == ("C2", "C0")
            lvl = 1 - 0.05 / 4 if primary else 0.95
            est, lo, hi, per_site = C.paired_boot(dec, f"{pol}|{a}", f"{pol}|{b}", level=lvl)
            f1e, f1l, f1h, _ = C.paired_boot(dec, f"{pol}|{a}", f"{pol}|{b}", col="f1")
            mae = C.paired_boot(dec, f"{pol}|{a}", f"{pol}|{b}", col="abs_count_error")[0]
            rows.append({"selector": pol, "comparacion": f"{a} - {b}", "primaria": primary, "nivel": lvl,
                         "d_recompensa": est, "ci_lo": lo, "ci_hi": hi, "d_f1": f1e, "f1_lo": f1l, "f1_hi": f1h,
                         "d_mae_conteo": mae, "sitios_mejor": int((per_site > 0).sum()),
                         "sitios_peor": int((per_site < 0).sum())})
    comp = pd.DataFrame(rows)
    comp.to_csv(C.OUT / "e01_comparaciones.csv", index=False)
    print(C.fmt(comp).to_string(index=False))

    # where does the decision change between C0 and C2 (LinUCB-F), and what does it gain or lose?
    a = dec[dec.policy == "LinUCB-F|C2"].set_index(["seed", "ctx"])
    b = dec[dec.policy == "LinUCB-F|C0"].set_index(["seed", "ctx"])
    j = a[["site", "action", "reward"]].join(b[["action", "reward"]], rsuffix="_C0")
    j["cambia"] = j.action != j.action_C0
    j["ganancia"] = j.reward - j.reward_C0
    ch = j.groupby("site").agg(contextos=("cambia", "size"), frac_cambia=("cambia", "mean"),
                               ganancia_media_si_cambia=("ganancia", lambda x: x[j.loc[x.index, "cambia"]].mean()),
                               ganancia_media=("ganancia", "mean"))
    ch.to_csv(C.OUT / "e01_cambios_por_sitio.csv")
    print(ch.round(3).to_string())


if __name__ == "__main__":
    main()
