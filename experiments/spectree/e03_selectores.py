#!/usr/bin/env python3
"""
E03 — What limits the selector: context, model or feedback?

Context (v0.5, 25 descriptors) and actions (RGB 40) held constant.
  * comparators: SBS, Random, cheap rule (classical vs DeepForest by one RGB feature), LinUCB (disjoint),
    LinUCB-F, LinUCB-F without exploration (alpha = 0), LinTS, Ridge (full information),
    ExtraTrees (full information, 4 configs by site-grouped inner validation), VBS
  * P2: the learned selector chosen PER FOLD by site-grouped inner validation vs SBS
  * 1 vs 3 passes; learning curve vs number of unique training plots (sensors kept together)
  * policy cost: decision time and size of the maintained matrices
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

LEARNED = ["CheapRule", "LinUCB", "LinUCB-F", "LinTS", "Ridge", "Trees"]


def inner_selected(T):
    """Per outer fold: pick the learned selector with the best site-grouped inner-validation reward."""
    folds = C.site_folds(T.sites, C.K)
    rows, picks = [], []
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(T.sites)), test)
        scores = {}
        for pol in LEARNED:
            vals = []
            for inner in C._inner_folds(T, train):
                tr = np.setdiff1d(train, inner)
                if pol == "Ridge":
                    ch = C._ridge(T, tr, inner)
                elif pol == "Trees":
                    ch, _ = C._trees(T, tr, inner, inner=False)
                elif pol == "CheapRule":
                    ch, _ = C._cheap_rule(T, tr, inner, [x for x in C.context_features("C0") if not x.startswith("sensor_")])
                else:
                    ch, _ = C._bandit(T, tr, inner, pol, 0)
                vals.append(T.R[inner, ch].mean())
            scores[pol] = float(np.mean(vals))
        best = max(scores, key=scores.get)
        picks.append({"fold": f, "elegido": best, **{f"iv_{k}": v for k, v in scores.items()}})
        for seed in C.SEEDS:
            if best == "Ridge":
                ch = C._ridge(T, train, test)
            elif best == "Trees":
                ch, _ = C._trees(T, train, test)
            elif best == "CheapRule":
                ch, _ = C._cheap_rule(T, train, test, [x for x in C.context_features("C0") if not x.startswith("sensor_")])
            else:
                ch, _ = C._bandit(T, train, test, best, seed)
            for i, a in zip(test, ch):
                rows.append(("E03-IV", f, seed, "Selector-VI", f"elegido en validación interna: {best}", i, T.images[i],
                             T.sites[i], T.sensors[i], T.arms[a], T.R[i, a], T.F1[i, a], T.CE[i, a], T.AE[i, a],
                             T.NP[i, a], T.TP[i, a], T.NT[i], T.SEC[i, a], ""))
    cols = ["tag", "fold", "seed", "policy", "info", "ctx", "image", "site", "sensor", "action", "reward", "f1",
            "rel_count_error", "abs_count_error", "n_pred", "tp", "n_true", "seconds", "meta"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(picks)


def main():
    T = C.build_table(C.RGB40, "V05")
    pols = ["SBS", "VBS", "Random", "CheapRule", "LinUCB", "LinUCB-F", "LinUCB-F-greedy", "LinTS", "Ridge", "Trees"]
    dec = C.cross_validate(T, pols, tag="E03")
    iv, picks = inner_selected(T)
    picks.to_csv(C.OUT / "e03_seleccion_interna.csv", index=False)
    d1 = C.cross_validate(T, ["LinUCB-F"], epochs=1, tag="E03-1pasada")
    d1["policy"] = "LinUCB-F (1 pasada)"
    C.INFO["LinUCB-F (1 pasada)"] = "bandido"
    C.INFO["Selector-VI"] = "según validación interna"
    alld = pd.concat([dec, iv, d1])
    alld.to_csv(C.OUT / "e03_decisiones.csv", index=False)
    s = C.summary(alld)
    # P2 with the Bonferroni level
    est, lo, hi, _ = C.paired_boot(alld, "Selector-VI", "SBS", level=1 - 0.05 / 4)
    s.loc[s.policy == "Selector-VI", ["ci_lo", "ci_hi"]] = [lo, hi]
    s["nivel_ic"] = np.where(s.policy == "Selector-VI", 1 - 0.05 / 4, 0.95)
    s.to_csv(C.OUT / "e03_resumen.csv", index=False)
    print(C.fmt(s).to_string(index=False))
    print(picks.to_string(index=False))
    rules = dec[dec.policy == "CheapRule"].groupby("fold").meta.first()
    print("Regla barata por partición:\n", rules.to_string())

    # family frequency
    dfam = alld.assign(fam=alld.action.map(C.family)).groupby(["policy", "fam"]).size().unstack(fill_value=0)
    (dfam.div(dfam.sum(axis=1), axis=0)).to_csv(C.OUT / "e03_frecuencia_familias.csv")

    # learning curve: number of unique training plots (sensors together)
    rows = []
    for n_plots in (10, 20, 40, 80, None):
        def subset(train_all, seed, n=n_plots):
            if n is None:
                return train_all
            ims = np.unique(T.images[train_all])
            pick = np.random.default_rng(1000 + seed).choice(ims, size=min(n, len(ims)), replace=False)
            return train_all[np.isin(T.images[train_all], pick)]
        for pol in ("LinUCB-F", "Ridge", "SBS"):
            d = C.cross_validate(T, [pol], train_subset=subset, tag="E03-curva")
            m = C.macro(d)[pol]
            rows.append({"parcelas_entrenamiento": n_plots or "todas", "selector": pol, "recompensa_macro": m,
                         "consultas": len(d[d.seed == 0]) and np.nan})
        print("curva", n_plots, flush=True)
    # queries / unique pairs for full training (bandits)
    folds = C.site_folds(T.sites, C.K)
    train = np.setdiff1d(np.arange(len(T.sites)), folds[0])
    for ep in (1, 3):
        _, meta = C._bandit(T, train, folds[0], "LinUCB-F", 0, epochs=ep)
        rows.append({"parcelas_entrenamiento": "todas", "selector": f"LinUCB-F consultas ({ep} pasadas)",
                     "recompensa_macro": np.nan, "consultas": meta["queries"], "pares_unicos": meta["unique_pairs"],
                     "contextos": len(train), "pares_posibles": len(train) * len(T.arms)})
    pd.DataFrame(rows).to_csv(C.OUT / "e03_curva_aprendizaje.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # policy cost: decision time and maintained matrices
    d = T.X.shape[1] + 1
    K = len(T.arms)
    g = C.FactoredLinUCB(K, d, np.random.default_rng(0), T.arms).G.shape[1]
    cost = []
    std = C.Standardizer().fit(T.X[train])
    Z = std(T.X[folds[0]])
    for name, pol in (("LinUCB", C.LinUCB(K, d, np.random.default_rng(0))),
                      ("LinUCB-F", C.FactoredLinUCB(K, d, np.random.default_rng(0), T.arms))):
        t0 = time.perf_counter()
        for z in Z:
            pol.select(z)
        cost.append({"selector": name, "ms_por_decision": 1000 * (time.perf_counter() - t0) / len(Z),
                     "coeficientes": K * d if name == "LinUCB" else d * g,
                     "entradas_matrices": K * d * d if name == "LinUCB" else (d * g) ** 2})
    t0 = time.perf_counter()
    C._ridge(T, train, folds[0])
    cost.append({"selector": "Ridge", "ms_por_decision": np.nan, "coeficientes": K * d, "entradas_matrices": K * d * d,
                 "s_entrenamiento": time.perf_counter() - t0})
    pd.DataFrame(cost).to_csv(C.OUT / "e03_costo_politica.csv", index=False)
    print(pd.DataFrame(cost).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
