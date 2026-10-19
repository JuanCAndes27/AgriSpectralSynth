#!/usr/bin/env python3
"""
E08 — Sensitivity to the reward weights and to the portfolio size.

  * grid kappa in {0, 0.1, 0.25, 0.5} x lambda in {0, 0.01, 0.05, 0.1, 0.2} 1/s (fixed before running)
    SBS and VBS are recomputed for every objective; absolute differences are reported
  * quality at time budgets: F1 and mean seconds of each policy/objective
  * reduced portfolio: k in {4, 8} representatives chosen ONLY on training (greedy forward selection
    maximising the training VBS), context fixed (v0.5), lambda = 0 and lambda = 0.1
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

KAPPAS = (0.0, 0.1, 0.25, 0.5)
LAMBDAS = (0.0, 0.01, 0.05, 0.1, 0.2)


def greedy_portfolio(R_train, k):
    chosen = []
    for _ in range(k):
        best, val = None, -np.inf
        for a in range(R_train.shape[1]):
            if a in chosen:
                continue
            v = R_train[:, chosen + [a]].max(axis=1).mean()
            if v > val:
                best, val = a, v
        chosen.append(best)
    return chosen


def cv_reduced(T, k, seeds=C.SEEDS):
    folds = C.site_folds(T.sites, C.K)
    rows = []
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(T.sites)), test)
        sub = greedy_portfolio(T.R[train], k)
        arms = [T.arms[a] for a in sub]
        Ts = C.Table(T.X, T.R[:, sub], T.F1[:, sub], T.CE[:, sub], T.AE[:, sub], T.NP[:, sub], T.TP[:, sub],
                     T.SEC[:, sub], T.NT, T.images, T.sites, T.sensors, arms, T.features)
        for seed in seeds:
            for pol in ("SBS", "VBS", "LinUCB-F", "Ridge"):
                if pol in ("SBS", "VBS", "Ridge") and seed != seeds[0]:
                    continue
                if pol == "SBS":
                    ch = np.full(len(test), int(np.argmax(Ts.R[train].mean(axis=0))))
                elif pol == "VBS":
                    ch = Ts.R[test].argmax(axis=1)
                elif pol == "Ridge":
                    ch = C._ridge(Ts, train, test)
                else:
                    ch, _ = C._bandit(Ts, train, test, pol, seed)
                for i, a in zip(test, ch):
                    rows.append({"fold": f, "seed": seed if pol not in ("SBS", "VBS", "Ridge") else -1,
                                 "policy": pol, "image": T.images[i], "site": T.sites[i], "sensor": T.sensors[i],
                                 "action": arms[a], "reward": Ts.R[i, a], "f1": Ts.F1[i, a], "tp": Ts.TP[i, a],
                                 "n_pred": Ts.NP[i, a], "n_true": T.NT[i], "abs_count_error": Ts.AE[i, a],
                                 "seconds": Ts.SEC[i, a], "portafolio": ";".join(arms)})
    return pd.DataFrame(rows)


def main():
    rows = []
    for kappa in KAPPAS:
        for lam in LAMBDAS:
            T = C.build_table(C.RGB40, "V05", kappa=kappa, lam=lam)
            d = C.cross_validate(T, ["SBS", "VBS", "Ridge", "LinUCB-F"], tag=f"E08-k{kappa}-l{lam}")
            mac, f1, sec = C.macro(d), C.macro(d, "f1"), C.per_plot(d, "seconds").groupby("policy").seconds.mean()
            mae = C.per_plot(d, "abs_count_error").groupby("policy").abs_count_error.mean()
            for pol in mac.index:
                est = lo = hi = np.nan
                if pol != "SBS":
                    est, lo, hi, _ = C.paired_boot(d, pol, "SBS")
                rows.append({"kappa": kappa, "lambda": lam, "policy": pol, "reward_macro": mac[pol], "f1_macro": f1[pol],
                             "count_mae": mae[pol], "seconds": sec[pol], "d_vs_SBS": est, "ci_lo": lo, "ci_hi": hi,
                             "brecha_SBS_VBS": mac["VBS"] - mac["SBS"],
                             "sbs_action": d[d.policy == "SBS"].groupby("fold").action.first().str.cat(sep=",")})
            print(kappa, lam, flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(C.OUT / "e08_grilla.csv", index=False)
    print(C.fmt(g.drop(columns="sbs_action")).to_string(index=False))

    out = []
    for lam in (0.0, 0.1):
        T = C.build_table(C.RGB40, "V05", lam=lam)
        full = C.cross_validate(T, ["SBS", "VBS", "Ridge", "LinUCB-F"], tag="full")
        for pol, v in C.macro(full).items():
            out.append({"lambda": lam, "portafolio": "40 (completo)", "policy": pol, "reward_macro": v,
                        "f1_macro": C.macro(full, "f1")[pol]})
        for k in (4, 8):
            d = cv_reduced(T, k)
            for pol, v in C.macro(d).items():
                out.append({"lambda": lam, "portafolio": f"{k} (codicioso en entrenamiento)", "policy": pol,
                            "reward_macro": v, "f1_macro": C.macro(d, "f1")[pol],
                            "acciones_fold0": d[d.fold == 0].portafolio.iloc[0]})
    p = pd.DataFrame(out)
    p.to_csv(C.OUT / "e08_portafolios.csv", index=False)
    print(C.fmt(p).to_string(index=False))


if __name__ == "__main__":
    main()
