"""
SpecTree — shared analysis code: reward tables, site-grouped CV with decision logging,
selectors (bandits, full-information, supervised, cheap rule), site-macro aggregation and
paired site bootstrap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from agrispectralsynth.agent.actions import ARMS, arm_names_in
from agrispectralsynth.agent.bandit import (FactoredLinUCB, FullInfoRidge, LinThompson, LinUCB, Standardizer)
from agrispectralsynth.agent.experiment import site_folds

ROOT = Path(__file__).resolve().parents[2] / "results" / "spectree"
S01 = ROOT / "s01"
OUT = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)

RGB40 = arm_names_in(["classical", "df"])
ALL58 = arm_names_in(["classical", "df", "hsi"])
ARM = {a.name: a for a in ARMS}
GROUPS = json.load(open(S01 / "feature_groups.json")) if (S01 / "feature_groups.json").exists() else {}
SEEDS = range(5)
K = 5
ALPHA_LINUCB, ETA, V_TS, EPOCHS = 0.5, 1.0, 0.2, 3
BOOT, BOOT_SEED = 5000, 12345


def family(name: str) -> str:
    a = ARM[name]
    if a.is_detector:
        return {"rgb": "DF", "ndvi": "DF+NDVIsint", "hsi": "DF+NDVIreal"}[a.signal]
    return {"exg": "Clas-RGB", "dark": "Clas-RGB", "ndvi": "Clas-NDVIsint", "hsi": "Clas-NDVIreal"}[a.signal]


def context_features(group: str) -> List[str]:
    """C0 | C1 | C2 | C3 | C4 | V05 (the 25 v0.5 descriptors)."""
    c0 = GROUPS["C0"]
    if group == "C0":
        return list(c0)
    if group == "V05":
        from agrispectralsynth.agent.features import FEATURE_NAMES

        return list(FEATURE_NAMES)
    if group.startswith("C0+"):          # C0 + an explicit list (used by E06)
        return list(c0) + group[3:].split(",")
    return list(c0) + [f for f in GROUPS[f"{group}_extra"] if f not in c0]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

_METRICS: Dict[tuple, pd.DataFrame] = {}


def metrics(iou: float = 0.4, matching: str = "hung") -> pd.DataFrame:
    key = (iou, matching)
    if key not in _METRICS:
        if "all" not in _METRICS:
            _METRICS["all"] = pd.read_csv(ROOT / "action_metrics.csv")
        m = _METRICS["all"]
        _METRICS[key] = m[(m.iou_thr == iou) & (m.matching == matching)].copy()
    return _METRICS[key]


def contexts() -> pd.DataFrame:
    return pd.read_csv(S01 / "contexts_ext.csv").sort_values(["image", "sensor"]).reset_index(drop=True)


@dataclass
class Table:
    X: np.ndarray
    R: np.ndarray
    F1: np.ndarray
    CE: np.ndarray          # relative count error |n_pred - n| / n
    AE: np.ndarray          # absolute count error |n_pred - n|
    NP: np.ndarray
    TP: np.ndarray
    SEC: np.ndarray
    NT: np.ndarray
    images: np.ndarray
    sites: np.ndarray
    sensors: np.ndarray
    arms: List[str]
    features: List[str]


def build_table(actions=RGB40, features="C2", kappa=0.25, lam=0.0, iou=0.4, matching="hung",
                extra_seconds: Optional[np.ndarray] = None, ctx: Optional[pd.DataFrame] = None) -> Table:
    ctx = contexts() if ctx is None else ctx
    feats = context_features(features) if isinstance(features, str) else list(features)
    m = metrics(iou, matching)
    m = m[m.action.isin(actions)]
    key = ["image", "sensor"]

    def piv(col):
        p = m.pivot_table(index=key, columns="action", values=col, aggfunc="first")
        p = p.reindex(pd.MultiIndex.from_frame(ctx[key]))[list(actions)]
        return p.to_numpy(dtype=np.float64)

    F1, CE, NP, TP, SEC = piv("f1"), piv("rel_count_error"), piv("n_pred"), piv("tp"), piv("seconds")
    NT = ctx["n_true"].to_numpy(float)
    if extra_seconds is not None:
        SEC = SEC + extra_seconds[:, None]
    R = F1 - kappa * np.minimum(1.0, CE) - lam * SEC
    return Table(ctx[feats].to_numpy(float), R, F1, CE, np.abs(NP - NT[:, None]), NP, TP, SEC, NT,
                 ctx.image.to_numpy(), ctx.site.to_numpy(), ctx.sensor.to_numpy(), list(actions), feats)


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

def _bandit(T: Table, train, test, kind: str, seed: int, epochs: int = EPOCHS, alpha: float = ALPHA_LINUCB):
    rng = np.random.default_rng(seed)
    std = Standardizer().fit(T.X[train])
    Ztr, Zte = std(T.X[train]), std(T.X[test])
    n_arms, dim = T.R.shape[1], Ztr.shape[1]
    if kind == "LinUCB":
        pol = LinUCB(n_arms, dim, np.random.default_rng(seed), alpha=alpha, lam=ETA)
    elif kind == "LinUCB-F":
        pol = FactoredLinUCB(n_arms, dim, np.random.default_rng(seed), T.arms, alpha=alpha, lam=ETA)
    elif kind == "LinTS":
        pol = LinThompson(n_arms, dim, np.random.default_rng(seed), v=V_TS, lam=ETA)
    else:
        raise ValueError(kind)
    queries, pairs = 0, set()
    for _ in range(epochs):
        for i in rng.permutation(len(train)):
            a = pol.select(Ztr[i])
            pol.update(Ztr[i], a, T.R[train[i], a])
            queries += 1
            pairs.add((train[i], a))
    return np.array([pol.greedy(z) for z in Zte]), {"queries": queries, "unique_pairs": len(pairs)}


def _ridge(T, train, test):
    std = Standardizer().fit(T.X[train])
    pol = FullInfoRidge(T.R.shape[1], T.X.shape[1] + 1, np.random.default_rng(0)).fit(std(T.X[train]), T.R[train])
    Zte = std(T.X[test])
    return np.array([pol.greedy(z) for z in Zte])


def _inner_folds(T, train, k=4):
    sub = T.sites[train]
    return [train[f] for f in site_folds(sub, min(k, len(np.unique(sub))))]


def _trees(T, train, test, inner=True):
    """Supervised non-linear selector: multi-output ExtraTrees predicting the reward of every action.
    4 configurations chosen by site-grouped inner validation."""
    from sklearn.ensemble import ExtraTreesRegressor

    grid = [dict(min_samples_leaf=l, max_features=f) for l in (5, 20) for f in (0.33, 1.0)]

    def fit_predict(tr, te, cfg):
        m = ExtraTreesRegressor(n_estimators=150, random_state=0, n_jobs=1, **cfg)
        m.fit(T.X[tr], T.R[tr])
        return m.predict(T.X[te]).argmax(axis=1)

    best = grid[0]
    if inner:
        scores = []
        for cfg in grid:
            s = []
            for f in _inner_folds(T, train):
                tr = np.setdiff1d(train, f)
                s.append(T.R[f, fit_predict(tr, f, cfg)].mean())
            scores.append(np.mean(s))
        best = grid[int(np.argmax(scores))]
    return fit_predict(train, test, best), best


def _cheap_rule(T, train, test, rgb_features: List[str]):
    """Decision stump between the best classical and the best DeepForest action of the training set,
    on ONE RGB-only feature (no DeepForest output is used to decide whether to run DeepForest)."""
    is_df = np.array([ARM[a].is_detector for a in T.arms])
    mean = T.R[train].mean(axis=0)
    a_c = int(np.flatnonzero(~is_df)[np.argmax(mean[~is_df])]) if (~is_df).any() else None
    a_d = int(np.flatnonzero(is_df)[np.argmax(mean[is_df])]) if is_df.any() else None
    if a_c is None or a_d is None:
        a = a_c if a_d is None else a_d
        return np.full(len(test), a), {"rule": "fixed"}
    cols = [T.features.index(f) for f in rgb_features if f in T.features]
    rc, rd = T.R[train, a_c], T.R[train, a_d]
    best = (max(rc.mean(), rd.mean()), None, None, None)
    for j in cols:
        x = T.X[train, j]
        for t in np.quantile(x, np.linspace(0.1, 0.9, 9)):
            for sgn in (1, -1):
                use_df = sgn * (x - t) > 0
                val = np.where(use_df, rd, rc).mean()
                if val > best[0] + 1e-12:
                    best = (val, j, t, sgn)
    _, j, t, sgn = best
    if j is None:
        a = a_d if rd.mean() >= rc.mean() else a_c
        return np.full(len(test), a), {"rule": f"fixed:{T.arms[a]}"}
    use_df = sgn * (T.X[test, j] - t) > 0
    return np.where(use_df, a_d, a_c), {"rule": f"{T.features[j]} {'>' if sgn > 0 else '<'} {t:.4g}",
                                        "classical": T.arms[a_c], "df": T.arms[a_d]}


POLICIES = ["SBS", "VBS", "Random", "Ridge", "LinUCB", "LinUCB-F", "LinUCB-F-greedy", "LinTS", "Trees", "CheapRule"]
INFO = {"SBS": "fijo (entrenamiento)", "VBS": "oráculo (prueba)", "Random": "ninguna", "Ridge": "completa",
        "Trees": "completa", "CheapRule": "completa", "LinUCB": "bandido", "LinUCB-F": "bandido",
        "LinUCB-F-greedy": "bandido sin exploración", "LinTS": "bandido"}


def cross_validate(T: Table, policies=("SBS", "VBS", "Ridge", "LinUCB-F"), seeds=SEEDS, train_subset=None,
                   epochs=EPOCHS, tag="") -> pd.DataFrame:
    """Outer site-grouped CV (the 5 v0.5 folds). Returns one row per (policy, seed, test context)."""
    folds = site_folds(T.sites, K)
    rows = []
    rgb_feats = [f for f in context_features("C0") if not f.startswith("sensor_")]
    for f, test in enumerate(folds):
        train_all = np.setdiff1d(np.arange(len(T.sites)), test)
        for seed in seeds:
            train = train_all if train_subset is None else train_subset(train_all, seed)
            for pol in policies:
                meta = {}
                if pol in ("SBS", "VBS", "Ridge", "Trees", "CheapRule") and seed != seeds[0] and train_subset is None:
                    continue                       # deterministic: computed once (seed column = -1)
                if pol == "SBS":
                    ch = np.full(len(test), int(np.argmax(T.R[train].mean(axis=0))))
                elif pol == "VBS":
                    ch = T.R[test].argmax(axis=1)
                elif pol == "Random":
                    ch = np.random.default_rng(seed).integers(T.R.shape[1], size=len(test))
                elif pol == "Ridge":
                    ch = _ridge(T, train, test)
                elif pol == "Trees":
                    ch, meta = _trees(T, train, test)
                elif pol == "CheapRule":
                    ch, meta = _cheap_rule(T, train, test, rgb_feats)
                elif pol == "LinUCB-F-greedy":
                    ch, meta = _bandit(T, train, test, "LinUCB-F", seed, epochs, alpha=0.0)
                else:
                    ch, meta = _bandit(T, train, test, pol, seed, epochs)
                det = pol in ("SBS", "VBS", "Ridge", "Trees", "CheapRule") and train_subset is None
                for i, a in zip(test, ch):
                    rows.append((tag, f, -1 if det else seed, pol, INFO[pol], i, T.images[i], T.sites[i], T.sensors[i],
                                 T.arms[a], T.R[i, a], T.F1[i, a], T.CE[i, a], T.AE[i, a], T.NP[i, a], T.TP[i, a],
                                 T.NT[i], T.SEC[i, a], json.dumps(meta) if meta else ""))
    return pd.DataFrame(rows, columns=["tag", "fold", "seed", "policy", "info", "ctx", "image", "site", "sensor",
                                       "action", "reward", "f1", "rel_count_error", "abs_count_error", "n_pred", "tp",
                                       "n_true", "seconds", "meta"])


# ---------------------------------------------------------------------------
# Aggregation and uncertainty
# ---------------------------------------------------------------------------

def per_plot(dec: pd.DataFrame, col="reward") -> pd.DataFrame:
    """Average over seeds and sensors within each plot -> one value per (policy, image)."""
    g = dec.groupby(["policy", "image", "site"], as_index=False)[col].mean()
    return g


def macro(dec: pd.DataFrame, col="reward") -> pd.Series:
    p = per_plot(dec, col)
    return p.groupby(["policy", "site"])[col].mean().groupby("policy").mean()


def micro(dec: pd.DataFrame, col="reward") -> pd.Series:
    return per_plot(dec, col).groupby("policy")[col].mean()


def f1_micro(dec: pd.DataFrame) -> pd.Series:
    g = dec.groupby("policy")[["tp", "n_pred", "n_true"]].sum()
    return 2 * g.tp / (g.n_pred + g.n_true)


def paired_boot(dec: pd.DataFrame, a: str, b: str, col="reward", B=BOOT, seed=BOOT_SEED, level=0.95):
    """Macro-by-site difference a - b with a paired site bootstrap. Returns (est, lo, hi, per-site Series)."""
    p = per_plot(dec[dec.policy.isin([a, b])], col).pivot_table(index=["image", "site"], columns="policy", values=col)
    d = (p[a] - p[b]).groupby(level="site").mean()
    est = float(d.mean())
    rng = np.random.default_rng(seed)
    vals = d.to_numpy()
    reps = vals[rng.integers(0, len(vals), size=(B, len(vals)))].mean(axis=1)
    q = (1 - level) / 2
    return est, float(np.quantile(reps, q)), float(np.quantile(reps, 1 - q)), d


def summary(dec: pd.DataFrame, ref: str = "SBS", level=0.95) -> pd.DataFrame:
    rows = []
    mac, mic = macro(dec), micro(dec)
    f1m, f1u = macro(dec, "f1"), f1_micro(dec)
    mae = per_plot(dec, "abs_count_error").groupby("policy").abs_count_error.mean()
    inv = dec.groupby("policy").apply(lambda g: (g.n_pred.sum() - g.n_true.sum()) / g.n_true.sum(), include_groups=False)
    secs = per_plot(dec, "seconds").groupby("policy").seconds.mean()
    vbs = mac.get("VBS", np.nan)
    sbs = mac.get(ref, np.nan)
    for pol in mac.index:
        est = lo = hi = np.nan
        if pol != ref and ref in mac.index:
            est, lo, hi, _ = paired_boot(dec, pol, ref, level=level)
        seed_sd = np.nan
        if (dec[dec.policy == pol].seed >= 0).any():
            ps = dec[dec.policy == pol].groupby(["seed", "image", "site"]).reward.mean().groupby(["seed", "site"]).mean()
            seed_sd = float(ps.groupby("seed").mean().std(ddof=1))
        rows.append({"policy": pol, "info": INFO.get(pol, ""), "reward_macro": mac[pol], "reward_micro": mic[pol],
                     "f1_macro": f1m[pol], "f1_micro": f1u[pol], "count_mae": mae[pol], "inventory_err": inv[pol],
                     "seconds": secs[pol], f"d_vs_{ref}": est, "ci_lo": lo, "ci_hi": hi,
                     "gap_closed": (mac[pol] - sbs) / (vbs - sbs) if vbs - sbs > 1e-9 else np.nan,
                     "seed_sd": seed_sd})
    return pd.DataFrame(rows).sort_values("reward_macro", ascending=False)


def fmt(df: pd.DataFrame, nd=3) -> pd.DataFrame:
    return df.round(nd)
