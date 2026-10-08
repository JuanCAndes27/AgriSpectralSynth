"""
Experiment protocol for the crown-delineation agent.

* Site-grouped cross-validation: every test fold contains whole NEON
  sites never seen in training, so the score measures generalisation
  to new forests, not memorisation of a site.
* Online training (bandit feedback, several passes over the training
  contexts in random order), then greedy evaluation on the test fold.
* Repeated over several random seeds; mean and standard deviation are
  reported.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from .bandit import (
    EpsilonGreedy,
    FactoredLinUCB,
    FixedArm,
    FullInfoRidge,
    LinThompson,
    LinUCB,
    RandomPolicy,
    Standardizer,
    TrainedAgent,
    UCB1,
)
from .rewards import RewardTable


def site_folds(sites: np.ndarray, k: int = 5) -> List[np.ndarray]:
    """Assign whole sites to k folds, balancing the number of contexts (largest first)."""
    uniq, counts = np.unique(sites, return_counts=True)
    order = np.argsort(-counts)
    load = np.zeros(k)
    fold_of: Dict[str, int] = {}
    for i in order:
        f = int(np.argmin(load))
        fold_of[uniq[i]] = f
        load[f] += counts[i]
    return [np.flatnonzero(np.array([fold_of[s] == f for s in sites])) for f in range(k)]


def image_folds(images: np.ndarray, k: int = 5, seed: int = 0) -> List[np.ndarray]:
    """Random folds grouped by image (all sensors of an image stay together)."""
    uniq = np.random.default_rng(seed).permutation(np.unique(images))
    fold_of = {im: i % k for i, im in enumerate(uniq)}
    return [np.flatnonzero(np.array([fold_of[i] == f for i in images])) for f in range(k)]


POLICIES: Dict[str, Callable] = {
    "LinUCB factorizado": lambda n, d, rng, arms=None: FactoredLinUCB(n, d, rng, arms, alpha=0.5),
    "LinUCB": lambda n, d, rng: LinUCB(n, d, rng, alpha=0.5),
    "LinThompson": lambda n, d, rng: LinThompson(n, d, rng, v=0.2),
    "EpsGreedy": lambda n, d, rng: EpsilonGreedy(n, d, rng),
    "UCB1": lambda n, d, rng: UCB1(n, d, rng),
    "Random": lambda n, d, rng: RandomPolicy(n, d, rng),
}


@dataclass
class FoldResult:
    policy: str
    seed: int
    fold: int
    test_reward: float
    test_f1: float
    test_count_err: float           # mean relative count error
    inventory_err: float            # (sum predicted - sum true) / sum true over the fold
    choices: np.ndarray             # arm index chosen per test context
    test_idx: np.ndarray
    train_regret: np.ndarray = field(default_factory=lambda: np.zeros(0))   # cumulative, per step


def _score(T: RewardTable, idx: np.ndarray, choice: np.ndarray, name: str, seed: int, fold: int, regret=None):
    r = T.R[idx, choice]
    f1 = T.F1[idx, choice]
    ce = T.count_err[idx, choice]
    npred = T.n_pred[idx, choice]
    inv = (npred.sum() - T.n_true[idx].sum()) / max(T.n_true[idx].sum(), 1)
    return FoldResult(name, seed, fold, float(r.mean()), float(f1.mean()), float(ce.mean()), float(inv),
                      choice, idx, regret if regret is not None else np.zeros(0))


def run_fold(T: RewardTable, train: np.ndarray, test: np.ndarray, seed: int, fold: int,
             epochs: int = 3, policies=None) -> List[FoldResult]:
    rng = np.random.default_rng(seed)
    std = Standardizer().fit(T.X[train])
    Ztr, Zte = std(T.X[train]), std(T.X[test])
    n_arms, dim = T.R.shape[1], Ztr.shape[1]
    out = []

    # References (no online learning)
    best = int(np.argmax(T.R[train].mean(axis=0)))
    out.append(_score(T, test, np.full(len(test), best), "Mejor método fijo", seed, fold))
    out.append(_score(T, test, T.R[test].argmax(axis=1), "Oráculo", seed, fold))
    ridge = FullInfoRidge(n_arms, dim, rng).fit(Ztr, T.R[train])
    out.append(_score(T, test, np.array([ridge.greedy(z) for z in Zte]), "Ridge (información completa)", seed, fold))

    for name, make in (POLICIES if policies is None else policies).items():
        if name.startswith("LinUCB factorizado"):
            pol = make(n_arms, dim, np.random.default_rng(seed), arms=T.arms)
        else:
            pol = make(n_arms, dim, np.random.default_rng(seed))
        regret = []
        cum = 0.0
        for _ in range(epochs):
            for i in rng.permutation(len(train)):
                ctx = train[i]
                a = pol.select(Ztr[i])
                r = T.R[ctx, a]
                pol.update(Ztr[i], a, r)
                cum += T.R[ctx].max() - r
                regret.append(cum)
        if name == "Random":
            choice = np.array([pol.select(z) for z in Zte])
        else:
            choice = np.array([pol.greedy(z) for z in Zte])
        out.append(_score(T, test, choice, name, seed, fold, np.asarray(regret)))
    return out


def cross_validate(T: RewardTable, k: int = 5, seeds=range(5), epochs: int = 3, policies=None,
                   grouping: str = "site") -> List[FoldResult]:
    """grouping='site': test sites never seen in training (generalisation to new forests).
    grouping='image': new images, possibly from sites seen in training."""
    results = []
    for seed in seeds:
        folds = site_folds(T.sites, k) if grouping == "site" else image_folds(T.images, k, seed)
        for f, test in enumerate(folds):
            train = np.setdiff1d(np.arange(len(T.sites)), test)
            results += run_fold(T, train, test, seed, f, epochs, policies)
    return results


def summarize(results: List[FoldResult]) -> List[dict]:
    """Mean over folds per seed, then mean and std over seeds."""
    by = defaultdict(lambda: defaultdict(list))
    for r in results:
        by[r.policy][r.seed].append(r)
    rows = []
    for pol, seeds in by.items():
        def agg(attr):
            per_seed = []
            for rs in seeds.values():
                w = np.array([len(r.test_idx) for r in rs], dtype=float)
                per_seed.append(np.average([getattr(r, attr) for r in rs], weights=w))
            return float(np.mean(per_seed)), float(np.std(per_seed))
        row = {"policy": pol}
        for attr in ("test_reward", "test_f1", "test_count_err", "inventory_err"):
            m, s = agg(attr)
            row[attr], row[attr + "_sd"] = round(m, 4), round(s, 4)
        rows.append(row)
    best = next(r for r in rows if r["policy"] == "Mejor método fijo")["test_reward"]
    orac = next(r for r in rows if r["policy"] == "Oráculo")["test_reward"]
    for r in rows:
        r["gap_closed"] = round((r["test_reward"] - best) / (orac - best), 3) if orac > best else 0.0
    order = {"Oráculo": 0}
    return sorted(rows, key=lambda r: (order.get(r["policy"], 1), -r["test_reward"]))


def train_final_agent(T: RewardTable, policy: str = "LinUCB", epochs: int = 3, seed: int = 0) -> TrainedAgent:
    """Train on every context (bandit feedback) and export the linear model."""
    rng = np.random.default_rng(seed)
    std = Standardizer().fit(T.X)
    Z = std(T.X)
    make = POLICIES[policy]
    if policy.startswith("LinUCB factorizado"):
        pol = make(T.R.shape[1], Z.shape[1], np.random.default_rng(seed), arms=T.arms)
    else:
        pol = make(T.R.shape[1], Z.shape[1], np.random.default_rng(seed))
    for _ in range(epochs):
        for i in rng.permutation(len(Z)):
            a = pol.select(Z[i])
            pol.update(Z[i], a, T.R[i, a])
    return TrainedAgent(T.arms, T.features, std.mean, std.std, pol.theta(),
                        {"policy": policy, "epochs": epochs, "contexts": int(len(Z)), "seed": seed})
