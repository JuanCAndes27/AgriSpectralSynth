"""
Contextual bandit policies for choosing a crown delineation method.

Each step: the agent sees the context vector x of a scene, chooses one
arm (method), and receives only that arm's reward. Over time it learns,
per arm, a linear model reward ~ theta_a . x.

Policies
--------
LinUCB          optimism in the face of uncertainty (Li et al., 2010)
LinThompson     Thompson sampling with Bayesian linear regression (Agrawal & Goyal, 2013)
EpsilonGreedy   ridge regression + random exploration with decaying epsilon
UCB1            context-free bandit (ignores x): converges to the best single arm
Random          uniform choice
FixedArm        always the same arm (the "tune one method once" practice)
FullInfoRidge   NOT a bandit: sees all arms' rewards during training
                (supervised selector); an upper reference for linear policies
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


class Standardizer:
    """z-score features (fitted on training contexts) and append a bias term."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean, self.std = mean, std

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(X) - self.mean) / self.std
        return np.hstack([Z, np.ones((Z.shape[0], 1))])


class Policy:
    name = "policy"
    learns = True

    def __init__(self, n_arms: int, dim: int, rng: np.random.Generator):
        self.n_arms, self.dim, self.rng = n_arms, dim, rng

    def select(self, x: np.ndarray) -> int:  # pragma: no cover
        raise NotImplementedError

    def update(self, x: np.ndarray, arm: int, r: float) -> None:
        pass

    def greedy(self, x: np.ndarray) -> int:
        """Exploitation-only choice used on the test set."""
        return self.select(x)


class _LinearArms(Policy):
    def __init__(self, n_arms, dim, rng, lam: float = 1.0):
        super().__init__(n_arms, dim, rng)
        self.A = np.stack([lam * np.eye(dim) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, dim))
        self._Ainv = np.stack([np.eye(dim) / lam for _ in range(n_arms)])

    def update(self, x, arm, r):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += r * x
        # Sherman-Morrison rank-1 update of the inverse
        Ai = self._Ainv[arm]
        Ax = Ai @ x
        self._Ainv[arm] = Ai - np.outer(Ax, Ax) / (1.0 + x @ Ax)

    def theta(self) -> np.ndarray:
        return np.einsum("aij,aj->ai", self._Ainv, self.b)

    def greedy(self, x):
        return int(np.argmax(self.theta() @ x))


class LinUCB(_LinearArms):
    name = "LinUCB"

    def __init__(self, n_arms, dim, rng, alpha: float = 0.5, lam: float = 1.0):
        super().__init__(n_arms, dim, rng, lam)
        self.alpha = alpha

    def select(self, x):
        mean = self.theta() @ x
        width = np.sqrt(np.einsum("i,aij,j->a", x, self._Ainv, x))
        return int(np.argmax(mean + self.alpha * width))


class LinThompson(_LinearArms):
    name = "LinThompson"

    def __init__(self, n_arms, dim, rng, v: float = 0.2, lam: float = 1.0):
        super().__init__(n_arms, dim, rng, lam)
        self.v = v

    def select(self, x):
        # Sampling theta_a ~ N(theta_hat_a, v^2 A_a^-1) and scoring theta_a . x is the
        # same as sampling the scalar score ~ N(theta_hat_a . x, v^2 x' A_a^-1 x).
        mean = self.theta() @ x
        sd = self.v * np.sqrt(np.einsum("i,aij,j->a", x, self._Ainv, x))
        return int(np.argmax(mean + sd * self.rng.standard_normal(self.n_arms)))


class EpsilonGreedy(_LinearArms):
    name = "EpsGreedy"

    def __init__(self, n_arms, dim, rng, eps0: float = 0.3, decay: float = 0.01, lam: float = 1.0):
        super().__init__(n_arms, dim, rng, lam)
        self.eps0, self.decay, self.t = eps0, decay, 0

    def select(self, x):
        self.t += 1
        if self.rng.random() < self.eps0 / (1.0 + self.decay * self.t):
            return int(self.rng.integers(self.n_arms))
        return self.greedy(x)


FACTOR_SIGNALS = ("ndvi", "exg", "dark", "hsi", "rgb")
FACTOR_ALGOS = ("cc", "wsd", "lmw", "df")
FACTOR_SIZES = ("2m", "4m", "7m")
FACTOR_SCORES = ("s0.1", "s0.2", "s0.3", "s0.4")


def arm_factors(arm_names: List[str]) -> np.ndarray:
    """One-hot design of each arm's components: bias, signal, algorithm, crown size, detector score, Otsu.

    Arm names follow actions.py, e.g. 'wsd_exg-otsu_4m', 'cc_dark', 'df_hsi_s0.2'.
    """
    rows = []
    for name in arm_names:
        parts = name.split("_")
        algo, sig = parts[0], parts[1]
        signal = sig.split("-")[0]
        otsu = sig.endswith("-otsu") or signal == "dark"
        third = parts[2] if len(parts) > 2 else ""
        rows.append([1.0]
                    + [signal == x for x in FACTOR_SIGNALS]
                    + [algo == x for x in FACTOR_ALGOS]
                    + [third == x for x in FACTOR_SIZES]
                    + [third == x for x in FACTOR_SCORES]
                    + [otsu])
    return np.asarray(rows, dtype=np.float64)


class FactoredLinUCB(Policy):
    """
    LinUCB with parameters shared across arms.

    reward(x, a) ~ w . (x  kron  g(a)),  g(a) = one-hot components of arm a
    (signal, algorithm, crown size, threshold). Arms that share a component
    share what is learned about it, so far fewer samples are needed than with
    one independent model per arm.
    """

    name = "LinUCB factorizado"

    def __init__(self, n_arms, dim, rng, arm_names: List[str], alpha: float = 0.5, lam: float = 1.0):
        super().__init__(n_arms, dim, rng)
        self.G = arm_factors(arm_names)                      # (n_arms, g)
        self.p = dim * self.G.shape[1]
        self.Ainv = np.eye(self.p) / lam
        self.b = np.zeros(self.p)
        self.alpha = alpha

    def _phi(self, x):
        return np.einsum("i,aj->aij", x, self.G).reshape(self.n_arms, -1)   # (n_arms, p)

    def _w(self):
        return self.Ainv @ self.b

    def select(self, x):
        P = self._phi(x)
        mean = P @ self._w()
        # phi_a = x kron g_a  ->  phi_a' A^-1 phi_a = g_a' S g_a  with  S = (x' A^-1 x) over the factor blocks
        g = self.G.shape[1]
        A4 = self.Ainv.reshape(self.dim, g, self.dim, g)
        S = np.einsum("i,ijkl,k->jl", x, A4, x)
        width = np.sqrt(np.maximum(np.einsum("aj,jl,al->a", self.G, S, self.G), 0))
        return int(np.argmax(mean + self.alpha * width))

    def greedy(self, x):
        return int(np.argmax(self._phi(x) @ self._w()))

    def update(self, x, arm, r):
        f = self._phi(x)[arm]
        Af = self.Ainv @ f
        self.Ainv -= np.outer(Af, Af) / (1.0 + f @ Af)
        self.b += r * f

    def theta(self) -> np.ndarray:
        """Equivalent per-arm linear models (n_arms, dim), for export."""
        W = self._w().reshape(-1, self.G.shape[1])            # (dim, g)
        return (W @ self.G.T).T


class UCB1(Policy):
    name = "UCB1 (sin contexto)"

    def __init__(self, n_arms, dim, rng, c: float = 0.5):
        super().__init__(n_arms, dim, rng)
        self.n = np.zeros(n_arms)
        self.s = np.zeros(n_arms)
        self.c = c

    def select(self, x):
        if (self.n == 0).any():
            return int(np.flatnonzero(self.n == 0)[0])
        t = self.n.sum()
        return int(np.argmax(self.s / self.n + self.c * np.sqrt(np.log(t) / self.n)))

    def update(self, x, arm, r):
        self.n[arm] += 1
        self.s[arm] += r

    def greedy(self, x):
        return int(np.argmax(self.s / np.maximum(self.n, 1)))


class RandomPolicy(Policy):
    name = "Aleatorio"
    learns = False

    def select(self, x):
        return int(self.rng.integers(self.n_arms))


class FixedArm(Policy):
    name = "Mejor método fijo"
    learns = False

    def __init__(self, n_arms, dim, rng, arm: int = 0):
        super().__init__(n_arms, dim, rng)
        self.arm = arm

    def select(self, x):
        return self.arm


class FullInfoRidge(_LinearArms):
    """Supervised selector: trained with the rewards of ALL arms (not a bandit)."""

    name = "Ridge (información completa)"
    learns = False

    def fit(self, Z: np.ndarray, R: np.ndarray):
        for a in range(self.n_arms):
            self.A[a] += Z.T @ Z
            self.b[a] += Z.T @ R[:, a]
            self._Ainv[a] = np.linalg.inv(self.A[a])
        return self

    def select(self, x):
        return self.greedy(x)


# ---------------------------------------------------------------------------
# Persistence (a trained agent you can use on new images)
# ---------------------------------------------------------------------------

@dataclass
class TrainedAgent:
    arms: List[str]
    features: List[str]
    mean: np.ndarray
    std: np.ndarray
    theta: np.ndarray            # (n_arms, dim + 1)
    meta: dict

    def choose(self, feature_vector: np.ndarray, allowed=None) -> str:
        """Best arm by expected reward; ``allowed`` restricts the choice (e.g. no hyperspectral at hand)."""
        z = Standardizer(self.mean, self.std)(feature_vector)[0]
        scores = self.theta @ z
        if allowed is not None:
            scores = np.where([a in allowed for a in self.arms], scores, -np.inf)
        return self.arms[int(np.argmax(scores))]

    def expected_rewards(self, feature_vector: np.ndarray) -> dict:
        z = Standardizer(self.mean, self.std)(feature_vector)[0]
        return dict(zip(self.arms, (self.theta @ z).round(4).tolist()))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "arms": self.arms, "features": self.features,
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "theta": self.theta.tolist(), "meta": self.meta,
        }, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrainedAgent":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["arms"], d["features"], np.array(d["mean"]), np.array(d["std"]), np.array(d["theta"]), d["meta"])
