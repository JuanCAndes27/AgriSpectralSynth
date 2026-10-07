"""
Crown detection metrics and the agent's reward.

Matching follows the NeonTreeEvaluation benchmark: a predicted crown
is a true positive when it can be paired one-to-one (Hungarian
assignment on IoU) with an annotated crown with IoU >= 0.4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_IOU = 0.4


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N, 4) and (M, 4) boxes (xmin, ymin, xmax, ymax)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


@dataclass
class DetectionScore:
    tp: int
    fp: int
    fn: int
    n_pred: int
    n_true: int
    precision: float
    recall: float
    f1: float
    count_error: int          # predicted - true (positive = over-counting)
    rel_count_error: float    # |count_error| / max(true, 1)
    mean_iou: float           # mean IoU of the matched pairs

    def to_dict(self) -> dict:
        return asdict(self)


def score_detections(pred: np.ndarray, true: np.ndarray, iou_threshold: float = DEFAULT_IOU) -> DetectionScore:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 4)
    true = np.asarray(true, dtype=np.float64).reshape(-1, 4)
    n_pred, n_true = len(pred), len(true)
    tp, ious = 0, []
    if n_pred and n_true:
        iou = box_iou(pred, true)
        rows, cols = linear_sum_assignment(-iou)
        matched = iou[rows, cols]
        ok = matched >= iou_threshold
        tp = int(ok.sum())
        ious = matched[ok]
    fp, fn = n_pred - tp, n_true - tp
    if n_pred == 0 and n_true == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_true if n_true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ce = n_pred - n_true
    return DetectionScore(
        tp, fp, fn, n_pred, n_true,
        round(precision, 4), round(recall, 4), round(f1, 4),
        ce, round(abs(ce) / max(n_true, 1), 4),
        round(float(np.mean(ious)) if len(ious) else 0.0, 4),
    )


@dataclass(frozen=True)
class RewardWeights:
    """
    reward = f1 - count_weight * min(1, |count error| / true count) - time_weight * seconds

    * F1 rewards finding the right crowns in the right place.
    * The count term adds what forestry inventories care about most:
      the number of trees. F1 alone can stay mediocre while the count is
      right (or the reverse); the weight sets the trade-off.
    * The time term lets a deployment trade accuracy for speed.
      It is 0 by default so results do not depend on the machine.
    """

    count_weight: float = 0.25
    time_weight: float = 0.0


def reward(score: DetectionScore, seconds: float = 0.0, w: RewardWeights = RewardWeights()) -> float:
    r = score.f1 - w.count_weight * min(1.0, score.rel_count_error) - w.time_weight * seconds
    return round(float(r), 5)
