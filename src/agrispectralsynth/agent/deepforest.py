"""
DeepForest as an action of the agent.

DeepForest (Weinstein et al., 2019, 2020) is a RetinaNet (ResNet-50 FPN)
tree-crown detector trained on NEON RGB imagery. This module runs the
official release weights (``NEON.pt``, DeepForest 1.0.0) with plain
torchvision, so the full ``deepforest`` package and its training stack
are not needed:

    pip install -e ".[deepforest]"            # torch + torchvision
    agrispectralsynth-agent download-deepforest

Prediction follows DeepForest 1.0: RGB scaled to [0, 1], RetinaNet with
score threshold 0.1 and NMS 0.05, then a second NMS across boxes. Images
larger than one patch are processed with overlapping 400 px windows
(10 cm imagery, the resolution the model was trained on).
"""

from __future__ import annotations

import os
import time
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..constants import PROJECT_ROOT

WEIGHTS_URL = "https://github.com/weecology/DeepForest/releases/download/1.0.0/NEON.pt"
WEIGHTS_VERSION = "DeepForest 1.0.0 (NEON.pt)"
SCORE_THRESHOLD = 0.1
NMS_THRESHOLD = 0.05
PATCH = 400
OVERLAP = 0.1


def default_weights_path() -> Path:
    env = os.environ.get("AGRISPECTRALSYNTH_DEEPFOREST_WEIGHTS")
    return Path(env) if env else PROJECT_ROOT / "models" / "deepforest" / "NEON.pt"


def download_weights(dest: Optional[Path] = None) -> Path:
    """Download the official DeepForest 1.0.0 NEON weights (~130 MB) from GitHub."""
    dest = Path(dest or default_weights_path())
    if dest.exists() and dest.stat().st_size > 1e7:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    tmp.replace(dest)
    return dest


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except ImportError:
        return False


class DeepForestDetector:
    """RetinaNet tree-crown detector with DeepForest 1.0 release weights."""

    def __init__(self, weights: Optional[Path] = None, threads: Optional[int] = None):
        import torch
        from torchvision.models.detection import retinanet_resnet50_fpn
        from torchvision.models.detection.retinanet import RetinaNet

        weights = Path(weights or default_weights_path())
        if not weights.exists():
            raise FileNotFoundError(
                f"DeepForest weights not found at {weights}. Run: agrispectralsynth-agent download-deepforest"
            )
        if threads:
            torch.set_num_threads(threads)
        backbone = retinanet_resnet50_fpn(weights=None, weights_backbone=None).backbone
        model = RetinaNet(backbone=backbone, num_classes=1)
        model.score_thresh = SCORE_THRESHOLD
        model.nms_thresh = NMS_THRESHOLD
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval()
        self.model = model
        self.torch = torch
        self.weights = weights

    def _predict_patch(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32)).permute(2, 0, 1) / 255.0
        with self.torch.no_grad():
            p = self.model([x])[0]
        return p["boxes"].numpy(), p["scores"].numpy()

    def predict(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        rgb: (H, W, 3) uint8 RGB. Returns (boxes (N, 4) xmin ymin xmax ymax, scores (N,), seconds).
        """
        from torchvision.ops import nms

        t0 = time.perf_counter()
        h, w = rgb.shape[:2]
        if max(h, w) <= int(PATCH * 1.5):
            boxes, scores = self._predict_patch(rgb)
        else:
            step = int(PATCH * (1 - OVERLAP))
            bs, ss = [], []
            for y in range(0, max(h - PATCH, 0) + step, step):
                for x in range(0, max(w - PATCH, 0) + step, step):
                    y0, x0 = min(y, max(h - PATCH, 0)), min(x, max(w - PATCH, 0))
                    b, s = self._predict_patch(rgb[y0:y0 + PATCH, x0:x0 + PATCH])
                    bs.append(b + np.array([x0, y0, x0, y0], np.float32))
                    ss.append(s)
            boxes, scores = np.concatenate(bs), np.concatenate(ss)
        boxes, scores = np.asarray(boxes, np.float32), np.asarray(scores, np.float32)
        if len(boxes):
            keep = nms(self.torch.from_numpy(boxes), self.torch.from_numpy(scores), NMS_THRESHOLD).numpy()
            boxes, scores = boxes[keep], scores[keep]
        return boxes.astype(np.float32).reshape(-1, 4), scores.astype(np.float32), time.perf_counter() - t0


@lru_cache(maxsize=2)
def get_detector(weights: Optional[str] = None, threads: Optional[int] = None) -> DeepForestDetector:
    """One detector per process (loading the weights takes ~1 s)."""
    return DeepForestDetector(Path(weights) if weights else None, threads)
