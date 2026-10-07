"""
The agent's actions: crown delineation methods ("arms").

Every arm turns one scene (RGB photo + synthetic bands of one sensor)
into a set of crown bounding boxes. The arms combine three classical,
training-free algorithms with two signals and three expected crown
sizes, so the right choice depends on the scene: dense small conifers,
sparse large oaks, shrubland...

Algorithms
----------
cc   threshold + connected components. Fast; merges touching crowns.
wsd  threshold + distance-transform watershed: splits touching blobs
     at their narrowest points; the crown size sets the minimum
     distance between markers.
lmw  local maxima + marker-controlled watershed: tree tops are the
     brightest points of a smoothed brightness image (NIR for the
     spectral signal, luminance for RGB); the crown size sets the
     smoothing and the minimum distance between tops.

Signals
-------
ndvi  synthetic NDVI of the sensor (mask) and its NIR band (tree tops)
exg   chromatic excess-green of the RGB photo and its luminance:
      the RGB-only baseline, identical for every sensor
dark  crowns as objects darker than the background (luminance), the
      classic cue in open woodland where trees are darker than dry soil

Thresholds
----------
fixed  a constant per signal (NDVI 0.45, ExG 0.06)
otsu   Otsu's threshold computed on each image (adapts to illumination
       and to dark crowns whose greenness is weak)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

THRESHOLDS = {"ndvi": 0.45, "exg": 0.06}
CROWN_SIZES_M = (2.0, 4.0, 7.0)


@dataclass(frozen=True)
class Arm:
    name: str
    algorithm: str            # cc | wsd | lmw
    signal: str               # ndvi | exg | dark
    crown_m: Optional[float]  # expected crown diameter (m); None for cc
    threshold: str = "fixed"  # fixed | otsu

    @property
    def sensor_dependent(self) -> bool:
        return self.signal == "ndvi"


def default_arms() -> List[Arm]:
    """32 arms: {ndvi, exg} x {fixed, otsu} x {cc, wsd x3, lmw x3}  +  dark/otsu x {cc, wsd x3}."""
    arms = []
    for signal in ("ndvi", "exg"):
        for thr in ("fixed", "otsu"):
            tag = f"{signal}{'' if thr == 'fixed' else '-otsu'}"
            arms.append(Arm(f"cc_{tag}", "cc", signal, None, thr))
            for algo in ("wsd", "lmw"):
                for d in CROWN_SIZES_M:
                    arms.append(Arm(f"{algo}_{tag}_{d:g}m", algo, signal, d, thr))
    arms.append(Arm("cc_dark", "cc", "dark", None, "otsu"))
    for d in CROWN_SIZES_M:
        arms.append(Arm(f"wsd_dark_{d:g}m", "wsd", "dark", d, "otsu"))
    return arms


ARMS: List[Arm] = default_arms()
ARM_INDEX: Dict[str, int] = {a.name: i for i, a in enumerate(ARMS)}


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

@dataclass
class Signals:
    """Per-scene rasters the arms work on (all float32, same shape)."""

    index: Dict[str, np.ndarray]      # ndvi, exg
    peaks: Dict[str, np.ndarray]      # brightness used to find tree tops


def exg_chromatic(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32)
    s = x.sum(axis=2) + 1e-6
    return (2 * x[..., 1] - x[..., 0] - x[..., 2]) / s


def luminance(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32)
    if x.max() > 1:
        x /= 255.0 if rgb.dtype == np.uint8 or x.max() <= 255 else 65535.0
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def make_signals(rgb: np.ndarray, ndvi: np.ndarray, nir: np.ndarray) -> Signals:
    lum = luminance(rgb)
    return Signals(
        index={"ndvi": ndvi.astype(np.float32), "exg": exg_chromatic(rgb),
               "dark": cv2.GaussianBlur(-lum, (0, 0), 1.0)},
        peaks={"ndvi": nir.astype(np.float32), "exg": lum, "dark": lum},
    )


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold of a float image (robust 1-99 % range)."""
    lo, hi = np.percentile(x, [1, 99])
    if hi - lo < 1e-6:
        return float(hi)
    hist, edges = np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(between)])


def arm_mask(arm: Arm, sig: Signals) -> np.ndarray:
    index = sig.index[arm.signal]
    thr = otsu_threshold(index) if arm.threshold == "otsu" else THRESHOLDS[arm.signal]
    return _clean(index > thr, 3)


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

def _clean(mask: np.ndarray, px: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, px | 1), max(3, px | 1)))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k).astype(bool)


def labels_to_boxes(labels: np.ndarray, min_area: int) -> np.ndarray:
    """(N, 4) boxes xmin, ymin, xmax, ymax of labelled regions with >= min_area pixels."""
    objs = ndi.find_objects(labels)
    areas = np.bincount(labels.ravel(), minlength=len(objs) + 1)
    out = []
    for i, sl in enumerate(objs, start=1):
        if sl is None or areas[i] < min_area:
            continue
        out.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
    return np.asarray(out, dtype=np.float32).reshape(-1, 4)


def run_arm(arm: Arm, sig: Signals, gsd_m: float = 0.1) -> np.ndarray:
    """Execute one arm. Returns crown boxes in pixel coordinates."""
    mask = arm_mask(arm, sig)

    if arm.algorithm == "cc":
        labels, _ = ndi.label(mask)
        min_area = int(np.pi * (1.0 / gsd_m) ** 2 * 0.25)         # crowns >= ~1 m diameter
        return labels_to_boxes(labels, min_area)

    radius_px = max(2.0, arm.crown_m / 2.0 / gsd_m)
    min_dist = max(2, int(round(0.8 * radius_px)))
    min_area = int(np.pi * (0.35 * radius_px) ** 2)

    if arm.algorithm == "wsd":
        dist = ndi.distance_transform_edt(mask)
        dist = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), max(1.0, 0.25 * radius_px))
        peaks = peak_local_max(dist, min_distance=min_dist, labels=mask.astype(np.int32), exclude_border=False)
        markers = np.zeros(mask.shape, np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        labels = watershed(-dist, markers, mask=mask)
        return labels_to_boxes(labels, min_area)

    if arm.algorithm == "lmw":
        bright = cv2.GaussianBlur(sig.peaks[arm.signal], (0, 0), max(1.0, 0.4 * radius_px))
        peaks = peak_local_max(bright, min_distance=min_dist, labels=mask.astype(np.int32), exclude_border=False)
        markers = np.zeros(mask.shape, np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        labels = watershed(-bright, markers, mask=mask)
        return labels_to_boxes(labels, min_area)

    raise ValueError(f"unknown algorithm {arm.algorithm}")
