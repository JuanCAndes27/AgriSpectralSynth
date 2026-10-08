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
hsi   REAL NDVI from the NEON hyperspectral cube (1 m, integrated with the
      sensor's bands and resampled to the RGB grid); only when available

Detector
--------
df    DeepForest (RetinaNet trained on NEON RGB, see deepforest.py). One
      inference per image; the arms differ in the minimum detection score
      kept (0.1 ... 0.4). ~100x slower than the classical arms on a CPU,
      which is what makes the time term of the reward meaningful.
      df_rgb_*  : detections as they come
      df_ndvi_* : detections whose median SYNTHETIC NDVI is >= 0.3
      df_hsi_*  : detections whose median REAL (hyperspectral) NDVI is >= 0.3

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

THRESHOLDS = {"ndvi": 0.45, "exg": 0.06, "hsi": 0.45}
DF_NDVI_FILTER = 0.3      # a priori vegetation threshold for filtering detections
CROWN_SIZES_M = (2.0, 4.0, 7.0)


DF_SCORES = (0.1, 0.2, 0.3, 0.4)


@dataclass(frozen=True)
class Arm:
    name: str
    algorithm: str            # cc | wsd | lmw | df
    signal: str               # ndvi | exg | dark | rgb (df)
    crown_m: Optional[float]  # expected crown diameter (m); None for cc / df
    threshold: str = "fixed"  # fixed | otsu
    score: Optional[float] = None   # df: minimum detection score

    @property
    def sensor_dependent(self) -> bool:
        return self.signal in ("ndvi", "hsi")

    @property
    def needs_hsi(self) -> bool:
        return self.signal == "hsi"

    @property
    def is_detector(self) -> bool:
        return self.algorithm == "df"


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


def deepforest_arms(filter_signal: str = "rgb") -> List[Arm]:
    return [Arm(f"df_{filter_signal}_s{s:g}", "df", filter_signal, None, "fixed", s) for s in DF_SCORES]


def hsi_arms() -> List[Arm]:
    """Same classical algorithms on the real (hyperspectral) NDVI."""
    arms = []
    for thr in ("fixed", "otsu"):
        tag = "hsi" if thr == "fixed" else "hsi-otsu"
        arms.append(Arm(f"cc_{tag}", "cc", "hsi", None, thr))
        for algo in ("wsd", "lmw"):
            for d in CROWN_SIZES_M:
                arms.append(Arm(f"{algo}_{tag}_{d:g}m", algo, "hsi", d, thr))
    return arms


CLASSICAL_ARMS: List[Arm] = default_arms()
DF_ARMS: List[Arm] = deepforest_arms("rgb") + deepforest_arms("ndvi")
HSI_ARMS: List[Arm] = hsi_arms()
DFH_ARMS: List[Arm] = deepforest_arms("hsi")
ARMS: List[Arm] = CLASSICAL_ARMS + DF_ARMS + HSI_ARMS + DFH_ARMS


def arms_for(deepforest: bool = False, hsi: bool = False) -> List[Arm]:
    """The arms of an experiment, depending on what is available."""
    arms = list(CLASSICAL_ARMS)
    if deepforest:
        arms += DF_ARMS
    if hsi:
        arms += HSI_ARMS + (DFH_ARMS if deepforest else [])
    return arms


ARM_GROUPS = ("classical", "df", "hsi")


def arm_group(arm: Arm) -> str:
    """classical | df | hsi  (DeepForest filtered by real NDVI counts as 'hsi': it needs the cube)."""
    if arm.needs_hsi:
        return "hsi"
    return "df" if arm.is_detector else "classical"


def arm_names_in(groups) -> List[str]:
    """Names of the arms in the given groups. A DeepForest+hyperspectral arm needs both 'df' and 'hsi'."""
    groups = set(groups)
    unknown = groups - set(ARM_GROUPS)
    if unknown:
        raise ValueError(f"unknown action groups {sorted(unknown)}; use {ARM_GROUPS}")
    out = []
    for a in ARMS:
        g = arm_group(a)
        if g in groups and (not (a.is_detector and a.needs_hsi) or "df" in groups):
            out.append(a.name)
    return out


ARM_INDEX: Dict[str, int] = {a.name: i for i, a in enumerate(ARMS)}


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

@dataclass
class Signals:
    """Per-scene rasters the arms work on (all float32, same shape)."""

    index: Dict[str, np.ndarray]      # ndvi, exg, dark (+ hsi when real hyperspectral is available)
    peaks: Dict[str, np.ndarray]      # brightness used to find tree tops
    rgb: Optional[np.ndarray] = None  # uint8 photo, for the detector
    detections: Optional[tuple] = None   # (boxes, scores, seconds) of DeepForest, computed once


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
        rgb=rgb,
    )


def filter_by_index(boxes: np.ndarray, index: np.ndarray, threshold: float) -> np.ndarray:
    """Keep boxes whose median index inside the box is >= threshold."""
    if len(boxes) == 0:
        return boxes
    h, w = index.shape
    keep = np.zeros(len(boxes), bool)
    for i, (x0, y0, x1, y1) in enumerate(np.round(boxes).astype(int)):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))
        keep[i] = np.median(index[y0:y1, x0:x1]) >= threshold
    return boxes[keep]


def detect(sig: Signals, weights: Optional[str] = None) -> tuple:
    """Run DeepForest once per scene and cache (boxes, scores, seconds) in the signals."""
    if sig.detections is None:
        from .deepforest import get_detector

        if sig.rgb is None:
            raise ValueError("DeepForest needs the RGB photo (Signals.rgb)")
        sig.detections = get_detector(weights).predict(sig.rgb)
    return sig.detections


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
    if arm.is_detector:
        boxes, scores, _ = detect(sig)
        boxes = boxes[scores >= arm.score]
        if arm.signal != "rgb":
            boxes = filter_by_index(boxes, sig.index[arm.signal], DF_NDVI_FILTER)
        return boxes

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
