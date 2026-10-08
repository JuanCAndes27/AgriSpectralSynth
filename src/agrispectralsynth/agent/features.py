"""
Context vector: what the agent "sees" before choosing an action.

Cheap, scene-level descriptors that should explain which delineation
method works: how much vegetation, how green, how dark the crowns are
relative to the ground, how textured the canopy is, and at what scale
the vegetation is organised (granulometry). No site name or location
is used, so the agent cannot memorise sites.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from .actions import Signals, otsu_threshold

SENSOR_FEATURES = ("dji_mavic3m", "dji_phantom4m", "micasense_rededge_mx", "parrot_sequoia_plus")

FEATURE_NAMES: List[str] = [
    "veg_fraction",        # mean vegetation fraction of the spectral engine
    "ndvi_mean", "ndvi_std", "ndvi_p90",
    "exg_mean", "exg_std",
    "lum_mean", "lum_std",
    "dark_fraction",       # pixels darker than the Otsu threshold of luminance
    "green_dark_corr",     # correlation between greenness and darkness (crowns = dark AND green?)
    "texture",             # mean gradient magnitude of luminance
    "gran_2m", "gran_4m", "gran_7m",   # share of the green mask surviving an opening of that diameter
    "otsu_exg",            # Otsu threshold of ExG (how separable vegetation is)
    # "Probe" features: a quick look at the objects in the scene. Blob density
    # and size decide the expected crown size; green vs dark blobs decide the signal.
    "green_blobs_ha", "green_blob_area_m2", "green_blob_fill",
    "dark_blobs_ha", "dark_blob_area_m2", "dark_blob_fill",
] + [f"sensor_{s}" for s in SENSOR_FEATURES]


def _granulometry(mask: np.ndarray, gsd_m: float) -> List[float]:
    total = mask.sum()
    if total == 0:
        return [0.0, 0.0, 0.0]
    out = []
    for d in (2.0, 4.0, 7.0):
        k = max(3, int(round(d / gsd_m)) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out.append(float(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, ker).sum() / total))
    return out


def _blob_stats(mask: np.ndarray, gsd_m: float):
    """(blobs per hectare, log10 median blob area in m2, solidity-like fill of blobs)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    min_px = (0.5 / gsd_m) ** 2                      # ignore blobs smaller than 0.25 m2
    keep = areas >= min_px
    areas = areas[keep]
    ha = mask.size * gsd_m ** 2 / 1e4
    if areas.size == 0:
        return 0.0, 0.0, 0.0
    boxes = stats[1:, cv2.CC_STAT_WIDTH][keep].astype(float) * stats[1:, cv2.CC_STAT_HEIGHT][keep]
    fill = float(np.median(areas / np.maximum(boxes, 1)))
    return float(areas.size / ha), float(np.log10(np.median(areas) * gsd_m ** 2)), fill


def scene_features(sig: Signals, veg_fraction: np.ndarray, sensor_id: str, gsd_m: float = 0.1) -> Dict[str, float]:
    ndvi, exg, lum = sig.index["ndvi"], sig.index["exg"], sig.peaks["exg"]
    dark = lum < otsu_threshold(lum)
    green = exg > otsu_threshold(exg)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    a, b = exg.ravel(), -lum.ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else 0.0
    g2, g4, g7 = _granulometry(green, gsd_m)
    f = {
        "veg_fraction": float(veg_fraction.mean()),
        "ndvi_mean": float(ndvi.mean()),
        "ndvi_std": float(ndvi.std()),
        "ndvi_p90": float(np.percentile(ndvi, 90)),
        "exg_mean": float(exg.mean()),
        "exg_std": float(exg.std()),
        "lum_mean": float(lum.mean()),
        "lum_std": float(lum.std()),
        "dark_fraction": float(dark.mean()),
        "green_dark_corr": corr,
        "texture": float(np.sqrt(gx ** 2 + gy ** 2).mean()),
        "gran_2m": g2, "gran_4m": g4, "gran_7m": g7,
        "otsu_exg": otsu_threshold(exg),
    }
    for tag, m in (("green", green), ("dark", dark)):
        n_ha, area, fill = _blob_stats(m, gsd_m)
        f[f"{tag}_blobs_ha"], f[f"{tag}_blob_area_m2"], f[f"{tag}_blob_fill"] = n_ha, area, fill
    for s in SENSOR_FEATURES:
        f[f"sensor_{s}"] = 1.0 if sensor_id == s else 0.0
    return f


def feature_vector(f: Dict[str, float]) -> np.ndarray:
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float64)
