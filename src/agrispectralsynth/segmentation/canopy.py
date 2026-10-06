"""
Tree canopy segmentation.

AgriSpectralSynth
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .masks import clean_mask


def remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than ``min_area`` pixels (vectorised)."""
    if min_area <= 1:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # background
    return keep[labels].astype(mask.dtype) * mask.max(initial=1)


def canopy_mask(
    ndvi: np.ndarray,
    threshold: float = 0.45,
    veg_fraction: Optional[np.ndarray] = None,
    min_fraction: float = 0.5,
    kernel_size: int = 5,
    min_area: int = 30,
) -> np.ndarray:
    """
    Canopy mask from NDVI (optionally gated by the vegetation fraction).

    Returns
    -------
    ndarray uint8 {0, 1}, same shape as ``ndvi``.
    """
    ndvi = np.asarray(ndvi)
    mask = np.nan_to_num(ndvi, nan=-1.0) > threshold
    if veg_fraction is not None:
        mask &= veg_fraction >= min_fraction
    mask = clean_mask(mask.astype(np.uint8), kernel_size)
    return remove_small_objects(mask, min_area)


class CanopySegmenter:
    """Instance-level crowns (connected components) from a binary mask."""

    def __init__(self, min_area: int = 500):
        self.min_area = min_area

    def segment(self, mask: np.ndarray):
        """
        Returns
        -------
        output : uint8 mask (0/255) with only the kept components
        objects : list of dicts with id, bbox, area, centroid
        """
        mask = (mask > 0).astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        keep = stats[:, cv2.CC_STAT_AREA] >= self.min_area
        keep[0] = False
        # One indexed read instead of a full-image comparison per component
        output = keep[labels].astype(np.uint8) * 255

        objects = [
            {
                "id": int(i),
                "bbox": tuple(int(v) for v in stats[i, :4]),
                "area": int(stats[i, cv2.CC_STAT_AREA]),
                "centroid": (float(centroids[i][0]), float(centroids[i][1])),
            }
            for i in np.flatnonzero(keep)
        ]
        return output, objects

    def contours(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw(self, rgb: np.ndarray, objects):
        image = rgb.copy()
        for obj in objects:
            x, y, w, h = obj["bbox"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image
