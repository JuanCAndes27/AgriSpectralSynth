"""
Binary mask generation for vegetation.

AgriSpectralSynth
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_binary_mask(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Threshold a single-band image (index, probability...) into a 0/1 mask.

    Returns
    -------
    ndarray uint8 with values {0, 1}
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {image.shape}")
    return (np.nan_to_num(image, nan=-np.inf) > threshold).astype(np.uint8)


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Morphological opening + closing (removes speckle, fills small holes)."""
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


class VegetationMask:
    """
    Vegetation mask directly from RGB using chromatic excess-green.

    ``threshold`` is applied to ExG_c = (2G - R - B) / (R + G + B), which
    does not depend on image brightness. (The previous version min-max
    normalised ExG per image, so the threshold meant something different
    in every image.)
    """

    def __init__(self, threshold: float = 0.06, kernel_size: int = 5):
        self.threshold = threshold
        self.kernel_size = kernel_size

    @staticmethod
    def normalize(rgb: np.ndarray) -> np.ndarray:
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1:
            rgb /= 255.0
        return rgb

    def excess_green(self, rgb: np.ndarray) -> np.ndarray:
        rgb = self.normalize(rgb)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        return (2 * g - r - b) / (r + g + b + 1e-6)

    def compute(self, rgb: np.ndarray) -> np.ndarray:
        """Return a 0/255 uint8 mask."""
        mask = (self.excess_green(rgb) > self.threshold).astype(np.uint8) * 255
        return clean_mask(mask, self.kernel_size)

    def save(self, mask: np.ndarray, filename):
        cv2.imwrite(str(Path(filename)), mask)
