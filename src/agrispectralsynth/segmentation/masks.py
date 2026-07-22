"""
Binary mask generation for vegetation.

AgriSpectralSynth
"""

from __future__ import annotations

import cv2
import numpy as np


class VegetationMask:

    def __init__(
        self,
        threshold: float = 0.25,
        kernel_size: int = 5,
    ):

        self.threshold = threshold
        self.kernel = np.ones(
            (kernel_size, kernel_size),
            np.uint8,
        )

    # ------------------------------------------------------------

    @staticmethod
    def normalize(rgb):

        rgb = rgb.astype(np.float32)

        if rgb.max() > 1:

            rgb /= 255.

        return rgb

    # ------------------------------------------------------------

    def excess_green(self, rgb):

        rgb = self.normalize(rgb)

        r = rgb[:, :, 0]

        g = rgb[:, :, 1]

        b = rgb[:, :, 2]

        exg = 2 * g - r - b

        exg -= exg.min()

        exg /= exg.max() + 1e-6

        return exg

    # ------------------------------------------------------------

    def compute(self, rgb):

        exg = self.excess_green(rgb)

        mask = exg > self.threshold

        mask = mask.astype(np.uint8) * 255

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel,
        )

        return mask

    # ------------------------------------------------------------

    def save(self, mask, filename):

        cv2.imwrite(filename, mask)