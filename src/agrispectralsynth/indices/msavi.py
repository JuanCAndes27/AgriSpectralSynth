"""
MSAVI2 - Modified Soil Adjusted Vegetation Index (Qi et al., 1994).

    MSAVI = (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2

For reflectances in [0, 1] the radicand equals (2*NIR - 1)^2 + 8*Red >= 0,
so no NaNs are produced.
"""

from __future__ import annotations

import numpy as np

from .base import VegetationIndex


class MSAVI(VegetationIndex):
    name = "MSAVI"
    formula = "(2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - RED))) / 2"

    def compute(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        red, nir = self._f32(red, nir)
        a = 2.0 * nir + 1.0
        radicand = np.maximum(a * a - 8.0 * (nir - red), 0.0)
        return self._finish((a - np.sqrt(radicand)) / 2.0)
