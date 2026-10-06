"""
SAVI - Soil Adjusted Vegetation Index (Huete, 1988).

    SAVI = (1 + L) * (NIR - Red) / (NIR + Red + L)
"""

from __future__ import annotations

import numpy as np

from .base import VegetationIndex


class SAVI(VegetationIndex):
    name = "SAVI"
    formula = "(1 + L) * (NIR - RED) / (NIR + RED + L)"

    def __init__(self, L: float = 0.5, epsilon: float = 1e-6):
        super().__init__(epsilon)
        self.L = L

    def compute(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        red, nir = self._f32(red, nir)
        return self._finish((1.0 + self.L) * (nir - red) / (nir + red + self.L))
