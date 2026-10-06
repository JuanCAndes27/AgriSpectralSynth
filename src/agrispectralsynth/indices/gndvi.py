"""
GNDVI - Green Normalized Difference Vegetation Index.

    GNDVI = (NIR - Green) / (NIR + Green)
"""

from __future__ import annotations

import numpy as np

from .base import VegetationIndex


class GNDVI(VegetationIndex):
    name = "GNDVI"
    formula = "(NIR - GREEN) / (NIR + GREEN)"

    def compute(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        green, nir = self._f32(green, nir)
        return self._finish((nir - green) / (nir + green + self.epsilon))
