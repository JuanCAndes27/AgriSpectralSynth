"""
NDVI - Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

Author
------
Juan Carlos Vega

License
-------
MIT
"""

from __future__ import annotations

import numpy as np

from .base import VegetationIndex


class NDVI(VegetationIndex):
    name = "NDVI"
    formula = "(NIR - RED) / (NIR + RED)"

    def compute(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        red, nir = self._f32(red, nir)
        return self._finish((nir - red) / (nir + red + self.epsilon))
