"""
NDRE - Normalized Difference Red Edge index.

    NDRE = (NIR - RedEdge) / (NIR + RedEdge)

More sensitive than NDVI to chlorophyll in dense canopies (NDVI
saturates); natural fit for the Mavic 3M red-edge band.
"""

from __future__ import annotations

import numpy as np

from .base import VegetationIndex


class NDRE(VegetationIndex):
    name = "NDRE"
    formula = "(NIR - REDEDGE) / (NIR + REDEDGE)"

    def compute(self, red_edge: np.ndarray, nir: np.ndarray) -> np.ndarray:
        red_edge, nir = self._f32(red_edge, nir)
        return self._finish((nir - red_edge) / (nir + red_edge + self.epsilon))
