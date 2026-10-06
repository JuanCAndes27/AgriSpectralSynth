"""
EVI - Enhanced Vegetation Index (Huete et al., 2002).

    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

On the Mavic 3M the blue band comes from the RGB camera (the
multispectral camera has no blue band).
"""

from __future__ import annotations

import numpy as np

from .base import VegetationIndex


class EVI(VegetationIndex):
    name = "EVI"
    formula = "2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)"

    def __init__(self, G=2.5, C1=6.0, C2=7.5, L=1.0, epsilon: float = 1e-6):
        super().__init__(epsilon)
        self.G, self.C1, self.C2, self.L = G, C1, C2, L

    def compute(self, blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        blue, red, nir = self._f32(blue, red, nir)
        denom = nir + self.C1 * red - self.C2 * blue + self.L
        denom = np.where(np.abs(denom) < self.epsilon, self.epsilon, denom)
        return self._finish(self.G * (nir - red) / denom)
