"""
GNDVI module.

Green Normalized Difference Vegetation Index
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine


class GNDVI:

    def __init__(self, epsilon: float = 1e-8):

        self.epsilon = epsilon

    def compute(self, green: np.ndarray, nir: np.ndarray):

        green = green.astype(np.float32)
        nir = nir.astype(np.float32)

        gndvi = (nir - green) / (nir + green + self.epsilon)

        return np.clip(gndvi, -1.0, 1.0)

    def normalize(self, image):

        return (image + 1.0) / 2.0

    def to_uint8(self, image):

        return (255 * self.normalize(image)).astype(np.uint8)

    def colorize(self, image):

        img = self.to_uint8(image)

        img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def save_jpg(self, image, filename, color=False):

        filename = Path(filename)

        if color:

            img = self.colorize(image)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        else:

            img = self.to_uint8(image)

        cv2.imwrite(str(filename), img)

    def save_geotiff(self, image, filename):

        filename = Path(filename)

        image = image.astype(np.float32)

        with rasterio.open(
            filename,
            "w",
            driver="GTiff",
            width=image.shape[1],
            height=image.shape[0],
            count=1,
            dtype="float32",
            transform=Affine.identity(),
            crs=None,
            compress="lzw",
        ) as dst:

            dst.write(image, 1)