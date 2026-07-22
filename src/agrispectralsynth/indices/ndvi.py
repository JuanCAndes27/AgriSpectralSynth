"""
NDVI computation module for AgriSpectralSynth.

Computes the Normalized Difference Vegetation Index (NDVI)
from synthetic Red and Near Infrared bands.

Author
------
Juan Carlos Vega
OpenAI Collaboration

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine


class NDVI:
    """
    Normalized Difference Vegetation Index.
    """

    def __init__(self, epsilon: float = 1e-8):

        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # NDVI
    # ------------------------------------------------------------------

    def compute(
        self,
        red: np.ndarray,
        nir: np.ndarray,
    ) -> np.ndarray:

        red = red.astype(np.float32)
        nir = nir.astype(np.float32)

        ndvi = (nir - red) / (nir + red + self.epsilon)

        return np.clip(ndvi, -1.0, 1.0)

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------

    def normalize(
        self,
        ndvi: np.ndarray,
    ) -> np.ndarray:

        return (ndvi + 1.0) / 2.0

    # ------------------------------------------------------------------
    # UINT8
    # ------------------------------------------------------------------

    def to_uint8(
        self,
        ndvi: np.ndarray,
    ) -> np.ndarray:

        img = self.normalize(ndvi)

        return (255 * img).astype(np.uint8)

    # ------------------------------------------------------------------
    # COLOR
    # ------------------------------------------------------------------

    def colorize(
        self,
        ndvi: np.ndarray,
    ) -> np.ndarray:

        gray = self.to_uint8(ndvi)

        color = cv2.applyColorMap(
            gray,
            cv2.COLORMAP_TURBO,
        )

        return cv2.cvtColor(
            color,
            cv2.COLOR_BGR2RGB,
        )

    # ------------------------------------------------------------------
    # SAVE JPG
    # ------------------------------------------------------------------

    def save_jpg(
        self,
        ndvi: np.ndarray,
        filename: str | Path,
        color: bool = False,
        quality: int = 95,
    ):

        filename = Path(filename)

        if color:

            image = self.colorize(ndvi)

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        else:

            image = self.to_uint8(ndvi)

        cv2.imwrite(
            str(filename),
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                quality,
            ],
        )

    # ------------------------------------------------------------------
    # SAVE GEOTIFF FLOAT32
    # ------------------------------------------------------------------

    def save_geotiff(
        self,
        ndvi: np.ndarray,
        filename: str | Path,
    ):
        """
        Saves NDVI as Float32 GeoTIFF.

        The image is intentionally saved WITHOUT
        georeferencing because synthetic datasets
        generated from RGB drone images may not
        contain spatial metadata.
        """

        filename = Path(filename)

        ndvi = ndvi.astype(np.float32)

        with rasterio.open(
            filename,
            "w",
            driver="GTiff",
            width=ndvi.shape[1],
            height=ndvi.shape[0],
            count=1,
            dtype="float32",
            transform=Affine.identity(),
            crs=None,
            compress="lzw",
        ) as dst:

            dst.write(ndvi, 1)

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------

    def statistics(
        self,
        ndvi: np.ndarray,
    ) -> dict:

        return {

            "minimum": float(np.min(ndvi)),

            "maximum": float(np.max(ndvi)),

            "mean": float(np.mean(ndvi)),

            "std": float(np.std(ndvi)),
        }

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    def summary(self):

        print("=" * 60)
        print("NDVI MODULE")
        print("=" * 60)
        print("Formula : (NIR - RED) / (NIR + RED)")
        print("Output  : Float32")
        print("Range   : [-1,1]")
        print("=" * 60)