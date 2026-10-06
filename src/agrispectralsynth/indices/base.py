"""
Common functionality shared by every vegetation index.

Each index only has to implement ``compute``; normalisation, colouring,
export and statistics live here (they used to be copy-pasted in every
module).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine

from ..utils.colormaps import colorize


class VegetationIndex:
    """Base class. Values are float32 in ``[vmin, vmax]`` (default [-1, 1])."""

    name: str = "INDEX"
    formula: str = ""
    vmin: float = -1.0
    vmax: float = 1.0

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    # ------------------------------------------------------------------

    def compute(self, *bands: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, *bands: np.ndarray) -> np.ndarray:
        return self.compute(*bands)

    @staticmethod
    def _f32(*arrays: np.ndarray):
        return [np.asarray(a, dtype=np.float32) for a in arrays]

    def _finish(self, index: np.ndarray) -> np.ndarray:
        index = np.nan_to_num(index, nan=0.0, posinf=self.vmax, neginf=self.vmin)
        return np.clip(index, self.vmin, self.vmax, out=index).astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def normalize(self, index: np.ndarray) -> np.ndarray:
        return (index - self.vmin) / (self.vmax - self.vmin)

    def to_uint8(self, index: np.ndarray) -> np.ndarray:
        return np.clip(self.normalize(index) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def colorize(self, index: np.ndarray, cmap: str = "RdYlGn") -> np.ndarray:
        """Return an RGB uint8 image using a 256-entry LUT (fast)."""
        return colorize(index, cmap=cmap, vmin=self.vmin, vmax=self.vmax)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_jpg(self, index, filename, color: bool = False, quality: int = 95, cmap: str = "RdYlGn"):
        img = cv2.cvtColor(self.colorize(index, cmap), cv2.COLOR_RGB2BGR) if color else self.to_uint8(index)
        cv2.imwrite(str(filename), img, [cv2.IMWRITE_JPEG_QUALITY, quality])

    def save_png(self, index, filename, color: bool = True, cmap: str = "RdYlGn"):
        img = cv2.cvtColor(self.colorize(index, cmap), cv2.COLOR_RGB2BGR) if color else self.to_uint8(index)
        cv2.imwrite(str(filename), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def save_geotiff(self, index, filename, transform=None, crs=None):
        """Float32 GeoTIFF. Keeps georeferencing when ``transform``/``crs`` are given."""
        import warnings

        from rasterio.errors import NotGeoreferencedWarning

        index = np.asarray(index, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            self._write_tif(index, filename, transform, crs)

    def _write_tif(self, index, filename, transform, crs):
        with rasterio.open(
            Path(filename),
            "w",
            driver="GTiff",
            width=index.shape[1],
            height=index.shape[0],
            count=1,
            dtype="float32",
            transform=transform if transform is not None else Affine.identity(),
            crs=crs,
            compress="deflate",
            predictor=3,
        ) as dst:
            dst.write(index, 1)
            dst.set_band_description(1, self.name)

    # ------------------------------------------------------------------

    def statistics(self, index: np.ndarray) -> dict:
        p5, p50, p95 = np.percentile(index, [5, 50, 95])
        return {
            "minimum": float(index.min()),
            "maximum": float(index.max()),
            "mean": float(index.mean()),
            "std": float(index.std()),
            "p5": float(p5),
            "median": float(p50),
            "p95": float(p95),
        }

    def summary(self):
        print("=" * 60)
        print(f"{self.name} MODULE")
        print("=" * 60)
        print(f"Formula : {self.formula}")
        print("Output  : Float32")
        print(f"Range   : [{self.vmin}, {self.vmax}]")
        print("=" * 60)
