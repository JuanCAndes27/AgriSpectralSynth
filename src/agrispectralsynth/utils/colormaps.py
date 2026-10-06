"""
Fast colour mapping through 256-entry lookup tables.

Calling a ``matplotlib`` colormap on a full image builds a float64 RGBA
array (4x the memory of the input) on every call. Here the colormap is
sampled once into a uint8 LUT and the image is mapped with OpenCV
(``applyColorMap`` with a user LUT), ~100x faster.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=16)
def get_lut(cmap: str = "RdYlGn") -> np.ndarray:
    """(256, 3) uint8 RGB lookup table for a matplotlib colormap name."""
    from matplotlib import colormaps  # lazy: only needed once per process

    rgba = colormaps[cmap](np.linspace(0.0, 1.0, 256))
    lut = np.round(rgba[:, :3] * 255.0).astype(np.uint8)
    lut.setflags(write=False)
    return lut


@lru_cache(maxsize=16)
def _bgr_lut(cmap: str) -> np.ndarray:
    return np.ascontiguousarray(get_lut(cmap)[:, ::-1].reshape(256, 1, 3))


def to_index(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map float values to uint8 LUT indices; NaN and values < vmin -> 0, > vmax -> 255."""
    values = np.asarray(values, dtype=np.float32)
    if np.isnan(values).any():
        values = np.nan_to_num(values, nan=vmin)
    alpha = 255.0 / (vmax - vmin)
    # convertScaleAbs saturates at 255; clamping below vmin first avoids the abs()
    return cv2.convertScaleAbs(np.maximum(values, vmin), alpha=alpha, beta=-vmin * alpha)


def colorize_bgr(values: np.ndarray, cmap: str = "RdYlGn", vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    """(H, W, 3) uint8 in BGR order, ready for cv2.imwrite / imencode."""
    return cv2.applyColorMap(to_index(values, vmin, vmax), _bgr_lut(cmap))


def colorize(values: np.ndarray, cmap: str = "RdYlGn", vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    """(H, W, 3) uint8 in RGB order."""
    return cv2.cvtColor(colorize_bgr(values, cmap, vmin, vmax), cv2.COLOR_BGR2RGB)
