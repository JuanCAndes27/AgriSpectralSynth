"""
Real NEON hyperspectral data for the counting experiments.

NeonTreeEvaluation ships, for each RGB plot, the co-registered NEON AOP
surface reflectance cube (426 bands, 380-2510 nm, 1 m/pixel, int16 x
10000). Integrating each pixel's spectrum with a sensor's response
functions gives the bands that sensor would really have measured
(at 1 m): a real counterpart of the synthetic bands.

Uses
----
* validate the synthetic NDVI against real NDVI, sensor by sensor;
* "hsi" signal for the agent's actions: real NDVI resampled to the RGB
  grid (10 cm), so the same delineation methods can run on real spectra;
* filter DeepForest detections by real NDVI (remove non-vegetation).
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..sensors.registry import load_sensor
from ..sensors.srf import WAVELENGTHS, band_average

REFLECTANCE_SCALE = 10000.0
NODATA = -9999


def hsi_path(hsi_dir: Path, stem: str) -> Optional[Path]:
    p = Path(hsi_dir) / f"{stem}_hyperspectral.tif"
    return p if p.exists() else None


@lru_cache(maxsize=4)
def read_band_table(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """neon_aop_bands.csv -> (wavelengths nm, noisy-band flags)."""
    wl, noise = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = float(r["nanometer"])
            wl.append(v * 1000.0 if v < 10 else v)          # the file stores micrometres
            noise.append(int(float(r.get("noise", 0) or 0)))
    return np.asarray(wl), np.asarray(noise, bool)


def read_cube(path: Path, bands_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the reflectance bands that fall in 400-1000 nm.
    Returns (cube (B, H, W) float32 reflectance 0-1, wavelengths (B,) nm).
    """
    import rasterio

    wl, noise = read_band_table(str(bands_csv))
    sel = np.flatnonzero((wl >= WAVELENGTHS[0] - 10) & (wl <= WAVELENGTHS[-1] + 10) & ~noise)
    with rasterio.open(path) as src:
        if src.count != len(wl):
            raise ValueError(f"{Path(path).name}: {src.count} bands, band table has {len(wl)}")
        cube = src.read((sel + 1).tolist()).astype(np.float32)
    bad = cube <= NODATA + 1
    cube /= REFLECTANCE_SCALE
    cube[bad] = np.nan
    return np.clip(cube, 0.0, 1.5), wl[sel]


@lru_cache(maxsize=16)
def _interp_matrix(wl_key: tuple) -> np.ndarray:
    """Linear interpolation matrix from the cube's wavelengths to the 1 nm grid."""
    wl = np.asarray(wl_key)
    M = np.zeros((WAVELENGTHS.size, wl.size))
    for j in range(wl.size):
        e = np.zeros(wl.size)
        e[j] = 1.0
        M[:, j] = np.interp(WAVELENGTHS, wl, e)
    return M


def sensor_bands(cube: np.ndarray, wl: np.ndarray, sensor) -> Dict[str, np.ndarray]:
    """Bands the sensor would have measured: SRF-weighted mean of each pixel spectrum."""
    sensor = load_sensor(sensor)
    B, H, W = cube.shape
    spectra = cube.reshape(B, -1).T                       # (N, B)
    valid = np.isfinite(spectra).all(axis=1)
    out = np.full((H * W, len(sensor.band_list)), np.nan, np.float32)
    if valid.any():
        s1nm = spectra[valid] @ _interp_matrix(tuple(np.round(wl, 3))).T     # (n, 601)
        out[valid] = band_average(sensor.srf_matrix(), s1nm)
    return {b.name: out[:, i].reshape(H, W) for i, b in enumerate(sensor.band_list)}


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (nir - red) / (nir + red)
    return np.clip(np.nan_to_num(v, nan=0.0), -1, 1).astype(np.float32)


def upsample(img: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """1 m raster -> RGB grid (bilinear), same footprint."""
    return cv2.resize(np.nan_to_num(img, nan=0.0).astype(np.float32), (shape[1], shape[0]),
                      interpolation=cv2.INTER_LINEAR)


def aggregate(img: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """RGB grid -> 1 m (area average), to compare synthetic with real pixels."""
    return cv2.resize(img.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


class RealSpectra:
    """Real bands of one plot for one sensor, at 1 m and resampled to the RGB grid."""

    def __init__(self, cube: np.ndarray, wl: np.ndarray, sensor, rgb_shape: Tuple[int, int]):
        s = load_sensor(sensor)
        self.bands_1m = sensor_bands(cube, wl, s)
        self.ndvi_1m = ndvi(self.bands_1m[s.roles["red"]], self.bands_1m[s.roles["nir"]])
        self.nir_1m = np.nan_to_num(self.bands_1m[s.roles["nir"]], nan=0.0)
        self.ndvi = upsample(self.ndvi_1m, rgb_shape)
        self.nir = upsample(self.nir_1m, rgb_shape)


def attach_real_ndvi(sig, hsi_file: Path, bands_csv: Path, sensor, rgb_shape: Tuple[int, int]) -> RealSpectra:
    """Add the real-NDVI signal ("hsi") of one plot to an agent's Signals."""
    cube, wl = read_cube(hsi_file, bands_csv)
    real = RealSpectra(cube, wl, sensor, rgb_shape)
    sig.index["hsi"], sig.peaks["hsi"] = real.ndvi, real.nir
    return real
