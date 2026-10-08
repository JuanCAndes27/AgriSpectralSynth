"""
Spectral response functions (SRF).

Every spectrum in AgriSpectralSynth is sampled on the same grid,
400-1000 nm every 1 nm. A band's SRF is a weight vector on that grid;
the simulated band value is the SRF-weighted mean of the spectrum:

    rho_band = sum_l S(l) * rho(l) / sum_l S(l)

Manufacturers of drone cameras only publish centre and width, so by
default the SRF is a Gaussian whose full width at half maximum (FWHM)
equals the published width. A measured curve can be given instead as a
CSV file (``srf_file``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

WL_MIN = 400.0
WL_MAX = 1000.0
WL_STEP = 1.0
WAVELENGTHS = np.arange(WL_MIN, WL_MAX + WL_STEP / 2, WL_STEP, dtype=np.float64)

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ~0.4247


def gaussian_srf(center: float, fwhm: float, wl: np.ndarray = WAVELENGTHS) -> np.ndarray:
    """Gaussian response with peak 1 at ``center`` and the given FWHM (nm)."""
    sigma = fwhm * FWHM_TO_SIGMA
    return np.exp(-0.5 * ((wl - center) / sigma) ** 2)


def ir_cut(edge_nm: float, width_nm: float = 10.0, wl: np.ndarray = WAVELENGTHS) -> np.ndarray:
    """Sigmoid IR-cut filter: ~1 below ``edge_nm``, ~0 above."""
    return 1.0 / (1.0 + np.exp((wl - edge_nm) / width_nm))


def tabulated_srf(path: Path, wl: np.ndarray = WAVELENGTHS) -> np.ndarray:
    """Load a measured SRF (CSV: wavelength_nm,response) and resample it on ``wl``."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path}: expected two columns wavelength_nm,response")
    order = np.argsort(data[:, 0])
    resp = np.interp(wl, data[order, 0], data[order, 1], left=0.0, right=0.0)
    if resp.max() <= 0:
        raise ValueError(f"{path}: response is zero inside {WL_MIN:.0f}-{WL_MAX:.0f} nm")
    return resp / resp.max()


def band_average(srf: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    """
    SRF-weighted mean of one or more spectra.

    srf : (n_bands, n_wl) or (n_wl,)
    spectra : (..., n_wl)
    returns : (..., n_bands)
    """
    srf = np.atleast_2d(srf)
    weights = srf / srf.sum(axis=1, keepdims=True)
    return spectra @ weights.T


def overlap(srf_a: np.ndarray, srf_b: np.ndarray) -> float:
    """Fraction of band A's response that falls inside band B (B scaled to peak 1)."""
    b = srf_b / srf_b.max()
    return float((srf_a * b).sum() / srf_a.sum())


def fwhm_of(srf: np.ndarray, wl: Optional[np.ndarray] = None) -> float:
    """Measured FWHM of a sampled response (for checks and documentation)."""
    wl = WAVELENGTHS if wl is None else wl
    half = srf.max() / 2.0
    above = np.flatnonzero(srf >= half)
    return float(wl[above[-1]] - wl[above[0]]) if above.size else 0.0
