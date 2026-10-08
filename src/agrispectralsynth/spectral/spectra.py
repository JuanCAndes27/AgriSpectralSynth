"""
Continuous reflectance spectra (400-1000 nm, 1 nm) and the on-disk
spectral library under ``spectral_library/``.

CSV format (one file per material)::

    # name: healthy
    # category: vegetation
    # ...any other "# key: value" metadata...
    wavelength_nm,reflectance
    400,0.01774
    ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from ..constants import PROJECT_ROOT
from ..sensors.srf import WAVELENGTHS


@dataclass(frozen=True)
class Spectrum:
    name: str
    reflectance: np.ndarray            # on the common WAVELENGTHS grid
    category: str = ""
    metadata: Dict[str, str] = field(default_factory=dict, compare=False)

    def at(self, wavelength_nm: float) -> float:
        return float(np.interp(wavelength_nm, WAVELENGTHS, self.reflectance))


def library_dir() -> Path:
    env = os.environ.get("AGRISPECTRALSYNTH_LIBRARY")
    path = Path(env) if env else PROJECT_ROOT / "spectral_library"
    if not path.is_dir():
        raise FileNotFoundError(f"Spectral library not found: {path}")
    return path


def read_spectrum_csv(path: Union[str, Path]) -> Spectrum:
    path = Path(path)
    meta: Dict[str, str] = {}
    rows: List[List[float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if ":" in line:
                    k, v = line[1:].split(":", 1)
                    meta[k.strip()] = v.strip()
                continue
            if line[0].isalpha():  # header
                continue
            wl, r = line.split(",")[:2]
            rows.append([float(wl), float(r)])
    if not rows:
        raise ValueError(f"{path}: no data rows")
    data = np.asarray(rows)
    order = np.argsort(data[:, 0])
    wl, r = data[order, 0], data[order, 1]
    if wl[0] > WAVELENGTHS[0] or wl[-1] < WAVELENGTHS[-1]:
        raise ValueError(f"{path}: must cover {WAVELENGTHS[0]:.0f}-{WAVELENGTHS[-1]:.0f} nm")
    resampled = np.interp(WAVELENGTHS, wl, r)
    if resampled.min() < 0 or resampled.max() > 1.5:
        raise ValueError(f"{path}: reflectance outside [0, 1.5]")
    return Spectrum(
        name=meta.get("name", path.stem),
        category=meta.get("category", path.parent.name),
        reflectance=resampled,
        metadata=meta,
    )


@lru_cache(maxsize=64)
def load_spectrum(name: str) -> Spectrum:
    """
    Load a library spectrum by ``"category/name"`` (e.g. ``"vegetation/healthy"``),
    by bare name if it is unique, or by path to a CSV file.
    """
    p = Path(name)
    if p.suffix == ".csv" and p.exists():
        return read_spectrum_csv(p)
    lib = library_dir()
    candidate = lib / f"{name}.csv"
    if candidate.exists():
        return read_spectrum_csv(candidate)
    matches = sorted(lib.rglob(f"{p.name}.csv"))
    if len(matches) == 1:
        return read_spectrum_csv(matches[0])
    if not matches:
        raise KeyError(f"Spectrum '{name}' not found in {lib}. Available: {', '.join(list_spectra())}")
    raise KeyError(f"Spectrum '{name}' is ambiguous: {[str(m.relative_to(lib)) for m in matches]}")


def list_spectra() -> List[str]:
    lib = library_dir()
    return sorted(str(p.relative_to(lib).with_suffix("")).replace(os.sep, "/") for p in lib.rglob("*.csv"))
