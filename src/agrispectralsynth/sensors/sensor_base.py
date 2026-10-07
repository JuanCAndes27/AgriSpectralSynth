"""
Sensor definitions for AgriSpectralSynth.

A sensor is plain data: its bands (centre, FWHM, optional measured
curve), the "roles" that tell the vegetation indices which band is red,
NIR, etc., and its ground sampling distance. Sensors are described in
YAML files under ``sensors/`` at the repository root, so adding a new
camera or satellite does not require writing Python.

Author
------
Juan Carlos Vega

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import srf as _srf

ROLES = ("blue", "green", "red", "red_edge", "nir")


# =============================================================================
# Spectral band
# =============================================================================

@dataclass(frozen=True)
class SpectralBand:
    """
    One band of a sensor.

    center : central wavelength (nm)
    fwhm : full width at half maximum (nm)
    auxiliary : True when the band does not come from the multispectral
        camera itself (e.g. the blue of the Mavic 3M comes from its RGB camera)
    gsd_m : band-specific ground sampling distance, if it differs from the sensor's
    srf_file : optional measured response (CSV), relative to the sensors folder
    """

    name: str
    center: float
    fwhm: float
    auxiliary: bool = False
    gsd_m: Optional[float] = None
    srf_file: Optional[str] = None

    @property
    def bandwidth(self) -> float:
        """Alias of ``fwhm`` (kept for compatibility with v0.1/v0.2 code)."""
        return self.fwhm

    def response(self, base_dir: Optional[Path] = None, ir_cut_nm: Optional[float] = None) -> np.ndarray:
        if self.srf_file:
            path = Path(self.srf_file)
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            r = _srf.tabulated_srf(path)
        else:
            r = _srf.gaussian_srf(self.center, self.fwhm)
        if ir_cut_nm is not None:
            r = r * _srf.ir_cut(ir_cut_nm)
        return r


# =============================================================================
# Sensor
# =============================================================================

@dataclass
class Sensor:
    id: str
    name: str
    manufacturer: str = ""
    platform: str = ""                     # drone | satellite | camera
    gsd_m: Optional[float] = None
    reference_altitude_m: Optional[float] = None
    srf_model: str = "gaussian"
    ir_cut_nm: Optional[float] = None
    band_list: List[SpectralBand] = field(default_factory=list)
    roles: Dict[str, str] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    notes: str = ""
    base_dir: Optional[Path] = None        # folder the YAML was loaded from

    def __post_init__(self):
        names = [b.name for b in self.band_list]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.id}: duplicated band names {names}")
        for role, band in self.roles.items():
            if role not in ROLES:
                raise ValueError(f"{self.id}: unknown role '{role}' (valid: {ROLES})")
            if band not in names:
                raise ValueError(f"{self.id}: role '{role}' points to missing band '{band}'")
        self._srf_cache: Optional[np.ndarray] = None

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict, base_dir: Optional[Path] = None) -> "Sensor":
        data = dict(data)
        bands = [SpectralBand(**b) for b in data.pop("bands")]
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = set(data) - known
        if extra:
            raise ValueError(f"Unknown keys in sensor '{data.get('id')}': {sorted(extra)}")
        return cls(band_list=bands, base_dir=base_dir, **data)

    # ---- band access ---------------------------------------------------------

    @property
    def bands(self) -> Dict[str, SpectralBand]:
        """Multispectral bands only (auxiliary bands such as RGB-camera blue excluded)."""
        return {b.name: b for b in self.band_list if not b.auxiliary}

    @property
    def all_bands(self) -> Dict[str, SpectralBand]:
        return {b.name: b for b in self.band_list}

    def band_names(self, include_auxiliary: bool = True) -> List[str]:
        return [b.name for b in self.band_list if include_auxiliary or not b.auxiliary]

    def band(self, name: str) -> SpectralBand:
        return self.all_bands[name]

    def wavelength(self, name: str) -> float:
        return self.all_bands[name].center

    def has_band(self, name: str) -> bool:
        return name in self.bands

    def role(self, role: str) -> Optional[str]:
        return self.roles.get(role)

    @property
    def spectral_range(self) -> tuple:
        c = [b.center for b in self.bands.values()]
        return (min(c), max(c))

    @property
    def spatial_resolution(self) -> Optional[float]:
        return self.gsd_m

    @property
    def band_count(self) -> int:
        return len(self.bands)

    # ---- spectral response ---------------------------------------------------

    def srf_matrix(self) -> np.ndarray:
        """(n_bands_including_auxiliary, n_wavelengths) response matrix, cached."""
        if self._srf_cache is None:
            self._srf_cache = np.stack(
                [b.response(self.base_dir, self.ir_cut_nm) for b in self.band_list]
            )
        return self._srf_cache

    # ---- display -------------------------------------------------------------

    def summary(self) -> None:
        print("=" * 64)
        print(f"{self.name}  [{self.id}]")
        print("=" * 64)
        if self.manufacturer:
            print(f"Manufacturer : {self.manufacturer}")
        if self.gsd_m:
            print(f"GSD          : {self.gsd_m:g} m/pixel")
        print(f"{'Band':10s}{'centre nm':>11s}{'FWHM nm':>10s}  role")
        print("-" * 64)
        inv = {v: k for k, v in self.roles.items()}
        for b in self.band_list:
            aux = " (aux)" if b.auxiliary else ""
            print(f"{b.name:10s}{b.center:11.1f}{b.fwhm:10.1f}  {inv.get(b.name, '')}{aux}")
        print("=" * 64)

    def __repr__(self) -> str:
        lo, hi = self.spectral_range
        return f"{self.name}(bands={self.band_count}, range={lo:g}-{hi:g} nm)"


# Backwards-compatible name used by earlier code
SensorBase = Sensor
