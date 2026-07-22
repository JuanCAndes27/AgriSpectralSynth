"""
Base sensor definition for AgriSpectralSynth.

Every multispectral sensor supported by the framework must inherit
from SensorBase and implement its abstract interface.

Author
------
Juan Carlos Vega
OpenAI Collaboration

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


# =============================================================================
# Spectral Band
# =============================================================================

@dataclass(frozen=True)
class SpectralBand:
    """
    Represents a spectral band of a multispectral sensor.

    Parameters
    ----------
    name : str
        Band name.

    center : float
        Central wavelength (nm).

    bandwidth : float
        Full Width Half Maximum (FWHM) in nm.
    """

    name: str

    center: float

    bandwidth: float


# =============================================================================
# Sensor Base
# =============================================================================

class SensorBase(ABC):
    """
    Abstract base class for every multispectral sensor.
    """

    def __init__(self):

        self._bands: Dict[str, SpectralBand] = {}

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Sensor name."""

    @property
    @abstractmethod
    def manufacturer(self) -> str:
        """Manufacturer."""

    @property
    @abstractmethod
    def spatial_resolution(self) -> float:
        """Ground Sampling Distance (meters/pixel)."""

    @property
    @abstractmethod
    def spectral_range(self) -> tuple:
        """Minimum and maximum wavelength."""

    # -------------------------------------------------------------------------
    # Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def create_bands(self) -> None:
        """
        Populate the internal spectral band dictionary.
        """

    # -------------------------------------------------------------------------
    # Generic utilities
    # -------------------------------------------------------------------------

    @property
    def bands(self) -> Dict[str, SpectralBand]:

        return self._bands

    def band_names(self) -> List[str]:

        return list(self._bands.keys())

    def band(self, name: str) -> SpectralBand:

        return self._bands[name]

    def wavelength(self, name: str) -> float:

        return self._bands[name].center

    def has_band(self, name: str) -> bool:

        return name in self._bands

    def summary(self) -> None:

        print("=" * 60)

        print(self.name)

        print("=" * 60)

        print(f"Manufacturer : {self.manufacturer}")

        print(f"Resolution   : {self.spatial_resolution:.3f} m/pixel")

        print(f"Spectral     : {self.spectral_range[0]} - {self.spectral_range[1]} nm")

        print()

        print("Bands")

        print("-" * 60)

        for band in self._bands.values():

            print(
                f"{band.name:10s}"
                f"{band.center:8.1f} nm"
                f"{band.bandwidth:8.1f} nm"
            )

        print("=" * 60)