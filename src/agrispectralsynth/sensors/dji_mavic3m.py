"""
DJI Mavic 3 Multispectral sensor definition.

This module defines the DJI Mavic 3 Multispectral camera using the
central wavelength of each spectral band.

Author
------
Juan Carlos Vega
OpenAI Collaboration

License
-------
MIT
"""

from __future__ import annotations

from .sensor_base import SensorBase, SpectralBand


class DJIMavic3M(SensorBase):
    """
    DJI Mavic 3 Multispectral sensor.

    Specifications
    --------------
    Manufacturer:
        DJI

    Spectral Bands:
        Blue      : 475 nm
        Green     : 560 nm
        Red       : 668 nm
        Red Edge  : 717 nm
        NIR       : 842 nm
    """

    def __init__(self):

        super().__init__()

        self.create_bands()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:

        return "DJI Mavic 3 Multispectral"

    @property
    def manufacturer(self) -> str:

        return "DJI"

    @property
    def spatial_resolution(self) -> float:
        """
        Default Ground Sampling Distance (meters/pixel).

        Assuming a flight altitude of 120 m.
        """

        return 0.05

    @property
    def spectral_range(self) -> tuple:

        return (475.0, 842.0)

    # ------------------------------------------------------------------
    # Band Definition
    # ------------------------------------------------------------------

    def create_bands(self) -> None:

        self._bands = {

            "Blue": SpectralBand(
                name="Blue",
                center=475.0,
                bandwidth=1.0,
            ),

            "Green": SpectralBand(
                name="Green",
                center=560.0,
                bandwidth=1.0,
            ),

            "Red": SpectralBand(
                name="Red",
                center=668.0,
                bandwidth=1.0,
            ),

            "RedEdge": SpectralBand(
                name="RedEdge",
                center=717.0,
                bandwidth=1.0,
            ),

            "NIR": SpectralBand(
                name="NIR",
                center=842.0,
                bandwidth=1.0,
            ),
        }

    # ------------------------------------------------------------------
    # Convenience Methods
    # ------------------------------------------------------------------

    @property
    def blue(self) -> float:
        return self.wavelength("Blue")

    @property
    def green(self) -> float:
        return self.wavelength("Green")

    @property
    def red(self) -> float:
        return self.wavelength("Red")

    @property
    def red_edge(self) -> float:
        return self.wavelength("RedEdge")

    @property
    def nir(self) -> float:
        return self.wavelength("NIR")

    @property
    def band_count(self) -> int:
        return len(self._bands)

    def __repr__(self) -> str:

        return (
            f"{self.name}"
            f"(bands={self.band_count}, "
            f"range={self.spectral_range[0]}-{self.spectral_range[1]} nm)"
        )