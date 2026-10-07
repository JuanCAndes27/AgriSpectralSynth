"""
DJI Mavic 3 Multispectral sensor definition.

Band centres and widths follow DJI's published specification for the
Mavic 3M multispectral camera:

    Green     560 nm +/- 16 nm
    Red       650 nm +/- 16 nm
    Red Edge  730 nm +/- 16 nm
    NIR       860 nm +/- 26 nm

The Mavic 3M has NO multispectral blue band (it was removed with respect
to the Phantom 4 Multispectral). A blue channel is still available from
the drone's RGB camera, which is what AgriSpectralSynth uses for EVI.

``bandwidth`` is stored as the full width (2 x the +/- value).

Author
------
Juan Carlos Vega

License
-------
MIT
"""

from __future__ import annotations

from .sensor_base import SensorBase, SpectralBand


class DJIMavic3M(SensorBase):
    """DJI Mavic 3 Multispectral (4 narrow bands)."""

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
        """Approximate multispectral GSD (m/pixel) at 120 m altitude."""
        return 0.0567

    @property
    def spectral_range(self) -> tuple:
        return (560.0, 860.0)

    # ------------------------------------------------------------------
    # Band definition
    # ------------------------------------------------------------------

    def create_bands(self) -> None:
        self._bands = {
            "Green": SpectralBand(name="Green", center=560.0, bandwidth=32.0),
            "Red": SpectralBand(name="Red", center=650.0, bandwidth=32.0),
            "RedEdge": SpectralBand(name="RedEdge", center=730.0, bandwidth=32.0),
            "NIR": SpectralBand(name="NIR", center=860.0, bandwidth=52.0),
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

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


# Alias kept for readability / backwards compatibility with the tests.
DJIMavic3Multispectral = DJIMavic3M
