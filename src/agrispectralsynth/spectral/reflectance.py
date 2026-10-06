"""
Reflectance model for AgriSpectralSynth.

Converts an RGB aerial image into synthetic narrow-band reflectance maps
(Blue*, Green, Red, RedEdge, NIR) emulating the DJI Mavic 3 Multispectral.
(*) Blue is taken from the RGB camera; the Mavic 3M has no MS blue band.

Model ("unmixing", default)
---------------------------
Each pixel is treated as a linear mixture of two end-members, vegetation
and background (soil / rock / built surfaces), with vegetation fraction
``f`` estimated from *chromatic* excess-green:

    r, g, b   = R/S, G/S, B/S        with S = R + G + B
    ExG_c     = 2g - r - b           (brightness independent)
    f         = smoothstep(t0, t1, ExG_c) * smoothstep(d0, d1, brightness)

Using a fixed scale (instead of per-image min-max normalisation) makes
``f`` comparable between images: an image with no trees no longer gets
"100 % vegetation" in its greenest pixel.

The bands are built from *linear* RGB (sRGB gamma removed, scaled so the
camera white maps to ``reflectance_scale`` = 0.6), because reflectance is
linear and PNG/JPG values are not. All bands are clipped to [0, 1]:

    Red     = R * (1 - a_chl * f)                  chlorophyll absorption
    NIR     = f * k_veg * G + (1 - f) * k_soil * R  NIR plateau vs soil line
    RedEdge = Red + (w_veg*f + w_bg*(1-f)) * (NIR - Red)
    Green   = 0.95 G,  Blue = 0.90 B

NIR is proportional to brightness for both end-members, so shadows stay
dark in every band and do NOT turn into high-NDVI artefacts (the old
``NIR = 1.6G - 0.4R + 0.1`` formula added a constant 0.1 that pushed dark
pixels to NDVI ~0.5 while sunlit trees stayed at ~0.25).

This is an empirical model, not radiative transfer; see ``prosail.py``
for the planned physically based option.

Author
------
Juan Carlos Vega

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from ..sensors.dji_mavic3m import DJIMavic3M
from ..sensors.sensor_base import SensorBase


@dataclass
class ReflectanceParams:
    """Tunable parameters of the empirical model (all exposed in YAML)."""

    model: str = "unmixing"        # "unmixing" | "legacy"
    linearize: bool = True         # undo the sRGB gamma of PNG/JPG before building bands
    reflectance_scale: float = 0.6 # linear camera white (1.0) -> 60 % reflectance (avoids NIR saturation)

    # Vegetation fraction from chromatic ExG
    exg_low: float = 0.02          # ExG_c below this -> f = 0
    exg_high: float = 0.10         # ExG_c above this -> f = 1
    dark_low: float = 0.04         # brightness below -> chromaticity unreliable
    dark_high: float = 0.12

    # End-member behaviour
    chlorophyll_absorption: float = 0.60   # narrow-band Red depth over vegetation
    k_veg: float = 2.0                     # NIR / Green over vegetation
    k_soil: float = 1.25                   # NIR / Red over background (soil line)
    rededge_veg: float = 0.30              # RedEdge position between Red and NIR
    rededge_bg: float = 0.50

    # Sensor noise (std of reflectance, 0 disables)
    noise_std: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Inverse sRGB transfer function (IEC 61966-2-1), float32 in [0, 1]."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


# uint8 -> linear float32 in a single table lookup (much cheaper than pow())
_SRGB_LUT_U8 = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)


def smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Cubic Hermite step from 0 (x <= lo) to 1 (x >= hi), float32, in place where possible."""
    t = (x - lo) * (1.0 / (hi - lo))
    np.clip(t, 0.0, 1.0, out=t)
    return t * t * (3.0 - 2.0 * t)


class ReflectanceModel:
    """
    Synthetic reflectance model.

    Parameters
    ----------
    sensor : SensorBase, optional
        Sensor definition (defaults to the DJI Mavic 3M).
    params : ReflectanceParams, optional
    """

    BANDS = ("Blue", "Green", "Red", "RedEdge", "NIR")

    def __init__(self, sensor: Optional[SensorBase] = None, params: Optional[ReflectanceParams] = None):
        self.sensor = sensor or DJIMavic3M()
        self.params = params or ReflectanceParams()

    # -----------------------------------------------------------------

    @staticmethod
    def normalize(rgb: np.ndarray) -> np.ndarray:
        """RGB (H, W, 3) -> float32 in [0, 1]. Integer input is scaled by its dtype max."""
        if np.issubdtype(rgb.dtype, np.integer):
            return rgb.astype(np.float32) * (1.0 / np.iinfo(rgb.dtype).max)
        rgb = rgb.astype(np.float32, copy=False)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        return np.clip(rgb, 0.0, 1.0)

    # -----------------------------------------------------------------

    def vegetation_fraction(self, rgb: np.ndarray) -> np.ndarray:
        """Vegetation abundance in [0, 1] from a normalised RGB float32 image."""
        p = self.params
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        s = r + g + b
        exg = (2.0 * g - r - b) / (s + 1e-6)
        f = smoothstep(exg, p.exg_low, p.exg_high)
        f *= smoothstep(s * (1.0 / 3.0), p.dark_low, p.dark_high)
        return f

    # Kept for backwards compatibility with earlier code
    def vegetation_probability(self, rgb: np.ndarray) -> np.ndarray:
        return self.vegetation_fraction(self.normalize(rgb))

    # -----------------------------------------------------------------

    def compute(
        self,
        rgb: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        return_fraction: bool = False,
    ):
        """
        Compute synthetic reflectance maps.

        Parameters
        ----------
        rgb : ndarray (H, W, 3), RGB order, uint8/uint16/float
        rng : numpy Generator used for sensor noise (pass a seeded one
              for reproducible results)
        return_fraction : also return the vegetation fraction map

        Returns
        -------
        dict[str, ndarray float32]  (and the fraction map if requested)
        """
        display = self.normalize(rgb)

        if self.params.model == "legacy":
            bands, f = self._legacy(display)
        else:
            # Vegetation fraction on display values (thresholds are calibrated there),
            # band radiometry on linear values (reflectance is linear, PNG/JPG are not).
            f = self.vegetation_fraction(display)
            if not self.params.linearize:
                linear = display
            elif rgb.dtype == np.uint8:
                linear = _SRGB_LUT_U8[rgb]
            else:
                linear = srgb_to_linear(display)
            if self.params.reflectance_scale != 1.0:
                linear = linear * np.float32(self.params.reflectance_scale)
            bands = self._unmixing(linear, f)

        # legacy = exact v0.1 behaviour, which had no sensor noise
        if self.params.noise_std > 0 and self.params.model != "legacy":
            rng = rng or np.random.default_rng()
            for band in bands.values():
                # Signal-dependent (shot-noise-like) model: sigma = noise_std * sqrt(signal).
                # Purely additive noise made dark shadows dominate the NDVI speckle.
                noise = rng.standard_normal(band.shape, dtype=np.float32)  # float32: ~2x faster than normal()
                noise *= np.sqrt(band) * np.float32(self.params.noise_std)
                band += noise
                np.clip(band, 0.0, 1.0, out=band)

        return (bands, f) if return_fraction else bands

    # -----------------------------------------------------------------

    def _unmixing(self, rgb: np.ndarray, f: np.ndarray) -> Dict[str, np.ndarray]:
        p = self.params
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        bg = 1.0 - f

        red = R * (1.0 - p.chlorophyll_absorption * f)
        nir = (p.k_veg * f) * G
        nir += (p.k_soil * bg) * R
        np.clip(nir, 0.0, 1.0, out=nir)

        w = p.rededge_veg * f + p.rededge_bg * bg
        red_edge = red + w * (nir - red)

        bands = {
            "Blue": np.clip(0.90 * B, 0.0, 1.0),
            "Green": np.clip(0.95 * G, 0.0, 1.0),
            "Red": np.clip(red, 0.0, 1.0, out=red),
            "RedEdge": np.clip(red_edge, 0.0, 1.0, out=red_edge),
            "NIR": nir,
        }
        return bands

    def _legacy(self, rgb: np.ndarray):
        """The formula used in tests/test_spectral_synthesis.py up to v0.1 (for comparison)."""
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        nir = np.clip(G * 1.6 - R * 0.4 + 0.1, 0.0, 1.0)
        red = R.copy()
        bands = {
            "Blue": B.copy(),
            "Green": G.copy(),
            "Red": red,
            "RedEdge": np.clip(0.5 * (red + nir), 0.0, 1.0),
            "NIR": nir,
        }
        f = (G > R).astype(np.float32)
        return bands, f

    # -----------------------------------------------------------------

    def summary(self):
        print("=" * 60)
        print("Reflectance Model")
        print("=" * 60)
        print(f"Sensor : {self.sensor.name}")
        print(f"Bands  : {list(self.BANDS)}")
        print(f"Model  : {self.params.model}")
        print("=" * 60)
