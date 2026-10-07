"""
Spectral engine: RGB image -> per-pixel spectrum -> any sensor.

Idea
----
1. From the RGB image estimate, per pixel, the vegetation fraction ``f``
   (chromatic excess-green, as in the v0.2 ``unmixing`` model).
2. Model the pixel spectrum as a linear mixture of two library spectra
   (end-members) times a RELATIVE brightness ``b``:

       rho(l) = b * [ f * E_veg(l) + (1 - f) * E_soil(l) ]

   Aerial RGB photos (NEON, NAIP, drone JPGs) are stretched for display,
   so their pixel values are not reflectance. The absolute level comes
   from the spectral library; the photo only says whether a pixel is
   brighter or darker than a typical pixel of its material:

       b = Y / Y_ref(f),   Y_ref(f) = f * Y_veg + (1 - f) * Y_soil

   where Y is the linear luminance and Y_veg / Y_soil are the image's
   median luminance over pure vegetation / pure background pixels
   (fixed defaults if the image has too few of either). Shadows get
   b << 1: dark in every band, with the NDVI of their material.
3. A sensor band is the SRF-weighted mean of rho(l). Because everything
   is linear, the end-members are integrated ONCE per sensor:

       rho_band = b * [ f * E_veg_band + (1 - f) * E_soil_band ] * c_band

   so the per-pixel cost is a few multiplications per band, whatever
   the sensor.
4. ``c_band`` keeps the colour detail of the photo. For each RGB channel
   c = (observed chromaticity) / (modelled chromaticity); the three
   factors define a curve c(l), linear between the channel centres and
   constant beyond them, and each band takes its SRF-weighted mean.
   Every band at or beyond the camera's red (red, red edge, NIR) gets
   the SAME factor, so NDVI and other red/NIR ratios depend only on the
   spectral library and the sensor's bands, never on the photo's colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..sensors.registry import load_sensor
from ..sensors.sensor_base import Sensor
from ..sensors.srf import WAVELENGTHS, band_average
from .reflectance import ReflectanceModel, ReflectanceParams, _SRGB_LUT_U8, srgb_to_linear
from .spectra import Spectrum, load_spectrum


# Typical linear luminance (R+G+B) of pure vegetation / background pixels in
# display-stretched aerial photos (medians over NEON tiles). Used only when an
# image has too few pixels of one class to estimate its own reference.
DEFAULT_Y_VEG = 0.86
DEFAULT_Y_SOIL = 0.60
MIN_CLASS_PIXELS = 200
MAX_BRIGHTNESS = 2.0
CHROMA_LIMITS = (0.5, 2.0)


@dataclass
class SceneState:
    """Sensor-independent per-pixel quantities (computed once per image)."""

    veg_fraction: np.ndarray       # (H, W) float32 in [0, 1]
    brightness: np.ndarray         # (H, W) relative brightness b (1 = typical pixel of its material)
    color_corr: np.ndarray         # (3, H, W) chromaticity ratio for R, G, B
    y_ref: Tuple[float, float] = (DEFAULT_Y_VEG, DEFAULT_Y_SOIL)

    @property
    def k(self) -> np.ndarray:     # backwards-compatible name
        return self.brightness


@dataclass
class SensorPlan:
    """Per-sensor constants precomputed from the SRFs (cheap, cached)."""

    sensor: Sensor
    veg: np.ndarray                # (n_bands,) end-member reflectance per band
    soil: np.ndarray               # (n_bands,)
    rgb_weights: np.ndarray        # (n_bands, 3) weights of the R, G, B colour factors, rows sum to 1


class SpectralEngine:
    """
    Parameters
    ----------
    vegetation, soil : library spectrum names (``spectral_library/<category>/<name>.csv``)
        or ``Spectrum`` objects.
    rgb_camera : sensor id of the camera that took the input photo.
    params : the vegetation-fraction and noise settings are shared with
        ``ReflectanceParams`` (exg_low/high, dark_low/high, linearize,
        reflectance_scale, noise_std).
    """

    def __init__(
        self,
        vegetation="vegetation/healthy",
        soil="soil/soil_mixed",
        rgb_camera="rgb_camera",
        params: Optional[ReflectanceParams] = None,
    ):
        self.params = params or ReflectanceParams()
        self.veg = vegetation if isinstance(vegetation, Spectrum) else load_spectrum(vegetation)
        self.soil = soil if isinstance(soil, Spectrum) else load_spectrum(soil)
        self.camera = load_sensor(rgb_camera)
        cam = self.camera
        self._cam_srf = np.stack([cam.band(cam.roles[c]).response(cam.base_dir, cam.ir_cut_nm) for c in ("red", "green", "blue")])
        ends = np.stack([self.veg.reflectance, self.soil.reflectance])
        self._cam_ends = band_average(self._cam_srf, ends)          # (2, 3): [veg|soil] x [R,G,B]
        # Linear-interpolation basis between the camera channel centres (R, G, B),
        # held constant outside them: shape (3, n_wavelengths), columns sum to 1.
        centres = [cam.band(cam.roles[c]).center for c in ("red", "green", "blue")]
        order = np.argsort(centres)
        basis = np.zeros((3, WAVELENGTHS.size))
        xs = np.asarray(centres)[order]
        for ch in range(3):
            y = (order == ch).astype(float)
            basis[ch] = np.interp(WAVELENGTHS, xs, y)
        self._color_basis = basis
        self._fraction = ReflectanceModel(params=self.params)       # reuse v0.2 vegetation fraction
        self._plans: Dict[str, SensorPlan] = {}

    # ------------------------------------------------------------------

    def plan(self, sensor) -> SensorPlan:
        sensor = load_sensor(sensor)
        if sensor.id not in self._plans:
            srf = sensor.srf_matrix()
            ends = band_average(srf, np.stack([self.veg.reflectance, self.soil.reflectance]))
            weights = band_average(srf, self._color_basis)          # (3, n_bands) -> transpose
            weights = weights.T / weights.T.sum(axis=1, keepdims=True)
            self._plans[sensor.id] = SensorPlan(sensor, ends[0], ends[1], weights)
        return self._plans[sensor.id]

    # ------------------------------------------------------------------

    def prepare(self, rgb: np.ndarray) -> SceneState:
        """Sensor-independent part (vegetation fraction, relative brightness, colour)."""
        p = self.params
        display = ReflectanceModel.normalize(rgb)
        f = self._fraction.vegetation_fraction(display)

        if not p.linearize:
            linear = display
        elif rgb.dtype == np.uint8:
            linear = _SRGB_LUT_U8[rgb]
        else:
            linear = srgb_to_linear(display)
        y = linear.sum(axis=2)

        # Per-image reference luminance of each material
        veg_px, bg_px = f > 0.9, f < 0.1
        y_veg = float(np.median(y[veg_px])) if veg_px.sum() >= MIN_CLASS_PIXELS else DEFAULT_Y_VEG
        y_bg = float(np.median(y[bg_px])) if bg_px.sum() >= MIN_CLASS_PIXELS else DEFAULT_Y_SOIL
        y_ref = f * np.float32(y_veg - y_bg) + np.float32(y_bg)
        b = np.clip(y / np.maximum(y_ref, 1e-4), 0.0, MAX_BRIGHTNESS)

        # Chromaticity ratio photo / model for R, G, B
        veg_c, soil_c = self._cam_ends.astype(np.float32)
        model = f[..., None] * veg_c + (1.0 - f)[..., None] * soil_c
        chroma_obs = linear / np.maximum(y, 1e-6)[..., None]
        chroma_mod = model / model.sum(axis=2, keepdims=True)
        corr = chroma_obs / np.maximum(chroma_mod, 1e-6)
        dark = y < 1e-3                                  # black pixels carry no colour
        corr[dark] = 1.0
        np.clip(corr, *CHROMA_LIMITS, out=corr)
        return SceneState(
            f,
            b.astype(np.float32),
            np.ascontiguousarray(np.moveaxis(corr, -1, 0), dtype=np.float32),
            (y_veg, y_bg),
        )

    # ------------------------------------------------------------------

    def render(
        self,
        state: SceneState,
        sensor,
        rng: Optional[np.random.Generator] = None,
        noise_std: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Bands (float32 reflectance 0-1) of one sensor for a prepared scene."""
        plan = self.plan(sensor)
        f, k = state.veg_fraction, state.brightness
        noise_std = self.params.noise_std if noise_std is None else noise_std
        out: Dict[str, np.ndarray] = {}
        for i, band in enumerate(plan.sensor.band_list):
            value = (f * np.float32(plan.veg[i] - plan.soil[i]) + np.float32(plan.soil[i])) * k
            w = plan.rgb_weights[i]
            if w.max() > 0.999:                       # single channel (e.g. every band >= camera red)
                value *= state.color_corr[int(w.argmax())]
            else:
                value *= np.tensordot(w.astype(np.float32), state.color_corr, axes=1)
            if noise_std and noise_std > 0:
                rng = rng or np.random.default_rng()
                noise = rng.standard_normal(value.shape, dtype=np.float32)
                noise *= np.sqrt(np.maximum(value, 0)) * np.float32(noise_std)
                value += noise
            np.clip(value, 0.0, 1.0, out=value)
            out[band.name] = value
        return out

    def compute(self, rgb: np.ndarray, sensor="dji_mavic3m", rng=None) -> Dict[str, np.ndarray]:
        """Convenience: prepare + render for a single sensor."""
        return self.render(self.prepare(rgb), sensor, rng=rng)

    # ------------------------------------------------------------------

    def endmember_table(self, sensor) -> Dict[str, Tuple[float, float]]:
        """{band: (vegetation, soil)} reflectance of the end-members, for documentation."""
        plan = self.plan(sensor)
        return {b.name: (float(plan.veg[i]), float(plan.soil[i])) for i, b in enumerate(plan.sensor.band_list)}
