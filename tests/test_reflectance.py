"""
AgriSpectralSynth - synthetic reflectance model.
"""

import numpy as np
import pytest

from agrispectralsynth.indices import NDVI
from agrispectralsynth.spectral import ReflectanceModel, ReflectanceParams

VEG = (110, 150, 70)     # green canopy (sRGB uint8)
SOIL = (170, 150, 120)   # bare soil
SHADOW = (25, 28, 35)    # dark, bluish shadow


def patch(color, size=16):
    return np.full((size, size, 3), color, dtype=np.uint8)


def ndvi_of(color, **params):
    bands = ReflectanceModel(params=ReflectanceParams(**params)).compute(patch(color))
    return float(NDVI().compute(bands["Red"], bands["NIR"]).mean())


def test_all_bands_present_and_valid():
    bands = ReflectanceModel().compute(patch(VEG))
    assert set(bands) == {"Blue", "Green", "Red", "RedEdge", "NIR"}
    for b in bands.values():
        assert b.dtype == np.float32 and b.min() >= 0 and b.max() <= 1


def test_vegetation_ndvi_much_higher_than_soil():
    veg, soil = ndvi_of(VEG), ndvi_of(SOIL)
    assert veg > 0.6
    assert 0.0 < soil < 0.25
    assert veg - soil > 0.4


def test_shadow_is_not_vegetation():
    # The v0.1 formula (NIR = 1.6G - 0.4R + 0.1) gave dark pixels NDVI ~0.5
    assert ndvi_of(SHADOW) < 0.3
    assert ndvi_of(SHADOW, model="legacy") > 0.4


def test_vegetation_fraction_is_image_independent():
    m = ReflectanceModel()
    soil_only = m.vegetation_fraction(m.normalize(patch(SOIL)))
    assert soil_only.max() < 0.05  # min-max normalisation would have produced 1.0 here


def test_red_edge_between_red_and_nir():
    b = ReflectanceModel().compute(patch(VEG))
    assert np.all(b["Red"] <= b["RedEdge"]) and np.all(b["RedEdge"] <= b["NIR"])


def test_noise_is_reproducible_with_seed():
    p = ReflectanceParams(noise_std=0.02)
    a = ReflectanceModel(params=p).compute(patch(VEG), rng=np.random.default_rng(7))
    b = ReflectanceModel(params=p).compute(patch(VEG), rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a["NIR"], b["NIR"])


@pytest.mark.parametrize("dtype,scale", [(np.uint8, 255), (np.uint16, 65535), (np.float32, 1.0)])
def test_accepts_common_dtypes(dtype, scale):
    rgb = (np.array(VEG, np.float64) / 255 * scale).astype(dtype)
    rgb = np.broadcast_to(rgb, (8, 8, 3)).copy()
    bands = ReflectanceModel().compute(rgb)
    assert bands["NIR"].shape == (8, 8)
