"""
AgriSpectralSynth - unit tests for the vegetation indices.

Bands are passed by keyword so a swapped argument order (a bug in the
previous version of these tests) cannot go unnoticed.
"""

import numpy as np
import pytest

from agrispectralsynth.indices import EVI, GNDVI, INDEX_REGISTRY, MSAVI, NDRE, NDVI, SAVI

rng = np.random.default_rng(0)


def bands(shape=(64, 64)):
    """Random reflectance per band ROLE (the keys INDEX_REGISTRY uses)."""
    return {k: rng.random(shape, dtype=np.float32) for k in ("blue", "green", "red", "red_edge", "nir")}


def test_ndvi_known_values():
    red = np.array([[0.1, 0.5]], dtype=np.float32)
    nir = np.array([[0.5, 0.1]], dtype=np.float32)
    ndvi = NDVI().compute(red=red, nir=nir)
    np.testing.assert_allclose(ndvi, [[2 / 3, -2 / 3]], atol=1e-5)


def test_ndvi_vegetation_is_positive():
    ndvi = NDVI().compute(red=np.full((4, 4), 0.05), nir=np.full((4, 4), 0.45))
    assert np.all(ndvi > 0.7)


def test_ndvi_zero_bands_no_nan():
    ndvi = NDVI().compute(red=np.zeros((8, 8)), nir=np.zeros((8, 8)))
    assert not np.isnan(ndvi).any()
    assert np.all(ndvi == 0)


@pytest.mark.parametrize("name", list(INDEX_REGISTRY))
def test_registry_indices_shape_range_dtype(name):
    cls, needed = INDEX_REGISTRY[name]
    b = bands((50, 40))
    out = cls().compute(*(b[k] for k in needed))
    assert out.shape == (50, 40)
    assert out.dtype == np.float32
    assert not np.isnan(out).any()
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_gndvi_ndre_formulas():
    g, re, nir = np.float32(0.1), np.float32(0.2), np.float32(0.5)
    assert GNDVI().compute(green=np.array([g]), nir=np.array([nir]))[0] == pytest.approx(0.4 / 0.6, abs=1e-5)
    assert NDRE().compute(red_edge=np.array([re]), nir=np.array([nir]))[0] == pytest.approx(0.3 / 0.7, abs=1e-5)


def test_savi_reduces_to_scaled_ndvi_when_L_zero():
    b = bands()
    np.testing.assert_allclose(
        SAVI(L=0.0).compute(b["red"], b["nir"]),
        NDVI(epsilon=0).compute(b["red"], b["nir"]),
        atol=1e-4,
    )


def test_msavi_no_nan_extremes():
    red = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    nir = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    assert not np.isnan(MSAVI().compute(red, nir)).any()


def test_evi_handles_zero_denominator():
    # nir + 6*red - 7.5*blue + 1 == 0
    out = EVI().compute(blue=np.array([0.2]), red=np.array([0.0]), nir=np.array([0.5]))
    assert np.isfinite(out).all()


def test_colorize_and_uint8():
    ndvi = np.linspace(-1, 1, 100, dtype=np.float32).reshape(10, 10)
    idx = NDVI()
    assert idx.to_uint8(ndvi).dtype == np.uint8
    rgb = idx.colorize(ndvi)
    assert rgb.shape == (10, 10, 3) and rgb.dtype == np.uint8


def test_save_geotiff_roundtrip(tmp_path):
    import rasterio

    ndvi = np.random.default_rng(1).uniform(-1, 1, (20, 30)).astype(np.float32)
    f = tmp_path / "ndvi.tif"
    NDVI().save_geotiff(ndvi, f)
    with rasterio.open(f) as src:
        np.testing.assert_array_equal(src.read(1), ndvi)


def test_exports_and_statistics(tmp_path):
    import cv2

    ndvi = np.linspace(-1, 1, 400, dtype=np.float32).reshape(20, 20)
    idx = NDVI()
    idx.save_png(ndvi, tmp_path / "c.png")
    idx.save_png(ndvi, tmp_path / "g.png", color=False)
    idx.save_jpg(ndvi, tmp_path / "c.jpg", color=True)
    assert cv2.imread(str(tmp_path / "c.png")).shape == (20, 20, 3)
    assert cv2.imread(str(tmp_path / "g.png"), cv2.IMREAD_UNCHANGED).ndim == 2
    assert (tmp_path / "c.jpg").exists()
    st = idx.statistics(ndvi)
    assert st["minimum"] == -1 and st["maximum"] == 1 and abs(st["median"]) < 0.01
    idx.summary()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_nan_and_inf_are_sanitised():
    out = NDVI().compute(np.array([np.nan, 0.1]), np.array([0.5, np.inf]))
    assert np.isfinite(out).all()
