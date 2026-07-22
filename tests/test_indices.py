"""
===========================================================
AgriSpectralSynth

Unit Tests
Vegetation Indices

Author:
Juan Carlos Vega
OpenAI Collaboration

License:
MIT
===========================================================
"""

import numpy as np

from agrispectralsynth.indices.ndvi import NDVI
from agrispectralsynth.indices.gndvi import GNDVI
from agrispectralsynth.indices.savi import SAVI
from agrispectralsynth.indices.evi import EVI
from agrispectralsynth.indices.msavi import MSAVI


# ---------------------------------------------------------
# NDVI
# ---------------------------------------------------------

def test_ndvi_shape():

    nir = np.ones((100, 100), dtype=np.float32)
    red = np.ones((100, 100), dtype=np.float32)

    ndvi = NDVI.compute(nir, red)

    assert ndvi.shape == nir.shape


def test_ndvi_range():

    nir = np.random.rand(100,100).astype(np.float32)
    red = np.random.rand(100,100).astype(np.float32)

    ndvi = NDVI.compute(nir, red)

    assert np.all(ndvi >= -1.0)
    assert np.all(ndvi <= 1.0)


# ---------------------------------------------------------
# GNDVI
# ---------------------------------------------------------

def test_gndvi_range():

    nir = np.random.rand(64,64).astype(np.float32)
    green = np.random.rand(64,64).astype(np.float32)

    gndvi = GNDVI.compute(nir, green)

    assert np.min(gndvi) >= -1
    assert np.max(gndvi) <= 1


# ---------------------------------------------------------
# SAVI
# ---------------------------------------------------------

def test_savi_shape():

    nir = np.random.rand(32,32).astype(np.float32)
    red = np.random.rand(32,32).astype(np.float32)

    savi = SAVI.compute(nir, red)

    assert savi.shape == nir.shape


# ---------------------------------------------------------
# EVI
# ---------------------------------------------------------

def test_evi_shape():

    nir = np.random.rand(50,50).astype(np.float32)
    red = np.random.rand(50,50).astype(np.float32)
    blue = np.random.rand(50,50).astype(np.float32)

    evi = EVI.compute(nir, red, blue)

    assert evi.shape == nir.shape


# ---------------------------------------------------------
# MSAVI
# ---------------------------------------------------------

def test_msavi_shape():

    nir = np.random.rand(80,80).astype(np.float32)
    red = np.random.rand(80,80).astype(np.float32)

    msavi = MSAVI.compute(nir, red)

    assert msavi.shape == nir.shape


# ---------------------------------------------------------
# NaN verification
# ---------------------------------------------------------

def test_indices_without_nan():

    nir = np.random.rand(100,100).astype(np.float32)
    red = np.random.rand(100,100).astype(np.float32)
    green = np.random.rand(100,100).astype(np.float32)
    blue = np.random.rand(100,100).astype(np.float32)

    assert not np.isnan(NDVI.compute(nir,red)).any()
    assert not np.isnan(GNDVI.compute(nir,green)).any()
    assert not np.isnan(SAVI.compute(nir,red)).any()
    assert not np.isnan(EVI.compute(nir,red,blue)).any()
    assert not np.isnan(MSAVI.compute(nir,red)).any()