"""
===========================================================
AgriSpectralSynth

Unit Tests
Segmentation Masks

Author:
Juan Carlos Vega
OpenAI Collaboration

License:
MIT
===========================================================
"""

import numpy as np

from agrispectralsynth.segmentation.masks import create_binary_mask
from agrispectralsynth.segmentation.canopy import canopy_mask


# ---------------------------------------------------------
# Binary mask
# ---------------------------------------------------------

def test_binary_mask_shape():

    image = np.random.rand(256,256).astype(np.float32)

    mask = create_binary_mask(image, threshold=0.5)

    assert mask.shape == image.shape


# ---------------------------------------------------------
# Binary values
# ---------------------------------------------------------

def test_binary_mask_values():

    image = np.random.rand(128,128)

    mask = create_binary_mask(image)

    unique = np.unique(mask)

    assert set(unique).issubset({0,1})


# ---------------------------------------------------------
# Canopy mask
# ---------------------------------------------------------

def test_canopy_shape():

    ndvi = np.random.rand(200,200)

    canopy = canopy_mask(ndvi)

    assert canopy.shape == ndvi.shape


# ---------------------------------------------------------
# Mask datatype
# ---------------------------------------------------------

def test_mask_dtype():

    image = np.random.rand(64,64)

    mask = create_binary_mask(image)

    assert mask.dtype in [
        np.uint8,
        np.bool_,
        bool
    ]


# ---------------------------------------------------------
# Empty image
# ---------------------------------------------------------

def test_empty_image():

    image = np.zeros((100,100),dtype=np.float32)

    mask = create_binary_mask(image)

    assert np.sum(mask)==0


# ---------------------------------------------------------
# Full vegetation
# ---------------------------------------------------------

def test_full_image():

    image = np.ones((100,100),dtype=np.float32)

    mask = create_binary_mask(image)

    assert np.sum(mask)==100*100