"""
AgriSpectralSynth - unit tests for segmentation masks.
"""

import numpy as np

from agrispectralsynth.segmentation import (
    CanopySegmenter,
    VegetationMask,
    canopy_mask,
    create_binary_mask,
    remove_small_objects,
)


def test_binary_mask_shape():
    image = np.random.rand(256, 256).astype(np.float32)
    assert create_binary_mask(image, threshold=0.5).shape == image.shape


def test_binary_mask_values_and_dtype():
    mask = create_binary_mask(np.random.rand(128, 128))
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask.dtype == np.uint8


def test_empty_and_full_image():
    assert create_binary_mask(np.zeros((100, 100), np.float32)).sum() == 0
    assert create_binary_mask(np.ones((100, 100), np.float32)).sum() == 100 * 100


def test_binary_mask_nan_is_background():
    img = np.full((4, 4), np.nan, dtype=np.float32)
    assert create_binary_mask(img).sum() == 0


def test_canopy_shape_and_values():
    ndvi = np.random.rand(200, 200)
    canopy = canopy_mask(ndvi)
    assert canopy.shape == ndvi.shape
    assert set(np.unique(canopy)).issubset({0, 1})


def test_canopy_detects_a_crown():
    ndvi = np.full((100, 100), 0.1, np.float32)
    yy, xx = np.mgrid[:100, :100]
    crown = (yy - 50) ** 2 + (xx - 50) ** 2 < 15**2
    ndvi[crown] = 0.8
    mask = canopy_mask(ndvi, threshold=0.45)
    iou = (mask.astype(bool) & crown).sum() / (mask.astype(bool) | crown).sum()
    assert iou > 0.9


def test_remove_small_objects():
    m = np.zeros((50, 50), np.uint8)
    m[5:7, 5:7] = 1          # 4 px
    m[20:40, 20:40] = 1      # 400 px
    out = remove_small_objects(m, min_area=10)
    assert out[5:7, 5:7].sum() == 0 and out[20:40, 20:40].all()


def test_segmenter_objects():
    m = np.zeros((60, 60), np.uint8)
    m[10:30, 10:30] = 255
    m[40:42, 40:42] = 255
    out, objs = CanopySegmenter(min_area=50).segment(m)
    assert len(objs) == 1 and objs[0]["area"] == 400
    assert out.max() == 255 and out[40:42, 40:42].sum() == 0


def test_vegetation_mask_from_rgb():
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[:] = (150, 130, 110)            # soil
    rgb[10:30, 10:30] = (60, 140, 50)   # vegetation
    mask = VegetationMask(kernel_size=3).compute(rgb)
    assert mask[20, 20] == 255 and mask[2, 2] == 0
