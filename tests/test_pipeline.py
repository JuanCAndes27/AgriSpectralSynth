"""
AgriSpectralSynth - end-to-end batch pipeline.
"""

import csv
import os
import time

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from agrispectralsynth.config import AppConfig, load_config
from agrispectralsynth.pipeline import guess_dataset, run_pipeline


def synthetic_scene(seed=0, size=96):
    """Soil background with a few round green crowns and their shadows."""
    rng = np.random.default_rng(seed)
    img = np.empty((size, size, 3), np.uint8)
    img[:] = (120, 135, 160)  # BGR soil
    for _ in range(4):
        x, y = rng.integers(15, size - 15, 2)
        cv2.circle(img, (int(x) + 4, int(y) + 4), 10, (35, 30, 25), -1)   # shadow
        cv2.circle(img, (int(x), int(y)), 10, (60, 150, 100), -1)         # crown
    return img


@pytest.fixture
def raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    for i in range(5):
        cv2.imwrite(str(d / f"{i}_Test_Source_2024.png"), synthetic_scene(i))
    return d


def test_pipeline_outputs_and_manifest(raw_dir, tmp_path):
    out = tmp_path / "out"
    summary = run_pipeline(AppConfig(), input_dir=raw_dir, output_dir=out, workers=1, progress=False)
    assert summary["ok"] == 5 and summary["error"] == 0

    stem = "0_Test_Source_2024"
    for rel in [
        f"ndvi_raw/{stem}_NDVI.tif",
        f"ndvi_visual/{stem}_NDVI.png",
        f"nir/{stem}_NIR.png",
        f"canopy_mask/{stem}_canopy.png",
        f"multispectral/{stem}_MS.tif",
    ]:
        assert (out / rel).exists(), rel

    with rasterio.open(out / f"multispectral/{stem}_MS.tif") as src:
        assert src.count == 5 and src.dtypes[0] == "uint16"
        assert src.descriptions == ("Blue", "Green", "Red", "RedEdge", "NIR")

    with open(out / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert rows[0]["dataset"] == "Test_Source_2024"
    assert 0 < float(rows[0]["canopy_fraction"]) < 0.6


def test_parallel_matches_sequential(raw_dir, tmp_path):
    run_pipeline(AppConfig(), input_dir=raw_dir, output_dir=tmp_path / "seq", workers=1, progress=False)
    run_pipeline(AppConfig(), input_dir=raw_dir, output_dir=tmp_path / "par", workers=2, progress=False)
    for f in (tmp_path / "seq" / "ndvi_raw").glob("*.tif"):
        with rasterio.open(f) as a, rasterio.open(tmp_path / "par" / "ndvi_raw" / f.name) as b:
            np.testing.assert_array_equal(a.read(1), b.read(1))


def test_incremental_skip_and_overwrite(raw_dir, tmp_path):
    out = tmp_path / "out"
    cfg = AppConfig()
    run_pipeline(cfg, input_dir=raw_dir, output_dir=out, workers=1, progress=False)

    s = run_pipeline(cfg, input_dir=raw_dir, output_dir=out, workers=1, progress=False)
    assert s["skipped"] == 5 and s["ok"] == 0

    # touching one source re-processes only that one
    src = raw_dir / "3_Test_Source_2024.png"
    future = time.time() + 10
    os.utime(src, (future, future))
    s = run_pipeline(cfg, input_dir=raw_dir, output_dir=out, workers=1, progress=False)
    assert s["ok"] == 1 and s["skipped"] == 4

    s = run_pipeline(cfg, input_dir=raw_dir, output_dir=out, workers=1, overwrite=True, progress=False)
    assert s["ok"] == 5

    # manifest keeps the statistics of skipped images
    with open(out / "manifest.csv", encoding="utf-8") as f:
        assert all(r["status"] == "ok" for r in csv.DictReader(f))


def test_generated_products_in_input_are_ignored(raw_dir, tmp_path):
    cv2.imwrite(str(raw_dir / "0_Test_Source_2024_NDVI.png"), synthetic_scene())
    s = run_pipeline(AppConfig(), input_dir=raw_dir, output_dir=tmp_path / "o", workers=1, progress=False)
    assert s["images"] == 5


def test_georeferencing_is_preserved(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rgb = np.moveaxis(synthetic_scene()[:, :, ::-1], -1, 0)
    transform = from_origin(500000, 4100000, 0.1, 0.1)
    with rasterio.open(
        raw / "geo.tif", "w", driver="GTiff", width=96, height=96, count=3, dtype="uint8",
        crs="EPSG:32617", transform=transform,
    ) as dst:
        dst.write(rgb)

    out = tmp_path / "out"
    run_pipeline(AppConfig(), input_dir=raw, output_dir=out, workers=1, progress=False)
    with rasterio.open(out / "ndvi_raw" / "geo_NDVI.tif") as src:
        assert src.crs.to_epsg() == 32617
        assert src.transform == transform


def test_unreadable_file_reported_not_raised(raw_dir, tmp_path):
    (raw_dir / "broken.png").write_bytes(b"not an image")
    s = run_pipeline(AppConfig(), input_dir=raw_dir, output_dir=tmp_path / "o", workers=1, progress=False)
    assert s["error"] == 1 and s["ok"] == 5


def test_missing_input_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(AppConfig(), input_dir=tmp_path / "nope", output_dir=tmp_path / "o", progress=False)


def test_default_yaml_loads():
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert cfg.reflectance.model == "unmixing"
    assert cfg.enabled_indices() == []
    assert cfg.output.compress == "zstd"


def test_extra_indices_when_enabled(raw_dir, tmp_path):
    cfg = AppConfig()
    cfg.simulation.generate_ndre = True
    cfg.simulation.generate_evi = True
    out = tmp_path / "o"
    run_pipeline(cfg, input_dir=raw_dir, output_dir=out, workers=1, limit=1, progress=False)
    assert (out / "indices" / "NDRE" / "0_Test_Source_2024_NDRE.tif").exists()
    assert (out / "indices" / "EVI" / "0_Test_Source_2024_EVI.tif").exists()


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("470_Velasquez_Camacho_et_al_2023", "Velasquez_Camacho_et_al_2023"),
        ("9c8d8103-naips9-0_8192_6_World_Resources_Institute", "World_Resources_Institute"),
        ("0_Zamboni_et_al_2021", "Zamboni_et_al_2021"),
    ],
)
def test_guess_dataset(stem, expected):
    assert guess_dataset(stem) == expected
