"""
AgriSpectralSynth - MillionTrees dataset manager.

The `milliontrees` package (and its download) is not needed: the
downloaded-dataset workflow is exercised with a small stand-in object
that has the same attributes the manager reads.
"""

import json
import sys

import cv2
import numpy as np
import pytest

from agrispectralsynth.datasets import BaseDataset, MillionTreesDataset


class FakePolygon:
    def __init__(self, coords):
        self.exterior = type("Ring", (), {"coords": coords})()
        self.is_empty = False
        self.wkt = "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in coords) + "))"


class FakeTreePolygons:
    """Mimics the attributes of milliontrees' TreePolygonsDataset that the manager uses."""

    def __init__(self, data_dir):
        self._data_dir = data_dir
        (data_dir / "images").mkdir(parents=True)
        self._input_array = ["a.png", "a.png", "b.png"]
        self._input_lookup = {"a.png": [0, 1], "b.png": [2]}
        self._y_array = [
            FakePolygon([(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)]),
            FakePolygon([(40, 40), (60, 40), (60, 60), (40, 60), (40, 40)]),
            FakePolygon([(5, 5), (15, 5), (15, 15), (5, 15), (5, 5)]),
        ]
        for name in ("a.png", "b.png"):
            cv2.imwrite(str(data_dir / "images" / name), np.zeros((80, 80, 3), np.uint8))

    def __len__(self):
        return len(self._input_array)


def test_invalid_version():
    with pytest.raises(ValueError):
        MillionTreesDataset("x", version="huge")


def test_is_a_base_dataset(tmp_path):
    ds = MillionTreesDataset(tmp_path / "mt")
    assert isinstance(ds, BaseDataset) and not ds.exists()
    ds.create_directory()
    assert ds.exists() and "MillionTrees" in repr(ds)


def test_local_folder_scan_with_limit(tmp_path):
    for i in range(5):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.zeros((4, 4, 3), np.uint8))
    (tmp_path / "notes.txt").write_text("x")
    ds = MillionTreesDataset(tmp_path, limit=3)
    paths = ds.load()
    assert len(paths) == 3 and all(p.suffix == ".png" for p in paths) and len(ds) == 3


def test_download_without_package_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "milliontrees", None)      # import milliontrees -> ImportError
    with pytest.raises(ImportError, match="milliontrees"):
        MillionTreesDataset(tmp_path).download()


def test_steps_require_download(tmp_path):
    ds = MillionTreesDataset(tmp_path)
    with pytest.raises(RuntimeError):
        ds.select_images()
    with pytest.raises(RuntimeError):
        ds.copy_rgb()


def test_rasterize_polygons():
    mask = MillionTreesDataset.rasterize([FakePolygon([(2, 2), (7, 2), (7, 7), (2, 7), (2, 2)])], (10, 10))
    assert mask.dtype == np.uint8 and mask[4, 4] == 255 and mask[0, 0] == 0


def test_full_workflow_after_download(tmp_path):
    ds = MillionTreesDataset(tmp_path / "mt", version="mini")
    ds.prepare()
    ds.dataset = FakeTreePolygons(tmp_path / "mt" / "raw" / "TreePolygons_v0.25")

    assert ds.select_images(limit=10) == ["a.png", "b.png"]          # unique images, original order
    assert len(ds.polygons_for("a.png")) == 2
    assert ds.copy_rgb() == 2 and ds.copy_rgb() == 0                 # second call: already copied
    ds.copy_polygons()
    ds.export_masks()
    ds.export_metadata()

    wkt = json.loads((ds.polygon_dir / "a.json").read_text())
    assert len(wkt) == 2 and wkt[0].startswith("POLYGON")
    mask = cv2.imread(str(ds.mask_dir / "a_canopy_gt.png"), cv2.IMREAD_GRAYSCALE)
    assert mask[20, 20] == 255 and mask[50, 50] == 255 and mask[35, 5] == 0
    meta = json.loads((ds.metadata_dir / "b.json").read_text())
    assert meta["n_polygons"] == 1 and meta["version"] == "mini"
    assert ds.verify_dataset() is True
    assert ds.statistics()["masks"] == 2
