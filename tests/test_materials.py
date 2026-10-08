"""
AgriSpectralSynth - v0.1 building blocks still in the package:
Material, SpectralLibrary (per-band JSON materials), BandSimulator and
the ProsailModel placeholder.
"""

import json

import cv2
import numpy as np
import pytest

from agrispectralsynth.spectral import BandSimulator, Material, ProsailModel, SpectralLibrary


def leaf(name="leaf", nir=0.5, cat="vegetation", chl=40.0):
    return Material(name, cat, {"Red": 0.05, "NIR": nir}, chlorophyll=chl, lai=3.0)


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

def test_material_reflectance_access():
    m = leaf()
    assert m.get_reflectance("NIR") == 0.5 and m.get_reflectance("Blue") == 0.0
    m.set_reflectance("Green", 0.1)
    assert set(m.bands()) == {"Red", "NIR", "Green"}
    assert "leaf" in repr(m)


def test_material_validation():
    leaf().validate()
    with pytest.raises(ValueError):
        Material("bad", "x", {"NIR": 1.4}).validate()


def test_material_dict_roundtrip():
    m = leaf()
    assert Material.from_dict(m.to_dict()) == m


# ---------------------------------------------------------------------------
# SpectralLibrary
# ---------------------------------------------------------------------------

def test_library_crud_and_categories():
    lib = SpectralLibrary()
    lib.add(leaf("a"))
    lib.add(leaf("b"))
    lib.add(Material("sand", "soil", {"Red": 0.3, "NIR": 0.35}))
    assert len(lib) == 3 and lib.names() == ["a", "b", "sand"]
    assert lib.categories() == ["soil", "vegetation"]
    assert len(lib.filter_by_category("vegetation")) == 2
    assert lib.random("soil").name == "sand"
    lib.remove("a")
    assert not lib.exists("a")
    with pytest.raises(KeyError):
        lib.get("a")
    with pytest.raises(ValueError):
        lib.random("water")
    lib.summary()
    lib.clear()
    assert len(lib) == 0


def test_library_load_folder(tmp_path):
    (tmp_path / "veg").mkdir()
    for m in (leaf("x"), leaf("y")):
        (tmp_path / "veg" / f"{m.name}.json").write_text(json.dumps(m.to_dict()))
    lib = SpectralLibrary()
    lib.load_folder(tmp_path)
    assert lib.names() == ["x", "y"]


def test_library_interpolation():
    lib = SpectralLibrary()
    mix = lib.interpolate(leaf("healthy", nir=0.5, chl=40), leaf("dry", nir=0.3, chl=10), 0.25)
    assert mix.get_reflectance("NIR") == pytest.approx(0.45)
    assert mix.chlorophyll == pytest.approx(32.5)
    with pytest.raises(ValueError):
        lib.interpolate(leaf(), leaf(), 1.5)


# ---------------------------------------------------------------------------
# BandSimulator
# ---------------------------------------------------------------------------

def test_band_simulator_conversion():
    img = np.full((8, 8), 0.5, np.float32)
    assert BandSimulator(noise_std=0).convert(img).dtype == np.uint8
    out16 = BandSimulator(output_dtype="uint16", noise_std=0).convert(img)
    assert out16.dtype == np.uint16 and out16[0, 0] == 32767
    noisy = BandSimulator(noise_std=0.05).add_noise(img)
    assert noisy.min() >= 0 and noisy.max() <= 1 and noisy.std() > 0


def test_band_simulator_export(tmp_path):
    bands = {"Red": np.full((10, 10), 0.1), "NIR": np.full((10, 10), 0.6)}
    BandSimulator(noise_std=0).export_all(bands, tmp_path / "bands", extension=".png")
    nir = cv2.imread(str(tmp_path / "bands" / "NIR.png"), cv2.IMREAD_UNCHANGED)
    assert nir.shape == (10, 10) and nir[0, 0] == 153
    BandSimulator(noise_std=0).export_band(bands["Red"], tmp_path / "red.jpg")
    assert (tmp_path / "red.jpg").exists()


# ---------------------------------------------------------------------------
# ProsailModel placeholder
# ---------------------------------------------------------------------------

def test_prosail_placeholder_points_to_library():
    p = ProsailModel()
    assert p.available is False
    with pytest.raises(NotImplementedError):
        p.compute(np.zeros((2, 2, 3)))
    p.summary()
