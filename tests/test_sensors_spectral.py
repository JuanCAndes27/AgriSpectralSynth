"""
AgriSpectralSynth - sensor definitions, spectral response functions,
spectral library and spectral engine.
"""

import numpy as np
import pytest

from agrispectralsynth.indices import NDVI
from agrispectralsynth.sensors import available_sensors, load_sensor, resolve_sensor_list
from agrispectralsynth.sensors.srf import WAVELENGTHS, band_average, fwhm_of, gaussian_srf
from agrispectralsynth.spectral.engine import SpectralEngine
from agrispectralsynth.spectral.spectra import list_spectra, load_spectrum

ALL = available_sensors()


# ---------------------------------------------------------------------------
# Spectral response functions
# ---------------------------------------------------------------------------

def test_gaussian_fwhm_is_respected():
    assert fwhm_of(gaussian_srf(650, 32)) == pytest.approx(32, abs=1.5)


def test_band_average_of_flat_spectrum():
    flat = np.full(WAVELENGTHS.size, 0.3)
    assert band_average(gaussian_srf(800, 40), flat)[0] == pytest.approx(0.3)


def test_band_average_picks_local_value():
    ramp = (WAVELENGTHS - 400) / 600          # 0 at 400 nm, 1 at 1000 nm
    assert band_average(gaussian_srf(700, 10), ramp)[0] == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# Sensor YAML files
# ---------------------------------------------------------------------------

def test_six_sensors_available():
    assert set(ALL) == {
        "dji_mavic3m", "dji_phantom4m", "micasense_rededge_mx",
        "parrot_sequoia_plus", "sentinel2a_msi", "landsat_oli",
    }


@pytest.mark.parametrize("sid", ALL)
def test_sensor_is_consistent(sid):
    s = load_sensor(sid)
    assert {"red", "nir"} <= set(s.roles)                    # NDVI always possible
    assert s.gsd_m and s.gsd_m > 0
    assert s.sources, "every sensor must cite where its specs come from"
    srf = s.srf_matrix()
    assert srf.shape == (len(s.band_list), WAVELENGTHS.size)
    for b in s.band_list:
        # half-maximum points inside the 400-1000 nm grid
        assert 400 <= b.center - b.fwhm / 2 and b.center + b.fwhm / 2 <= 1000
    assert s.wavelength(s.roles["red"]) < s.wavelength(s.roles["nir"])


@pytest.mark.parametrize(
    "sid,band,center,fwhm",
    [
        ("dji_mavic3m", "NIR", 860, 52),
        ("dji_phantom4m", "Blue", 450, 32),
        ("micasense_rededge_mx", "Red", 668, 14),
        ("parrot_sequoia_plus", "RedEdge", 735, 10),
        ("sentinel2a_msi", "B8", 832.8, 106),
        ("landsat_oli", "B4", 654.5, 37),
    ],
)
def test_published_specs(sid, band, center, fwhm):
    b = load_sensor(sid).band(band)
    assert (b.center, b.fwhm) == (center, fwhm)


def test_resolve_groups():
    assert set(resolve_sensor_list("drones")) == {s for s in ALL if load_sensor(s).platform == "drone"}
    assert resolve_sensor_list("dji_mavic3m,landsat_oli") == ["dji_mavic3m", "landsat_oli"]
    assert resolve_sensor_list(None) == ["dji_mavic3m"]


def test_load_sensor_from_yaml_path(tmp_path):
    f = tmp_path / "my_cam.yaml"
    f.write_text(
        "id: my_cam\nname: Mi cámara\nplatform: drone\ngsd_m: 0.05\nsources: [test]\n"
        "bands:\n  - {name: R, center: 660, fwhm: 20}\n  - {name: N, center: 850, fwhm: 30}\n"
        "roles: {red: R, nir: N}\n",
        encoding="utf-8",
    )
    s = load_sensor(str(f))
    assert s.id == "my_cam" and s.band_count == 2


def test_bad_role_is_rejected(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("id: bad\nname: x\nbands:\n  - {name: R, center: 660, fwhm: 20}\nroles: {nir: N}\n")
    with pytest.raises(ValueError):
        load_sensor(str(f))


# ---------------------------------------------------------------------------
# Spectral library
# ---------------------------------------------------------------------------

def test_library_contents():
    names = list_spectra()
    for n in ("vegetation/healthy", "vegetation/stressed", "vegetation/dry", "vegetation/dead", "soil/soil_mixed"):
        assert n in names


def test_vegetation_states_are_ordered():
    """Healthier vegetation -> higher NIR/Red contrast (PROSAIL presets)."""
    ndvi = {}
    for state in ("healthy", "stressed", "dry", "dead"):
        r = load_spectrum(f"vegetation/{state}")
        red, nir = r.at(650), r.at(860)
        ndvi[state] = (nir - red) / (nir + red)
    assert ndvi["healthy"] > ndvi["stressed"] > ndvi["dry"] > ndvi["dead"]
    assert ndvi["healthy"] > 0.85


def test_vegetation_has_red_edge():
    r = load_spectrum("vegetation/healthy").reflectance
    i680, i760 = 280, 360
    assert r[i760] > 5 * r[i680]                               # steep rise 680 -> 760 nm


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

VEG, SOIL, SHADOW = (110, 150, 70), (170, 150, 120), (25, 28, 35)


def scene():
    img = np.zeros((30, 30, 3), np.uint8)
    img[:, :10], img[:, 10:20], img[:, 20:] = VEG, SOIL, SHADOW
    return img


@pytest.mark.parametrize("sid", ALL)
def test_engine_separates_materials_for_every_sensor(sid):
    s = load_sensor(sid)
    b = SpectralEngine().compute(scene(), s)
    ndvi = NDVI().compute(b[s.roles["red"]], b[s.roles["nir"]])
    veg, soil, shadow = ndvi[:, :10].mean(), ndvi[:, 10:20].mean(), ndvi[:, 20:].mean()
    assert veg > 0.6
    assert soil < 0.3 and shadow < 0.3
    for band in b.values():
        assert band.dtype == np.float32 and band.min() >= 0 and band.max() <= 1


def test_engine_shadow_is_darker_but_same_material():
    s = load_sensor("dji_mavic3m")
    b = SpectralEngine().compute(scene(), s)
    assert b["NIR"][:, 20:].mean() < 0.3 * b["NIR"][:, 10:20].mean()


def test_vegetation_endmember_changes_ndvi():
    s = load_sensor("dji_mavic3m")
    img = np.full((8, 8, 3), VEG, np.uint8)
    nd = {}
    for state in ("healthy", "dry"):
        b = SpectralEngine(vegetation=f"vegetation/{state}").compute(img, s)
        nd[state] = NDVI().compute(b["Red"], b["NIR"]).mean()
    assert nd["healthy"] > nd["dry"] + 0.2


def test_prepare_once_render_many():
    eng = SpectralEngine()
    st = eng.prepare(scene())
    a = eng.render(st, "dji_mavic3m", noise_std=0)
    b = eng.render(st, "dji_mavic3m", noise_std=0)
    np.testing.assert_array_equal(a["NIR"], b["NIR"])


def test_visible_bands_keep_photo_detail():
    """Two greens of different hue must stay different in the visible bands."""
    img = np.zeros((4, 8, 3), np.uint8)
    img[:, :4], img[:, 4:] = (90, 160, 60), (120, 160, 60)
    b = SpectralEngine().compute(img, "micasense_rededge_mx")
    assert b["Red"][:, 4:].mean() > b["Red"][:, :4].mean()


@pytest.mark.parametrize("sid", ALL)
def test_pure_vegetation_ndvi_equals_endmember_ndvi(sid):
    """Photo colour must not leak into NDVI: a fully vegetated pixel gets exactly
    the NDVI of the vegetation spectrum integrated with that sensor's bands."""
    s = load_sensor(sid)
    eng = SpectralEngine()
    img = np.full((4, 4, 3), VEG, np.uint8)
    st = eng.prepare(img)
    assert st.veg_fraction.min() == 1.0
    b = eng.render(st, s, noise_std=0)
    red, nir = eng.endmember_table(s)[s.roles["red"]][0], eng.endmember_table(s)[s.roles["nir"]][0]
    expected = (nir - red) / (nir + red)
    got = NDVI().compute(b[s.roles["red"]], b[s.roles["nir"]])
    np.testing.assert_allclose(got, expected, atol=1e-4)
