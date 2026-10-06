"""
AgriSpectralSynth - DJI Mavic 3 Multispectral sensor.

Reference: DJI Mavic 3M multispectral camera
G 560±16, R 650±16, RE 730±16, NIR 860±26 nm (no multispectral blue band).
"""

from agrispectralsynth.sensors import DJIMavic3M, DJIMavic3Multispectral, SpectralBand


def test_alias_is_same_class():
    assert DJIMavic3Multispectral is DJIMavic3M


def test_sensor_name():
    assert DJIMavic3M().name == "DJI Mavic 3 Multispectral"


def test_number_of_bands():
    assert len(DJIMavic3M().bands) == 4


def test_band_centres():
    s = DJIMavic3M()
    assert s.green == 560
    assert s.red == 650
    assert s.red_edge == 730
    assert s.nir == 860


def test_no_multispectral_blue():
    assert not DJIMavic3M().has_band("Blue")


def test_bandwidths():
    s = DJIMavic3M()
    assert s.band("NIR").bandwidth == 52
    assert all(isinstance(b, SpectralBand) for b in s.bands.values())


def test_band_range():
    for band in DJIMavic3M().bands.values():
        assert 400 <= band.center <= 900


def test_string_representation():
    assert "4" in repr(DJIMavic3M())
