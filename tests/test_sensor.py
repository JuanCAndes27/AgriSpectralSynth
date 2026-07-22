"""
===========================================================
AgriSpectralSynth

Unit Tests
DJI Mavic 3 Multispectral Sensor

Author:
Juan Carlos Vega
OpenAI Collaboration

License:
MIT
===========================================================
"""

import pytest

from agrispectralsynth.sensors.dji_mavic3m import DJIMavic3Multispectral


# ---------------------------------------------------------
# Sensor creation
# ---------------------------------------------------------

def test_sensor_creation():

    sensor = DJIMavic3Multispectral()

    assert sensor is not None


# ---------------------------------------------------------
# Sensor name
# ---------------------------------------------------------

def test_sensor_name():

    sensor = DJIMavic3Multispectral()

    assert sensor.name == "DJI Mavic 3 Multispectral"


# ---------------------------------------------------------
# Number of spectral bands
# ---------------------------------------------------------

def test_number_of_bands():

    sensor = DJIMavic3Multispectral()

    assert len(sensor.bands) == 5


# ---------------------------------------------------------
# RGB Band
# ---------------------------------------------------------

def test_green_band():

    sensor = DJIMavic3Multispectral()

    assert sensor.bands["green"] == 560


def test_red_band():

    sensor = DJIMavic3Multispectral()

    assert sensor.bands["red"] == 650


def test_red_edge_band():

    sensor = DJIMavic3Multispectral()

    assert sensor.bands["red_edge"] == 730


def test_nir_band():

    sensor = DJIMavic3Multispectral()

    assert sensor.bands["nir"] == 860


# ---------------------------------------------------------
# Band validity
# ---------------------------------------------------------

def test_band_range():

    sensor = DJIMavic3Multispectral()

    for wavelength in sensor.bands.values():

        assert 400 <= wavelength <= 900


# ---------------------------------------------------------
# String representation
# ---------------------------------------------------------

def test_string_representation():

    sensor = DJIMavic3Multispectral()

    assert isinstance(str(sensor), str)


# ---------------------------------------------------------
# Sensor metadata
# ---------------------------------------------------------

def test_sensor_has_metadata():

    sensor = DJIMavic3Multispectral()

    assert hasattr(sensor, "bands")
    assert hasattr(sensor, "name")