"""
Reflectance model for AgriSpectralSynth.

Transforms an RGB image into synthetic reflectance maps
for the DJI Mavic 3 Multispectral sensor.

Author
------
Juan Carlos Vega
OpenAI Collaboration

License
-------
MIT
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from ..sensors.sensor_base import SensorBase


class ReflectanceModel:
    """
    Synthetic reflectance model.

    Converts RGB images into physically-consistent synthetic
    reflectance maps.

    Parameters
    ----------
    sensor : SensorBase
        Sensor definition.
    """

    def __init__(self, sensor: SensorBase):

        self.sensor = sensor

    # -----------------------------------------------------------------

    @staticmethod
    def normalize(rgb: np.ndarray) -> np.ndarray:
        """
        Normalize RGB image to [0,1].
        """

        rgb = rgb.astype(np.float32)

        if rgb.max() > 1:
            rgb /= 255.0

        return np.clip(rgb, 0.0, 1.0)

    # -----------------------------------------------------------------

    @staticmethod
    def vegetation_probability(rgb: np.ndarray) -> np.ndarray:
        """
        Estimate vegetation probability from RGB.

        Returns values between 0 and 1.
        """

        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        exg = 2 * g - r - b

        exg = cv2.GaussianBlur(exg, (5, 5), 0)

        exg -= exg.min()

        exg /= (exg.max() + 1e-6)

        return exg

    # -----------------------------------------------------------------

    def compute(self, rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute synthetic reflectance maps.

        Parameters
        ----------
        rgb : ndarray

        Returns
        -------
        dict
            Dictionary with all spectral bands.
        """

        rgb = self.normalize(rgb)

        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        vegetation = self.vegetation_probability(rgb)

        blue = 0.85 * b

        green = 0.95 * g

        red = 0.90 * r

        red_edge = (
            0.45 * r +
            0.45 * g +
            0.10 * vegetation
        )

        nir = (
            0.25 * r +
            0.55 * g +
            0.20 * vegetation
        )

        # Incremento en vegetación
        nir += vegetation * 0.25

        # Atenuar suelo
        nir *= (0.75 + 0.25 * vegetation)

        # Limitar rango
        blue = np.clip(blue, 0, 1)

        green = np.clip(green, 0, 1)

        red = np.clip(red, 0, 1)

        red_edge = np.clip(red_edge, 0, 1)

        nir = np.clip(nir, 0, 1)

        return {

            "Blue": blue,

            "Green": green,

            "Red": red,

            "RedEdge": red_edge,

            "NIR": nir

        }

    # -----------------------------------------------------------------

    def summary(self):

        print("=" * 60)

        print("Reflectance Model")

        print("=" * 60)

        print(f"Sensor : {self.sensor.name}")

        print(f"Bands  : {self.sensor.band_names()}")

        print("=" * 60)