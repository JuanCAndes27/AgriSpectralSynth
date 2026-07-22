"""
Band simulator for AgriSpectralSynth.

Converts synthetic reflectance maps into exportable spectral images.

Author
------
Juan Carlos Vega
OpenAI Collaboration
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np


class BandSimulator:
    """
    Converts reflectance maps into 8-bit or 16-bit images.
    """

    def __init__(
        self,
        output_dtype: str = "uint8",
        jpeg_quality: int = 95,
        noise_std: float = 0.01,
    ):

        self.output_dtype = output_dtype
        self.jpeg_quality = jpeg_quality
        self.noise_std = noise_std

    # ------------------------------------------------------------------

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Adds Gaussian noise.
        """

        if self.noise_std <= 0:
            return image

        noise = np.random.normal(
            0,
            self.noise_std,
            image.shape,
        )

        image = image + noise

        return np.clip(image, 0, 1)

    # ------------------------------------------------------------------

    def to_uint8(self, image: np.ndarray) -> np.ndarray:

        return (255 * image).astype(np.uint8)

    # ------------------------------------------------------------------

    def to_uint16(self, image: np.ndarray) -> np.ndarray:

        return (65535 * image).astype(np.uint16)

    # ------------------------------------------------------------------

    def convert(self, image: np.ndarray):

        image = self.add_noise(image)

        if self.output_dtype == "uint16":
            return self.to_uint16(image)

        return self.to_uint8(image)

    # ------------------------------------------------------------------

    def export_band(
        self,
        image: np.ndarray,
        filename: str | Path,
    ):

        filename = Path(filename)

        image = self.convert(image)

        suffix = filename.suffix.lower()

        if suffix in [".jpg", ".jpeg"]:

            cv2.imwrite(
                str(filename),
                image,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    self.jpeg_quality,
                ],
            )

        else:

            cv2.imwrite(str(filename), image)

    # ------------------------------------------------------------------

    def export_all(
        self,
        bands: Dict[str, np.ndarray],
        output_folder: str | Path,
        extension: str = ".jpg",
    ):

        output_folder = Path(output_folder)

        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name, image in bands.items():

            filename = output_folder / f"{name}{extension}"

            self.export_band(image, filename)