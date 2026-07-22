"""
Material definition for AgriSpectralSynth.

Represents the physical properties of a terrain element (vegetation,
soil, water, etc.) together with its spectral response.

Author
------
Juan Carlos Vega
OpenAI Collaboration

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Material:
    """
    Represents one material in the synthetic scene.

    Parameters
    ----------
    name : str
        Material name.

    category : str
        vegetation, soil, water, urban...

    reflectance : dict
        Dictionary containing the reflectance values
        for each spectral band.

    chlorophyll : float

    water_content : float

    lai : float

    stress : float
    """

    name: str

    category: str

    reflectance: Dict[str, float] = field(default_factory=dict)

    chlorophyll: float = 0.0

    water_content: float = 0.0

    lai: float = 0.0

    stress: float = 0.0

    # ------------------------------------------------------------------

    def get_reflectance(self, band: str) -> float:
        """
        Returns reflectance for one band.
        """

        return self.reflectance.get(band, 0.0)

    # ------------------------------------------------------------------

    def set_reflectance(self, band: str, value: float) -> None:
        """
        Updates reflectance value.
        """

        self.reflectance[band] = float(value)

    # ------------------------------------------------------------------

    def bands(self):
        """
        Returns available bands.
        """

        return list(self.reflectance.keys())

    # ------------------------------------------------------------------

    def validate(self):
        """
        Checks reflectance values.
        """

        for band, value in self.reflectance.items():

            if value < 0 or value > 1:

                raise ValueError(
                    f"Reflectance of {band} must be between 0 and 1."
                )

    # ------------------------------------------------------------------

    def to_dict(self):
        """
        Converts material to dictionary.
        """

        return {

            "name": self.name,

            "category": self.category,

            "chlorophyll": self.chlorophyll,

            "water_content": self.water_content,

            "lai": self.lai,

            "stress": self.stress,

            "reflectance": self.reflectance,
        }

    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates Material from dictionary.
        """

        return cls(**data)

    # ------------------------------------------------------------------

    def __repr__(self):

        return (
            f"Material("
            f"name='{self.name}', "
            f"category='{self.category}', "
            f"bands={len(self.reflectance)})"
        )