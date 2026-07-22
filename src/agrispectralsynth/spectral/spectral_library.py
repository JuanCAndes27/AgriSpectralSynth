"""
Spectral Library Manager

AgriSpectralSynth

This module manages the spectral material database used by the
synthetic multispectral simulator.

Author
------
Juan Carlos Vega
OpenAI Collaboration

License
-------
MIT
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from .material import Material


class SpectralLibrary:
    """
    Spectral material manager.
    """

    def __init__(self):

        self.materials: Dict[str, Material] = {}

    # ------------------------------------------------------------------

    def __len__(self):

        return len(self.materials)

    # ------------------------------------------------------------------

    def clear(self):

        self.materials.clear()

    # ------------------------------------------------------------------

    def add(self, material: Material):

        self.materials[material.name] = material

    # ------------------------------------------------------------------

    def remove(self, name: str):

        if name in self.materials:

            del self.materials[name]

    # ------------------------------------------------------------------

    def get(self, name: str) -> Material:

        if name not in self.materials:

            raise KeyError(f"Material '{name}' not found.")

        return self.materials[name]

    # ------------------------------------------------------------------

    def exists(self, name: str):

        return name in self.materials

    # ------------------------------------------------------------------

    def names(self):

        return sorted(self.materials.keys())

    # ------------------------------------------------------------------

    def categories(self):

        categories = set()

        for material in self.materials.values():

            categories.add(material.category)

        return sorted(categories)

    # ------------------------------------------------------------------

    def filter_by_category(self, category: str):

        return [

            material

            for material in self.materials.values()

            if material.category == category

        ]

    # ------------------------------------------------------------------

    def random(self, category: Optional[str] = None):

        if category is None:

            return random.choice(list(self.materials.values()))

        materials = self.filter_by_category(category)

        if len(materials) == 0:

            raise ValueError(f"No materials in category '{category}'.")

        return random.choice(materials)

    # ------------------------------------------------------------------

    def load_json(self, filename):

        filename = Path(filename)

        with open(filename, "r", encoding="utf-8") as f:

            data = json.load(f)

        material = Material.from_dict(data)

        self.add(material)

    # ------------------------------------------------------------------

    def load_folder(self, folder):

        folder = Path(folder)

        json_files = sorted(folder.rglob("*.json"))

        for file in json_files:

            self.load_json(file)

    # ------------------------------------------------------------------

    def summary(self):

        print("=" * 60)

        print("Spectral Library")

        print("=" * 60)

        print(f"Materials : {len(self)}")

        print(f"Categories: {len(self.categories())}")

        print()

        for category in self.categories():

            n = len(self.filter_by_category(category))

            print(f"{category:15s} {n}")

        print("=" * 60)

    # ------------------------------------------------------------------

    def interpolate(
        self,
        material1: Material,
        material2: Material,
        alpha: float,
    ) -> Material:
        """
        Linear interpolation between two materials.

        alpha = 0 -> material1

        alpha = 1 -> material2
        """

        if not 0 <= alpha <= 1:

            raise ValueError("alpha must be between 0 and 1.")

        reflectance = {}

        bands = material1.reflectance.keys()

        for band in bands:

            r1 = material1.get_reflectance(band)

            r2 = material2.get_reflectance(band)

            reflectance[band] = (1 - alpha) * r1 + alpha * r2

        return Material(

            name=f"{material1.name}_{material2.name}",

            category=material1.category,

            chlorophyll=(1 - alpha) * material1.chlorophyll
            + alpha * material2.chlorophyll,

            water_content=(1 - alpha) * material1.water_content
            + alpha * material2.water_content,

            lai=(1 - alpha) * material1.lai
            + alpha * material2.lai,

            stress=(1 - alpha) * material1.stress
            + alpha * material2.stress,

            reflectance=reflectance,
        )