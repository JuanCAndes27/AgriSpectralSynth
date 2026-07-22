"""
PROSAIL interface.

This module is reserved for a future implementation of the
PROSAIL radiative transfer model.

Version 1.0 uses an empirical reflectance model.
"""

from __future__ import annotations

import numpy as np


class ProsailModel:
    """
    Placeholder interface for future PROSAIL implementation.
    """

    def __init__(self):

        self.available = False

    def compute(
        self,
        rgb: np.ndarray,
    ):
        """
        Future implementation.

        Raises
        ------
        NotImplementedError
        """

        raise NotImplementedError(

            "PROSAIL is not implemented in version 1. "

            "Use ReflectanceModel instead."
        )

    def summary(self):

        print()

        print("=" * 60)

        print("PROSAIL MODEL")

        print("=" * 60)

        print()

        print("Status : Placeholder")

        print("Version: 2.0")

        print()

        print("=" * 60)