"""
AgriSpectralSynth
=================

Open-source Synthetic Multispectral Dataset Generator
for Precision Agriculture and Computer Vision.

This package provides tools to generate physically-consistent
synthetic multispectral datasets from RGB drone imagery,
emulating sensors such as the DJI Mavic 3 Multispectral.

Main Features
-------------
- Spectral simulation
- Vegetation modeling
- Multispectral band synthesis
- Vegetation indices
- Canopy segmentation
- YOLO annotation generation
- Scientific reports

Author
------
Juan Carlos Vega
OpenAI Collaboration

License
-------
MIT License
"""

from importlib.metadata import version, PackageNotFoundError

# -----------------------------------------------------------------------------
# Package Information
# -----------------------------------------------------------------------------

PACKAGE_NAME = "agrispectralsynth"

try:
    __version__ = version(PACKAGE_NAME)
except PackageNotFoundError:
    # Development mode (package not installed)
    __version__ = "0.1.0-dev"

__author__ = "Juan Carlos Vega"
__email__ = ""
__license__ = "MIT"

__description__ = (
    "Synthetic Multispectral Dataset Generator for Precision Agriculture."
)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__description__",
]