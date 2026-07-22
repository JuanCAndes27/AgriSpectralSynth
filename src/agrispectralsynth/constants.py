"""
Global constants for AgriSpectralSynth.

This module centralizes all framework-wide constants to avoid
hard-coded values throughout the codebase.

Author
------
Juan Carlos Vega
OpenAI Collaboration

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# PROJECT INFORMATION
# =============================================================================

PROJECT_NAME: str = "AgriSpectralSynth"

PROJECT_DESCRIPTION: str = (
    "Synthetic Multispectral Dataset Generator for Precision Agriculture"
)

VERSION: str = "0.1.0"

LICENSE: str = "MIT"

# =============================================================================
# DIRECTORY NAMES
# =============================================================================

INPUT_DIR = "input"

OUTPUT_DIR = "output"

DATASETS_DIR = "datasets"

CONFIG_DIR = "configs"

SPECTRAL_LIBRARY_DIR = "spectral_library"

REPORTS_DIR = "reports"

MASKS_DIR = "masks"

YOLO_DIR = "yolo"

# =============================================================================
# IMAGE EXTENSIONS
# =============================================================================

SUPPORTED_IMAGE_FORMATS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
)

# =============================================================================
# EXPORT FORMATS
# =============================================================================

SUPPORTED_EXPORT_FORMATS = (
    "jpg",
    "png",
    "tif",
    "tiff",
)

# =============================================================================
# SENSOR NAMES
# =============================================================================

DJI_MAVIC_3M = "DJI_Mavic_3_Multispectral"

SUPPORTED_SENSORS = (
    DJI_MAVIC_3M,
)

# =============================================================================
# DJI MAVIC 3 MULTISPECTRAL
# Central wavelength (nm)
# =============================================================================

BAND_BLUE = 475

BAND_GREEN = 560

BAND_RED = 668

BAND_RED_EDGE = 717

BAND_NIR = 842

DJI_BANDS = {
    "Blue": BAND_BLUE,
    "Green": BAND_GREEN,
    "Red": BAND_RED,
    "RedEdge": BAND_RED_EDGE,
    "NIR": BAND_NIR,
}

# =============================================================================
# NDVI LIMITS
# =============================================================================

NDVI_MIN = -1.0

NDVI_MAX = 1.0

# =============================================================================
# RGB VALUES
# =============================================================================

RGB_MAX = 255

RGB_MIN = 0

# =============================================================================
# RANDOM SEED
# =============================================================================

DEFAULT_RANDOM_SEED = 42

# =============================================================================
# DATA TYPES
# =============================================================================

FLOAT32 = "float32"

UINT8 = "uint8"

UINT16 = "uint16"

# =============================================================================
# VEGETATION CLASSES
# =============================================================================

HEALTHY = "healthy"

STRESSED = "stressed"

DRY = "dry"

DEAD = "dead"

SOIL = "soil"

WATER = "water"

SHADOW = "shadow"

WEEDS = "weeds"

# =============================================================================
# SUPPORTED VEGETATION INDICES
# =============================================================================

SUPPORTED_INDICES = (
    "NDVI",
    "GNDVI",
    "SAVI",
    "MSAVI",
    "EVI",
)

# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

DEFAULT_GSD = 0.05          # meters/pixel

DEFAULT_ALTITUDE = 120.0    # meters

DEFAULT_SUN_ELEVATION = 45.0

DEFAULT_NOISE_STD = 0.01

DEFAULT_JPEG_QUALITY = 95

# =============================================================================
# PACKAGE ROOT
# =============================================================================

PACKAGE_ROOT = Path(__file__).resolve().parent

PROJECT_ROOT = PACKAGE_ROOT.parent.parent
