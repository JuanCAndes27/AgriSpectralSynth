"""
Configuration management for AgriSpectralSynth.

Loads configuration parameters from YAML files and validates
them using Pydantic models.

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
from typing import List

import yaml
from pydantic import BaseModel, Field


# =============================================================================
# GENERAL
# =============================================================================

class GeneralConfig(BaseModel):
    project_name: str = "AgriSpectralSynth"
    random_seed: int = 42
    verbose: bool = True


# =============================================================================
# SENSOR
# =============================================================================

class SensorConfig(BaseModel):
    name: str = "DJI_Mavic_3_Multispectral"
    altitude: float = 120.0
    gsd: float = 0.05


# =============================================================================
# SIMULATION
# =============================================================================

class SimulationConfig(BaseModel):
    generate_red: bool = True
    generate_nir: bool = True
    generate_ndvi: bool = True
    generate_gndvi: bool = True
    generate_savi: bool = True
    generate_msavi: bool = True
    generate_evi: bool = True

    noise_std: float = 0.01

    jpeg_quality: int = 95


# =============================================================================
# VEGETATION
# =============================================================================

class VegetationConfig(BaseModel):

    healthy: float = 0.70

    stressed: float = 0.15

    dry: float = 0.10

    dead: float = 0.05


# =============================================================================
# OUTPUT
# =============================================================================

class OutputConfig(BaseModel):

    export_format: str = "jpg"

    save_masks: bool = True

    save_yolo: bool = True

    save_reports: bool = True


# =============================================================================
# ROOT CONFIGURATION
# =============================================================================

class AppConfig(BaseModel):

    general: GeneralConfig = Field(default_factory=GeneralConfig)

    sensor: SensorConfig = Field(default_factory=SensorConfig)

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    vegetation: VegetationConfig = Field(default_factory=VegetationConfig)

    output: OutputConfig = Field(default_factory=OutputConfig)


# =============================================================================
# YAML LOADER
# =============================================================================

def load_config(config_file: str | Path) -> AppConfig:
    """
    Load YAML configuration.

    Parameters
    ----------
    config_file : str or Path

        Path to YAML configuration.

    Returns
    -------
    AppConfig
    """

    config_file = Path(config_file)

    with open(config_file, "r", encoding="utf-8") as f:

        data = yaml.safe_load(f)

    return AppConfig(**data)