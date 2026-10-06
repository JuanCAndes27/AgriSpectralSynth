"""
Configuration management for AgriSpectralSynth.

Loads configuration parameters from YAML files and validates them
with Pydantic models. Every section is optional in the YAML file;
missing values fall back to the defaults below.

Author
------
Juan Carlos Vega

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

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
# SIMULATION  (which products are generated)
# =============================================================================

class SimulationConfig(BaseModel):
    generate_red: bool = True        # kept for compatibility (Red is always inside the MS GeoTIFF)
    generate_nir: bool = True        # NIR preview PNG
    generate_ndvi: bool = True       # NDVI float GeoTIFF + coloured PNG
    # Extra indices: off by default because they can be recomputed at any time
    # from multispectral/*_MS.tif and they multiply disk usage (~4 MB each per 1000x1000 image).
    generate_gndvi: bool = False
    generate_ndre: bool = False
    generate_savi: bool = False
    generate_msavi: bool = False
    generate_evi: bool = False

    noise_std: float = 0.01
    jpeg_quality: int = 95


# =============================================================================
# REFLECTANCE MODEL
# =============================================================================

class ReflectanceConfig(BaseModel):
    model: Literal["unmixing", "legacy"] = "unmixing"
    linearize: bool = True
    reflectance_scale: float = 0.6
    exg_low: float = 0.02
    exg_high: float = 0.10
    dark_low: float = 0.04
    dark_high: float = 0.12
    chlorophyll_absorption: float = 0.60
    k_veg: float = 2.0
    k_soil: float = 1.25
    rededge_veg: float = 0.30
    rededge_bg: float = 0.50


# =============================================================================
# CANOPY
# =============================================================================

class CanopyConfig(BaseModel):
    ndvi_threshold: float = 0.45
    min_fraction: float = 0.5
    kernel_size: int = 5
    min_area: int = 30


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
    export_format: Literal["png", "jpg"] = "png"   # format of the coloured previews
    save_multispectral: bool = True                # 5-band uint16 GeoTIFF (reflectance x 10000)
    save_masks: bool = True
    save_yolo: bool = False
    save_reports: bool = True                      # manifest.csv with per-image statistics
    compress: Literal["none", "deflate", "zstd", "lzw"] = "zstd"   # zstd: ~same speed as none, ~20 % smaller


# =============================================================================
# PIPELINE  (batch execution)
# =============================================================================

class PipelineConfig(BaseModel):
    input_dir: str = "data/raw"
    output_dir: str = "data/processed"
    recursive: bool = False
    limit: Optional[int] = None
    workers: int = 0                  # 0 = all CPU cores minus one
    overwrite: bool = False           # False = skip images whose outputs are up to date
    colormap: str = "RdYlGn"          # any matplotlib colormap ("jet" reproduces the old look)
    ndvi_vmin: float = -0.2
    ndvi_vmax: float = 1.0


# =============================================================================
# ROOT CONFIGURATION
# =============================================================================

class AppConfig(BaseModel):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    sensor: SensorConfig = Field(default_factory=SensorConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    reflectance: ReflectanceConfig = Field(default_factory=ReflectanceConfig)
    canopy: CanopyConfig = Field(default_factory=CanopyConfig)
    vegetation: VegetationConfig = Field(default_factory=VegetationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    def enabled_indices(self) -> List[str]:
        s = self.simulation
        flags = {
            "GNDVI": s.generate_gndvi,
            "NDRE": s.generate_ndre,
            "SAVI": s.generate_savi,
            "MSAVI": s.generate_msavi,
            "EVI": s.generate_evi,
        }
        return [name for name, on in flags.items() if on]


# =============================================================================
# YAML LOADER
# =============================================================================

def load_config(config_file: str | Path | None = None) -> AppConfig:
    """
    Load a YAML configuration. ``None`` returns the defaults.
    """
    if config_file is None:
        return AppConfig()

    with open(Path(config_file), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig(**data)
