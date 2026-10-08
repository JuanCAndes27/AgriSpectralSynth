from .base import VegetationIndex
from .evi import EVI
from .gndvi import GNDVI
from .msavi import MSAVI
from .ndre import NDRE
from .ndvi import NDVI
from .savi import SAVI

# name -> (class, band ROLES it needs, in the order compute() expects them).
# Each sensor YAML maps roles to its own band names (e.g. Sentinel-2: red -> B4).
INDEX_REGISTRY = {
    "NDVI": (NDVI, ("red", "nir")),
    "GNDVI": (GNDVI, ("green", "nir")),
    "NDRE": (NDRE, ("red_edge", "nir")),
    "SAVI": (SAVI, ("red", "nir")),
    "MSAVI": (MSAVI, ("red", "nir")),
    "EVI": (EVI, ("blue", "red", "nir")),
}

__all__ = ["VegetationIndex", "NDVI", "GNDVI", "NDRE", "SAVI", "MSAVI", "EVI", "INDEX_REGISTRY"]
