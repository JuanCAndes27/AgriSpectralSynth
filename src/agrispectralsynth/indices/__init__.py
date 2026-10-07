from .base import VegetationIndex
from .evi import EVI
from .gndvi import GNDVI
from .msavi import MSAVI
from .ndre import NDRE
from .ndvi import NDVI
from .savi import SAVI

# name -> (class, bands it needs, in the order compute() expects them)
INDEX_REGISTRY = {
    "NDVI": (NDVI, ("Red", "NIR")),
    "GNDVI": (GNDVI, ("Green", "NIR")),
    "NDRE": (NDRE, ("RedEdge", "NIR")),
    "SAVI": (SAVI, ("Red", "NIR")),
    "MSAVI": (MSAVI, ("Red", "NIR")),
    "EVI": (EVI, ("Blue", "Red", "NIR")),
}

__all__ = ["VegetationIndex", "NDVI", "GNDVI", "NDRE", "SAVI", "MSAVI", "EVI", "INDEX_REGISTRY"]
