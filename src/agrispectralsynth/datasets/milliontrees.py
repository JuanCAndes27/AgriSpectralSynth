"""
===========================================================
AgriSpectralSynth

MillionTrees Dataset Manager

Two ways of using it:

1. Local folder (no extra dependency): images you already have
   in ``root_dir`` (e.g. ``data/raw``)::

       ds = MillionTreesDataset("data/raw", limit=100)
       ds.load()
       ds.image_paths

2. Download with the official ``milliontrees`` package
   (``pip install -e ".[milliontrees]"``, needs Python 3.10-3.12)::

       ds = MillionTreesDataset("datasets/milliontrees", version="mini")
       ds.run(limit=100)    # download, copy RGB, export polygons,
                            # rasterise ground-truth crown masks

Note: the ``milliontrees`` package returns samples as the tuple
``(metadata, image, targets)``; the image file names and polygons are
read from its internal ``_input_array`` / ``_input_lookup`` /
``_y_array`` attributes (current upstream layout).

Author:
Juan Carlos Vega

License:
MIT
===========================================================
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..constants import SUPPORTED_IMAGE_FORMATS
from .base_dataset import BaseDataset

logger = logging.getLogger("MillionTrees")


class MillionTreesDataset(BaseDataset):
    """Dataset manager for MillionTrees (TreePolygons)."""

    VERSIONS = ("mini", "small", "full")

    def __init__(self, root_dir="datasets/milliontrees", version: str = "small", limit: Optional[int] = None):
        super().__init__(root_dir)
        if version not in self.VERSIONS:
            raise ValueError(f"version must be one of {self.VERSIONS}")
        self.version = version
        self.limit = limit
        self.dataset = None
        self.image_paths: List[Path] = []
        self.selected_files: List[str] = []

    # --------------------------------------------------
    # Folders
    # --------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        return self.root_dir / "raw"

    @property
    def rgb_dir(self) -> Path:
        return self.root_dir / "rgb"

    @property
    def polygon_dir(self) -> Path:
        return self.root_dir / "polygons"

    @property
    def mask_dir(self) -> Path:
        return self.root_dir / "masks"

    @property
    def metadata_dir(self) -> Path:
        return self.root_dir / "metadata"

    # --------------------------------------------------
    # Download (optional dependency)
    # --------------------------------------------------

    def download(self):
        try:
            from milliontrees import get_dataset
        except ImportError as e:
            raise ImportError(
                "The 'milliontrees' package is not installed. "
                "Install it with: pip install -e \".[milliontrees]\" (Python 3.10-3.12)."
            ) from e

        kwargs = {"root_dir": str(self.raw_dir), "download": True}
        if self.version == "mini":
            kwargs["mini"] = True
        elif self.version == "small":
            kwargs["small"] = True

        logger.info("Downloading MillionTrees TreePolygons (%s) into %s ...", self.version, self.raw_dir)
        self.dataset = get_dataset("TreePolygons", **kwargs)
        logger.info("Dataset ready: %d samples", len(self.dataset))
        return self.dataset

    def prepare(self):
        for folder in (self.raw_dir, self.rgb_dir, self.polygon_dir, self.mask_dir, self.metadata_dir):
            folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Loading
    # --------------------------------------------------

    def scan(self, folder: Optional[Path] = None) -> List[Path]:
        """List local images (sorted, honouring ``limit``)."""
        folder = Path(folder or self.root_dir)
        paths = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        )
        self.image_paths = paths[: self.limit] if self.limit else paths
        return self.image_paths

    def load(self):
        """Downloaded dataset if available, otherwise the local images in ``root_dir``."""
        if self.dataset is not None:
            return self.dataset
        return self.scan()

    # --------------------------------------------------
    # Working with the downloaded dataset
    # --------------------------------------------------

    def _require_dataset(self):
        if self.dataset is None:
            raise RuntimeError("Run download() first.")

    @property
    def images_dir(self) -> Path:
        self._require_dataset()
        return Path(self.dataset._data_dir) / "images"

    def select_images(self, limit: Optional[int] = None) -> List[str]:
        """First N unique image file names of the dataset."""
        self._require_dataset()
        limit = limit or self.limit
        unique = list(dict.fromkeys(self.dataset._input_array))
        self.selected_files = unique[:limit] if limit else unique
        logger.info("%d images selected", len(self.selected_files))
        return self.selected_files

    def polygons_for(self, filename: str) -> list:
        """Shapely polygons annotated for one image (pixel coordinates)."""
        self._require_dataset()
        idx = self.dataset._input_lookup.get(filename, [])
        return [self.dataset._y_array[i] for i in idx]

    def copy_rgb(self) -> int:
        if not self.selected_files:
            raise RuntimeError("Run select_images() first.")
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for name in self.selected_files:
            dst = self.rgb_dir / name
            if not dst.exists():
                shutil.copy2(self.images_dir / name, dst)
                copied += 1
        logger.info("%d RGB images copied to %s", copied, self.rgb_dir)
        return copied

    def copy_polygons(self) -> int:
        """Export polygons as WKT lists in JSON (one file per image)."""
        if not self.selected_files:
            raise RuntimeError("Run select_images() first.")
        self.polygon_dir.mkdir(parents=True, exist_ok=True)
        for name in self.selected_files:
            wkt = [poly.wkt for poly in self.polygons_for(name)]
            with open(self.polygon_dir / f"{Path(name).stem}.json", "w", encoding="utf-8") as f:
                json.dump(wkt, f)
        return len(self.selected_files)

    @staticmethod
    def rasterize(polygons: list, shape) -> np.ndarray:
        """Ground-truth crown mask (uint8 0/255) from shapely polygons."""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for poly in polygons:
            geoms = getattr(poly, "geoms", [poly])
            for g in geoms:
                if g.is_empty:
                    continue
                pts = np.round(np.asarray(g.exterior.coords)[:, :2]).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
        return mask

    def export_masks(self) -> int:
        """Rasterise annotated crowns -> masks/<stem>_canopy_gt.png (real labels, not NDVI)."""
        if not self.selected_files:
            raise RuntimeError("Run select_images() first.")
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        for name in self.selected_files:
            img = cv2.imread(str(self.images_dir / name), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            mask = self.rasterize(self.polygons_for(name), img.shape)
            cv2.imwrite(str(self.mask_dir / f"{Path(name).stem}_canopy_gt.png"), mask)
        return len(self.selected_files)

    def export_metadata(self) -> None:
        if not self.selected_files:
            raise RuntimeError("Run select_images() first.")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        for name in self.selected_files:
            p = Path(name)
            meta = {
                "dataset": "MillionTrees",
                "version": self.version,
                "image": p.name,
                "stem": p.stem,
                "n_polygons": len(self.polygons_for(name)),
                "synthetic": False,
                "sensor": "RGB",
                "annotations": "TreePolygons",
            }
            with open(self.metadata_dir / f"{p.stem}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    # --------------------------------------------------
    # Reporting
    # --------------------------------------------------

    def statistics(self) -> Dict[str, int]:
        stats = {
            "rgb": len([p for p in self.rgb_dir.glob("*") if p.is_file()]) if self.rgb_dir.exists() else 0,
            "polygons": len(list(self.polygon_dir.glob("*.json"))) if self.polygon_dir.exists() else 0,
            "masks": len(list(self.mask_dir.glob("*.png"))) if self.mask_dir.exists() else 0,
            "metadata": len(list(self.metadata_dir.glob("*.json"))) if self.metadata_dir.exists() else 0,
        }
        print("=" * 60)
        print(f"MillionTrees ({self.version}) - {self.root_dir}")
        for k, v in stats.items():
            print(f"  {k:9s}: {v}")
        print("=" * 60)
        return stats

    def verify_dataset(self) -> bool:
        s = self.statistics()
        ok = s["rgb"] == s["polygons"] == s["metadata"]
        if not ok:
            logger.warning("RGB / polygons / metadata counts differ")
        return ok

    def run(self, limit: Optional[int] = 100):
        self.prepare()
        self.download()
        self.select_images(limit)
        self.copy_rgb()
        self.copy_polygons()
        self.export_masks()
        self.export_metadata()
        self.verify_dataset()

    def __len__(self):
        return len(self.dataset) if self.dataset is not None else len(self.image_paths)

    def __repr__(self):
        return f"MillionTreesDataset(version='{self.version}', root='{self.root_dir}')"
