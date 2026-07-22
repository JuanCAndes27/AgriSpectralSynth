"""
===========================================================
AgriSpectralSynth

MillionTrees Dataset Manager

Author:
Juan Carlos Vega
OpenAI Collaboration

License:
MIT
===========================================================
"""

from pathlib import Path
from typing import List, Dict
import json
import shutil
import logging

from milliontrees.datasets import TreePolygonsDataset

from .base_dataset import BaseDataset


class MillionTreesDataset(BaseDataset):
    """
    Dataset manager for MillionTrees.

    Responsibilities
    ----------------
    ✓ Download dataset
    ✓ Prepare folder structure
    ✓ Select subset of images
    ✓ Export metadata
    ✓ Verify integrity
    """

    # --------------------------------------------------

    def __init__(
        self,
        root_dir="datasets/milliontrees",
        version="small"
    ):

        super().__init__(root_dir)

        self.version = version

        self.dataset = None

        self.logger = logging.getLogger(
            "MillionTrees"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s"
        )

    # --------------------------------------------------

    @property
    def raw_dir(self):

        return self.root_dir / "raw"

    @property
    def rgb_dir(self):

        return self.root_dir / "rgb"

    @property
    def polygon_dir(self):

        return self.root_dir / "polygons"

    @property
    def mask_dir(self):

        return self.root_dir / "masks"

    @property
    def metadata_dir(self):

        return self.root_dir / "metadata"

    @property
    def synthetic_dir(self):

        return self.root_dir / "synthetic"

    # --------------------------------------------------

    def download(self):

        """
        Download MillionTrees dataset.
        """

        self.logger.info(
            "Downloading MillionTrees..."
        )

        kwargs = {

            "download": True

        }

        if self.version == "mini":

            kwargs["mini"] = True

        elif self.version == "small":

            kwargs["small"] = True

        elif self.version == "full":

            pass

        else:

            raise ValueError(
                "Version must be "
                "'mini', 'small' or 'full'."
            )

        self.dataset = TreePolygonsDataset(
            **kwargs
        )

        self.logger.info(
            "Dataset downloaded."
        )

    # --------------------------------------------------

    def prepare(self):

        """
        Create AgriSpectralSynth structure.
        """

        folders = [

            self.raw_dir,

            self.rgb_dir,

            self.polygon_dir,

            self.mask_dir,

            self.metadata_dir,

            self.synthetic_dir,

            self.synthetic_dir / "red",

            self.synthetic_dir / "green",

            self.synthetic_dir / "red_edge",

            self.synthetic_dir / "nir",

            self.synthetic_dir / "ndvi",

            self.synthetic_dir / "gndvi",

            self.synthetic_dir / "savi",

            self.synthetic_dir / "evi",

            self.synthetic_dir / "msavi",

            self.synthetic_dir / "yolo"

        ]

        for folder in folders:

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

        self.logger.info(
            "Folder structure created."
        )

    # --------------------------------------------------

    def load(self):

        """
        Returns downloaded dataset.
        """

        if self.dataset is None:

            raise RuntimeError(
                "Dataset not downloaded."
            )

        return self.dataset

    # --------------------------------------------------

    def statistics(self):

        """
        Print dataset information.
        """

        print()

        print("=" * 60)

        print("MillionTrees Dataset")

        print("=" * 60)

        print(f"Version : {self.version}")

        print(f"Root    : {self.root_dir}")

        if self.dataset is not None:

            try:

                print(
                    f"Images  : {len(self.dataset)}"
                )

            except Exception:

                print(
                    "Images  : Unknown"
                )

        print("=" * 60)
    # --------------------------------------------------
    def select_images(self, limit=100):
        """
        Select the first N samples from the dataset.

        Parameters
        ----------
        limit : int
            Maximum number of samples.
        """

        if self.dataset is None:
            raise RuntimeError(
                "Dataset not downloaded."
            )

        self.logger.info(
            f"Selecting first {limit} images..."
        )

        self.selected_samples = []

        total = min(limit, len(self.dataset))

        for i in range(total):

            sample = self.dataset[i]

            self.selected_samples.append(sample)

        self.logger.info(
            f"{len(self.selected_samples)} samples selected."
        )

        return self.selected_samples

    # --------------------------------------------------
    def copy_rgb(self):
        """
        Copy RGB images into AgriSpectralSynth folder.
        """

        if not hasattr(self, "selected_samples"):

            raise RuntimeError(
                "Run select_images() first."
            )

        self.logger.info("Copying RGB images...")

        copied = 0

        for sample in self.selected_samples:

            image_path = Path(sample["image"])

            destination = self.rgb_dir / image_path.name

            if destination.exists():

                continue

            shutil.copy2(
                image_path,
                destination
            )

            copied += 1

        self.logger.info(
            f"{copied} RGB images copied."
        )

    # --------------------------------------------------
    def copy_polygons(self):
        """
        Export polygon annotations.
        """

        if not hasattr(self, "selected_samples"):

            raise RuntimeError(
                "Run select_images() first."
            )

        self.logger.info(
            "Copying polygon annotations..."
        )

        exported = 0

        for sample in self.selected_samples:

            polygon_file = (
                self.polygon_dir /
                (Path(sample["image"]).stem + ".json")
            )

            polygons = sample["polygons"]

            with open(
                polygon_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    polygons,
                    f,
                    indent=4
                )

            exported += 1

        self.logger.info(
            f"{exported} polygon files exported."
        )

    # --------------------------------------------------
    def export_metadata(self):
        """
        Export metadata for every selected sample.
        """

        if not hasattr(self, "selected_samples"):

            raise RuntimeError(
                "Run select_images() first."
            )

        self.logger.info(
            "Generating metadata..."
        )

        for sample in self.selected_samples:

            image = Path(sample["image"])

            metadata = {

                "dataset": "MillionTrees",

                "version": self.version,

                "image": image.name,

                "stem": image.stem,

                "suffix": image.suffix,

                "synthetic": False,

                "sensor": "RGB",

                "annotations": "TreePolygons"

            }

            metadata_file = (
                self.metadata_dir /
                f"{image.stem}.json"
            )

            with open(
                metadata_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    metadata,
                    f,
                    indent=4
                )

        self.logger.info(
            "Metadata exported."
        )

    # --------------------------------------------------
    def verify_dataset(self):
        """
        Verify dataset integrity.
        """

        rgb = len(
            list(
                self.rgb_dir.glob("*")
            )
        )

        polygons = len(
            list(
                self.polygon_dir.glob("*.json")
            )
        )

        metadata = len(
            list(
                self.metadata_dir.glob("*.json")
            )
        )

        print()

        print("=" * 60)

        print("Dataset verification")

        print("=" * 60)

        print(f"RGB images : {rgb}")

        print(f"Polygons   : {polygons}")

        print(f"Metadata   : {metadata}")

        print("=" * 60)

        if rgb != polygons:

            self.logger.warning(
                "RGB/Polygon mismatch."
            )

        if rgb != metadata:

            self.logger.warning(
                "RGB/Metadata mismatch."
            )
    # --------------------------------------------------
    def clean(self):
        """
        Remove generated files while preserving
        the folder structure.
        """

        self.logger.info("Cleaning dataset folders...")

        folders = [

            self.rgb_dir,
            self.polygon_dir,
            self.mask_dir,
            self.metadata_dir,

            self.synthetic_dir / "red",
            self.synthetic_dir / "green",
            self.synthetic_dir / "red_edge",
            self.synthetic_dir / "nir",
            self.synthetic_dir / "ndvi",
            self.synthetic_dir / "gndvi",
            self.synthetic_dir / "savi",
            self.synthetic_dir / "evi",
            self.synthetic_dir / "msavi",
            self.synthetic_dir / "yolo"

        ]

        for folder in folders:

            if not folder.exists():
                continue

            for file in folder.iterdir():

                if file.is_file():

                    file.unlink()

        self.logger.info("Dataset cleaned.")

    # --------------------------------------------------
    def summary(self):
        """
        Print dataset summary.
        """

        print()

        print("=" * 70)
        print("AgriSpectralSynth")
        print("MillionTrees Summary")
        print("=" * 70)

        print(f"Dataset : MillionTrees")
        print(f"Version : {self.version}")
        print(f"Root    : {self.root_dir}")

        if hasattr(self, "selected_samples"):

            print(
                f"Selected samples : {len(self.selected_samples)}"
            )

        print("=" * 70)

    # --------------------------------------------------
    def run(self, limit=100):
        """
        Complete pipeline.

        Parameters
        ----------
        limit : int
            Number of images to prepare.
        """

        self.download()

        self.prepare()

        self.statistics()

        self.select_images(limit)

        self.copy_rgb()

        self.copy_polygons()

        self.export_metadata()

        self.verify_dataset()

        self.summary()

    # --------------------------------------------------
    def __len__(self):

        if self.dataset is None:

            return 0

        return len(self.dataset)

    # --------------------------------------------------
    def __repr__(self):

        return (
            f"MillionTreesDataset("
            f"version='{self.version}', "
            f"root='{self.root_dir}')"
        )