"""
===========================================================
AgriSpectralSynth

MillionTrees Dataset Interface

Supports:

    • mini
    • small
    • full

Author:
Juan Carlos Vega
OpenAI Collaboration

License:
MIT
===========================================================
"""

from pathlib import Path

from milliontrees.datasets import TreePolygonsDataset

from .base_dataset import BaseDataset


class MillionTreesDataset(BaseDataset):

    def __init__(
        self,
        root_dir="datasets/milliontrees",
        version="small"
    ):

        super().__init__(root_dir)

        self.version = version

        self.dataset = None

    # --------------------------------------------------

    def download(self):

        print("=" * 60)
        print("Downloading MillionTrees Dataset")
        print("=" * 60)

        if self.version not in [
            "mini",
            "small",
            "full"
        ]:
            raise ValueError(
                "Version must be: mini, small or full"
            )

        self.create_directory()

        kwargs = {
            "download": True
        }

        if self.version == "mini":
            kwargs["mini"] = True

        elif self.version == "small":
            kwargs["small"] = True

        self.dataset = TreePolygonsDataset(
            **kwargs
        )

        print("Download completed.")

    # --------------------------------------------------

    def prepare(self):

        print("Preparing dataset...")

        folders = [

            "raw",
            "rgb",
            "polygons",
            "masks",
            "metadata",
            "synthetic"

        ]

        for folder in folders:

            Path(
                self.root_dir / folder
            ).mkdir(
                parents=True,
                exist_ok=True
            )

        print("Folders ready.")

    # --------------------------------------------------

    def load(self):

        if self.dataset is None:

            raise RuntimeError(
                "Dataset not downloaded."
            )

        return self.dataset

    # --------------------------------------------------

    def statistics(self):

        print()

        print("=" * 60)

        print("MillionTrees Dataset")

        print("=" * 60)

        print(f"Version : {self.version}")

        print(f"Root    : {self.root_dir}")

        if self.dataset is None:

            print("Dataset not loaded.")

        else:

            try:

                print(
                    f"Images  : {len(self.dataset)}"
                )

            except Exception:

                print(
                    "Unable to determine number of images."
                )

        print("=" * 60)