#!/usr/bin/env python3
"""
Download MillionTrees (TreePolygons) and prepare RGB images, polygon
annotations, ground-truth crown masks and metadata.

    pip install -e ".[milliontrees]"        # Python 3.10 - 3.12
    python scripts/download_milliontrees.py --version mini --limit 100

Then generate the synthetic products:

    agrispectralsynth -i datasets/milliontrees/rgb -o data/processed
"""

import argparse
import logging
import sys

from agrispectralsynth.datasets import MillionTreesDataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and prepare the MillionTrees dataset.")
    parser.add_argument("--version", default="mini", choices=["mini", "small", "full"])
    parser.add_argument("--output", default="datasets/milliontrees")
    parser.add_argument("--limit", default=100, type=int, help="Number of images to prepare")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        MillionTreesDataset(args.output, version=args.version).run(limit=args.limit)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
    except ImportError as e:
        print(f"\nERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
