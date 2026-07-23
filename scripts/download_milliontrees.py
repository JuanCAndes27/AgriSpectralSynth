#!/usr/bin/env python3
"""
===========================================================
AgriSpectralSynth

Download and Prepare MillionTrees Dataset

Author:
Juan Carlos Vega
OpenAI Collaboration

License:
MIT
===========================================================
"""

from pathlib import Path
import argparse
import sys

from datasets.milliontrees import MillionTreesDataset


# ---------------------------------------------------------

def banner():

    print("\n" + "=" * 65)
    print("AgriSpectralSynth")
    print("MillionTrees Downloader")
    print("=" * 65)


# ---------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Download MillionTrees dataset."
    )

    parser.add_argument(
        "--version",
        default="small",
        choices=["mini", "small", "full"],
        help="Dataset version."
    )

    parser.add_argument(
        "--output",
        default="datasets/milliontrees",
        help="Output directory."
    )

    parser.add_argument(
        "--limit",
        default=100,
        type=int,
        help="Maximum number of RGB images to prepare."
    )

    return parser.parse_args()


# ---------------------------------------------------------

def main():

    args = parse_arguments()

    banner()

    print(f"Version : {args.version}")
    print(f"Output  : {args.output}")
    print(f"Limit   : {args.limit} images\n")

    dataset = MillionTreesDataset(
        root_dir=args.output,
        version=args.version
    )

    print("[1/4] Downloading dataset...")
    dataset.download()

    print("[2/4] Preparing folder structure...")
    dataset.prepare()

    print("[3/4] Reading metadata...")
    dataset.statistics()

    print("[4/4] Selecting first images...")
    dataset.select_images(limit=args.limit)

    print("\nDataset ready.\n")


# ---------------------------------------------------------

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\nCancelled by user.")
        sys.exit(0)

    except Exception as e:

        print(f"\nERROR:\n{e}")
        sys.exit(1)