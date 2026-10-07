#!/usr/bin/env python3
"""
Download the NeonTreeEvaluation benchmark (RGB + crown annotations).

    python scripts/download_neontree_benchmark.py            # -> data/benchmarks/NeonTreeEvaluation
    python scripts/download_neontree_benchmark.py --with-hyperspectral

Only the 194 evaluation plots that have hand annotations are fetched
(~110 MB RGB). Requires git >= 2.27. Data licence: CC0 1.0.

Benchmark: Weinstein, B. G. et al. (2021). A benchmark dataset for canopy
crown detection and delineation in co-registered airborne RGB, LiDAR and
hyperspectral imagery from the National Ecological Observation Network.
PLOS Computational Biology 17(7): e1009180.
Repository: https://github.com/weecology/NeonTreeEvaluation
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/weecology/NeonTreeEvaluation.git"
ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "NeonTreeEvaluation")
    ap.add_argument("--with-hyperspectral", action="store_true", help="also fetch the co-registered NEON hyperspectral tiles")
    ap.add_argument("--with-chm", action="store_true", help="also fetch the LiDAR canopy height model tiles")
    a = ap.parse_args()

    if shutil.which("git") is None:
        print("git is required: https://git-scm.com/downloads")
        return 1
    out = a.out
    if not (out / ".git").exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning file list into {out} ...")
        git("clone", "--depth", "1", "--filter=blob:none", "--no-checkout", REPO, str(out))
    git("config", "gc.auto", "0", cwd=out)

    files = git("ls-tree", "-r", "--name-only", "HEAD", cwd=out).splitlines()
    ann = {Path(f).stem for f in files if f.startswith("annotations/") and f.endswith(".xml")}
    rgb = {Path(f).stem for f in files if f.startswith("evaluation/RGB/") and f.endswith(".tif")}
    stems = sorted(ann & rgb)
    paths = [f"annotations/{s}.xml" for s in stems] + [f"evaluation/RGB/{s}.tif" for s in stems]
    if a.with_hyperspectral:
        paths += [f for f in files if f.startswith("evaluation/Hyperspectral/") and Path(f).stem.replace("_hyperspectral", "") in stems]
    if a.with_chm:
        paths += [f for f in files if f.startswith("evaluation/CHM/") and Path(f).stem.replace("_CHM", "") in stems]
    paths += ["LICENSE", "README.md"]

    print(f"{len(stems)} annotated plots; fetching {len(paths)} files ...")
    subprocess.run(["git", "sparse-checkout", "set", "--no-cone", "--stdin"], cwd=out, input="\n".join(paths),
                   text=True, check=True)
    subprocess.run(["git", "checkout", "-q", "HEAD"], cwd=out, check=True)
    n_rgb = len(list((out / "evaluation" / "RGB").glob("*.tif")))
    print(f"Done: {n_rgb} RGB images, annotations in {out / 'annotations'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
