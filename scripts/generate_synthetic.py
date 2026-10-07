#!/usr/bin/env python3
"""
Generate synthetic multispectral products for every RGB image in a folder.

Equivalent to the ``agrispectralsynth`` command installed by
``pip install -e .``. Examples (from the repo root):

    python scripts/generate_synthetic.py                       # uses configs/default.yaml
    python scripts/generate_synthetic.py -i data/raw -o data/processed --workers 8
    python scripts/generate_synthetic.py --limit 20 --overwrite
    python scripts/generate_synthetic.py --model legacy --cmap jet   # old v0.1 look

On Windows the ``if __name__ == "__main__"`` guard below is required for
multiprocessing; do not remove it.
"""

import sys
from pathlib import Path

try:
    from agrispectralsynth.cli import main
except ImportError:  # package not installed: fall back to the src/ layout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from agrispectralsynth.cli import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    default_cfg = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    if "-c" not in argv and "--config" not in argv and default_cfg.exists():
        argv = ["--config", str(default_cfg)] + argv
    sys.exit(main(argv))
