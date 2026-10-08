"""
Command line interface.

    agrispectralsynth --config configs/default.yaml
    agrispectralsynth -i data/raw -o data/processed --workers 8 --overwrite
    python -m agrispectralsynth -i data/raw --limit 20

Every flag overrides the corresponding value in the YAML file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agrispectralsynth",
        description="Generate synthetic multispectral bands, NDVI and canopy masks from RGB images.",
    )
    p.add_argument("-c", "--config", type=Path, default=None, help="YAML config (default: built-in defaults)")
    p.add_argument("-i", "--input", type=Path, default=None, help="Folder with RGB images (default: pipeline.input_dir)")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output folder (default: pipeline.output_dir)")
    p.add_argument("-w", "--workers", type=int, default=None, help="Parallel processes (0 = all cores - 1, 1 = sequential)")
    p.add_argument("-n", "--limit", type=int, default=None, help="Process only the first N images")
    p.add_argument("--overwrite", action="store_true", help="Regenerate even if outputs are up to date")
    p.add_argument("--recursive", action="store_true", help="Search sub-folders too")
    p.add_argument("-s", "--sensors", default=None,
                   help="Sensor ids separated by commas, or 'drones', 'satellites', 'all' (default: config)")
    p.add_argument("--list-sensors", action="store_true", help="Show available sensors and exit")
    p.add_argument("--simulate-gsd", action="store_true", help="Aggregate pixels to each sensor's ground resolution")
    p.add_argument("--source-gsd", type=float, default=None, help="GSD of the input photos in metres (PNG/JPG)")
    p.add_argument("--model", choices=["spectral", "unmixing", "legacy"], default=None, help="Reflectance model")
    p.add_argument("--cmap", default=None, help="Colormap for NDVI previews (RdYlGn, jet, viridis...)")
    p.add_argument("--all-indices", action="store_true", help="Also write GNDVI, NDRE, SAVI, MSAVI and EVI GeoTIFFs")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_sensors:
        from .sensors import available_sensors, load_sensor

        for sid in available_sensors():
            s = load_sensor(sid)
            bands = ", ".join(f"{b.name} {b.center:g}/{b.fwhm:g}" for b in s.band_list)
            print(f"{sid:22s} {s.platform:9s} {str(s.gsd_m or ''):>6s} m   {bands}")
        return 0

    cfg = load_config(args.config)
    if args.sensors:
        cfg.pipeline.sensors = [x.strip() for x in args.sensors.split(",")]
    if args.simulate_gsd:
        cfg.pipeline.simulate_gsd = True
    if args.source_gsd:
        cfg.pipeline.source_gsd_m = args.source_gsd
    if args.recursive:
        cfg.pipeline.recursive = True
    if args.model:
        cfg.reflectance.model = args.model
    if args.cmap:
        cfg.pipeline.colormap = args.cmap
    if args.all_indices:
        for flag in ("generate_gndvi", "generate_ndre", "generate_savi", "generate_msavi", "generate_evi"):
            setattr(cfg.simulation, flag, True)

    try:
        summary = run_pipeline(
            cfg,
            input_dir=args.input,
            output_dir=args.output,
            workers=args.workers,
            limit=args.limit,
            overwrite=True if args.overwrite else None,
            progress=not args.quiet,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        logging.error(str(e))
        return 1

    print(
        f"\nSensors: {', '.join(summary['sensors'])}"
        f"\nDone: {summary['ok']} processed, {summary['skipped']} up to date (skipped), "
        f"{summary['error']} errors in {summary['seconds']} s "
        f"({summary['ms_per_image']} ms/img)\nOutputs: {summary['output_dir']}"
    )
    return 0 if summary["error"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
