#!/usr/bin/env python3
"""
Generate the spectral library (CSV, 400-1000 nm, 1 nm) from PROSAIL.

    pip install -e ".[prosail]"
    python scripts/build_spectral_library.py

Writes:
    spectral_library/vegetation/<preset>.csv    (healthy, stressed, dry, dead)
    spectral_library/soil/soil_dry.csv, soil_wet.csv, soil_mixed.csv

The CSV files are committed to the repository, so the simulator itself
does not need PROSAIL installed. Re-run this script only to change the
presets in spectral_library/vegetation/prosail_presets.yaml.

PROSAIL implementation: https://github.com/jgomezdans/prosail (GPLv3).
It is an optional dependency; only the generated spectra are stored here.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "spectral_library"
WL = np.arange(400, 1001)


def write_csv(path: Path, values: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        f.write("wavelength_nm,reflectance\n")
        for wl, r in zip(WL, values):
            f.write(f"{wl},{r:.5f}\n")
    print(f"  {path.relative_to(ROOT)}  NIR/Red = {values[860 - 400] / values[650 - 400]:.2f}")


def main() -> int:
    try:
        import prosail
    except ImportError:
        print('PROSAIL is not installed: pip install -e ".[prosail]"')
        return 1

    cfg = yaml.safe_load(open(LIB / "vegetation" / "prosail_presets.yaml", encoding="utf-8"))
    geo, soil = cfg["geometry"], cfg["soil"]
    today = date.today().isoformat()
    idx = slice(0, WL.size)  # PROSAIL output starts at 400 nm, 1 nm step

    print("Soil")
    s = prosail.spectral_lib.soil
    dry, wet = s.rsoil1[idx], s.rsoil2[idx]
    mixed = soil["rsoil"] * (soil["psoil"] * dry + (1 - soil["psoil"]) * wet)
    common = {"source": "PROSAIL soil spectra (prosail 2.0.5 package)", "generated": today, "units": "reflectance 0-1"}
    write_csv(LIB / "soil" / "soil_dry.csv", dry, {"name": "soil_dry", "category": "soil", **common})
    write_csv(LIB / "soil" / "soil_wet.csv", wet, {"name": "soil_wet", "category": "soil", **common})
    write_csv(
        LIB / "soil" / "soil_mixed.csv",
        mixed,
        {"name": "soil_mixed", "category": "soil", "mix": f"{soil['psoil']} dry + {1 - soil['psoil']:.1f} wet", **common},
    )

    print("Vegetation")
    for name, p in cfg["presets"].items():
        r = prosail.run_prosail(
            p["N"], p["cab"], p["car"], p["cbrown"], p["cw"], p["cm"], p["lai"],
            p["ala"], p["hspot"], geo["tts"], geo["tto"], geo["psi"],
            ant=p["ant"], prospect_version="D", typelidf=2,
            rsoil=soil["rsoil"], psoil=soil["psoil"],
        )[idx]
        params = ", ".join(f"{k}={v}" for k, v in p.items() if k != "description")
        write_csv(
            LIB / "vegetation" / f"{name}.csv",
            r,
            {
                "name": name,
                "category": "vegetation",
                "description": p["description"],
                "model": "PROSPECT-D + 4SAIL (prosail 2.0.5)",
                "parameters": params,
                "geometry": f"tts={geo['tts']}, tto={geo['tto']}, psi={geo['psi']}",
                "generated": today,
                "units": "reflectance 0-1 (directional, SDR)",
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
