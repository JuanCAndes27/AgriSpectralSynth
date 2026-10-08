"""
Find and load sensor definitions (YAML) from the ``sensors/`` folder.

Search order for the folder:
1. the ``AGRISPECTRALSYNTH_SENSORS`` environment variable
2. ``<repository root>/sensors`` (works with ``pip install -e .``)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from ..constants import PROJECT_ROOT
from .sensor_base import Sensor


def sensors_dir() -> Path:
    env = os.environ.get("AGRISPECTRALSYNTH_SENSORS")
    path = Path(env) if env else PROJECT_ROOT / "sensors"
    if not path.is_dir():
        raise FileNotFoundError(
            f"Sensor folder not found: {path}. Run from a clone of the repository "
            "(pip install -e .) or set AGRISPECTRALSYNTH_SENSORS."
        )
    return path


def _index(folder: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for f in sorted(folder.rglob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            sid = (yaml.safe_load(fh) or {}).get("id")
        if sid:
            if sid in index:
                raise ValueError(f"Sensor id '{sid}' defined twice: {index[sid]} and {f}")
            index[sid] = f
    return index


def available_sensors(include_cameras: bool = False) -> List[str]:
    """Ids of all sensors defined in the sensors folder."""
    ids = []
    for sid, path in _index(sensors_dir()).items():
        if not include_cameras and sid == "rgb_camera":
            continue
        ids.append(sid)
    return ids


@lru_cache(maxsize=64)
def _load_path(path: str) -> Sensor:
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Sensor.from_dict(data, base_dir=sensors_dir() if p.is_relative_to(sensors_dir()) else p.parent)


def load_sensor(sensor: Union[str, Path, Sensor]) -> Sensor:
    """Load a sensor by id (``"dji_mavic3m"``) or by path to a YAML file."""
    if isinstance(sensor, Sensor):
        return sensor
    s = str(sensor)
    if s.endswith((".yaml", ".yml")) or os.sep in s or "/" in s:
        return _load_path(str(Path(s).resolve()))
    index = _index(sensors_dir())
    if s not in index:
        raise KeyError(f"Unknown sensor '{s}'. Available: {', '.join(sorted(index))}")
    return _load_path(str(index[s].resolve()))


def resolve_sensor_list(spec: Optional[Union[str, List[str]]]) -> List[str]:
    """'all' | 'drones' | 'satellites' | 'a,b' | [a, b]  ->  list of ids."""
    if spec is None:
        return ["dji_mavic3m"]
    if isinstance(spec, str):
        spec = [s.strip() for s in spec.split(",") if s.strip()]
    out: List[str] = []
    for item in spec:
        if item == "all":
            out += available_sensors()
        elif item in ("drones", "satellites"):
            platform = item[:-1]
            out += [s for s in available_sensors() if load_sensor(s).platform == platform]
        else:
            load_sensor(item)  # validates
            out.append(item)
    return list(dict.fromkeys(out))
