"""
DJI Mavic 3 Multispectral.

The definition lives in ``sensors/drones/dji_mavic3m.yaml``; this class
is a convenience wrapper kept for backwards compatibility:

    DJIMavic3M()  ==  load_sensor("dji_mavic3m")
"""

from __future__ import annotations

from .registry import load_sensor
from .sensor_base import Sensor


class DJIMavic3M(Sensor):
    def __init__(self):
        s = load_sensor("dji_mavic3m")
        super().__init__(**{k: getattr(s, k) for k in Sensor.__dataclass_fields__})  # type: ignore[attr-defined]

    @property
    def green(self) -> float:
        return self.wavelength("Green")

    @property
    def red(self) -> float:
        return self.wavelength("Red")

    @property
    def red_edge(self) -> float:
        return self.wavelength("RedEdge")

    @property
    def nir(self) -> float:
        return self.wavelength("NIR")


DJIMavic3Multispectral = DJIMavic3M
