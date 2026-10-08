"""
Ground-truth tree crowns.

Supported annotation formats (all become an (N, 4) array of boxes
``xmin, ymin, xmax, ymax`` in pixel coordinates):

* Pascal VOC XML  - NeonTreeEvaluation benchmark, DeepForest samples
* CSV with xmin, ymin, xmax, ymax columns (DeepForest format; an
  ``image_path`` column selects the rows of one image)
* CSV with a WKT ``polygon`` / ``geometry`` column (MillionTrees
  TreePolygons); each polygon is reduced to its bounding box, which is
  what the box-based benchmark metric uses.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass
class GroundTruth:
    image: str                       # image file name (stem used for matching)
    boxes: np.ndarray                # (N, 4) float32 xmin, ymin, xmax, ymax
    width: Optional[int] = None
    height: Optional[int] = None
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.boxes.shape[0])


def _as_boxes(rows: List[List[float]]) -> np.ndarray:
    b = np.asarray(rows, dtype=np.float32).reshape(-1, 4)
    # normalise corner order and drop degenerate boxes
    b = np.stack([np.minimum(b[:, 0], b[:, 2]), np.minimum(b[:, 1], b[:, 3]),
                  np.maximum(b[:, 0], b[:, 2]), np.maximum(b[:, 1], b[:, 3])], axis=1)
    keep = (b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)
    return b[keep]


def read_voc_xml(path: Union[str, Path]) -> GroundTruth:
    root = ET.parse(path).getroot()
    rows = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        rows.append([float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
    size = root.find("size")
    w = int(size.findtext("width")) if size is not None and size.findtext("width") else None
    h = int(size.findtext("height")) if size is not None and size.findtext("height") else None
    fname = root.findtext("filename") or Path(path).with_suffix(".tif").name
    return GroundTruth(fname, _as_boxes(rows), w, h, {"source": str(path)})


def wkt_bounds(wkt: str) -> List[float]:
    """Bounding box of a WKT POLYGON / MULTIPOLYGON without needing shapely."""
    nums = np.asarray([float(x) for x in _NUM.findall(wkt)], dtype=np.float64)
    if nums.size < 6 or nums.size % 2:
        raise ValueError(f"cannot parse WKT: {wkt[:60]}")
    xy = nums.reshape(-1, 2)
    return [xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()]


def read_csv_annotations(path: Union[str, Path], image: Optional[str] = None) -> Dict[str, GroundTruth]:
    """
    Read a CSV of annotations, grouped by image. Columns recognised:
    image_path | filename | image ; xmin, ymin, xmax, ymax ; or polygon | geometry (WKT).
    """
    groups: Dict[str, List[List[float]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = r.get("image_path") or r.get("filename") or r.get("image") or Path(path).stem
            name = Path(name).name
            if image and Path(name).stem != Path(image).stem:
                continue
            if all(k in r and r[k] != "" for k in ("xmin", "ymin", "xmax", "ymax")):
                box = [float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
            elif r.get("polygon") or r.get("geometry"):
                box = wkt_bounds(r.get("polygon") or r.get("geometry"))
            else:
                continue
            groups.setdefault(name, []).append(box)
    return {n: GroundTruth(n, _as_boxes(rows), meta={"source": str(path)}) for n, rows in groups.items()}


def load_annotations(folder: Union[str, Path]) -> Dict[str, GroundTruth]:
    """All annotations in a folder, keyed by image stem."""
    folder = Path(folder)
    out: Dict[str, GroundTruth] = {}
    for f in sorted(folder.glob("*.xml")):
        gt = read_voc_xml(f)
        out[Path(gt.image).stem] = gt
    for f in sorted(folder.glob("*.csv")):
        for name, gt in read_csv_annotations(f).items():
            out.setdefault(Path(name).stem, gt)
    return out


def site_of(stem: str) -> str:
    """NEON site code from a NeonTreeEvaluation / NEON file name.

    'SJER_002_2018' -> 'SJER';  '2018_TEAK_3_315000_4096000_image_163' -> 'TEAK'
    """
    for tok in stem.split("_"):
        if len(tok) == 4 and tok.isalpha() and tok.isupper():
            return tok
    return "UNKNOWN"
