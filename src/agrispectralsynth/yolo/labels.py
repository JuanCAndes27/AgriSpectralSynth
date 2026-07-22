"""
YOLO Segmentation Label Generator

AgriSpectralSynth

Generates YOLO segmentation labels from binary canopy masks.

Author
------
Juan Carlos Vega
OpenAI Collaboration

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np


class YOLOLabelGenerator:
    """
    Generates YOLO Segmentation labels.

    Output format

    class x1 y1 x2 y2 x3 y3 ...

    where coordinates are normalized to [0,1]
    """

    def __init__(
        self,
        class_id: int = 0,
        min_points: int = 10,
        epsilon_factor: float = 0.003,
    ):

        self.class_id = class_id
        self.min_points = min_points
        self.epsilon_factor = epsilon_factor

    # ----------------------------------------------------------

    @staticmethod
    def normalize_points(points, width, height):

        normalized = []

        for p in points:

            x = p[0][0] / width
            y = p[0][1] / height

            normalized.extend([x, y])

        return normalized

    # ----------------------------------------------------------

    def contour_to_polygon(self, contour):

        perimeter = cv2.arcLength(contour, True)

        epsilon = self.epsilon_factor * perimeter

        polygon = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )

        return polygon

    # ----------------------------------------------------------

    def generate(
        self,
        mask: np.ndarray,
    ) -> List[str]:

        h, w = mask.shape

        contours, _ = cv2.findContours(

            mask,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE,

        )

        labels = []

        for contour in contours:

            polygon = self.contour_to_polygon(contour)

            if len(polygon) < self.min_points:

                continue

            coords = self.normalize_points(
                polygon,
                w,
                h,
            )

            line = str(self.class_id)

            for value in coords:

                line += f" {value:.6f}"

            labels.append(line)

        return labels

    # ----------------------------------------------------------

    def save(
        self,
        mask: np.ndarray,
        filename: str | Path,
    ):

        filename = Path(filename)

        labels = self.generate(mask)

        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as f:

            for line in labels:

                f.write(line + "\n")

    # ----------------------------------------------------------

    def statistics(
        self,
        mask: np.ndarray,
    ):

        contours, _ = cv2.findContours(

            mask,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE,

        )

        return {

            "objects": len(contours),

            "class_id": self.class_id,

        }