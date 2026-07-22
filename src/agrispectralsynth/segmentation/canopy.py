"""
Tree canopy segmentation.

AgriSpectralSynth
"""

from __future__ import annotations

import cv2
import numpy as np


class CanopySegmenter:

    def __init__(
        self,
        min_area: int = 500,
    ):

        self.min_area = min_area

    # ------------------------------------------------------------

    def segment(self, mask):

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        output = np.zeros_like(mask)

        objects = []

        for i in range(1, num_labels):

            area = stats[i, cv2.CC_STAT_AREA]

            if area < self.min_area:

                continue

            component = labels == i

            output[component] = 255

            x = stats[i, cv2.CC_STAT_LEFT]

            y = stats[i, cv2.CC_STAT_TOP]

            w = stats[i, cv2.CC_STAT_WIDTH]

            h = stats[i, cv2.CC_STAT_HEIGHT]

            cx, cy = centroids[i]

            objects.append({

                "id": i,

                "bbox": (x, y, w, h),

                "area": int(area),

                "centroid": (float(cx), float(cy))

            })

        return output, objects

    # ------------------------------------------------------------

    def contours(self, mask):

        contours, _ = cv2.findContours(

            mask,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE,

        )

        return contours

    # ------------------------------------------------------------

    def draw(self, rgb, objects):

        image = rgb.copy()

        for obj in objects:

            x, y, w, h = obj["bbox"]

            cv2.rectangle(

                image,

                (x, y),

                (x + w, y + h),

                (0, 255, 0),

                2,

            )

        return image