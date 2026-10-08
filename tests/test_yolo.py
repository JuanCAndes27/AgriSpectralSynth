"""
AgriSpectralSynth - YOLO segmentation labels from canopy masks.
"""

import cv2
import numpy as np

from agrispectralsynth.yolo import YOLOLabelGenerator


def two_crowns():
    m = np.zeros((100, 200), np.uint8)
    cv2.circle(m, (50, 50), 20, 255, -1)
    cv2.circle(m, (150, 50), 25, 255, -1)
    m[90:92, 190:192] = 255          # 4-pixel speck: below min_area
    return m


def test_one_line_per_crown():
    lines = YOLOLabelGenerator(class_id=3).generate(two_crowns())
    assert len(lines) == 2
    for line in lines:
        parts = line.split()
        assert parts[0] == "3"
        coords = np.array(parts[1:], dtype=float)
        assert len(coords) % 2 == 0 and len(coords) >= 6          # >= 3 vertices
        assert coords.min() >= 0 and coords.max() <= 1             # normalised


def test_coordinates_are_x_then_y():
    m = np.zeros((100, 200), np.uint8)
    cv2.rectangle(m, (150, 10), (190, 40), 255, -1)                 # right half, top
    xy = np.array(YOLOLabelGenerator().generate(m)[0].split()[1:], dtype=float).reshape(-1, 2)
    assert xy[:, 0].min() > 0.7 and xy[:, 1].max() < 0.45


def test_min_area_and_empty_mask():
    gen = YOLOLabelGenerator(min_area=5000)
    assert gen.generate(two_crowns()) == []
    assert YOLOLabelGenerator().generate(np.zeros((50, 50), np.uint8)) == []


def test_save_and_statistics(tmp_path):
    gen = YOLOLabelGenerator()
    f = tmp_path / "img.txt"
    gen.save(two_crowns(), f)
    assert len(f.read_text().strip().splitlines()) == 2
    assert gen.statistics(two_crowns())["objects"] == 3        # statistics counts every contour
