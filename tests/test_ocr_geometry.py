"""Rotation mapping and tag scoring.

Vertical tags were the single largest OCR failure mode (15 of 19 inspected
misses), so the rotation maths is pinned here.
"""
import numpy as np
import cv2
import pytest
from src.edi.vision.ocr import _unrotate_pt, _score_external, _score_tag


@pytest.mark.parametrize("deg,rot", [(90, cv2.ROTATE_90_CLOCKWISE),
                                     (-90, cv2.ROTATE_90_COUNTERCLOCKWISE)])
def test_unrotate_matches_cv2(deg, rot):
    h, w = 9, 13
    for (x, y) in [(2, 1), (0, 0), (12, 8), (7, 4)]:
        img = np.zeros((h, w), np.uint8); img[y, x] = 255
        r = cv2.rotate(img, rot)
        ry, rx = np.argwhere(r == 255)[0]
        assert _unrotate_pt(float(rx), float(ry), deg, h, w) == pytest.approx((x, y))


def test_unrotate_identity():
    assert _unrotate_pt(5.0, 7.0, 0, 10, 10) == (5.0, 7.0)


def test_complete_tags_score_high():
    for t in ["IJ-43075", "UV-05201", "ST-47622", "ERV-8-20", "INS ( 45C )"]:
        assert _score_external(t) >= 0.9, t


def test_truncated_tags_score_low():
    """These are real observed truncations - each must be flagged for retry."""
    for t in ["IJ 4307", "ST 476", "ERV", "GH", "-44323"]:
        assert _score_external(t) < 0.9, t


def test_prefix_only_is_worst():
    assert _score_external("ERV") < _score_external("ST 476")


def test_internal_tag_scoring_uses_ontology_prefix():
    assert _score_tag("ZLO 433", 31) > _score_tag("XQ 433", 31)
    assert _score_tag("", 31) == 0.0
