"""Tiling geometry: symbols are ~33x62 px on a 7168x4562 sheet, so full-sheet
resizing destroys them. These tests pin the grid and the coordinate mapping."""
import numpy as np
from src.edi.ingest.tiling import plan_tiles, cut, yolo_to_abs, boxes_for_tile, nms, Tile


def test_real_sheet_grid():
    t = plan_tiles(7168, 4562, 1024, 768)
    assert len(t) == 54
    assert max(x.x1 for x in t) == 7168, "right edge must be covered"
    assert max(x.y1 for x in t) == 4562, "bottom edge must be covered"


def test_small_image_single_tile():
    assert len(plan_tiles(500, 400, 1024, 768)) == 1


def test_tiles_overlap():
    t = plan_tiles(4000, 1024, 1024, 768)
    xs = sorted({x.x0 for x in t})
    assert xs[1] - xs[0] == 768 < 1024


def test_cut_pads_to_square():
    img = np.zeros((4562, 7168), np.uint8)
    last = plan_tiles(7168, 4562, 1024, 768)[-1]
    assert cut(img, last, 1024).shape == (1024, 1024)


def test_yolo_to_abs():
    out = yolo_to_abs([[3, 0.5, 0.5, 0.1, 0.2]], 1000, 500)
    assert out[0].tolist() == [3, 450, 200, 550, 300]


def test_box_dropped_when_mostly_outside():
    """A symbol sliced by a tile seam must not become a half-symbol target."""
    abs_boxes = yolo_to_abs([[0, 0.5, 0.5, 0.1, 0.1]], 1000, 1000)  # 450..550
    mostly_in = boxes_for_tile(abs_boxes, Tile(0, 0, 0, 540, 1000), 1024, 0.4)
    barely_in = boxes_for_tile(abs_boxes, Tile(0, 0, 0, 460, 1000), 1024, 0.4)
    assert len(mostly_in) == 1
    assert len(barely_in) == 0


def test_nms_dedupes_overlapping_tiles():
    """25% overlap means a symbol can be detected up to 4x; duplicates would
    corrupt every count-based answer."""
    d = [{"yolo_class": 5, "bbox": [100, 100, 140, 160], "confidence": 0.9},
         {"yolo_class": 5, "bbox": [102, 101, 142, 161], "confidence": 0.8},
         {"yolo_class": 5, "bbox": [800, 800, 840, 860], "confidence": 0.7}]
    assert len(nms(d, 0.55)) == 2


def test_nms_keeps_different_classes():
    d = [{"yolo_class": 1, "bbox": [10, 10, 50, 50], "confidence": 0.9},
         {"yolo_class": 2, "bbox": [10, 10, 50, 50], "confidence": 0.9}]
    assert len(nms(d, 0.55)) == 2
