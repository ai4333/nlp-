"""
Sheet tiling.

A P&ID is ~7168x4562 px and its symbols are ~33x62 px.  Resizing the sheet to a
detector's native 640/1024 input shrinks a symbol to 3-6 px, which is
unrecoverable.  So the sheet is cut into overlapping tiles, the detector runs per
tile, and detections are mapped back to sheet coordinates and de-duplicated.

Everything here is pure geometry - no model dependencies - so it is unit-testable.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Sequence
import cv2
import numpy as np


@dataclass(frozen=True)
class Tile:
    idx: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int: return self.x1 - self.x0
    @property
    def h(self) -> int: return self.y1 - self.y0


def _starts(extent: int, tile: int, stride: int) -> list[int]:
    """Tile origins along one axis; the final tile is clamped flush to the edge
    so the far border is never left uncovered."""
    if extent <= tile:
        return [0]
    pos = list(range(0, extent - tile + 1, stride))
    if pos[-1] + tile < extent:
        pos.append(extent - tile)
    return pos


def plan_tiles(width: int, height: int, tile: int, stride: int) -> list[Tile]:
    """Deterministic tile grid for a sheet of the given size."""
    out, i = [], 0
    for y0 in _starts(height, tile, stride):
        for x0 in _starts(width, tile, stride):
            out.append(Tile(i, x0, y0, min(x0 + tile, width), min(y0 + tile, height)))
            i += 1
    return out


def cut(image: np.ndarray, t: Tile, tile_size: int, pad: bool = True) -> np.ndarray:
    """Crop one tile, padding to a square tile_size when the sheet edge is hit."""
    crop = image[t.y0:t.y1, t.x0:t.x1]
    if not pad or (crop.shape[0] == tile_size and crop.shape[1] == tile_size):
        return crop
    ph, pw = tile_size - crop.shape[0], tile_size - crop.shape[1]
    border = 255 if crop.ndim == 2 else (255, 255, 255)
    return cv2.copyMakeBorder(crop, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=border)


# --------------------------------------------------------------------------
# label remapping (training-time)
# --------------------------------------------------------------------------

def yolo_to_abs(rows: Sequence[Sequence[float]], W: int, H: int) -> np.ndarray:
    """(cls,cx,cy,w,h) normalised  ->  (cls,x0,y0,x1,y1) absolute pixels."""
    if len(rows) == 0:
        return np.zeros((0, 5), dtype=np.float64)
    a = np.asarray(rows, dtype=np.float64)
    cx, cy, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    return np.column_stack([a[:, 0], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def boxes_for_tile(abs_boxes: np.ndarray, t: Tile, tile_size: int,
                   min_visible: float = 0.4) -> list[tuple[int, float, float, float, float]]:
    """Clip absolute boxes into a tile and renormalise to tile coordinates.

    A box is kept only if at least `min_visible` of its area survives the crop -
    this is what stops a symbol sliced by a tile seam from becoming two bogus
    half-symbol training targets.
    """
    if abs_boxes.shape[0] == 0:
        return []
    cls = abs_boxes[:, 0].astype(int)
    x0 = np.clip(abs_boxes[:, 1], t.x0, t.x1)
    y0 = np.clip(abs_boxes[:, 2], t.y0, t.y1)
    x1 = np.clip(abs_boxes[:, 3], t.x0, t.x1)
    y1 = np.clip(abs_boxes[:, 4], t.y0, t.y1)

    orig = (abs_boxes[:, 3] - abs_boxes[:, 1]) * (abs_boxes[:, 4] - abs_boxes[:, 2])
    kept = (x1 - x0) * (y1 - y0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(orig > 0, kept / orig, 0.0)

    keep = (frac >= min_visible) & (kept > 0)
    out = []
    for c, a, b, cc, d in zip(cls[keep], x0[keep], y0[keep], x1[keep], y1[keep]):
        lx0, ly0, lx1, ly1 = a - t.x0, b - t.y0, cc - t.x0, d - t.y0
        out.append((int(c),
                    ((lx0 + lx1) / 2) / tile_size,
                    ((ly0 + ly1) / 2) / tile_size,
                    (lx1 - lx0) / tile_size,
                    (ly1 - ly0) / tile_size))
    return out


# --------------------------------------------------------------------------
# detection remapping (inference-time)
# --------------------------------------------------------------------------

def to_sheet_coords(dets: Sequence[dict], t: Tile) -> list[dict]:
    """Shift tile-local pixel boxes into sheet coordinates."""
    out = []
    for d in dets:
        x0, y0, x1, y1 = d["bbox"]
        out.append({**d, "bbox": [x0 + t.x0, y0 + t.y0, x1 + t.x0, y1 + t.y0]})
    return out


def nms(dets: list[dict], iou_thr: float = 0.55) -> list[dict]:
    """Class-wise greedy NMS over sheet coordinates.

    Overlapping tiles mean the same symbol is detected up to 4x (a 25% overlap
    grid double-covers seams in both axes).  Without this the graph gets
    duplicate nodes and every count-based answer is wrong.
    """
    if not dets:
        return []
    keep: list[dict] = []
    for cls in {d["yolo_class"] for d in dets}:
        group = sorted([d for d in dets if d["yolo_class"] == cls],
                       key=lambda d: -d["confidence"])
        boxes = np.array([g["bbox"] for g in group], dtype=np.float64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        alive = np.ones(len(group), dtype=bool)
        for i in range(len(group)):
            if not alive[i]:
                continue
            keep.append(group[i])
            if i + 1 >= len(group):
                continue
            rest = np.where(alive[i + 1:])[0] + i + 1
            if rest.size == 0:
                continue
            xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
            iou = inter / (area[i] + area[rest] - inter + 1e-9)
            alive[rest[iou > iou_thr]] = False
    return sorted(keep, key=lambda d: (d["bbox"][1], d["bbox"][0]))


def write_offsets(tiles: list[Tile], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(t) for t in tiles], indent=1))
