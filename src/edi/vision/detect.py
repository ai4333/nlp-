"""
Symbol detection over a full sheet.

Detectors share one interface so the rest of the pipeline never knows which is
running:

  YoloDetector          - the real thing: tiles the sheet, runs YOLO per tile,
                          maps boxes back to sheet coordinates, global NMS.
  GroundTruthDetector   - reads the dataset's own YOLO labels and returns them
                          as if they were predictions.

GroundTruthDetector is not a stub for missing work.  It isolates downstream
failures: if a graph or QA answer is wrong while running on ground-truth
detections, the bug is in relation extraction or the LLM, never in the detector.
It also gives the "perfect detection" ceiling for the ablation table.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Protocol
import cv2
import numpy as np

from ..config import load, yolo_to_pidqa
from ..ingest.tiling import plan_tiles, cut, to_sheet_coords, nms, yolo_to_abs

CFG = load()


class Detector(Protocol):
    def detect(self, image: np.ndarray, sheet_id: str) -> list[dict]: ...


def _record(yolo_class: int, bbox, conf: float, idx: int) -> dict:
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    return {
        "id": f"d{idx:04d}",
        "yolo_class": int(yolo_class),
        "pidqa_class": yolo_to_pidqa(int(yolo_class)),
        "bbox": [x0, y0, x1, y1],
        "confidence": round(float(conf), 4),
    }


class YoloDetector:
    def __init__(self, weights: str | Path | None = None, device: str | None = None):
        from ultralytics import YOLO
        self.weights = str(weights or CFG.get_path("detection.weights"))
        self.device = device or CFG.dotted("detection.device", "mps")
        self.model = YOLO(self.weights)

        # Take the tile size from the MODEL, not the config. A model trained on
        # 512px tiles must be fed 512px tiles: running it over 1024px tiles at
        # imgsz 1024 halves the apparent symbol size and quietly wrecks recall,
        # with nothing to indicate anything is wrong. Whoever trains it can pick
        # any size; this follows.
        trained = self._trained_imgsz()
        self.imgsz = trained or CFG.dotted("detection.imgsz")
        self.tile = trained or CFG.dotted("tiling.tile_size")
        self.stride = int(self.tile * 0.75)
        self.conf = CFG.dotted("detection.conf_threshold")
        self.iou = CFG.dotted("detection.iou_threshold")
        self.merge_iou = CFG.dotted("detection.merge_iou")

    def _trained_imgsz(self) -> int | None:
        """The imgsz the checkpoint was trained with, if it recorded one."""
        for src in (getattr(self.model, "overrides", None) or {},
                    getattr(getattr(self.model, "model", None), "args", None) or {}):
            try:
                v = src.get("imgsz")
            except AttributeError:
                continue
            if isinstance(v, (list, tuple)):
                v = v[0] if v else None
            if isinstance(v, (int, float)) and 128 <= int(v) <= 4096:
                return int(v)
        return None

    def describe(self) -> dict:
        return {"weights": Path(self.weights).name, "imgsz": self.imgsz,
                "tile": self.tile, "stride": self.stride, "device": self.device}

    def detect(self, image: np.ndarray, sheet_id: str = "") -> list[dict]:
        H, W = image.shape[:2]
        tiles = plan_tiles(W, H, self.tile, self.stride)
        crops = [cut(image, t, self.tile) for t in tiles]
        if crops and crops[0].ndim == 2:
            crops = [cv2.cvtColor(c, cv2.COLOR_GRAY2BGR) for c in crops]

        raw: list[dict] = []
        B = 8
        for s in range(0, len(crops), B):
            batch, btiles = crops[s:s + B], tiles[s:s + B]
            results = self.model.predict(batch, imgsz=self.imgsz, conf=self.conf,
                                         iou=self.iou, device=self.device, verbose=False)
            for res, t in zip(results, btiles):
                local = []
                if res.boxes is not None and len(res.boxes):
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    cf = res.boxes.conf.cpu().numpy()
                    for b, c, f in zip(xyxy, cls, cf):
                        local.append({"yolo_class": int(c), "bbox": b.tolist(),
                                      "confidence": float(f)})
                raw.extend(to_sheet_coords(local, t))

        merged = nms(raw, self.merge_iou)
        return [_record(d["yolo_class"], d["bbox"], d["confidence"], i)
                for i, d in enumerate(merged)]


class GroundTruthDetector:
    """Replays dataset labels as detections (confidence 1.0)."""

    def __init__(self, split: str | None = None):
        self.root = CFG.get_path("paths.dataset")
        self.split = split

    def _label_path(self, sheet_id: str) -> Path | None:
        for sp in ([self.split] if self.split else ["train", "val"]):
            p = self.root / "labels" / sp / f"{sheet_id}.txt"
            if p.exists():
                return p
        return None

    def detect(self, image: np.ndarray, sheet_id: str) -> list[dict]:
        p = self._label_path(sheet_id)
        if p is None:
            return []
        H, W = image.shape[:2]
        rows = [[float(x) for x in ln.split()] for ln in p.read_text().split("\n")
                if len(ln.split()) == 5]
        abs_boxes = yolo_to_abs(rows, W, H)
        return [_record(int(r[0]), r[1:], 1.0, i) for i, r in enumerate(abs_boxes)]


def get_detector(kind: str = "auto", **kw) -> Detector:
    """kind: 'yolo' | 'gt' | 'auto' (yolo if weights exist, else gt)."""
    if kind == "auto":
        kind = "yolo" if Path(CFG.get_path("detection.weights")).exists() else "gt"
    return YoloDetector(**kw) if kind == "yolo" else GroundTruthDetector(**kw)


def load_sheet(sheet_id: str, split: str | None = None) -> tuple[np.ndarray, str]:
    root = CFG.get_path("paths.dataset")
    for sp in ([split] if split else ["train", "val"]):
        p = root / "images" / sp / f"{sheet_id}.jpg"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            return img, sp
    raise FileNotFoundError(f"sheet {sheet_id} not found")


def save(dets: list[dict], sheet_id: str, image_shape) -> Path:
    out = CFG.get_path("paths.interim") / sheet_id
    out.mkdir(parents=True, exist_ok=True)
    p = out / "detections.json"
    p.write_text(json.dumps({
        "sheet_id": sheet_id,
        "image_size": [int(image_shape[1]), int(image_shape[0])],
        "tiling": {"tile": CFG.dotted("tiling.tile_size"),
                   "stride": CFG.dotted("tiling.stride")},
        "n_detections": len(dets),
        "detections": dets,
    }, indent=1))
    return p
