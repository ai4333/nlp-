#!/usr/bin/env python
"""Fine-tune YOLO on P&ID symbol tiles (Apple Silicon / MPS).

Tuned to actually FINISH on an 18 GB machine: a previous run died mid-validation
under memory pressure with no checkpoint saved. Hence save_period=1, a reduced
validation split, and conservative workers/batch.
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="data/tiles_fastval/data.yaml")
    ap.add_argument("--model",  default="yolov8s.pt")
    ap.add_argument("--imgsz",  type=int, default=640)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--fraction", type=float, default=0.45)
    ap.add_argument("--name",   default="pid_v2")
    ap.add_argument("--no-val", action="store_true",
                    help="skip per-epoch validation (it crashes on MPS with a "
                         "leaked-semaphore fault); score afterwards with "
                         "scripts/eval_detection.py, which measures full-sheet "
                         "accuracy rather than per-tile anyway")
    a = ap.parse_args()

    YOLO(a.model).train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
        fraction=a.fraction,          # subsample tiles per epoch; data is repetitive
        device="mps", workers=2, project="runs", name=a.name, exist_ok=True,
        patience=100, cache=False, val=not a.no_val, plots=False, seed=0,
        save_period=1,                # keep a checkpoint every epoch
        # P&ID symbols have fixed orientation and no colour - disable the
        # augmentations that would teach wrong invariances.
        degrees=0.0, fliplr=0.0, flipud=0.0, mosaic=0.4,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15, scale=0.12, translate=0.04,
    )

if __name__ == "__main__":
    main()
