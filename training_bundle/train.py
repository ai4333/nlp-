#!/usr/bin/env python
"""Train the P&ID symbol detector."""
import argparse, sys
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, help="0 for first GPU, or 'cpu'")
    ap.add_argument("--name", default="pid")
    ap.add_argument("--no-val", action="store_true",
                    help="skip per-epoch validation (only needed on Apple Silicon)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if a.device is None:
        a.device = 0 if torch.cuda.is_available() else "cpu"
    if a.device == "cpu":
        print("!! No GPU detected — this will take many hours.")
        print("   If you have an NVIDIA card, install the CUDA build:")
        print("   pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu121\n")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)\n")

    data = HERE / "tiles" / "data.yaml"
    if not data.exists():
        sys.exit("No tiles. Run:  python fetch_data.py  then  python build_tiles.py")

    YOLO(a.model).train(
        data=str(data), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
        device=a.device, project=str(HERE / "runs"), name=a.name, exist_ok=True,
        resume=a.resume, val=not a.no_val, save_period=5, patience=15,
        plots=True, seed=0, workers=4,
        # P&ID symbols have a fixed orientation and no colour, so the usual
        # flip/rotate/hue augmentations would teach wrong invariances.
        degrees=0.0, fliplr=0.0, flipud=0.0, mosaic=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, scale=0.15, translate=0.05,
    )
    print("\nDone. Now run:  python verify.py")
    print(f"Then send back:  runs/{a.name}/weights/best.pt  and  runs/{a.name}/results.csv")


if __name__ == "__main__":
    main()
