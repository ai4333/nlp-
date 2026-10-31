#!/usr/bin/env python
"""How long will training take on THIS machine?

Runs ~40 real training steps and extrapolates. Takes about a minute and tells
you your own number instead of a guess from someone else's hardware.
"""
import argparse, sys, time
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()

    data = HERE / "tiles" / "data.yaml"
    if not data.exists():
        sys.exit("No tiles yet. Run:  python fetch_data.py  then  python build_tiles.py")

    n_tiles = len(list((HERE / "tiles" / "images" / "train").glob("*.jpg")))
    import torch
    from ultralytics import YOLO

    dev = 0 if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("GPU: none detected — CPU training is roughly 20x slower.\n")
    else:
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}  ({p.total_memory/1e9:.1f} GB)\n")

    print(f"{n_tiles:,} training tiles · batch {a.batch} · "
          f"{-(-n_tiles // a.batch):,} steps per epoch")
    print("timing ~40 steps…\n")

    t0 = time.time()
    YOLO(a.model).train(
        data=str(data), imgsz=a.imgsz, batch=a.batch, device=dev,
        epochs=1, fraction=min(1.0, (40 * a.batch) / max(n_tiles, 1)),
        val=False, plots=False, save=False, workers=4, verbose=False,
        project=str(HERE / "runs"), name="_estimate", exist_ok=True,
    )
    dt = time.time() - t0

    steps = 40
    per_step = dt / steps
    per_epoch = per_step * (-(-n_tiles // a.batch))
    total = per_epoch * a.epochs

    def hms(s):
        h, m = int(s // 3600), int((s % 3600) // 60)
        return f"{h}h {m:02d}m" if h else f"{m} min"

    print("\n" + "=" * 46)
    print(f"  per step    {per_step:.2f} s")
    print(f"  per epoch   {hms(per_epoch)}")
    print(f"  {a.epochs} epochs  {hms(total)}   <-- your estimate")
    print("=" * 46)
    print("\n(startup and validation add ~10%. Stop any time — every 5th epoch")
    print(" is saved, and mAP is usually good well before the last epoch.)")


if __name__ == "__main__":
    main()
