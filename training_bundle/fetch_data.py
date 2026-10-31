#!/usr/bin/env python
"""Download the P&ID dataset (500 drawings + labelled symbols)."""
from pathlib import Path
from huggingface_hub import snapshot_download

DEST = Path(__file__).parent / "dataset"

def main():
    print("downloading Dataset-P&ID (500 sheets, ~1.3 GB)…")
    snapshot_download(repo_id="hamzas/digitize-pid-yolo", repo_type="dataset",
                      local_dir=str(DEST))
    root = DEST / "DigitizePID_Dataset"
    tr = len(list((root / "images" / "train").glob("*.jpg")))
    va = len(list((root / "images" / "val").glob("*.jpg")))
    lt = len(list((root / "labels" / "train").glob("*.txt")))
    lv = len(list((root / "labels" / "val").glob("*.txt")))
    print(f"\nimages: {tr} train + {va} val")
    print(f"labels: {lt} train + {lv} val")
    if tr == lt and va == lv and tr > 0:
        print("\nOK — now run:  python build_tiles.py")
    else:
        print("\nSomething is missing. Re-run this script.")

if __name__ == "__main__":
    main()
