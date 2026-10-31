#!/usr/bin/env python
"""Cut the 500 sheets into YOLO-ready 1024px tiles (parallel)."""
from __future__ import annotations
import sys, json, random, argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.edi.config import load
from src.edi.ingest.tiling import plan_tiles, cut, yolo_to_abs, boxes_for_tile

import os
CFG = load(os.environ.get('EDI_CONFIG'))
TILE   = CFG.dotted("tiling.tile_size")
STRIDE = CFG.dotted("tiling.stride")
MINVIS = CFG.dotted("tiling.min_box_visible")
SRC    = CFG.get_path("paths.dataset")
DST    = CFG.get_path("paths.tiles")
EMPTY_KEEP = 0.08          # keep a few background tiles as negatives


def one_sheet(args):
    split, stem = args
    img_p = SRC / "images" / split / f"{stem}.jpg"
    lbl_p = SRC / "labels" / split / f"{stem}.txt"
    img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (stem, 0, 0, 0)
    H, W = img.shape[:2]

    rows = []
    if lbl_p.exists():
        for ln in lbl_p.read_text().split("\n"):
            p = ln.split()
            if len(p) == 5:
                rows.append([float(x) for x in p])
    abs_boxes = yolo_to_abs(rows, W, H)

    rng = random.Random(hash(stem) & 0xFFFF)
    im_dir = DST / "images" / split
    lb_dir = DST / "labels" / split
    written = kept_boxes = empties = 0

    for t in plan_tiles(W, H, TILE, STRIDE):
        bxs = boxes_for_tile(abs_boxes, t, TILE, MINVIS)
        if not bxs:
            if rng.random() > EMPTY_KEEP:
                continue
            empties += 1
        name = f"{stem}_{t.idx:03d}"
        cv2.imwrite(str(im_dir / f"{name}.jpg"), cut(img, t, TILE),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        (lb_dir / f"{name}.txt").write_text(
            "\n".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}" for c, a, b, w, h in bxs))
        written += 1
        kept_boxes += len(bxs)
    return (stem, written, kept_boxes, empties)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="sheets per split (0 = all)")
    a = ap.parse_args()

    jobs = []
    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
        stems = sorted((SRC / "images" / split).glob("*.jpg"), key=lambda p: int(p.stem))
        if a.limit:
            stems = stems[:a.limit]
        jobs += [(split, p.stem) for p in stems]

    print(f"tiling {len(jobs)} sheets on {min(cpu_count(),8)} workers -> {DST}", flush=True)
    tot_t = tot_b = tot_e = 0
    with Pool(min(cpu_count(), 8)) as pool:
        for i, (stem, w, b, e) in enumerate(pool.imap_unordered(one_sheet, jobs, chunksize=2), 1):
            tot_t += w; tot_b += b; tot_e += e
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} sheets | {tot_t} tiles | {tot_b} boxes", flush=True)

    names = {i: f"class_{i+1}" for i in range(32)}   # PIDQA 1-based names
    (DST / "data.yaml").write_text(
        f"path: {DST}\ntrain: images/train\nval: images/val\nnc: 32\n"
        f"names:\n" + "\n".join(f"  {k}: {v}" for k, v in names.items()) + "\n")
    print(f"\nDONE  tiles={tot_t}  boxes={tot_b}  empty_negatives={tot_e}")
    print(f"data.yaml -> {DST/'data.yaml'}")


if __name__ == "__main__":
    main()
