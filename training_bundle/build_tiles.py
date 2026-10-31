#!/usr/bin/env python
"""Cut each 7168x4562 drawing into overlapping 512px tiles.

Symbols are ~33x62 px. Resizing a whole sheet to the detector's input would
shrink them to about six pixels, which is unrecoverable — so the sheet is tiled
at native resolution instead and the detector runs per tile.
"""
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count
import cv2, numpy as np

HERE = Path(__file__).parent
SRC = HERE / "dataset" / "DigitizePID_Dataset"
DST = HERE / "tiles"
TILE, STRIDE, MIN_VISIBLE, EMPTY_KEEP = 512, 384, 0.4, 0.08


def starts(extent, tile, stride):
    if extent <= tile:
        return [0]
    pos = list(range(0, extent - tile + 1, stride))
    if pos[-1] + tile < extent:
        pos.append(extent - tile)
    return pos


def one(args):
    split, stem = args
    img = cv2.imread(str(SRC / "images" / split / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    H, W = img.shape[:2]
    lp = SRC / "labels" / split / f"{stem}.txt"
    rows = []
    if lp.exists():
        for ln in lp.read_text().split("\n"):
            p = ln.split()
            if len(p) == 5:
                rows.append([float(x) for x in p])
    if rows:
        a = np.asarray(rows, float)
        cx, cy, w, h = a[:, 1]*W, a[:, 2]*H, a[:, 3]*W, a[:, 4]*H
        boxes = np.column_stack([a[:, 0], cx-w/2, cy-h/2, cx+w/2, cy+h/2])
    else:
        boxes = np.zeros((0, 5))

    rng = random.Random(hash(stem) & 0xFFFF)
    written = 0
    for y0 in starts(H, TILE, STRIDE):
        for x0 in starts(W, TILE, STRIDE):
            x1, y1 = min(x0+TILE, W), min(y0+TILE, H)
            keep = []
            if len(boxes):
                bx0 = np.clip(boxes[:, 1], x0, x1); by0 = np.clip(boxes[:, 2], y0, y1)
                bx1 = np.clip(boxes[:, 3], x0, x1); by1 = np.clip(boxes[:, 4], y0, y1)
                orig = (boxes[:, 3]-boxes[:, 1])*(boxes[:, 4]-boxes[:, 2])
                kept = (bx1-bx0)*(by1-by0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = np.where(orig > 0, kept/orig, 0.0)
                m = (frac >= MIN_VISIBLE) & (kept > 0)
                for c, a_, b_, cc, d_ in zip(boxes[m, 0].astype(int), bx0[m], by0[m], bx1[m], by1[m]):
                    keep.append((int(c), ((a_-x0+cc-x0)/2)/TILE, ((b_-y0+d_-y0)/2)/TILE,
                                 (cc-a_)/TILE, (d_-b_)/TILE))
            if not keep and rng.random() > EMPTY_KEEP:
                continue
            crop = img[y0:y1, x0:x1]
            if crop.shape[0] != TILE or crop.shape[1] != TILE:
                crop = cv2.copyMakeBorder(crop, 0, TILE-crop.shape[0], 0, TILE-crop.shape[1],
                                          cv2.BORDER_CONSTANT, value=255)
            name = f"{stem}_{y0}_{x0}"
            cv2.imwrite(str(DST/"images"/split/f"{name}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            (DST/"labels"/split/f"{name}.txt").write_text(
                "\n".join(f"{c} {a_:.6f} {b_:.6f} {w_:.6f} {h_:.6f}" for c, a_, b_, w_, h_ in keep))
            written += 1
    return written


def main():
    jobs = []
    for split in ("train", "val"):
        (DST/"images"/split).mkdir(parents=True, exist_ok=True)
        (DST/"labels"/split).mkdir(parents=True, exist_ok=True)
        jobs += [(split, p.stem) for p in sorted((SRC/"images"/split).glob("*.jpg"))]
    if not jobs:
        print("No images. Run  python fetch_data.py  first."); return
    print(f"tiling {len(jobs)} drawings…")
    total = 0
    with Pool(min(cpu_count(), 8)) as pool:
        for i, n in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            total += n
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} drawings, {total} tiles")
    (DST/"data.yaml").write_text(
        f"path: {DST.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnc: 32\nnames:\n"
        + "\n".join(f"  {i}: class_{i+1}" for i in range(32)) + "\n")
    print(f"\n{total} tiles written.\nNow run:  python train.py")


if __name__ == "__main__":
    main()
