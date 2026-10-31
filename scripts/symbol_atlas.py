#!/usr/bin/env python
"""Crop real instances of each of the 32 symbol classes into contact sheets,
so the classes can be named by eye instead of guessed."""
import sys, random
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.edi.config import load

CFG = load(); SRC = CFG.get_path("paths.dataset")
OUT = Path("scratch/atlas"); OUT.mkdir(parents=True, exist_ok=True)
CELL, NEX, PAD = 130, 6, 10

want = defaultdict(list)
sheets = sorted((SRC/"images"/"train").glob("*.jpg"), key=lambda p: int(p.stem))[:60]
for sp in sheets:
    lp = SRC/"labels"/"train"/f"{sp.stem}.txt"
    if not lp.exists(): continue
    for ln in lp.read_text().split("\n"):
        p = ln.split()
        if len(p)==5:
            c=int(p[0])
            if len(want[c]) < NEX*3:
                want[c].append((sp.stem, *[float(x) for x in p[1:]]))
    if all(len(want[c])>=NEX*3 for c in range(32)): break

cache={}
def sheet(stem):
    if stem not in cache:
        if len(cache)>6: cache.clear()
        cache[stem]=cv2.imread(str(SRC/"images"/"train"/f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
    return cache[stem]

for grp in range(4):
    classes = list(range(grp*8, grp*8+8))
    canvas = np.full(((CELL+30)*len(classes), CELL*NEX+90, 3), 255, np.uint8)
    for r,c in enumerate(classes):
        y = r*(CELL+30)
        cv2.putText(canvas, f"c{c}", (6, y+CELL//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,180), 2)
        cv2.putText(canvas, f"(pid {c+1})", (4, y+CELL//2+26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1)
        picks = sorted(want[c], key=lambda t:(t[0]))[:NEX*3]
        random.Random(c).shuffle(picks); picks=picks[:NEX]
        for i,(stem,cx,cy,w,h) in enumerate(picks):
            img = sheet(stem)
            if img is None: continue
            H,W = img.shape
            x0=max(0,int((cx-w/2)*W)-PAD); x1=min(W,int((cx+w/2)*W)+PAD)
            y0=max(0,int((cy-h/2)*H)-PAD); y1=min(H,int((cy+h/2)*H)+PAD)
            crop=img[y0:y1, x0:x1]
            if crop.size==0: continue
            s=min((CELL-8)/max(crop.shape[0],1), (CELL-8)/max(crop.shape[1],1))
            crop=cv2.resize(crop,(max(1,int(crop.shape[1]*s)),max(1,int(crop.shape[0]*s))),interpolation=cv2.INTER_NEAREST)
            crop=cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            ox=90+i*CELL+(CELL-crop.shape[1])//2; oy=y+(CELL-crop.shape[0])//2
            canvas[oy:oy+crop.shape[0], ox:ox+crop.shape[1]]=crop
            cv2.rectangle(canvas,(90+i*CELL+2,y+2),(90+i*CELL+CELL-2,y+CELL-2),(215,215,215),1)
    cv2.imwrite(str(OUT/f"atlas_{grp}.png"), canvas)
    print("wrote", OUT/f"atlas_{grp}.png")
