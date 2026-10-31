#!/usr/bin/env python
"""Run what CAN run on a real drawing with no trained detector, and report honestly.

Detection needs a model trained on this drawing's symbol legend. OCR, line
tracing and topology do not — they are geometric. This measures which parts of
the pipeline transfer to a real offshore drawing today.
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from src.edi.ingest.acquire import load_any
from src.edi.ingest.tiling import plan_tiles, cut
from src.edi.vision import ocr as O
from src.edi.graph import relations as R

sheet = open("scratch/offshore_id.txt").read().strip()
img = load_any(sheet)
H, W = img.shape[:2]
print(f"sheet {sheet}  {W}x{H}\n")

# ---- 1. OCR over the whole sheet -----------------------------------------
eng = O.OcrEngine.get()
tiles = plan_tiles(W, H, 1024, 820)
crops = [(cv2.cvtColor(cut(img, t, 1024), cv2.COLOR_GRAY2BGR), t.x0, t.y0) for t in tiles]
t0 = time.time()
lines = O._collect_rotated(eng, crops, 2.0)
dt = time.time() - t0
print(f"1. OCR      {len(tiles)} tiles in {dt:.0f}s -> {len(lines)} raw text lines")

texts = O._dedupe(lines)
print(f"            {len(texts)} after dedupe")

import re
TAG = re.compile(r"^[A-Z]{1,3}[-–][0-9]{3,5}([A-Z/]{0,6})$")
INST = re.compile(r"^(PC|LC|FC|XC|TC|PI|LI|FI|TI|PT|LT|FT|TT|PSV|PV|LV|FV)$")
tags = sorted({t["text"].strip().upper().replace("–", "-") for t in texts})
equip = sorted(t for t in tags if TAG.match(t))
inst = sorted(t for t in tags if INST.match(t))
print(f"\n2. TAGS     {len(equip)} equipment tags matched an engineering pattern")
print("            " + ", ".join(equip[:22]))
print(f"            {len(inst)} instrument bubbles: {', '.join(inst[:14])}")

# ---- 3. line tracing (geometry only, no model) ---------------------------
t0 = time.time()
mask = R._ink(img)
h = R._runs(mask, "h", 30, 14)
v = R._runs(mask, "v", 30, 14)
print(f"\n3. LINES    {len(h)} horizontal + {len(v)} vertical pipe runs "
      f"traced in {time.time()-t0:.1f}s")

segs = h + []
for s in v:
    s.id = len(segs); segs.append(s)
for i, s in enumerate(segs):
    s.id = i
nets = R._nets(segs, 10)
print(f"            {len(set(nets))} distinct pipe networks after junction merging")

out = {"sheet": sheet, "size": [W, H], "n_text": len(texts),
       "equipment_tags": equip, "instrument_tags": inst,
       "h_runs": len(h), "v_runs": len(v), "nets": len(set(nets))}
Path("reports").mkdir(exist_ok=True)
Path("reports/real_drawing.json").write_text(json.dumps(out, indent=1))
print(f"\nreport -> reports/real_drawing.json")
