#!/usr/bin/env python
"""Render the OCR failures as a contact sheet so they can be inspected by eye."""
import sys, ast, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np, pandas as pd
from src.edi.config import load
from src.edi.ingest.acquire import load_any

CFG = load()

def gt_tags(sheet: str) -> dict[int, set]:
    df = pd.read_csv(CFG.get_path("paths.pidqa") / "Value/value_based.csv")
    out = defaultdict(set)
    for _, r in df[df["P&ID_number"] == int(sheet)].iterrows():
        try: v = ast.literal_eval(r["GT"])
        except Exception: continue
        out[int(r["Symbol_Class"])].update(v if isinstance(v, (list, tuple)) else [v])
    return out

def norm(s):
    import re, unicodedata
    s = unicodedata.normalize("NFKD", str(s)).replace("°", "")
    return " ".join(re.sub(r"[^A-Za-z0-9]+", " ", s).split()).upper()

def main(sheets, out_path="scratch/ocr_fail.png", per_sheet=6):
    cells = []
    for sh in sheets:
        recs = json.load(open(CFG.get_path("paths.interim") / sh / "ocr.json"))["records"]
        gt = gt_tags(sh)
        ours = defaultdict(set)
        for r in recs:
            if r.get("tag"): ours[r["pidqa_class"]].add(norm(r["tag"]))
        img = load_any(sh)
        picked = 0
        for cls, want in sorted(gt.items()):
            missing = {norm(t) for t in want} - ours[cls]
            if not missing: continue
            # symbols of this class whose tag is empty or not in GT
            bad = [r for r in recs if r["pidqa_class"] == cls
                   and (not r.get("tag") or norm(r["tag"]) not in {norm(t) for t in want})]
            for r in bad[:2]:
                if picked >= per_sheet: break
                x0,y0,x1,y1 = r["bbox"]; p = 150
                c = img[max(0,y0-p):y1+p, max(0,x0-p):x1+p]
                if c.size == 0: continue
                cells.append((f"s{sh} c{cls}", r.get("tag") or "(none)",
                              sorted(missing)[0][:18], c))
                picked += 1
            if picked >= per_sheet: break
    if not cells:
        print("no failures found"); return
    W, H, cols = 300, 240, 4
    rows = (len(cells)+cols-1)//cols
    canvas = np.full((rows*(H+46), cols*W, 3), 22, np.uint8)
    for i,(who,got,want,c) in enumerate(cells):
        r,cc = divmod(i, cols); ox,oy = cc*W, r*(H+46)
        s = min((W-10)/c.shape[1], (H-10)/c.shape[0])
        c2 = cv2.resize(c, (max(1,int(c.shape[1]*s)), max(1,int(c.shape[0]*s))))
        if c2.ndim==2: c2 = cv2.cvtColor(c2, cv2.COLOR_GRAY2BGR)
        canvas[oy+40:oy+40+c2.shape[0], ox+5:ox+5+c2.shape[1]] = c2
        cv2.putText(canvas,f"{who}  got:{got[:16]}",(ox+6,oy+16),cv2.FONT_HERSHEY_SIMPLEX,.42,(120,190,255),1)
        cv2.putText(canvas,f"want:{want}",(ox+6,oy+32),cv2.FONT_HERSHEY_SIMPLEX,.42,(120,255,160),1)
    cv2.imwrite(out_path, canvas)
    print(f"{len(cells)} failure crops -> {out_path}")

if __name__ == "__main__":
    main(sys.argv[1:] or ["0","2","5"])
