#!/usr/bin/env python
"""Benchmark serial vs pooled OCR. Must be a real file: the spawn context
re-imports __main__ in each worker, which is impossible from a stdin heredoc."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.edi.vision.detect import load_sheet
import src.edi.vision.ocr as O
from src.edi.vision import ocr_pool as P


def main():
    img, _ = load_sheet("0")
    dets = json.load(open("data/interim/0/detections.json"))["detections"][:40]
    crops = [O._prep(img, *d["bbox"], 130)[0] for d in dets]

    eng = O.OcrEngine.get()
    t0 = time.time(); base = eng._run(crops); t_ser = time.time() - t0
    print(f"serial : {len(crops)} crops {t_ser:6.2f}s", flush=True)

    key = lambda r: [[l["text"] for l in x] for x in r]
    for w in (3, 5):
        P.start(w)
        t0 = time.time(); par = P.run(crops); t_par = time.time() - t0
        print(f"pool {w}: {len(crops)} crops {t_par:6.2f}s  "
              f"speedup {t_ser/max(t_par,1e-6):.2f}x  identical={key(base)==key(par)}", flush=True)
        P.stop()


if __name__ == "__main__":
    main()
