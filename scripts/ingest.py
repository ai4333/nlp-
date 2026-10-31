#!/usr/bin/env python
"""Run the full offline ingestion pipeline over a set of sheets."""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.edi.config import load
from src.edi.vision.detect import get_detector, load_sheet, save as save_dets
from src.edi.vision import ocr as O
from src.edi.graph import relations as R
from src.edi.graph.kg import KG

CFG = load()


def ingest_sheet(sheet_id: str, detector, engine, kg: KG | None,
                 use_vlm: bool = False) -> dict:
    t0 = time.time()
    img, split = load_sheet(sheet_id)
    dets = detector.detect(img, sheet_id)
    save_dets(dets, sheet_id, img.shape)
    t1 = time.time()
    recs = O.ocr_symbols(img, dets, engine)
    n_vlm = 0
    if use_vlm:
        from src.edi.vision.vlm import repair as vlm_repair
        n_vlm = vlm_repair(img, recs, O._score_external)
    O.save(recs, sheet_id)
    t2 = time.time()
    rels = R.extract(img, recs)
    R.save(rels, sheet_id)
    t3 = time.time()
    if kg is not None:
        kg.add_sheet(sheet_id, recs, rels)
    return {"sheet": sheet_id, "split": split, "symbols": len(dets),
            "tagged": sum(1 for r in recs if r["tag"]), "edges": len(rels),
            "t_detect": round(t1 - t0, 1), "t_ocr": round(t2 - t1, 1),
            "t_rel": round(t3 - t2, 1), "vlm_fixed": n_vlm,
            "t_total": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default="", help="comma list, e.g. 0,1,2")
    ap.add_argument("--n", type=int, default=0, help="first N val sheets")
    ap.add_argument("--split", default="val")
    ap.add_argument("--detector", default="gt", choices=["gt", "yolo", "auto"])
    ap.add_argument("--graph", default="")   # empty = JSON only (recommended)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--vlm", action="store_true",
                    help="re-read unreadable tags with qwen3-vl (slow, +7GB RAM)")
    a = ap.parse_args()

    if a.sheets:
        sheets = [s.strip() for s in a.sheets.split(",") if s.strip()]
    else:
        root = CFG.get_path("paths.dataset") / "images" / a.split
        sheets = [p.stem for p in sorted(root.glob("*.jpg"), key=lambda p: int(p.stem))][:a.n or 10]

    det = get_detector(a.detector) if a.detector != "gt" else get_detector("gt")
    eng = O.OcrEngine.get()
    # Default to JSON-only. Kuzu holds an exclusive lock, so writing during a
    # long ingest blocks the app and evaluator; run scripts/build_graph.py after.
    kg = KG(a.graph, fresh=a.fresh) if a.graph else None

    print(f"ingesting {len(sheets)} sheets | detector={a.detector} | graph={a.graph}", flush=True)
    rows = []
    for i, s in enumerate(sheets, 1):
        try:
            r = ingest_sheet(s, det, eng, kg, use_vlm=a.vlm)
            rows.append(r)
            print(f"  [{i}/{len(sheets)}] sheet {s:>4}: {r['symbols']:3d} sym, "
                  f"{r['tagged']:3d} tagged, {r['edges']:3d} edges "
                  f"({r['t_total']}s: det {r['t_detect']} ocr {r['t_ocr']} rel {r['t_rel']}"
                  + (f", vlm fixed {r['vlm_fixed']}" if a.vlm else "") + ")", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(sheets)}] sheet {s}: FAILED {type(e).__name__}: {e}", flush=True)

    if kg is not None:
        print("\n", kg.counts())
    out = CFG.get_path("paths.reports"); out.mkdir(parents=True, exist_ok=True)
    (out / "ingest_summary.json").write_text(json.dumps(rows, indent=1))
    if kg is not None:
        kg.close()


if __name__ == "__main__":
    main()
