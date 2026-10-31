#!/usr/bin/env python
"""Sweep relation-extraction settings against the PIDQA spatial questions.

Relation extraction is the stage with no pretrained model and the most guesswork,
so its geometry is tuned by measurement rather than intuition. Runs are cheap
(~0.3s/sheet) and scored on the two question types that actually depend on
topology.
"""
import sys, json, itertools, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.edi.config import load
from src.edi.ingest.acquire import load_any
from src.edi.graph import relations as R
from src.edi.graph.kg import KG
from src.edi.eval.run_pidqa import evaluate_L1

CFG = load()
SHEETS = ["0", "2", "5", "6", "9"]


def score(mode, params, sheets=SHEETS):
    tmp = Path("data/graph/_sweep")
    kg = KG(tmp, fresh=True)
    total_edges = 0
    for sh in sheets:
        ents = json.load(open(CFG.get_path("paths.interim") / sh / "ocr.json"))["records"]
        img = load_any(sh)
        rels = R.extract(img, ents, mode=mode, params=params)
        total_edges += len(rels)
        kg.add_sheet(sh, ents, rels)
    rep = evaluate_L1(kg, sheets, per_type=32)
    kg.close()
    bt = rep["by_type"]
    return {"edges": total_edges,
            "spatial_counting": bt.get("spatial_counting", {}).get("accuracy", 0),
            "spatial_connection": bt.get("spatial_connection", {}).get("accuracy", 0),
            "overall": rep["overall_accuracy"]}


def main():
    grid = []
    for mode in ("mst", "walk"):
        for erase in (3, 8, 14):
            for snap in (18, 28, 40):
                grid.append((mode, {"erase_pad": erase, "snap": snap}))
    print(f"{'mode':6} {'erase':>5} {'snap':>5} {'edges':>6} "
          f"{'sp.count':>9} {'sp.conn':>8} {'overall':>8}")
    best = None
    for mode, p in grid:
        try:
            r = score(mode, p)
        except Exception as e:
            print(f"{mode:6} {p['erase_pad']:5} {p['snap']:5}  FAILED {type(e).__name__}: {str(e)[:50]}")
            continue
        key = r["spatial_counting"] + r["spatial_connection"]
        star = ""
        if best is None or key > best[0]:
            best = (key, mode, p, r); star = "  <-"
        print(f"{mode:6} {p['erase_pad']:5} {p['snap']:5} {r['edges']:6} "
              f"{r['spatial_counting']:9.3f} {r['spatial_connection']:8.3f} "
              f"{r['overall']:8.4f}{star}", flush=True)
    if best:
        print(f"\nBEST  mode={best[1]}  {best[2]}  -> {best[3]}")
        Path("reports").mkdir(exist_ok=True)
        Path("reports/relation_sweep.json").write_text(
            json.dumps({"mode": best[1], "params": best[2], "scores": best[3]}, indent=2))
    shutil.rmtree("data/graph/_sweep", ignore_errors=True)


if __name__ == "__main__":
    main()
