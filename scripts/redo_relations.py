#!/usr/bin/env python
"""Re-extract relations from existing OCR output. Seconds, not minutes -
OCR is the expensive stage and it does not need redoing to retune topology."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.edi.config import load
from src.edi.ingest.acquire import load_any
from src.edi.graph import relations as R
from src.edi.graph.semantics import enrich

CFG = load()


def _sheet_sort(name: str):
    """Dataset sheets are numeric ("0", "41"); uploaded ones are not
    ("u_pid_real_p1_52af52"). Sorting with int() crashes on the second kind."""
    return (0, int(name), "") if name.isdigit() else (1, 0, name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default="")
    ap.add_argument("--mode", default=None)
    a = ap.parse_args()
    interim = CFG.get_path("paths.interim")
    ids = ([s.strip() for s in a.sheets.split(",") if s.strip()] if a.sheets else
           sorted([d.name for d in interim.glob("*") if (d / "ocr.json").exists()],
                  key=_sheet_sort))
    tot = 0
    for i, sid in enumerate(ids, 1):
        ents = json.load(open(interim / sid / "ocr.json"))["records"]
        img = load_any(sid)
        rels = R.extract(img, ents, mode=a.mode)
        R.save(rels, sid); tot += len(rels)
        derived = enrich(img, ents, rels)
        (interim / sid / "derived.json").write_text(json.dumps(derived, indent=1))
        print(f"  [{i}/{len(ids)}] {sid}: {len(rels)} connected, "
              f"{len(derived['CONTROLS'])} controls, {len(derived['FEEDS_TO'])} feeds", flush=True)
    print(f"total {tot} edges across {len(ids)} sheets")

if __name__ == "__main__":
    main()
