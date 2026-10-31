#!/usr/bin/env python
"""Stage 7: load interim JSON into the knowledge graph.

Kept separate from ingest.py on purpose.  Kuzu is single-writer with an
exclusive file lock, so holding the database open for the whole ingest blocks
the app and the evaluator.  Ingest writes JSON; this step opens the DB, loads,
and closes - a few seconds of lock instead of minutes.
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.edi.config import load
from src.edi.graph.kg import KG
from src.edi.edits import load_sheet_edited, load_edits, has_edits, count
from src.edi.ingest.connectors import cross_sheet_links

CFG = load()


def _sheet_sort(name: str):
    """Dataset sheets are numeric ("0", "41"); uploaded ones are not
    ("u_pid_real_p1_52af52"). Sorting with int() crashes on the second kind."""
    return (0, int(name), "") if name.isdigit() else (1, 0, name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(CFG.get_path("graph.path")))
    ap.add_argument("--fresh", action="store_true", default=True)
    ap.add_argument("--sheets", default="")
    a = ap.parse_args()

    interim = CFG.get_path("paths.interim")
    if a.sheets:
        ids = [s.strip() for s in a.sheets.split(",") if s.strip()]
    else:
        ids = sorted([d.name for d in interim.glob("*")
                      if (d / "ocr.json").exists() and (d / "relations.json").exists()],
                     key=_sheet_sort)
    kg = KG(a.graph, fresh=a.fresh)
    n_edited = 0
    all_entities = {}
    for i, sid in enumerate(ids, 1):
        # load_sheet_edited replays edits.json over the pipeline output, so
        # human corrections survive re-ingestion and land in the graph.
        ents, rels, derived = load_sheet_edited(sid)
        ed = load_edits(sid)
        mark = ""
        if has_edits(ed):
            n_edited += 1
            c = count(ed)
            mark = (f"  [edited: +{c['added']} -{c['removed']} ~{c['modified']} "
                    f"rel +{c['rel_added']} -{c['rel_removed']}]")
        all_entities[sid] = ents
        kg.add_sheet(sid, ents, rels, derived)
        print(f"  [{i}/{len(ids)}] sheet {sid}: {len(ents)} symbols, "
              f"{len(rels)} edges{mark}", flush=True)
    if n_edited:
        print(f"  ({n_edited} sheet(s) carried manual edits)")

    # Off-page connectors span sheets, so they can only be joined once every
    # sheet is loaded — two drawings that both reference "HP FLARE HEADER" are
    # plumbed together through it even though neither says so.
    links = cross_sheet_links(all_entities)
    if links:
        made = kg.add_cross_sheet(links)
        vias = sorted({l["via"] for l in links})
        print(f"  cross-sheet: {made} CONTINUES_ON via {len(vias)} shared point(s)"
              f" — {', '.join(vias[:5])}{' …' if len(vias) > 5 else ''}")
    print("graph:", kg.counts())
    kg.close()

if __name__ == "__main__":
    main()
