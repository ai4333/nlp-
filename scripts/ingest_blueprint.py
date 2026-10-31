#!/usr/bin/env python
"""Ingest a real drawing with no trained detector (blueprint mode)."""
import sys, argparse, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.edi.ingest.acquire import acquire, load_any
from src.edi.ingest import blueprint as BP
from src.edi.graph.kg import KG
from src.edi.config import load

CFG = load()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="PDF or image; omit to reuse --sheet")
    ap.add_argument("--sheet", default="")
    ap.add_argument("--upscale", type=float, default=1.6)
    ap.add_argument("--tile", type=int, default=1400)
    ap.add_argument("--graph", action="store_true", help="load into the knowledge graph")
    ap.add_argument("--graph-only", action="store_true",
                    help="skip OCR: load the interim files already on disk into the "
                         "graph. Use after a classification change, or when the graph "
                         "was locked by a running app during the original ingest.")
    a = ap.parse_args()

    ids = []
    if a.file:
        for s in acquire(a.file):
            ids.append(s.sheet_id)
            print(f"acquired {s.sheet_id}  {s.width}x{s.height}  {s.source_name} p{s.page}")
    elif a.sheet:
        ids = [a.sheet]
    else:
        ap.error("give a file or --sheet")

    if a.graph_only:
        for sid in ids:
            base = CFG.get_path("paths.interim") / sid
            ents = json.loads((base / "ocr.json").read_text())["records"]
            rels = json.loads((base / "relations.json").read_text())["relations"]
            der = json.loads((base / "derived.json").read_text())
            kg = KG(CFG.get_path("graph.path"))
            try:
                kg.conn.execute("MATCH (s:Symbol) WHERE s.sheet_id=$s DETACH DELETE s",
                                {"s": sid})
            except Exception:
                pass
            kg.add_sheet(sid, ents, rels, der)
            print(f"{sid}: {len(ents)} nodes, {len(rels)} edges -> {kg.counts()}")
            kg.close()
        return

    for sid in ids:
        img = load_any(sid)
        t0 = time.time()
        res = BP.extract(img, sid, upscale=a.upscale, tile=a.tile,
                         stride=int(a.tile * 0.84),
                         on_progress=lambda m: print(f"   … {m}", flush=True))
        BP.save(res, sid)
        s = res["stats"]
        print(f"\n{sid}")
        print(f"  text lines   {s['text_lines']}")
        print(f"  tags found   {s['tags']}   {s['by_category']}")
        print(f"  connections  {s['relations']}  ·  controls {s['controls']}")
        print(f"  time         {s['t_total']}s  (ocr {s['t_ocr']}s)")
        if a.graph:
            kg = KG(CFG.get_path("graph.path"))
            try:
                kg.conn.execute("MATCH (s:Symbol) WHERE s.sheet_id=$s DETACH DELETE s", {"s": sid})
            except Exception:
                pass
            kg.add_sheet(sid, res["entities"], res["relations"], res["derived"])
            print(f"  graph        {kg.counts()}")
            kg.close()

if __name__ == "__main__":
    main()
