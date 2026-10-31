#!/usr/bin/env python
"""
Build an oil-rig P&ID from a sentence, without the app.

    python scripts/rig_build.py "three stage separation with gas compression and flare"
    python scripts/rig_build.py --template drilling
    python scripts/rig_build.py --list
    python scripts/rig_build.py "gas compression" --model deepseek-coder-v2:16b

Writes the drawing to data/uploads/<sheet_id>.png, the entity/relation contracts
to data/interim/<sheet_id>/, and loads the knowledge graph, so the sheet is
queryable in the app immediately (`--no-graph` skips the graph write, which is
what you want while an ingest holds the Kuzu lock).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.edi.rig import catalog as CAT
from src.edi.rig import templates as TPL
from src.edi.rig import spec as SP
from src.edi.rig import build as BLD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="?", help="what to build, in plain English")
    ap.add_argument("--template", help="build a standard package by name")
    ap.add_argument("--list", action="store_true", help="list packages and item types")
    ap.add_argument("--model", help="Ollama model (default: config llm.rig_model)")
    ap.add_argument("--no-llm", action="store_true", help="templates only, no model")
    ap.add_argument("--no-graph", action="store_true", help="skip the graph write")
    ap.add_argument("--no-auto", action="store_true",
                    help="do not add relief, ESD, check valves or instrument loops")
    ap.add_argument("--sheet-id")
    a = ap.parse_args()

    if a.list:
        print("PACKAGES")
        for k, v in TPL.TEMPLATES.items():
            print(f"  {k:<18} {v['title']}")
        print(f"\nCATALOGUE — {len(CAT.CATALOG)} item types")
        for sysname, items in sorted(CAT.systems().items()):
            print(f"  {sysname:<12} " + ", ".join(sorted(i.key for i in items)))
        return

    if a.template:
        if a.template not in TPL.TEMPLATES:
            sys.exit(f"unknown package: {a.template} (try --list)")
        raw, src = TPL.expand(a.template), "package"
    elif a.request and not a.no_llm:
        from src.edi.rig.chat import RigChat
        t = time.time()
        out = RigChat(model=a.model).propose(a.request)
        raw, src = out["spec"], out["source"]
        print(f"· spec from {src} in {time.time()-t:.1f}s"
              + (f" — {out['note']}" if out.get("note") else ""))
    elif a.request:
        names = TPL.match(a.request) or ["production_train"]
        raw = TPL.merged(names[:2]) if len(names) > 1 else TPL.expand(names[0])
        src = "package"
        print(f"· matched package(s): {', '.join(names[:2])}")
    else:
        ap.error("give a request, --template NAME, or --list")

    sp = SP.build(raw, auto=not a.no_auto)
    res = BLD.build_sheet(sp, sheet_id=a.sheet_id, write_graph=not a.no_graph)

    print(f"\n{sp.title}")
    print(f"  sheet      {res['sheet_id']}  ({res['width']}x{res['height']})")
    print(f"  drawing    {res['png']}")
    print(f"  content    {res['n_equipment']} equipment · {res['n_valves']} valves · "
          f"{res['n_instruments']} instruments · {res['n_lines']} lines")
    print(f"  graph      {res['n_entities']} nodes · {res['n_relations']} edges · "
          f"{res['n_controls']} control loops"
          + ("  (not written)" if a.no_graph else ""))
    if res["issues"]:
        print(f"\n  design findings ({len(res['issues'])}):")
        for i in res["issues"]:
            print(f"    [{i['severity']:<6}] {i['code']:<15} {i['finding']}")
    else:
        print("  review     no findings")

    print(f"\n  equipment list ({len(SP.equipment_list(sp))} rows), "
          f"line list ({len(SP.line_list(sp))}), valve list ({len(SP.valve_list(sp))}), "
          f"instrument index ({len(SP.instrument_index(sp))}) — see the app to export")


if __name__ == "__main__":
    main()
