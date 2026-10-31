"""
Spec → a sheet the rest of the system already knows how to handle.

This is the join. The generated drawing is written to the same upload store a
scanned PDF lands in, and the generated entities/relations/derived files use the
exact `ocr.json` / `relations.json` / `derived.json` contracts that ingest
produces. Nothing downstream — knowledge graph, question answering, evidence
crops, manual editing, export, cross-drawing search — knows or cares that this
sheet was generated rather than scanned.

Two consequences worth stating:

* the graph is exact. Positions, tags and connections come from the spec, so a
  generated sheet has no OCR error and no detector miss. It is the one place in
  this system where ground truth is free.
* the drawing is honest. It is a real 7000px raster with real printed tags, so
  running the normal blueprint ingest over it exercises the whole pipeline —
  which is how you show a customer what their own drawings will do.
"""
from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np

from ..config import load
from .spec import Spec
from .layout import Box, place
from . import render as RD

CFG = load()

CATEGORY_CLASS = {"equipment": 101, "valve": 102, "instrument": 103,
                  "annotation": 104, "connector": 105, "line": 106, "unknown": 107}


def sheet_id_for(sp: Spec) -> str:
    base = "".join(c if c.isalnum() else "_" for c in sp.title.lower())[:32].strip("_")
    return f"u_rig_{base or 'sheet'}"


def _uploads() -> Path:
    d = CFG.get_path("paths.interim").parent / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def to_entities(sp: Spec, boxes: dict[str, Box]) -> list[dict]:
    """Graph nodes, one per drawn item, positioned exactly where they are drawn."""
    out = []
    for n in sp.all_nodes():
        b = boxes.get(n.id)
        if b is None:
            continue
        out.append({
            "id": n.id,
            "pidqa_class": CATEGORY_CLASS.get(n.category, 107),
            "category": n.category,
            "name": n.name,
            "bbox": b.tup(),
            "tag": n.tag,
            "confidence": 1.0,
            "tag_confidence": 1.0,
            "tag_location": "internal",
            "needs_review": False,
            "source": "rig_builder",
            "system": n.system,
            "zone": n.zone,
            "design_p": n.design_p,
            "design_t": n.design_t,
            "label": n.label,
        })
    # each pipe run is a node too, so "which lines are 6 inch or larger" is an
    # exact query rather than a string guess — same treatment blueprint mode
    # gives a printed line number
    for s in sp.streams:
        a, b = boxes.get(s.src), boxes.get(s.dst)
        if a is None or b is None:
            continue
        cx, cy = (a.cx + b.cx) // 2, (a.cy + b.cy) // 2
        out.append({
            "id": f"L{s.id}",
            "pidqa_class": 106, "category": "line",
            "name": RD_line_name(s),
            "bbox": [cx - 150, cy - 26, cx + 150, cy + 26],
            "tag": s.tag, "confidence": 1.0, "tag_confidence": 1.0,
            "tag_location": "internal", "needs_review": False,
            "source": "rig_builder", "size_in": s.size_in,
            "system": "piping", "zone": "",
            "line": {"size_in": s.size_in, "service": s.service,
                     "service_name": _svc(s.service), "area": s.area,
                     "number": s.number, "spec": s.spec,
                     "insulation": s.insulation, "raw": s.tag},
        })
    return out


def _svc(code: str) -> str:
    from ..ingest.lines import SERVICE
    return (SERVICE.get(code) or {}).get("name", code)


def RD_line_name(s) -> str:
    return f'{s.size_in:g}" {_svc(s.service)}'


def to_relations(sp: Spec) -> tuple[list[dict], dict]:
    """CONNECTED_TO through the line node, plus CONTROLS and FEEDS_TO."""
    rels, feeds = [], []
    for s in sp.streams:
        lid = f"L{s.id}"
        rels.append({"source": s.src, "target": lid, "type": "CONNECTED_TO",
                     "evidence": "generated", "confidence": 1.0})
        rels.append({"source": lid, "target": s.dst, "type": "CONNECTED_TO",
                     "evidence": "generated", "confidence": 1.0})
        # the direct item-to-item edge as well, so a two-hop question does not
        # have to know the pipe node is in the middle
        rels.append({"source": s.src, "target": s.dst, "type": "CONNECTED_TO",
                     "evidence": "generated", "confidence": 1.0})
        feeds.append({"source": s.src, "target": s.dst, "type": "FEEDS_TO",
                      "evidence": "flow direction", "confidence": 1.0})

    ctrl = [{"source": a, "target": b, "type": "CONTROLS",
             "evidence": "instrument loop", "confidence": 1.0}
            for a, b in sp.controls]
    # an instrument is also physically connected to what it measures
    for a, b in sp.controls:
        rels.append({"source": a, "target": b, "type": "CONNECTED_TO",
                     "evidence": "instrument connection", "confidence": 1.0})

    cont = [{"source": _first_feeder(sp, c.id), "target": c.id, "type": "CONTINUES_ON",
             "evidence": "off-page connector", "confidence": 1.0}
            for c in sp.connectors if _first_feeder(sp, c.id)]
    return rels, {"CONTROLS": ctrl, "FEEDS_TO": feeds, "CONTINUES_ON": cont}


def _first_feeder(sp: Spec, cid: str) -> str | None:
    for s in sp.streams:
        if s.dst == cid:
            return s.src
    return None


def build_sheet(sp: Spec, sheet_id: str | None = None,
                write_graph: bool = True) -> dict:
    """Render, write the interim contracts, and load the graph. Returns a summary."""
    sid = sheet_id or sheet_id_for(sp)
    boxes = place(sp)
    img, boxes = RD.draw(sp, boxes)

    png = _uploads() / f"{sid}.png"
    cv2.imwrite(str(png), img)

    ents = to_entities(sp, boxes)
    rels, derived = to_relations(sp)

    base = CFG.get_path("paths.interim") / sid
    base.mkdir(parents=True, exist_ok=True)
    (base / "ocr.json").write_text(json.dumps(
        {"sheet_id": sid, "mode": "rig_builder", "n": len(ents), "records": ents}, indent=1))
    (base / "relations.json").write_text(json.dumps(
        {"sheet_id": sid, "n": len(rels), "relations": rels}, indent=1))
    (base / "derived.json").write_text(json.dumps(derived, indent=1))
    (base / "rig_spec.json").write_text(json.dumps(sp.dict(), indent=1))

    graph_error = ""
    if write_graph:
        try:
            load_graph(sid, ents, rels, derived)
        except RuntimeError as e:
            # Kuzu is single-writer and another app instance or an ingest may
            # hold a reader. The sheet is already on disk, so this is recoverable:
            # report it and let the caller retry rather than losing the drawing.
            if "lock" not in str(e).lower():
                raise
            graph_error = ("The knowledge graph is locked by another process "
                           "(a second app window, or an ingest). The drawing and "
                           "its data are saved — load it into the graph when the "
                           "lock clears.")

    return {"sheet_id": sid, "png": str(png), "graph_error": graph_error, "width": img.shape[1], "height": img.shape[0],
            "n_entities": len(ents), "n_relations": len(rels),
            "n_controls": len(derived["CONTROLS"]),
            "n_equipment": len(sp.equipment()),
            "n_valves": len(sp.of_category("valve")),
            "n_instruments": len(sp.of_category("instrument")),
            "n_lines": len(sp.streams),
            "issues": sp.issues, "interim": str(base),
            "entities": ents, "relations": rels, "derived": derived}


def load_graph(sheet_id: str, entities: list[dict], relations: list[dict],
               derived: dict) -> int:
    """Replace this sheet in the knowledge graph. Raises if the lock is held."""
    from ..graph.kg import KG
    kg = KG(CFG.get_path("graph.path"))
    try:
        try:
            kg.conn.execute("MATCH (s:Symbol) WHERE s.sheet_id=$s DETACH DELETE s",
                            {"s": sheet_id})
        except Exception:
            pass
        kg.add_sheet(sheet_id, entities, relations, derived)
        return kg.counts()["symbols"]
    finally:
        kg.close()
