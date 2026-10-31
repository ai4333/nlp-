"""
Ingestion orchestrator with progress reporting.

Wraps the six offline stages behind one call so the UI can show a live stage
tracker (wireframe 1e) instead of a blank spinner, and so the CLI and the app
run exactly the same code path.
"""
from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np

from .config import load
from .ingest.acquire import load_any
from .vision.detect import get_detector, save as save_dets
from .vision import ocr as O
from .graph import relations as R

CFG = load()

STAGES = [
    ("preprocess", "preprocessing · load & normalise"),
    ("detect",     "component detection"),
    ("ocr",        "OCR · tags & dimensions"),
    ("resolve",    "entity resolution"),
    ("relations",  "relation extraction"),
    ("graph",      "graph build"),
]


@dataclass
class Progress:
    stage: str
    label: str
    index: int
    total: int
    status: str                    # pending | running | done
    detail: str = ""
    elapsed: float = 0.0


@dataclass
class SheetResult:
    sheet_id: str
    width: int
    height: int
    n_symbols: int = 0
    n_tagged: int = 0
    n_relations: int = 0
    n_review: int = 0
    timings: dict = field(default_factory=dict)
    entities: list = field(default_factory=list)
    relations: list = field(default_factory=list)


def _review_count(recs: list[dict]) -> int:
    thr = CFG.dotted("entity_resolution.review_below", 0.70)
    return sum(1 for r in recs
               if r.get("tag_location") != "none"
               and (not r.get("tag") or r.get("tag_confidence", 0) < thr))


def run_sheet(sheet_id: str, detector=None, engine=None,
              on_progress: Callable[[Progress], None] | None = None,
              use_vlm: bool = False, fast: bool = False) -> SheetResult:
    def emit(i, status, detail="", t0=None):
        if on_progress:
            key, label = STAGES[i]
            on_progress(Progress(key, label, i, len(STAGES), status, detail,
                                 round(time.time() - t0, 1) if t0 else 0.0))

    t_all = time.time()
    timings = {}
    O.set_fast_mode(fast)
    O.set_workers(CFG.dotted('ocr.workers', 0))

    # 1 preprocess ---------------------------------------------------------
    t = time.time(); emit(0, "running")
    img = load_any(sheet_id)
    if img is None:
        raise FileNotFoundError(f"sheet {sheet_id} not found")
    H, W = img.shape[:2]
    timings["preprocess"] = round(time.time() - t, 2)
    emit(0, "done", f"{W}x{H} px", t)

    # 2 detect -------------------------------------------------------------
    t = time.time(); emit(1, "running")
    det = detector or get_detector("auto")
    dets = det.detect(img, sheet_id)
    # An uploaded sheet has no ground-truth labels. If the GT backend is active
    # because no model is trained yet, say so instead of silently building an
    # empty graph that looks like a detection failure.
    if not dets and det.__class__.__name__ == "GroundTruthDetector":
        raise RuntimeError(
            "No trained detector found and this sheet has no ground-truth labels. "
            "Train one first:  python scripts/train_yolo.py   then copy "
            "runs/pid_v2/weights/best.pt to models/yolo_pid.pt")
    save_dets(dets, sheet_id, img.shape)
    timings["detect"] = round(time.time() - t, 2)
    emit(1, "done", f"{len(dets)} found", t)

    # 3 ocr ----------------------------------------------------------------
    t = time.time(); emit(2, "running", f"{len(dets)} regions")
    eng = engine or O.OcrEngine.get()
    recs = O.ocr_symbols(img, dets, eng)
    timings["ocr"] = round(time.time() - t, 2)
    emit(2, "done", f"{sum(1 for r in recs if r['tag'])} tags read", t)

    # 4 resolve ------------------------------------------------------------
    t = time.time(); emit(3, "running")
    if use_vlm:
        from .vision.vlm import repair as vlm_repair
        n = vlm_repair(img, recs, O._score_external)
        emit(3, "running", f"VLM re-read {n}")
    O.save(recs, sheet_id)
    n_review = _review_count(recs)
    timings["resolve"] = round(time.time() - t, 2)
    emit(3, "done", f"{n_review} need review", t)

    # 5 relations ----------------------------------------------------------
    t = time.time(); emit(4, "running")
    rels = R.extract(img, recs)
    R.save(rels, sheet_id)
    from .graph.semantics import enrich
    derived = enrich(img, recs, rels)
    (CFG.get_path("paths.interim") / sheet_id / "derived.json").write_text(
        json.dumps({k: v for k, v in derived.items()}, indent=1))
    timings["relations"] = round(time.time() - t, 2)
    emit(4, "done", f"{len(rels)} connected · {len(derived['CONTROLS'])} controls · "
                    f"{len(derived['FEEDS_TO'])} feeds", t)

    # 6 graph --------------------------------------------------------------
    t = time.time(); emit(5, "running")
    from .graph.kg import KG
    kg = KG(CFG.get_path("graph.path"))
    try:
        kg.conn.execute("MATCH (s:Symbol) WHERE s.sheet_id=$s DETACH DELETE s",
                        {"s": str(sheet_id)})
    except Exception:
        pass
    kg.add_sheet(sheet_id, recs, rels, derived)
    counts = kg.counts()
    kg.close()
    timings["graph"] = round(time.time() - t, 2)
    emit(5, "done", f"{counts['symbols']} nodes total", t)

    timings["total"] = round(time.time() - t_all, 2)
    return SheetResult(sheet_id, W, H, len(dets),
                       sum(1 for r in recs if r["tag"]), len(rels), n_review,
                       timings, recs, rels)


def sheet_manifest() -> list[dict]:
    """Every sheet that has been ingested, newest first."""
    interim = CFG.get_path("paths.interim")
    out = []
    for d in interim.glob("*"):
        ocr_p, rel_p = d / "ocr.json", d / "relations.json"
        if not (ocr_p.exists() and rel_p.exists()):
            continue
        try:
            recs = json.loads(ocr_p.read_text())["records"]
            rels = json.loads(rel_p.read_text())["relations"]
        except Exception:
            continue
        out.append({
            "sheet_id": d.name,
            "n_symbols": len(recs),
            "n_tagged": sum(1 for r in recs if r.get("tag")),
            "n_relations": len(rels),
            "n_review": _review_count(recs),
            "uploaded": d.name.startswith("u_"),
            "mtime": ocr_p.stat().st_mtime,
        })
    return sorted(out, key=lambda r: (-r["mtime"],))
