"""
Detector-free ingestion for real drawings ("blueprint mode").

The symbol detector only recognises the 32 shapes it was trained on, so a
customer's drawing is unreadable to it until a model is trained on their legend.
That is a chicken-and-egg problem for a demo: they will not hand over drawings
until they have seen it work on drawings like theirs.

This path breaks that. On a real P&ID the **tag is the identity** — an engineer
says "V-1103A", never "that bowtie shape" — and tag prefixes follow industry
convention (see `config/tag_conventions.yaml`). So:

    OCR the whole sheet  →  keep text that matches a tag pattern
                         →  read the equipment type from the prefix
                         →  trace pipe runs geometrically (no model)
                         →  connect tags that share a pipe network

No detector, no training, works on the first drawing you are handed.

It emits exactly the same `ocr.json` / `relations.json` / `derived.json`
contracts as the trained path, so the knowledge graph, question answering, app,
export and manual editing all work on the result without a single change.

**What it cannot do:** identify an untagged symbol. A valve with no tag printed
beside it is invisible here. That is what the trained detector adds.
"""
from __future__ import annotations
import json, re, time
from pathlib import Path
from typing import Callable
import cv2
import numpy as np
import yaml

from ..config import load
from .tiling import plan_tiles, cut
from ..vision import ocr as O
from ..graph import relations as R
from . import lines as LN
from . import connectors as CX

CFG = load()
_CONV = yaml.safe_load(open(Path(__file__).resolve().parents[3] / "config" / "tag_conventions.yaml"))
PATTERNS = [re.compile(p) for p in _CONV["patterns"]]
EXCLUDE = [re.compile(p, re.I) for p in _CONV.get("exclude_patterns", [])]
PREFIXES: dict = _CONV["prefixes"]

# class ids for blueprint entities are synthetic: real drawings have no 1..32
# legend, so category drives colour/behaviour instead.
CATEGORY_CLASS = {"equipment": 101, "valve": 102, "instrument": 103,
                  "annotation": 104, "connector": 105, "line": 106,
                  "unknown": 107}


# letters, then digits — with an optional dash, space and trailing suffix
_TAG_SHAPE = re.compile(r"^[A-Z]{1,5}[-\s]?\d{1,6}[A-Z]?(?:/[A-Z])*$")


def normalise(t: str) -> str:
    return (t or "").strip().upper().replace("–", "-").replace("—", "-").replace(" -", "-")


def classify(tag: str) -> tuple[str, str]:
    """(equipment name, category) from the tag prefix. Longest prefix wins.

    A clipped pipe-line number is checked FIRST. Without that guard the prefix
    walk resolves its leading letter and `DC266` (closed drain line 266) comes
    back as a drum — see `ingest/lines.parse_fragment`.
    """
    t = normalise(tag)
    frag = LN.parse_fragment(t)
    if frag is not None:
        return f"{frag.service_name} (partial tag)", "line"
    # The prefix walk resolves on the FIRST letter, so any word at all would
    # classify: "FROBNICATOR" came back as a filter. A tag is letters followed by
    # a number — anything else is prose, and prose is not equipment.
    if not _TAG_SHAPE.match(t):
        return "unidentified item", "unknown"
    head = re.split(r"[-\s0-9]", t, maxsplit=1)[0]

    # The customer's own convention first, longest prefix wins — but only for
    # MULTI-letter prefixes. A single letter is the last resort, because it is
    # what turned `TA-11003` (temperature alarm) into a tank: T is "tank" in the
    # table, and the walk never got as far as asking what TA means.
    for n in range(len(head), 1, -1):
        hit = PREFIXES.get(head[:n])
        if hit:
            return hit["name"], hit["category"]

    # ISA-5.1 decodes anything the table has not been told about, which on a
    # customer's first drawing is most of the instrument tags.
    from ..knowledge import isa as ISA
    d = ISA.decode(t)
    if d is not None:
        return d.name, d.category

    hit = PREFIXES.get(head[:1])
    if hit:
        return hit["name"], hit["category"]
    return "unidentified item", "unknown"


def is_tag(text: str) -> bool:
    t = normalise(text)
    if len(t) < 3 or len(t) > 22:
        return False
    if any(p.match(t) for p in EXCLUDE):
        return False
    return any(p.match(t) for p in PATTERNS)


def _read_sheet(img: np.ndarray, upscale: float, tile: int, stride: int,
                on_progress=None) -> list[dict]:
    H, W = img.shape[:2]
    tiles = plan_tiles(W, H, tile, stride)
    crops = [(cv2.cvtColor(cut(img, t, tile), cv2.COLOR_GRAY2BGR), t.x0, t.y0)
             for t in tiles]
    if on_progress:
        on_progress(f"{len(tiles)} tiles")
    eng = O.OcrEngine.get()
    O.set_workers(CFG.dotted("ocr.workers", 0))
    return O._collect_rotated(eng, crops, upscale)


# ISA instrument bubbles print the function code above the loop number, so OCR
# returns them as two separate lines ("PC" then "1103"). Neither is a tag on its
# own, which is why a first pass over a real drawing found 32 items of equipment
# and not one instrument.
_ISA_CODE = re.compile(r"^[A-Z]{2,4}$")
_ISA_NUM = re.compile(r"^[0-9]{3,5}[A-Z]?$")


def _pair_bubbles(lines: list[dict], max_dy: float = 78, max_dx: float = 46) -> list[dict]:
    """Join a stacked instrument code and loop number into one tag."""
    codes = [l for l in lines if _ISA_CODE.match(normalise(l["text"]))]
    nums = [l for l in lines if _ISA_NUM.match(normalise(l["text"]))]
    used, out = set(), []
    for c in codes:
        best, bd = None, 1e9
        for i, n in enumerate(nums):
            if i in used:
                continue
            dx, dy = abs(n["sx"] - c["sx"]), n["sy"] - c["sy"]
            if dx <= max_dx and 0 < dy <= max_dy and dy < bd:
                best, bd = i, dy
        if best is not None:
            used.add(best)
            n = nums[best]
            out.append({"text": f'{normalise(c["text"])}-{normalise(n["text"])}',
                        "score": min(c["score"], n["score"]),
                        "sx": (c["sx"] + n["sx"]) / 2, "sy": (c["sy"] + n["sy"]) / 2,
                        "deg": c.get("deg", 0), "bubble": True})
    return out


def _drop_truncations(tags: list[dict]) -> list[dict]:
    """OCR sometimes clips a tag ("V-1201" read as "V-120"). If a shorter tag is
    a strict prefix of a longer one on the same sheet, it is almost certainly the
    same item read badly, not a different item."""
    full = {normalise(t["text"]) for t in tags}
    keep = []
    for t in tags:
        n = normalise(t["text"])
        if any(o != n and o.startswith(n) and len(o) > len(n) for o in full):
            continue
        keep.append(t)
    return keep


def _merge_same_tag(entities: list[dict]) -> list[dict]:
    """One tag is one physical item.

    A P&ID prints the same tag in several places - beside the symbol, in a note,
    in the stream table. Left as separate nodes they would inflate every count
    ("how many vessels?") and split the item's connections across duplicates.
    The most confident instance keeps its position; the rest become extra
    evidence locations on the same node.
    """
    by: dict[str, list[dict]] = {}
    for e in entities:
        by.setdefault(e["tag"], []).append(e)
    out = []
    for tag, group in by.items():
        group.sort(key=lambda e: -e["tag_confidence"])
        head = dict(group[0])
        if len(group) > 1:
            head["also_at"] = [g["bbox"] for g in group[1:]]
            head["n_occurrences"] = len(group)
        out.append(head)
    for i, e in enumerate(sorted(out, key=lambda e: (e["bbox"][1], e["bbox"][0]))):
        e["id"] = f"b{i:04d}"
    return out


def _dedupe_tags(cands: list[dict], tol: float = 34.0) -> list[dict]:
    """Overlapping tiles read the same text several times; keep one per place."""
    kept: list[dict] = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if any(abs(c["sx"] - k["sx"]) < tol and abs(c["sy"] - k["sy"]) < tol
               and normalise(c["text"]) == normalise(k["text"]) for k in kept):
            continue
        kept.append(c)
    return kept


def extract(img: np.ndarray, sheet_id: str, upscale: float = 1.6,
            tile: int = 1400, stride: int = 1180,
            box_px: int = 46, on_progress: Callable[[str], None] | None = None) -> dict:
    """Full detector-free extraction. Returns entities, relations, derived, stats."""
    t0 = time.time()
    H, W = img.shape[:2]

    lines = _read_sheet(img, upscale, tile, stride, on_progress)
    bubbles = _pair_bubbles(lines)
    tags = _dedupe_tags([l for l in lines if is_tag(l["text"])] +
                        [b for b in bubbles if is_tag(b["text"])])
    tags = [t for t in _drop_truncations(tags)
            if not LN.is_line_number(t["text"])
            and not LN.is_line_fragment(t["text"])
            and not CX.is_connector(t["text"])]
    if on_progress:
        on_progress(f"{len(lines)} text lines, {len(bubbles)} bubbles, {len(tags)} tags")

    # Pipe line numbers are separate entities: "what size is that line?" is the
    # most common question asked of a P&ID, and the answer is printed on it.
    line_ents = []
    used_line = set()
    for l in lines:
        ln = LN.parse(l["text"])
        if ln is None:
            continue
        key = (ln.raw, round(l["sx"] / 40), round(l["sy"] / 40))
        if key in used_line:
            continue
        used_line.add(key)
        line_ents.append(LN.to_entity(ln, len(line_ents), l["sx"], l["sy"], l["score"]))

    # Clipped line numbers ("DC266"). They are lines, not equipment, and they
    # are flagged for review because the size and spec were never read.
    frag_ents = []
    complete = {(e["line"]["service"], e["line"]["number"]) for e in line_ents}
    seen_frag = set()
    for l in lines:
        fr = LN.parse_fragment(l["text"])
        if fr is None:
            continue
        k = (fr.service, fr.number)
        if k in complete or k in seen_frag:
            continue                    # the same pipe, already read in full
        seen_frag.add(k)
        frag_ents.append(LN.fragment_entity(fr, len(line_ents) + len(frag_ents),
                                            l["sx"], l["sy"], l["score"]))
    # one node per distinct line number, however many times it is printed
    # De-duplicate on the pipe's IDENTITY, not its printed string: the same
    # line read twice with different OCR noise in the spec code is one pipe.
    seen_ln, dedup = set(), []
    for e in sorted(line_ents, key=lambda e: -e["tag_confidence"]):
        L = e["line"]
        k = f'{L["size_in"]:g}|{L["service"]}|{L["area"]}|{L["number"]}'
        if k in seen_ln:
            continue
        seen_ln.add(k); dedup.append(e)
    line_ents = dedup + frag_ents

    # Off-page connectors: where this sheet hands off to another drawing.
    conn_ents, seen_cx = [], set()
    for l in lines:
        c = CX.parse(l["text"])
        if c is None:
            continue
        if c.target in seen_cx:
            continue
        seen_cx.add(c.target)
        conn_ents.append(CX.to_entity(c, len(conn_ents), l["sx"], l["sy"], l["score"]))

    entities = []
    for i, t in enumerate(sorted(tags, key=lambda t: (t["sy"], t["sx"]))):
        name, cat = classify(t["text"])
        cx, cy = t["sx"], t["sy"]
        entities.append({
            "id": f"b{i:04d}",
            "pidqa_class": CATEGORY_CLASS.get(cat, 107),
            "category": cat, "name": name,
            "bbox": [max(0, int(cx - box_px)), max(0, int(cy - box_px // 2)),
                     min(W, int(cx + box_px)), min(H, int(cy + box_px // 2))],
            "tag": normalise(t["text"]),
            "confidence": 1.0,
            "tag_confidence": round(float(t["score"]), 3),
            "tag_location": "internal",
            "source": "blueprint",
        })

    entities = _merge_same_tag(entities)
    for j, e in enumerate(line_ents):
        e["id"] = f"l{j:04d}"
    for j, e in enumerate(conn_ents):
        e["id"] = f"x{j:04d}"
    entities += line_ents + conn_ents
    t_ocr = time.time() - t0
    rels = _connect(img, entities)
    derived = _derive(entities, rels)

    return {
        "entities": entities, "relations": rels, "derived": derived,
        "stats": {"text_lines": len(lines), "tags": len(entities),
                  "line_numbers": len(line_ents),
                  "partial_line_tags": len(frag_ents),
                  "connectors": len(conn_ents),
                  "relations": len(rels),
                  "controls": len(derived["CONTROLS"]),
                  "feeds_to": len(derived["FEEDS_TO"]),
                  "by_category": _counts(entities),
                  "t_ocr": round(t_ocr, 1),
                  "t_total": round(time.time() - t0, 1)},
    }


def _counts(entities) -> dict:
    out: dict[str, int] = {}
    for e in entities:
        out[e["category"]] = out.get(e["category"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _connect(img: np.ndarray, entities: list[dict]) -> list[dict]:
    """Connect tags that sit on the same traced pipe network."""
    if len(entities) < 2:
        return []
    mask = R._ink(img)
    mask = R._erase(mask, [e["bbox"] for e in entities], pad=6)

    min_len = CFG.dotted("relations.min_line_length_px", 25)
    gap = CFG.dotted("relations.max_line_gap_px", 14)
    tol = CFG.dotted("relations.junction_tolerance_px", 10)

    segs = R._runs(mask, "h", min_len, gap)
    v = R._runs(mask, "v", min_len, gap)
    for s in v:
        s.id = len(segs); segs.append(s)
    for i, s in enumerate(segs):
        s.id = i
    if not segs:
        return []

    nets = R._nets(segs, tol)
    # a tag sits further from its pipe than a symbol does, so reach further
    attach = R._attach(entities, segs, nets, snap=120)

    by_net: dict[int, list[str]] = {}
    for eid, ns in attach.items():
        for n in ns:
            by_net.setdefault(n, []).append(eid)

    centre = {e["id"]: ((e["bbox"][0] + e["bbox"][2]) / 2,
                        (e["bbox"][1] + e["bbox"][3]) / 2) for e in entities}
    max_edge = CFG.dotted("relations.max_edge_px", 1600)
    out, seen = [], set()
    for net, members in by_net.items():
        if len(members) < 2 or len(members) > 40:      # a huge net is the border
            continue
        for a, b in R._mst(members, centre):
            k = tuple(sorted((a, b)))
            if k in seen:
                continue
            ax, ay = centre[k[0]]; bx, by = centre[k[1]]
            if max_edge and abs(ax - bx) + abs(ay - by) > max_edge:
                continue
            seen.add(k)
            out.append({"source": k[0], "target": k[1], "type": "CONNECTED_TO",
                        "evidence": "pipe_net", "confidence": 0.7})
    return out


def _derive(entities: list[dict], relations: list[dict]) -> dict:
    """Instrument adjacent to a valve is a control loop — same rule as the
    trained path, but driven by tag category rather than symbol class.

    Flow direction comes from `graph.flow`, which reads it out of the drawing's
    own labelling (off-page connectors, relief paths, service codes) rather than
    from detected arrow symbols the way the trained path does. Before this,
    `FEEDS_TO` was empty on every real drawing and no question about upstream,
    downstream or "where does this go" could be answered at all.
    """
    by = {e["id"]: e for e in entities}
    ctl = []
    for r in relations:
        a, b = by.get(r["source"]), by.get(r["target"])
        if not a or not b:
            continue
        if a["category"] == "instrument" and b["category"] in ("valve", "equipment"):
            s, t = a, b
        elif b["category"] == "instrument" and a["category"] in ("valve", "equipment"):
            s, t = b, a
        else:
            continue
        ctl.append({"source": s["id"], "target": t["id"], "type": "CONTROLS",
                    "evidence": "instrument_adjacent_to_item", "confidence": 0.65})
    from ..graph import flow as FLOW
    feeds = FLOW.infer(entities, relations)
    return {"CONTROLS": ctl, "FEEDS_TO": feeds}


def save(result: dict, sheet_id: str) -> Path:
    base = CFG.get_path("paths.interim") / sheet_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "ocr.json").write_text(json.dumps(
        {"sheet_id": sheet_id, "mode": "blueprint",
         "n": len(result["entities"]), "records": result["entities"]}, indent=1))
    (base / "relations.json").write_text(json.dumps(
        {"sheet_id": sheet_id, "n": len(result["relations"]),
         "relations": result["relations"]}, indent=1))
    (base / "derived.json").write_text(json.dumps(result["derived"], indent=1))
    (base / "blueprint_stats.json").write_text(json.dumps(result["stats"], indent=1))
    return base
