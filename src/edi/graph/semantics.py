"""
Semantic relation typing.

The project objectives call for `connected_to`, `controls`, `contains` and
`feeds_to`. Pipe tracing only ever yields the first: an undirected "these two are
plumbed together". The rest are inferred from what the symbols ARE and which way
the drawing says fluid moves.

Everything here is rule-based and auditable — each derived edge records the rule
that produced it, so a wrong one can be traced rather than argued about.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import yaml

from ..config import load

CFG = load()
_ONT = yaml.safe_load(open(Path(__file__).resolve().parents[3] / "config" / "symbol_ontology.yaml"))
CLASSES = _ONT["classes"]

ARROW_CLASSES = {c for c, v in CLASSES.items() if v.get("category") == "annotation"
                 and "arrow" in v.get("name", "").lower()}


def category(pidqa_class: int) -> str:
    return CLASSES.get(pidqa_class, {}).get("category", "fitting")


# ---------------------------------------------------------------- CONTROLS
def derive_controls(entities: list[dict], relations: list[dict]) -> list[dict]:
    """An instrument wired to a valve or actuator controls it.

    On a P&ID an instrument bubble adjacent to a valve is a control loop element,
    not merely a pipe neighbour. The direction is unambiguous: instrument -> valve.
    """
    by = {e["id"]: e for e in entities}
    out = []
    for r in relations:
        a, b = by.get(r["source"]), by.get(r["target"])
        if not a or not b:
            continue
        ca, cb = category(a["pidqa_class"]), category(b["pidqa_class"])
        if ca == "instrument" and cb in ("valve", "actuator"):
            src, tgt = a, b
        elif cb == "instrument" and ca in ("valve", "actuator"):
            src, tgt = b, a
        else:
            continue
        out.append({"source": src["id"], "target": tgt["id"], "type": "CONTROLS",
                    "evidence": "instrument_adjacent_to_valve",
                    "confidence": round(float(r.get("confidence", 0.8)) * 0.9, 3)})
    return out


# ---------------------------------------------------------------- FEEDS_TO
def arrow_direction(image: np.ndarray, bbox) -> tuple[float, float] | None:
    """Which way a solid flow arrow points.

    A filled triangle carries most of its ink at the base, so the apex lies
    opposite the ink centroid. The vector from centroid to bbox centre therefore
    points downstream. Snapped to the dominant axis, since P&ID arrows are drawn
    square.
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    crop = image[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return None
    ink = (crop < 128).astype(np.uint8)
    if ink.sum() < 12:
        return None
    ys, xs = np.nonzero(ink)
    cx, cy = xs.mean(), ys.mean()
    mx, my = (crop.shape[1] - 1) / 2, (crop.shape[0] - 1) / 2
    dx, dy = mx - cx, my - cy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return (1.0, 0.0) if abs(dx) >= abs(dy) else (0.0, 1.0), (dx, dy)


def derive_feeds_to(image: np.ndarray, entities: list[dict],
                    relations: list[dict], radius: int = 420) -> list[dict]:
    """Orient CONNECTED_TO into directed FEEDS_TO using nearby flow arrows."""
    arrows = []
    for e in entities:
        if e["pidqa_class"] not in ARROW_CLASSES:
            continue
        d = arrow_direction(image, e["bbox"])
        if not d:
            continue
        (_, _), (dx, dy) = d
        x0, y0, x1, y1 = e["bbox"]
        arrows.append({"cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
                       "dx": dx, "dy": dy})
    if not arrows:
        return []

    by = {e["id"]: e for e in entities}
    out = []
    for r in relations:
        a, b = by.get(r["source"]), by.get(r["target"])
        if not a or not b:
            continue
        ax = (a["bbox"][0] + a["bbox"][2]) / 2; ay = (a["bbox"][1] + a["bbox"][3]) / 2
        bx = (b["bbox"][0] + b["bbox"][2]) / 2; by_ = (b["bbox"][1] + b["bbox"][3]) / 2
        mx, my = (ax + bx) / 2, (ay + by_) / 2
        near = min(arrows, key=lambda w: (w["cx"] - mx) ** 2 + (w["cy"] - my) ** 2)
        if (near["cx"] - mx) ** 2 + (near["cy"] - my) ** 2 > radius ** 2:
            continue
        vx, vy = bx - ax, by_ - ay
        # does the a->b vector agree with the arrow?
        agree = vx * near["dx"] + vy * near["dy"]
        if abs(agree) < 1e-6:
            continue
        src, tgt = (a, b) if agree > 0 else (b, a)
        out.append({"source": src["id"], "target": tgt["id"], "type": "FEEDS_TO",
                    "evidence": "flow_arrow", "confidence": 0.7})
    return out


def enrich(image: np.ndarray, entities: list[dict],
           relations: list[dict]) -> dict[str, list[dict]]:
    """All derived relation types, keyed by type."""
    return {
        "CONTROLS": derive_controls(entities, relations),
        "FEEDS_TO": derive_feeds_to(image, entities, relations),
    }
