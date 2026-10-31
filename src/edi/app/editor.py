"""
Interactive canvas editor — a bidirectional Streamlit component.

The first editing UI asked the user to type pixel coordinates for a bounding
box. That is not something an engineer should ever have to do: they are looking
at the symbol, so they should be able to draw around it.

Streamlit's `components.html` is one-way, so this is a declared custom component
instead. It speaks the Streamlit component postMessage protocol in plain
JavaScript — no npm, no build step — and returns the user's action to Python:

    {"action": "select",  "id": "d0007"}
    {"action": "draw",    "bbox": [x0, y0, x1, y1]}    # in SHEET pixels
    {"action": "connect", "a": "d0001", "b": "d0009"}
    {"action": "delete",  "id": "d0003"}

Coordinates come back in original sheet pixels, not display pixels: the
component divides out the raster downscale before returning, so an edit made at
any zoom level lands on the right part of the drawing.
"""
from __future__ import annotations
import base64
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_ONT = yaml.safe_load(open(_ROOT / "config" / "symbol_ontology.yaml"))
CLASSES = _ONT["classes"]
CAT_COLOR = {
    "valve": "#5aa9ff", "instrument": "#ff9d5c", "fitting": "#7f8794",
    "actuator": "#c78bff", "equipment": "#4fd08a", "annotation": "#e3c05a",
}
BASE_WIDTH = 2200

_component = components.declare_component(
    "edi_editor", path=str(Path(__file__).parent / "components" / "editor"))


def _encode(img: np.ndarray, width: int = BASE_WIDTH) -> tuple[str, float]:
    h, w = img.shape[:2]
    scale = min(1.0, width / w)
    small = (cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
             if scale < 1 else img)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(), scale


@st.cache_data(show_spinner=False)
def build_spec(sheet_id: str, entities: list, relations: list, derived: dict,
               _img: np.ndarray, _rev: int = 0) -> dict:
    uri, scale = _encode(_img)
    H, W = _img.shape[:2]
    nodes = []
    for e in entities:
        x0, y0, x1, y1 = [v * scale for v in e["bbox"]]
        cat = CLASSES.get(e["pidqa_class"], {}).get("category", "fitting")
        nodes.append({
            "id": e["id"], "x": round(x0, 1), "y": round(y0, 1),
            "w": round(x1 - x0, 1), "h": round(y1 - y0, 1),
            "cx": round((x0 + x1) / 2, 1), "cy": round((y0 + y1) / 2, 1),
            "c": e["pidqa_class"],
            "n": CLASSES.get(e["pidqa_class"], {}).get("name", f"class {e['pidqa_class']}"),
            "t": e.get("tag") or "", "tc": round(float(e.get("tag_confidence", 0)), 2),
            "dc": round(float(e.get("confidence", 1.0)), 2),
            "col": CAT_COLOR.get(cat, "#7f8794"),
            "ed": bool(e.get("edited")),
        })
    ids = {n["id"] for n in nodes}
    edges = [{"a": r["source"], "b": r["target"], "t": "c"}
             for r in relations if r["source"] in ids and r["target"] in ids]
    edges += [{"a": r["source"], "b": r["target"], "t": "k"}
              for r in (derived or {}).get("CONTROLS", [])
              if r["source"] in ids and r["target"] in ids]
    return {"uri": uri, "scale": scale, "vw": int(W * scale), "vh": int(H * scale),
            "nodes": nodes, "edges": edges}


def canvas(spec: dict, mode: str = "inspect", height: int = 620,
           selected: str | None = None, key: str = "editor"):
    """Render the canvas. Returns the user's last action, or None."""
    return _component(spec=spec, mode=mode, height=height,
                      selected=selected, key=key, default=None)
