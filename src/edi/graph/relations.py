"""
Relation extraction: deciding which symbols are connected by pipe.

This is the stage with no pretrained model and the one most likely to sink the
project, so the approach is deliberately simple and inspectable.

Observed drawing structure (see scratch/region.png): symbols sit INLINE on
axis-aligned runs; solid strokes are process pipe, dashed strokes are
instrument signal; text floats alongside.

Method - netlist tracing:
  1. Binarise to an ink mask.
  2. Erase every symbol box and every OCR text box.  Ink that remains is pipe.
  3. Recover horizontal and vertical runs separately with directional
     morphological opening.  Doing the axes separately is what stops two merely
     CROSSING perpendicular pipes from being fused into one net - a plain
     connected-components pass over the raw mask would join them and silently
     wire together half the sheet.
  4. Join runs only at T-junctions (an endpoint of one lying on the span of the
     other), never at simple crossings.
  5. Each maximal set of joined runs is a `net`.
  6. Symbols touching the same net are CONNECTED_TO each other.

Step 3 is the crux: erasing the symbols first means runs terminate exactly at
symbol boundaries, so a net is precisely the pipe between symbols.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from ..config import load

CFG = load()


@dataclass
class Segment:
    id: int
    orient: str          # 'h' or 'v'
    x0: int; y0: int; x1: int; y1: int

    @property
    def span(self):      # (start, end) along the run axis
        return (self.x0, self.x1) if self.orient == "h" else (self.y0, self.y1)

    @property
    def cross(self):     # constant coordinate across the run
        return (self.y0 + self.y1) // 2 if self.orient == "h" else (self.x0 + self.x1) // 2


def _ink(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0).astype(np.uint8)


def _erase(mask: np.ndarray, boxes, pad: int = 2) -> np.ndarray:
    out = mask.copy()
    H, W = mask.shape
    for b in boxes:
        x0, y0, x1, y1 = (int(v) for v in b)
        out[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)] = 0
    return out


def _runs(mask: np.ndarray, orient: str, min_len: int, gap: int,
          line_class: str = "any") -> list[Segment]:
    """Directional opening -> closing (bridges dashes) -> connected components.

    `line_class` separates the two things a P&ID draws with lines:
      solid   process pipe      - survives a long directional opening on its own
      dashed  instrument signal - only appears once the gaps are closed
    They mean different things (fluid vs information), so treating them alike
    creates edges between an instrument and everything its signal wire passes.
    """
    k = (min_len, 1) if orient == "h" else (1, min_len)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, k))
    kc = (gap, 1) if orient == "h" else (1, gap)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, kc))

    if line_class == "solid":
        work = opened
    elif line_class == "dashed":
        # present after closing but not before: the gaps were doing the work
        short = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, kc))
        short = cv2.morphologyEx(short, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, k))
        work = cv2.subtract(short, opened)
    else:
        work = closed

    n, _, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    segs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        length = w if orient == "h" else h
        if length < min_len:
            continue
        segs.append(Segment(len(segs), orient, int(x), int(y), int(x + w), int(y + h)))
    return segs


def _touches(a: Segment, b: Segment, tol: int) -> bool:
    """True only for a genuine T/L junction, not an X crossing."""
    if a.orient == b.orient:
        # collinear and nearly abutting
        if abs(a.cross - b.cross) > tol:
            return False
        (s1, e1), (s2, e2) = a.span, b.span
        return not (e1 + tol < s2 or e2 + tol < s1)
    h, v = (a, b) if a.orient == "h" else (b, a)
    hx0, hx1, hy = h.x0, h.x1, h.cross
    vy0, vy1, vx = v.y0, v.y1, v.cross
    if not (hx0 - tol <= vx <= hx1 + tol and vy0 - tol <= hy <= vy1 + tol):
        return False
    # require an ENDPOINT of one run to sit at the meeting point
    end_of_v = min(abs(hy - vy0), abs(hy - vy1)) <= tol
    end_of_h = min(abs(vx - hx0), abs(vx - hx1)) <= tol
    return end_of_v or end_of_h


def _nets(segs: list[Segment], tol: int) -> list[int]:
    """Union-find over junction-connected runs."""
    parent = list(range(len(segs)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b: parent[a] = b

    # bucket by cross-coordinate to avoid an O(n^2) sweep over long sheets
    for i, a in enumerate(segs):
        for j in range(i + 1, len(segs)):
            b = segs[j]
            if abs(a.cross - b.cross) > 6000:
                continue
            if _touches(a, b, tol):
                union(i, j)
    return [find(i) for i in range(len(segs))]


def _attach(entities, segs, nets, snap: int) -> dict[str, set[int]]:
    """Map each symbol to the nets its bounding box touches."""
    out: dict[str, set[int]] = {}
    for e in entities:
        x0, y0, x1, y1 = e["bbox"]
        hit = set()
        for s in segs:
            if (s.x0 - snap <= x1 and s.x1 + snap >= x0 and
                    s.y0 - snap <= y1 and s.y1 + snap >= y0):
                hit.add(nets[s.id])
        if hit:
            out[e["id"]] = hit
    return out


def _walk(seg_of_symbol: dict[str, set[int]], adj: dict[int, set[int]],
          owner: dict[int, set[str]]) -> list[tuple[str, str]]:
    """Netlist walk with blocking.

    From each symbol, breadth-first through the pipe network. When a segment is
    attached to ANOTHER symbol, record the connection and stop expanding through
    it - the pipe carries on past that symbol, but anything beyond is connected
    to IT, not to us. This is the difference between true adjacency and a net
    clique, and it is what stops A-B-C on one run from also emitting A-C.
    """
    edges = set()
    for sid, segs in seg_of_symbol.items():
        seen = set(segs)
        frontier = list(segs)
        while frontier:
            nxt = []
            for sg in frontier:
                others = owner.get(sg, set()) - {sid}
                if others:
                    for o in others:
                        edges.add(tuple(sorted((sid, o))))
                    continue                      # blocked: do not expand past
                for nb in adj.get(sg, ()):
                    if nb not in seen:
                        seen.add(nb); nxt.append(nb)
            frontier = nxt
    return sorted(edges)


def extract(image: np.ndarray, entities: list[dict],
            text_boxes: list | None = None,
            mode: str | None = None, params: dict | None = None) -> list[dict]:
    P = params or {}
    mode = mode or CFG.dotted("relations.mode", "walk")
    min_len = P.get("min_len", CFG.dotted("relations.min_line_length_px", 25))
    gap = P.get("gap", CFG.dotted("relations.max_line_gap_px", 14))
    tol = P.get("tol", CFG.dotted("relations.junction_tolerance_px", 10))
    snap = P.get("snap", CFG.dotted("relations.endpoint_snap_px", 28))
    erase_pad = P.get("erase_pad", CFG.dotted("relations.symbol_erase_pad_px", 3))
    max_edge = P.get("max_edge", CFG.dotted("relations.max_edge_px", 0))  # 0 = off

    mask = _ink(image)
    mask = _erase(mask, [e["bbox"] for e in entities], pad=erase_pad)
    if text_boxes:
        mask = _erase(mask, text_boxes, pad=3)

    lc = P.get("line_class", CFG.dotted("relations.line_class", "any"))
    segs = _runs(mask, "h", min_len, gap, lc) + []
    v = _runs(mask, "v", min_len, gap, lc)
    for s in v:
        s.id = len(segs); segs.append(s)
    for i, s in enumerate(segs):
        s.id = i

    if mode == "segment":
        # Strictest reading of "directly connected": both symbols touch the SAME
        # pipe run, not merely the same net reached through junctions.
        owner: dict[int, set[str]] = {}
        for e in entities:
            x0, y0, x1, y1 = e["bbox"]
            for sg in segs:
                if (sg.x0 - snap <= x1 and sg.x1 + snap >= x0 and
                        sg.y0 - snap <= y1 and sg.y1 + snap >= y0):
                    owner.setdefault(sg.id, set()).add(e["id"])
        edges = set()
        for members in owner.values():
            m = sorted(members)
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    edges.add((m[i], m[j]))
        return [{"source": a, "target": b, "type": "CONNECTED_TO",
                 "evidence": "shared_segment", "confidence": 0.9}
                for a, b in sorted(edges)]

    if mode == "walk":
        adj: dict[int, set[int]] = {}
        for i, a in enumerate(segs):
            for j in range(i + 1, len(segs)):
                if _touches(a, segs[j], tol):
                    adj.setdefault(i, set()).add(j); adj.setdefault(j, set()).add(i)
        seg_of_symbol: dict[str, set[int]] = {}
        owner: dict[int, set[str]] = {}
        for e in entities:
            x0, y0, x1, y1 = e["bbox"]
            for sg in segs:
                if (sg.x0 - snap <= x1 and sg.x1 + snap >= x0 and
                        sg.y0 - snap <= y1 and sg.y1 + snap >= y0):
                    seg_of_symbol.setdefault(e["id"], set()).add(sg.id)
                    owner.setdefault(sg.id, set()).add(e["id"])
        return [{"source": a, "target": b, "type": "CONNECTED_TO",
                 "evidence": "netlist_walk", "confidence": 0.85}
                for a, b in _walk(seg_of_symbol, adj, owner)]

    nets = _nets(segs, tol)
    attach = _attach(entities, segs, nets, snap)

    by_net: dict[int, list[str]] = {}
    for eid, ns in attach.items():
        for n in ns:
            by_net.setdefault(n, []).append(eid)

    centre = {e["id"]: ((e["bbox"][0] + e["bbox"][2]) / 2,
                        (e["bbox"][1] + e["bbox"][3]) / 2) for e in entities}

    rels, seen = [], set()
    for net, members in by_net.items():
        if len(members) < 2:
            continue
        # A pipe run is a CHAIN, not a clique.  Emitting every pair inside a net
        # was measured to over-count on 10/10 failing spatial-count questions
        # (never under-count), so reduce each net to its minimum spanning tree:
        # n-1 edges joining nearest neighbours instead of n(n-1)/2.
        pairs = ([(members[i], members[j])
                  for i in range(len(members)) for j in range(i + 1, len(members))]
                 if mode == "clique" else _mst(members, centre))
        for a, b in pairs:
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            if max_edge:
                # Symbols that are genuinely adjacent on a run sit close together;
                # a very long MST link usually means the net merged two runs.
                ax, ay = centre[a]; bx, by = centre[b]
                if abs(ax - bx) + abs(ay - by) > max_edge:
                    continue
            seen.add(key)
            rels.append({"source": key[0], "target": key[1], "type": "CONNECTED_TO",
                         "evidence": "net_trace_mst", "net": int(net),
                         "confidence": 0.8})
    return rels


def _mst(members: list[str], centre: dict) -> list[tuple[str, str]]:
    """Prim's MST over net members, using Manhattan distance between symbol
    centres (pipes are axis-aligned, so Manhattan approximates run length)."""
    if len(members) < 2:
        return []
    inside = {members[0]}
    outside = set(members[1:])
    edges = []
    while outside:
        best = None
        for u in inside:
            ux, uy = centre[u]
            for v in outside:
                vx, vy = centre[v]
                d = abs(ux - vx) + abs(uy - vy)
                if best is None or d < best[0]:
                    best = (d, u, v)
        _, u, v = best
        edges.append((u, v))
        inside.add(v); outside.discard(v)
    return edges


def stats(segs_h: int, segs_v: int, rels: list[dict]) -> dict:
    return {"h_runs": segs_h, "v_runs": segs_v, "relations": len(rels)}


def save(rels: list[dict], sheet_id: str) -> Path:
    out = CFG.get_path("paths.interim") / sheet_id
    out.mkdir(parents=True, exist_ok=True)
    p = out / "relations.json"
    p.write_text(json.dumps({"sheet_id": sheet_id, "n": len(rels), "relations": rels}, indent=1))
    return p
