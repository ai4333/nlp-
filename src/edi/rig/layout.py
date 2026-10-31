"""
Sheet layout — where every item sits on the drawing.

A P&ID reads left to right: feed enters on the left, product leaves on the
right, relief goes up. So layout is a layered graph drawing over the *process*
nodes only:

  * column  = how far downstream the item is (longest path from a feed point)
  * row     = a lane, chosen to sit near whatever feeds it
  * cell    = each process item owns a column x row cell, and everything that
              belongs to it — its instrument bubbles, its relief valve — is
              stacked inside that cell above the symbol

That last rule is what keeps a generated sheet readable. Relief devices and
bubbles laid out as ordinary flow nodes produce a plate of spaghetti; owned by
their parent's cell they produce the vertical "instrument stack over equipment"
that a real drawing has.

Recycle streams (anti-surge, glycol circulation) make the flow graph cyclic, so
back edges are removed before layering — they are still drawn.

Coordinates come out in real sheet pixels, which is what makes the drawing and
the knowledge graph agree: the same box is the symbol's footprint in the PNG and
its `bbox` in the graph.
"""
from __future__ import annotations
from dataclasses import dataclass

from .spec import Spec, Node

SHEET_W, SHEET_H = 7000, 4600
MARGIN = 70
FRAME = 34                     # gap between outer border and inner frame
TITLE_W, TITLE_H = 2000, 620   # title block, bottom right
NOTES_W = 1220                 # notes column, right edge

# devices that hang off a parent instead of standing in the flow
SATELLITE = {"psv", "bdv", "rupture_disc", "antisurge_valve"}


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cx(self) -> int: return (self.x0 + self.x1) // 2

    @property
    def cy(self) -> int: return (self.y0 + self.y1) // 2

    def tup(self) -> list[int]: return [self.x0, self.y0, self.x1, self.y1]


def is_satellite(n: Node) -> bool:
    return n.item in SATELLITE


def _break_cycles(nodes: list[str], edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop back edges found by DFS, so layering sees a DAG."""
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].append(b)
    colour = {n: 0 for n in nodes}       # 0 white · 1 grey · 2 black
    back: set[tuple[str, str]] = set()

    def visit(u):
        colour[u] = 1
        for v in adj[u]:
            if colour[v] == 1:
                back.add((u, v))
            elif colour[v] == 0:
                visit(v)
        colour[u] = 2

    for n in nodes:
        if colour[n] == 0:
            visit(n)
    return [(a, b) for a, b in edges if (a, b) not in back]


def _layers(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """Longest-path layering."""
    edges = _break_cycles(nodes, edges)
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    indeg = {n: 0 for n in nodes}
    seen = set()
    for a, b in edges:
        if a not in adj or b not in adj or (a, b) in seen:
            continue
        seen.add((a, b))
        adj[a].append(b)
        indeg[b] += 1

    layer = {n: 0 for n in nodes}
    q = [n for n in nodes if indeg[n] == 0] or [nodes[0]]
    ind = dict(indeg)
    while q:
        n = q.pop(0)
        for m in adj[n]:
            layer[m] = max(layer[m], layer[n] + 1)
            ind[m] -= 1
            if ind[m] == 0:
                q.append(m)
    return layer


def _flow_edges(sp: Spec, keep: set[str]) -> list[tuple[str, str]]:
    return [(s.src, s.dst) for s in sp.streams if s.src in keep and s.dst in keep]


def place(sp: Spec) -> dict[str, Box]:
    """Assign every node a box in sheet pixels."""
    main = [n for n in sp.all_nodes()
            if n.category in ("equipment", "valve", "connector") and not is_satellite(n)]
    if not main:
        return {}
    ids = [n.id for n in main]
    keep = set(ids)
    layer = _layers(ids, _flow_edges(sp, keep))

    for n in main:                       # off-page connectors belong at the edge
        if n.category == "connector":
            layer[n.id] = max(layer.values())

    cols: dict[int, list[str]] = {}
    for i in ids:
        cols.setdefault(layer[i], []).append(i)
    order = sorted(cols)
    remap = {c: k for k, c in enumerate(order)}
    layer = {i: remap[c] for i, c in layer.items()}
    cols = {remap[c]: v for c, v in cols.items()}

    # rows: follow the feed so pipes stay short
    src_of: dict[str, str] = {}
    for s in sp.streams:
        src_of.setdefault(s.dst, s.src)
    row_of: dict[str, int] = {}
    for c in sorted(cols):
        members = cols[c]
        members.sort(key=lambda i: row_of.get(src_of.get(i, ""), 99))
        for r, i in enumerate(members):
            row_of[i] = r

    n_cols = len(cols)
    n_rows = max((len(v) for v in cols.values()), default=1)

    ax0 = MARGIN + FRAME + 110
    ax1 = SHEET_W - MARGIN - FRAME - NOTES_W - 70
    ay0 = MARGIN + FRAME + 250          # the flare header runs above this
    ay1 = SHEET_H - MARGIN - FRAME - 200

    by_id = {n.id: n for n in sp.all_nodes()}

    # a column of nothing but valves needs far less width than a column of vessels
    weight = {c: (0.5 if all(by_id[i].category == "valve" for i in v) else 1.0)
              for c, v in cols.items()}
    total = sum(weight.values()) or 1
    xs, acc = {}, 0.0
    span = ax1 - ax0
    for c in sorted(cols):
        w = span * weight[c] / total
        xs[c] = (ax0 + acc, w)
        acc += w

    # A cell taller than this wastes the sheet: one linear train would put the
    # symbols on the floor and their relief valves near the border, joined by
    # 2000px of empty riser. Cap it and centre the block instead.
    avail = ay1 - ay0
    row_h = min(avail / max(1, n_rows), 1450)
    y_top = ay0 + max(0.0, (avail - row_h * n_rows) / 2)
    out: dict[str, Box] = {}

    for n in main:
        c, r = layer[n.id], row_of[n.id]
        x0c, col_w = xs[c]
        cx = int(x0c + col_w / 2)
        # the symbol sits in the LOWER part of its cell; bubbles and relief
        # devices stack above it
        cy = int(y_top + row_h * r + row_h * 0.74)
        w = int(min(col_w * 0.66, 430))
        h = int(min(row_h * 0.30, 300))
        if n.category == "valve":
            w, h = int(min(col_w * 0.5, 130)), int(min(row_h * 0.13, 90))
        elif n.category == "connector":
            w, h = int(min(col_w * 0.8, 340)), int(min(row_h * 0.16, 110))
        elif n.symbol in ("flare", "column", "vessel_v", "riser", "mol_sieve"):
            w, h = int(w * 0.55), int(min(row_h * 0.46, 420))
        out[n.id] = Box(cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
        n.col, n.row = c, r

    # ---- instrument bubbles, in a band between the relief row and the symbol
    R = int(max(44, min(64, row_h * 0.075)))
    by_parent: dict[str, list[Node]] = {}
    for n in sp.of_category("instrument"):
        by_parent.setdefault(n.parent, []).append(n)

    def _bubbles(parent: str, group: list[Node], below: bool):
        pb = out[parent]
        r = int(R * 0.72) if below else R
        for k, n in enumerate(group):
            per = 2 if below else 4
            row, col = k // per, k % per
            span_k = min(per, len(group) - row * per)
            x = pb.cx + int((col - (span_k - 1) / 2) * 2.45 * r)
            if below:
                y = pb.y1 + int(2.0 * r) + row * int(2.3 * r)
            else:
                y = pb.y0 - int(row_h * 0.13) - row * int(2.3 * r)
                y = max(int(y_top) + 30, y)
            out[n.id] = Box(x - r, y - r, x + r, y + r)

    strip_x = ax0
    by_id_all = {n.id: n for n in sp.all_nodes()}
    deferred: list[str] = []
    for parent, group in by_parent.items():
        if parent not in out:
            # its parent is a relief device, placed further down this function
            deferred.append(parent)
            continue
        pnode = by_id_all.get(parent)
        # A valve's position switches belong beside the valve, not floating over
        # the equipment behind it — so they hang BELOW, into empty sheet.
        _bubbles(parent, group, below=pnode is not None and pnode.category == "valve")

    # ---- relief devices sit ABOVE the instrument stack, still inside the cell
    for n in [x for x in sp.all_nodes() if is_satellite(x)]:
        pb = out.get(n.parent)
        if pb is None:
            continue
        sibs = [x for x in sp.all_nodes() if is_satellite(x) and x.parent == n.parent]
        k = sibs.index(n)
        bubbles = [out[b.id] for b in by_parent.get(n.parent, []) if b.id in out]
        ceiling = min([b.y0 for b in bubbles], default=pb.y0)
        w, h = 120, 84
        x = pb.cx + int((k - (len(sibs) - 1) / 2) * 300)
        y = max(int(y_top) + 60, ceiling - 150)
        out[n.id] = Box(x - w // 2, y - h // 2, x + w // 2, y + h // 2)

    for parent in deferred:
        if parent in out:
            _bubbles(parent, by_parent[parent], below=True)
        else:
            for n in by_parent[parent]:          # truly unattached: a top strip
                out[n.id] = Box(strip_x, int(y_top) - 190, strip_x + 2 * R,
                                int(y_top) - 190 + 2 * R)
                strip_x += int(2.6 * R)
    return out


def flare_header_y() -> int:
    return MARGIN + FRAME + 150


def draw_area() -> Box:
    return Box(MARGIN + FRAME, MARGIN + FRAME,
               SHEET_W - MARGIN - FRAME - NOTES_W - 20, SHEET_H - MARGIN - FRAME)


def title_block() -> Box:
    return Box(SHEET_W - MARGIN - FRAME - TITLE_W, SHEET_H - MARGIN - FRAME - TITLE_H,
               SHEET_W - MARGIN - FRAME, SHEET_H - MARGIN - FRAME)


def notes_block() -> Box:
    return Box(SHEET_W - MARGIN - FRAME - NOTES_W, MARGIN + FRAME,
               SHEET_W - MARGIN - FRAME, SHEET_H - MARGIN - FRAME - TITLE_H - 40)
