"""
The drawing engine — spec + layout → a real P&ID sheet.

Draws to ISA-5.1 / EPC convention at full sheet resolution (7000 x 4600, the
size the detector was trained on), so the output is not a diagram of a P&ID but
a drawing the rest of this system can read back: OCR finds the tags, the tiler
finds the symbols, blueprint mode re-derives the same graph.

Everything is deliberately vector-crisp black on white. No anti-aliased grey
wash, no gradients — a scanned rig drawing is bilevel, and the OCR and detector
stages are tuned for that.
"""
from __future__ import annotations
import datetime as _dt
import math
import cv2
import numpy as np

from .spec import Spec, Node
from .layout import (Box, SHEET_W, SHEET_H, MARGIN, FRAME, place, title_block,
                     notes_block, flare_header_y, is_satellite)

BLACK, WHITE, GREY = 0, 255, 130
F = cv2.FONT_HERSHEY_SIMPLEX
FD = cv2.FONT_HERSHEY_DUPLEX


# ───────────────────────────────────────────────────────────── primitives
def _txt(img, s, x, y, scale=1.0, th=2, anchor="c", font=F, colour=BLACK):
    s = str(s)
    (w, h), _ = cv2.getTextSize(s, font, scale, th)
    if anchor == "c":
        x -= w // 2
    elif anchor == "r":
        x -= w
    cv2.putText(img, s, (int(x), int(y + h // 2)), font, scale, colour, th, cv2.LINE_AA)
    return w, h


def _rect(img, b: Box, th=4):
    cv2.rectangle(img, (b.x0, b.y0), (b.x1, b.y1), BLACK, th, cv2.LINE_AA)


def _dashed(img, p0, p1, th=2, dash=26, gap=18):
    x0, y0 = p0; x1, y1 = p1
    d = math.hypot(x1 - x0, y1 - y0)
    if d < 1:
        return
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    t = 0.0
    while t < d:
        a = (int(x0 + ux * t), int(y0 + uy * t))
        e = min(t + dash, d)
        b = (int(x0 + ux * e), int(y0 + uy * e))
        cv2.line(img, a, b, BLACK, th, cv2.LINE_AA)
        t = e + gap


def _arrow(img, p, direction, size=26, th=4):
    x, y = p
    dx, dy = direction
    n = math.hypot(dx, dy) or 1
    dx, dy = dx / n, dy / n
    px, py = -dy, dx
    a = (int(x), int(y))
    b = (int(x - dx * size + px * size * 0.5), int(y - dy * size + py * size * 0.5))
    c = (int(x - dx * size - px * size * 0.5), int(y - dy * size - py * size * 0.5))
    cv2.drawContours(img, [np.array([a, b, c])], 0, BLACK, -1, cv2.LINE_AA)


# ───────────────────────────────────────────────────────────── symbols
def _vessel_h(img, b: Box):
    r = (b.y1 - b.y0) // 2
    cv2.line(img, (b.x0 + r, b.y0), (b.x1 - r, b.y0), BLACK, 5, cv2.LINE_AA)
    cv2.line(img, (b.x0 + r, b.y1), (b.x1 - r, b.y1), BLACK, 5, cv2.LINE_AA)
    cv2.ellipse(img, (b.x0 + r, b.cy), (r, r), 0, 90, 270, BLACK, 5, cv2.LINE_AA)
    cv2.ellipse(img, (b.x1 - r, b.cy), (r, r), 0, -90, 90, BLACK, 5, cv2.LINE_AA)
    # liquid level and weir — what makes it read as a separator
    ly = b.y0 + int((b.y1 - b.y0) * 0.62)
    _dashed(img, (b.x0 + r, ly), (b.x1 - r, ly), th=2, dash=34, gap=22)
    wx = b.x0 + int((b.x1 - b.x0) * 0.72)
    cv2.line(img, (wx, b.y1), (wx, b.y0 + int((b.y1 - b.y0) * 0.35)), BLACK, 4, cv2.LINE_AA)


def _vessel_v(img, b: Box):
    r = (b.x1 - b.x0) // 2
    cv2.line(img, (b.x0, b.y0 + r), (b.x0, b.y1 - r), BLACK, 5, cv2.LINE_AA)
    cv2.line(img, (b.x1, b.y0 + r), (b.x1, b.y1 - r), BLACK, 5, cv2.LINE_AA)
    cv2.ellipse(img, (b.cx, b.y0 + r), (r, r), 0, 180, 360, BLACK, 5, cv2.LINE_AA)
    cv2.ellipse(img, (b.cx, b.y1 - r), (r, r), 0, 0, 180, BLACK, 5, cv2.LINE_AA)
    ly = b.y1 - int((b.y1 - b.y0) * 0.28)
    _dashed(img, (b.x0, ly), (b.x1, ly), th=2, dash=30, gap=20)


def _column(img, b: Box):
    _vessel_v(img, b)
    step = max(28, (b.y1 - b.y0) // 8)
    for y in range(b.y0 + step * 2, b.y1 - step, step):
        cv2.line(img, (b.x0 + 8, y), (b.x1 - 8, y), BLACK, 2, cv2.LINE_AA)


def _pump(img, b: Box):
    r = min(b.x1 - b.x0, b.y1 - b.y0) // 2
    cv2.circle(img, (b.cx, b.cy), r, BLACK, 5, cv2.LINE_AA)
    pts = np.array([(b.cx, b.cy - r), (b.cx + int(r * 1.25), b.cy - r),
                    (b.cx + int(r * 1.25), b.cy - int(r * 0.15)), (b.cx + r, b.cy)])
    cv2.polylines(img, [pts], False, BLACK, 5, cv2.LINE_AA)
    cv2.line(img, (b.cx - r, b.cy), (b.cx - int(r * 1.3), b.cy), BLACK, 5, cv2.LINE_AA)


def _pump_recip(img, b: Box):
    _rect(img, Box(b.x0, b.cy - (b.y1 - b.y0) // 3, b.x1, b.cy + (b.y1 - b.y0) // 3), 5)
    cv2.line(img, (b.x0 + (b.x1 - b.x0) // 3, b.cy - (b.y1 - b.y0) // 3),
             (b.x0 + (b.x1 - b.x0) // 3, b.cy + (b.y1 - b.y0) // 3), BLACK, 4, cv2.LINE_AA)


def _compressor(img, b: Box):
    r = min(b.x1 - b.x0, b.y1 - b.y0) // 2
    cv2.circle(img, (b.cx, b.cy), r, BLACK, 5, cv2.LINE_AA)
    # centrifugal: a tapered blade across the body
    pts = np.array([(b.cx - int(r * 0.8), b.cy - int(r * 0.55)),
                    (b.cx + int(r * 0.8), b.cy - int(r * 0.20)),
                    (b.cx + int(r * 0.8), b.cy + int(r * 0.20)),
                    (b.cx - int(r * 0.8), b.cy + int(r * 0.55))])
    cv2.polylines(img, [pts], True, BLACK, 4, cv2.LINE_AA)


def _compressor_recip(img, b: Box):
    _rect(img, b, 5)
    cv2.circle(img, (b.x0 + (b.x1 - b.x0) // 4, b.cy),
               max(10, (b.y1 - b.y0) // 4), BLACK, 4, cv2.LINE_AA)
    cv2.line(img, (b.cx, b.y0), (b.cx, b.y1), BLACK, 4, cv2.LINE_AA)


def _exchanger(img, b: Box):
    r = min(b.x1 - b.x0, b.y1 - b.y0) // 2
    cv2.circle(img, (b.cx, b.cy), r, BLACK, 5, cv2.LINE_AA)
    z = []
    for i in range(7):
        z.append((b.cx - r + int(2 * r * i / 6), b.cy + (-1) ** i * int(r * 0.45)))
    cv2.polylines(img, [np.array(z)], False, BLACK, 4, cv2.LINE_AA)


def _air_cooler(img, b: Box):
    _rect(img, b, 5)
    n = 3
    for i in range(n):
        cx = b.x0 + int((b.x1 - b.x0) * (i + 0.5) / n)
        rr = min((b.x1 - b.x0) // (2 * n + 1), (b.y1 - b.y0) // 3)
        cv2.circle(img, (cx, b.y0 - rr - 6), rr, BLACK, 4, cv2.LINE_AA)
        cv2.line(img, (cx - rr, b.y0 - rr - 6), (cx + rr, b.y0 - rr - 6), BLACK, 3, cv2.LINE_AA)


def _tank(img, b: Box):
    _rect(img, b, 5)
    cv2.line(img, (b.x0, b.y0 + 20), (b.x1, b.y0 + 20), BLACK, 3, cv2.LINE_AA)
    ly = b.y1 - int((b.y1 - b.y0) * 0.35)
    _dashed(img, (b.x0, ly), (b.x1, ly), th=2, dash=30, gap=20)


def _filter(img, b: Box):
    _rect(img, b, 5)
    for i in range(1, 6):
        x = b.x0 + int((b.x1 - b.x0) * i / 6)
        cv2.line(img, (x, b.y0 + 10), (x, b.y1 - 10), BLACK, 2, cv2.LINE_AA)


def _package(img, b: Box):
    _dashed(img, (b.x0, b.y0), (b.x1, b.y0), th=4, dash=40, gap=24)
    _dashed(img, (b.x1, b.y0), (b.x1, b.y1), th=4, dash=40, gap=24)
    _dashed(img, (b.x1, b.y1), (b.x0, b.y1), th=4, dash=40, gap=24)
    _dashed(img, (b.x0, b.y1), (b.x0, b.y0), th=4, dash=40, gap=24)


def _manifold(img, b: Box):
    h = max(18, (b.y1 - b.y0) // 5)
    cv2.rectangle(img, (b.x0, b.cy - h), (b.x1, b.cy + h), BLACK, 5, cv2.LINE_AA)
    n = 5
    for i in range(n):
        x = b.x0 + int((b.x1 - b.x0) * (i + 0.5) / n)
        cv2.line(img, (x, b.cy - h), (x, b.y0), BLACK, 4, cv2.LINE_AA)


def _hydrocyclone(img, b: Box):
    pts = np.array([(b.x0, b.y0), (b.x1, b.y0),
                    (b.cx + (b.x1 - b.x0) // 8, b.y1), (b.cx - (b.x1 - b.x0) // 8, b.y1)])
    cv2.polylines(img, [pts], True, BLACK, 5, cv2.LINE_AA)


def _turbine(img, b: Box):
    pts = np.array([(b.x0, b.y0), (b.x1, b.y0 + (b.y1 - b.y0) // 4),
                    (b.x1, b.y1 - (b.y1 - b.y0) // 4), (b.x0, b.y1)])
    cv2.polylines(img, [pts], True, BLACK, 5, cv2.LINE_AA)
    cv2.line(img, (b.cx, b.y0 + 12), (b.cx, b.y1 - 12), BLACK, 4, cv2.LINE_AA)


def _bop(img, b: Box):
    n = 3
    hh = (b.y1 - b.y0) // n
    for i in range(n):
        y0 = b.y0 + i * hh
        cv2.rectangle(img, (b.x0, y0 + 4), (b.x1, y0 + hh - 4), BLACK, 4, cv2.LINE_AA)
        cv2.line(img, (b.x0, y0 + hh // 2), (b.x1, y0 + hh // 2), BLACK, 3, cv2.LINE_AA)


def _pig_trap(img, b: Box):
    r = (b.y1 - b.y0) // 2
    cv2.line(img, (b.x0 + r, b.y0), (b.x1, b.y0), BLACK, 5, cv2.LINE_AA)
    cv2.line(img, (b.x0 + r, b.y1), (b.x1, b.y1), BLACK, 5, cv2.LINE_AA)
    cv2.ellipse(img, (b.x0 + r, b.cy), (r, r), 0, 90, 270, BLACK, 5, cv2.LINE_AA)
    cv2.line(img, (b.x1, b.y0), (b.x1, b.y1), BLACK, 7, cv2.LINE_AA)   # closure
    cv2.line(img, (b.x0 + r, b.cy - r // 2), (b.x1 - 20, b.cy - r // 2), BLACK, 2, cv2.LINE_AA)


def _flare(img, b: Box):
    cv2.line(img, (b.cx - 22, b.y1), (b.cx - 22, b.y0 + 60), BLACK, 6, cv2.LINE_AA)
    cv2.line(img, (b.cx + 22, b.y1), (b.cx + 22, b.y0 + 60), BLACK, 6, cv2.LINE_AA)
    pts = np.array([(b.cx - 40, b.y0 + 60), (b.cx, b.y0), (b.cx + 40, b.y0 + 60)])
    cv2.polylines(img, [pts], False, BLACK, 6, cv2.LINE_AA)
    for i in range(3):
        y = b.y1 - int((b.y1 - b.y0) * (i + 1) / 4)
        cv2.line(img, (b.cx - 60, y), (b.cx + 60, y), BLACK, 2, cv2.LINE_AA)


def _riser(img, b: Box):
    cv2.line(img, (b.cx, b.y0), (b.cx, b.y1), BLACK, 7, cv2.LINE_AA)
    cv2.ellipse(img, (b.cx, b.y1), ((b.x1 - b.x0) // 3, (b.y1 - b.y0) // 8),
                0, 0, 180, BLACK, 5, cv2.LINE_AA)


def _xmas_tree(img, b: Box):
    cv2.line(img, (b.cx, b.y0), (b.cx, b.y1), BLACK, 6, cv2.LINE_AA)
    cv2.line(img, (b.x0, b.cy), (b.x1, b.cy), BLACK, 6, cv2.LINE_AA)
    for p in ((b.cx, b.y0 + (b.y1 - b.y0) // 4), (b.x0 + (b.x1 - b.x0) // 4, b.cy),
              (b.x1 - (b.x1 - b.x0) // 4, b.cy)):
        _bowtie(img, Box(p[0] - 26, p[1] - 20, p[0] + 26, p[1] + 20))
    cv2.line(img, (b.x0 + 30, b.y1), (b.x1 - 30, b.y1), BLACK, 8, cv2.LINE_AA)


def _wellhead(img, b: Box):
    _xmas_tree(img, b)


def _bowtie(img, b: Box, th=5):
    pts = np.array([(b.x0, b.y0), (b.x0, b.y1), (b.x1, b.y0), (b.x1, b.y1)])
    cv2.polylines(img, [pts], True, BLACK, th, cv2.LINE_AA)


def _gate_valve(img, b: Box):
    _bowtie(img, b)


def _ball_valve(img, b: Box):
    _bowtie(img, b)
    cv2.circle(img, (b.cx, b.cy), max(8, (b.y1 - b.y0) // 4), BLACK, 4, cv2.LINE_AA)


def _globe_valve(img, b: Box):
    _bowtie(img, b)
    cv2.circle(img, (b.cx, b.cy), max(8, (b.y1 - b.y0) // 4), BLACK, -1, cv2.LINE_AA)


def _butterfly_valve(img, b: Box):
    _bowtie(img, b)
    cv2.line(img, (b.x0 + 6, b.y1 - 6), (b.x1 - 6, b.y0 + 6), BLACK, 4, cv2.LINE_AA)


def _check_valve(img, b: Box):
    _bowtie(img, b)
    cv2.line(img, (b.x1, b.y0), (b.x1, b.y1), BLACK, 6, cv2.LINE_AA)


def _control_valve(img, b: Box):
    _bowtie(img, b)
    # diaphragm actuator
    top = b.y0 - (b.y1 - b.y0)
    cv2.line(img, (b.cx, b.y0), (b.cx, top + 14), BLACK, 4, cv2.LINE_AA)
    cv2.ellipse(img, (b.cx, top + 14), ((b.x1 - b.x0) // 3, (b.y1 - b.y0) // 3),
                0, 180, 360, BLACK, 4, cv2.LINE_AA)
    cv2.line(img, (b.cx - (b.x1 - b.x0) // 3, top + 14),
             (b.cx + (b.x1 - b.x0) // 3, top + 14), BLACK, 4, cv2.LINE_AA)


def _esd_valve(img, b: Box):
    _bowtie(img, b)
    top = b.y0 - (b.y1 - b.y0)
    cv2.line(img, (b.cx, b.y0), (b.cx, top + 20), BLACK, 4, cv2.LINE_AA)
    cv2.rectangle(img, (b.cx - (b.x1 - b.x0) // 3, top), (b.cx + (b.x1 - b.x0) // 3, top + 20),
                  BLACK, 4, cv2.LINE_AA)
    _txt(img, "S", b.cx, top + 10, 0.7, 2)


def _psv(img, b: Box):
    # angle body relief valve with a spring bonnet
    cv2.line(img, (b.x0, b.cy), (b.cx, b.cy), BLACK, 5, cv2.LINE_AA)
    pts = np.array([(b.cx, b.cy), (b.cx - 24, b.cy + 22), (b.cx + 24, b.cy + 22)])
    cv2.polylines(img, [pts], True, BLACK, 5, cv2.LINE_AA)
    cv2.line(img, (b.cx, b.cy), (b.cx, b.y0 + 18), BLACK, 5, cv2.LINE_AA)
    zz = []
    for i in range(6):
        zz.append((b.cx + (-1) ** i * 20, b.y0 + 18 + i * 10))
    cv2.polylines(img, [np.array(zz)], False, BLACK, 3, cv2.LINE_AA)
    cv2.line(img, (b.cx - 26, b.y0), (b.cx + 26, b.y0), BLACK, 5, cv2.LINE_AA)


def _rupture_disc(img, b: Box):
    cv2.line(img, (b.x0, b.cy), (b.x1, b.cy), BLACK, 5, cv2.LINE_AA)
    cv2.ellipse(img, (b.cx, b.cy), ((b.x1 - b.x0) // 4, (b.y1 - b.y0) // 2),
                0, 90, 270, BLACK, 5, cv2.LINE_AA)


def _orifice(img, b: Box):
    cv2.line(img, (b.cx - 8, b.y0), (b.cx - 8, b.y1), BLACK, 5, cv2.LINE_AA)
    cv2.line(img, (b.cx + 8, b.y0), (b.cx + 8, b.y1), BLACK, 5, cv2.LINE_AA)


def _strainer(img, b: Box):
    _bowtie(img, b)
    cv2.line(img, (b.cx, b.y1), (b.cx + 20, b.y1 + 26), BLACK, 4, cv2.LINE_AA)


def _choke(img, b: Box):
    _bowtie(img, b)
    cv2.line(img, (b.x0, b.y1), (b.x1, b.y0), BLACK, 3, cv2.LINE_AA)


def _offpage(img, b: Box):
    d = (b.y1 - b.y0) // 2
    pts = np.array([(b.x0, b.y0), (b.x1 - d, b.y0), (b.x1, b.cy),
                    (b.x1 - d, b.y1), (b.x0, b.y1)])
    cv2.polylines(img, [pts], True, BLACK, 5, cv2.LINE_AA)


SYMBOLS = {
    "vessel_h": _vessel_h, "vessel_v": _vessel_v, "column": _column,
    "pump": _pump, "pump_recip": _pump_recip, "compressor": _compressor,
    "compressor_recip": _compressor_recip, "exchanger": _exchanger,
    "air_cooler": _air_cooler, "tank": _tank, "filter": _filter,
    "package": _package, "manifold": _manifold, "hydrocyclone": _hydrocyclone,
    "turbine": _turbine, "bop": _bop, "pig_trap": _pig_trap, "flare": _flare,
    "riser": _riser, "xmas_tree": _xmas_tree, "wellhead": _wellhead,
    "gate_valve": _gate_valve, "ball_valve": _ball_valve, "globe_valve": _globe_valve,
    "butterfly_valve": _butterfly_valve, "check_valve": _check_valve,
    "control_valve": _control_valve, "esd_valve": _esd_valve, "psv": _psv,
    "rupture_disc": _rupture_disc, "orifice": _orifice, "strainer": _strainer,
    "choke": _choke, "offpage": _offpage,
}


# ───────────────────────────────────────────────────────────── pipe routing
def _port(b: Box, side: str) -> tuple[int, int]:
    return {"r": (b.x1, b.cy), "l": (b.x0, b.cy),
            "t": (b.cx, b.y0), "b": (b.cx, b.y1)}[side]


def _route(a: Box, b: Box, lane: int = 0) -> list[tuple[int, int]]:
    """Orthogonal run from a to b — right-out, left-in, with one dog-leg."""
    if b.cx >= a.cx:
        p0, p1 = _port(a, "r"), _port(b, "l")
        mx = (p0[0] + p1[0]) // 2 + lane * 22
        if abs(p0[1] - p1[1]) < 6:
            return [p0, p1]
        return [p0, (mx, p0[1]), (mx, p1[1]), p1]
    # a recycle or return run: drop below and come back
    p0, p1 = _port(a, "b"), _port(b, "b")
    y = max(a.y1, b.y1) + 90 + lane * 26
    return [p0, (p0[0], y), (p1[0], y), p1]


def _draw_line(img, pts, th, dashed=False):
    for i in range(len(pts) - 1):
        if dashed:
            _dashed(img, pts[i], pts[i + 1], th=max(2, th - 2))
        else:
            cv2.line(img, pts[i], pts[i + 1], BLACK, th, cv2.LINE_AA)


# ───────────────────────────────────────────────────────────── the sheet
def _border(img, sp: Spec):
    cv2.rectangle(img, (MARGIN, MARGIN), (SHEET_W - MARGIN, SHEET_H - MARGIN),
                  BLACK, 4, cv2.LINE_AA)
    cv2.rectangle(img, (MARGIN + FRAME, MARGIN + FRAME),
                  (SHEET_W - MARGIN - FRAME, SHEET_H - MARGIN - FRAME),
                  BLACK, 3, cv2.LINE_AA)
    # zone references, as on any issued drawing
    cols = 8
    for i in range(cols):
        x = MARGIN + int((SHEET_W - 2 * MARGIN) * (i + 0.5) / cols)
        _txt(img, str(i + 1), x, MARGIN + FRAME // 2, 0.9, 2)
        _txt(img, str(i + 1), x, SHEET_H - MARGIN - FRAME // 2, 0.9, 2)
    rows = 6
    for i in range(rows):
        y = MARGIN + int((SHEET_H - 2 * MARGIN) * (i + 0.5) / rows)
        _txt(img, "ABCDEF"[i], MARGIN + FRAME // 2, y, 0.9, 2)
        _txt(img, "ABCDEF"[i], SHEET_W - MARGIN - FRAME // 2, y, 0.9, 2)


def _title(img, sp: Spec):
    b = title_block()
    _rect(img, b, 4)
    rows = [b.y0 + 120, b.y0 + 250, b.y0 + 370, b.y0 + 470]
    for y in rows:
        cv2.line(img, (b.x0, y), (b.x1, y), BLACK, 2, cv2.LINE_AA)
    _txt(img, sp.facility.upper()[:38], b.x0 + 30, b.y0 + 45, 1.15, 3, "l", FD)
    _txt(img, "ENGINEERING DRAWING INTELLIGENCE", b.x0 + 30, b.y0 + 90, 0.7, 2, "l")
    _txt(img, "PIPING & INSTRUMENTATION DIAGRAM", b.x0 + 30, b.y0 + 165, 0.78, 2, "l")
    _txt(img, sp.title.upper()[:44], b.x0 + 30, b.y0 + 215, 1.0, 3, "l", FD)

    cv2.line(img, (b.x0 + 1180, b.y0 + 250), (b.x0 + 1180, b.y1), BLACK, 2, cv2.LINE_AA)
    cv2.line(img, (b.x0 + 1600, b.y0 + 250), (b.x0 + 1600, b.y1), BLACK, 2, cv2.LINE_AA)
    _txt(img, "DRAWING No.", b.x0 + 30, b.y0 + 285, 0.62, 2, "l")
    _txt(img, sp.drawing_no.upper(), b.x0 + 30, b.y0 + 335, 0.95, 3, "l", FD)
    _txt(img, "REV", b.x0 + 1210, b.y0 + 285, 0.62, 2, "l")
    _txt(img, sp.revision.upper(), b.x0 + 1210, b.y0 + 335, 0.95, 3, "l", FD)
    _txt(img, "SHEET", b.x0 + 1630, b.y0 + 285, 0.62, 2, "l")
    _txt(img, "1 OF 1", b.x0 + 1630, b.y0 + 335, 0.85, 2, "l")

    _txt(img, "SCALE  NTS", b.x0 + 30, b.y0 + 420, 0.66, 2, "l")
    _txt(img, f"UNIT  {sp.area}", b.x0 + 500, b.y0 + 420, 0.66, 2, "l")
    _txt(img, f"DATE  {_dt.date.today():%d-%m-%Y}", b.x0 + 900, b.y0 + 420, 0.66, 2, "l")
    _txt(img, "DRAWN  EDI", b.x0 + 1630, b.y0 + 420, 0.62, 2, "l")
    _txt(img, "GENERATED FROM SPEC · ALL TAGS PER ISA-5.1 · DIMENSIONS IN INCHES",
         b.x0 + 30, b.y0 + 530, 0.6, 2, "l")


def _notes(img, sp: Spec, stats: dict):
    b = notes_block()
    _rect(img, b, 3)
    y = b.y0 + 46
    _txt(img, "NOTES", b.x0 + 26, y, 0.95, 3, "l", FD); y += 62
    notes = sp.notes or [
        "ALL VALVES SHOWN CLOSED ARE NORMALLY CLOSED.",
        "RELIEF DEVICES DISCHARGE TO THE HP FLARE HEADER.",
        "INSTRUMENT AIR FAILURE POSITION SHOWN AT EACH VALVE.",
        "LINE NUMBERS PER PROJECT PIPING SPECIFICATION.",
        "ESD VALVES FAIL CLOSED ON LOSS OF SIGNAL.",
    ]
    for i, n in enumerate(notes[:8], 1):
        for ln in _wrap(f"{i}. {n.upper()}", 30):
            _txt(img, ln, b.x0 + 26, y, 0.58, 2, "l"); y += 38
        y += 8

    y += 30
    cv2.line(img, (b.x0 + 20, y), (b.x1 - 20, y), BLACK, 2, cv2.LINE_AA); y += 44
    _txt(img, "SHEET CONTENT", b.x0 + 26, y, 0.85, 3, "l", FD); y += 56
    for k, v in stats.items():
        _txt(img, k.upper(), b.x0 + 26, y, 0.58, 2, "l")
        _txt(img, str(v), b.x1 - 30, y, 0.62, 2, "r"); y += 40

    y += 30
    cv2.line(img, (b.x0 + 20, y), (b.x1 - 20, y), BLACK, 2, cv2.LINE_AA); y += 44
    _txt(img, "LEGEND", b.x0 + 26, y, 0.85, 3, "l", FD); y += 20
    legend = [("gate_valve", "GATE / BLOCK VALVE"), ("control_valve", "CONTROL VALVE"),
              ("esd_valve", "ESD / SDV"), ("psv", "RELIEF VALVE"),
              ("check_valve", "CHECK VALVE"), ("offpage", "OFF-PAGE CONNECTOR")]
    for key, label in legend:
        y += 76
        sb = Box(b.x0 + 40, y - 26, b.x0 + 110, y + 26)
        SYMBOLS[key](img, sb)
        _txt(img, label, b.x0 + 170, y, 0.56, 2, "l")
    y += 80
    _dashed(img, (b.x0 + 40, y), (b.x0 + 110, y), th=2)
    _txt(img, "INSTRUMENT SIGNAL", b.x0 + 170, y, 0.56, 2, "l")
    y += 50
    cv2.line(img, (b.x0 + 40, y), (b.x0 + 110, y), BLACK, 6, cv2.LINE_AA)
    _txt(img, "PROCESS LINE", b.x0 + 170, y, 0.56, 2, "l")


def _wrap(s: str, n: int) -> list[str]:
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > n:
            out.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def _bubble(img, b: Box, n: Node):
    r = (b.x1 - b.x0) // 2
    cv2.circle(img, (b.cx, b.cy), r, BLACK, 4, cv2.LINE_AA)
    code, num = (n.tag.split("-", 1) + [""])[:2]
    # ISA prints the function code above the loop number, split by a rule.
    # blueprint mode's `_pair_bubbles` reads exactly this shape back.
    cv2.line(img, (b.cx - r, b.cy), (b.cx + r, b.cy), BLACK, 3, cv2.LINE_AA)
    _txt(img, code, b.cx, b.cy - r // 2, 0.78, 2)
    _txt(img, num, b.cx, b.cy + r // 2, 0.78, 2)


def draw(sp: Spec, boxes: dict[str, Box] | None = None) -> tuple[np.ndarray, dict[str, Box]]:
    """Render the sheet. Returns the image and the box of every node on it."""
    img = np.full((SHEET_H, SHEET_W), WHITE, np.uint8)
    boxes = boxes or place(sp)
    _border(img, sp)

    by_id = {n.id: n for n in sp.all_nodes()}
    relief = {n.id for n in sp.all_nodes() if n.item in ("psv", "bdv", "rupture_disc")}
    sat = {n.id for n in sp.all_nodes() if is_satellite(n)}
    flare = {n.id for n in sp.all_nodes()
             if n.item in ("flare_stack", "hp_flare_ko", "lp_flare_ko",
                           "vent_stack", "flare_header")}
    # the header sits just above the highest relief device, so the risers stay
    # short whatever the sheet's shape
    tops = [boxes[i].y0 for i in relief if i in boxes]
    hy = max(flare_header_y(), min(tops) - 150) if tops else flare_header_y()

    # hazardous area envelope around the Zone 1 process items
    z1 = [boxes[n.id] for n in sp.equipment() if n.zone == "Zone 1" and n.id in boxes]
    if len(z1) >= 2:
        x0 = min(b.x0 for b in z1) - 90; y0 = min(b.y0 for b in z1) - 150
        x1 = max(b.x1 for b in z1) + 90; y1 = max(b.y1 for b in z1) + 150
        y0 = max(hy + 70, y0)
        for p, q in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                     ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            _dashed(img, p, q, th=2, dash=48, gap=32)
        _txt(img, "HAZARDOUS AREA  ZONE 1", x0 + 250, y0 - 26, 0.66, 2, "c")

    # ---- pipes (drawn first so symbols sit on top of them) --------------
    header_x: list[int] = []
    lane: dict[int, int] = {}
    for s in sp.streams:
        a, b = boxes.get(s.src), boxes.get(s.dst)
        if a is None or b is None:
            continue
        th = 7 if s.size_in >= 10 else (5 if s.size_in >= 4 else 4)
        na, nb = by_id.get(s.src), by_id.get(s.dst)

        # equipment -> its own relief device: a short riser up the cell
        if s.dst in sat and nb is not None and nb.parent == s.src:
            _draw_line(img, [(a.cx + 60, a.y0), (a.cx + 60, b.cy), (b.x0, b.cy)], th)
            _txt(img, f'{s.size_in:g}"', a.cx + 96, (a.y0 + b.cy) // 2, 0.6, 2, "l")
            continue
        # anti-surge recycle back into the machine
        if s.src in sat and nb is not None and na is not None and na.parent == s.dst:
            _draw_line(img, [(a.x0, a.cy), (b.cx - 90, a.cy), (b.cx - 90, b.y0)], th)
            continue
        # relief discharge: straight up to the flare header
        if s.src in relief and s.dst in flare:
            _draw_line(img, [(a.cx, a.y0), (a.cx, hy)], th)
            header_x.append(a.cx)
            continue

        k = min(a.cy, b.cy) // 40
        lane[k] = lane.get(k, 0) + 1
        pts = _route(a, b, lane[k] % 3 - 1)
        _draw_line(img, pts, th)
        e = pts[-1]; d = (e[0] - pts[-2][0], e[1] - pts[-2][1])
        _arrow(img, e, d if (d[0] or d[1]) else (1, 0), 30, th)
        best, blen = None, 0
        for i in range(len(pts) - 1):
            if pts[i][1] == pts[i + 1][1] and abs(pts[i + 1][0] - pts[i][0]) > blen:
                best, blen = i, abs(pts[i + 1][0] - pts[i][0])
        if best is not None and blen > 300:
            mx = (pts[best][0] + pts[best + 1][0]) // 2
            _txt(img, s.tag, mx, pts[best][1] - 32, 0.62, 2)

    # ---- the flare header itself ---------------------------------------
    if header_x:
        dest = next((boxes[i] for i in flare if i in boxes), None)
        x_end = dest.cx if dest is not None else max(header_x) + 400
        x0 = min(min(header_x), x_end) - 120
        x1 = max(max(header_x), x_end)
        cv2.line(img, (x0, hy), (x1, hy), BLACK, 9, cv2.LINE_AA)
        _txt(img, f'24"-FL-15-100-F1A   HP FLARE HEADER', (x0 + x1) // 2, hy - 40, 0.72, 2)
        if dest is not None:
            cv2.line(img, (dest.cx, hy), (dest.cx, dest.y0), BLACK, 9, cv2.LINE_AA)
            _arrow(img, (dest.cx, dest.y0), (0, 1), 32, 9)

    # ---- instrument signal lines ----------------------------------------
    for n in sp.of_category("instrument"):
        nb, pb = boxes.get(n.id), boxes.get(n.parent)
        if nb is None or pb is None:
            continue
        if nb.cy <= pb.cy:
            _dashed(img, (nb.cx, nb.y1), (pb.cx, pb.y0), th=2)
        else:
            _dashed(img, (nb.cx, nb.y0), (pb.cx, pb.y1), th=2)

    # ---- symbols ---------------------------------------------------------
    for n in sp.all_nodes():
        b = boxes.get(n.id)
        if b is None:
            continue
        if n.category == "instrument":
            _bubble(img, b, n)
            continue
        fn = SYMBOLS.get(n.symbol)
        if fn is None:
            _rect(img, b, 5)
        else:
            fn(img, b)
        if n.category == "connector":
            _txt(img, n.tag[:24], b.cx - 12, b.cy, 0.58, 2)
            continue
        if is_satellite(n):
            # a relief valve's tag goes above it — below is where its bubbles are
            _txt(img, n.tag, b.cx, b.y0 - 34, 0.68, 2)
        else:
            _txt(img, n.tag, b.cx, b.y1 + 44,
                 0.95 if n.category == "equipment" else 0.68,
                 3 if n.category == "equipment" else 2)
        if n.category == "equipment":
            _txt(img, (n.label or n.name).upper()[:30], b.cx, b.y1 + 94, 0.6, 2)

    stats = {"equipment": len(sp.equipment()),
             "valves": len(sp.of_category("valve")),
             "instruments": len(sp.of_category("instrument")),
             "lines": len(sp.streams),
             "control loops": len(sp.controls)}
    _notes(img, sp, stats)
    _title(img, sp)
    return img, boxes
