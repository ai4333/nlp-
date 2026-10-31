"""
Pipe line numbers — size, service and spec.

"What size is that line?" is the single most common question asked of a P&ID,
and until now the system could not answer it at all. Line numbers carry it in
plain text, in the EPC convention every operator uses some variant of:

    12"-DC-11-185-P1A-HE
     │   │   │   │   │  └ insulation / tracing code
     │   │   │   │   └── piping material spec
     │   │   │   └────── line sequence number
     │   │   └────────── area or unit
     │   └────────────── service code (process, drain, vent, flare…)
     └────────────────── nominal size in inches

Read off a real offshore P&ID (ALPHA-AA-0000-PR-PD-0030-01): 1"-VL-16-350,
2"-DC-28-083-D1C, 3"-PL-11-004-P1A-HE, 12"-DC-11-185-P1A-HE.

The convention varies between operators — the patterns and service codes live in
`config/tag_conventions.yaml` so they can be matched to a customer's standard
without touching code.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

_CONV = yaml.safe_load(open(Path(__file__).resolve().parents[3] / "config" / "tag_conventions.yaml"))
SERVICE = _CONV.get("service_codes", {}) or {}

# size " service - area - number [- spec] [- insulation]
_LINE = re.compile(
    r"""^\s*
    (?P<size>\d{1,2}(?:\.\d)?)\s*["'”]?\s*[-–]\s*      # 12"
    (?P<service>[A-Z]{1,3})\s*[-–]\s*                  # DC
    (?P<area>\d{1,3})\s*[-–]\s*                        # 11
    (?P<number>\d{2,4})                                # 185
    (?:\s*[-–]\s*(?P<spec>[A-Z0-9]{2,5}))?             # P1A
    (?:\s*[-–]\s*(?P<insul>[A-Z]{1,3}))?               # HE
    \s*$""", re.X)

# shorter variant some drawings use:  6"-LD-753
_LINE_SHORT = re.compile(
    r"""^\s*(?P<size>\d{1,2}(?:\.\d)?)\s*["'”]\s*[-–]\s*
    (?P<service>[A-Z]{1,3})\s*[-–]\s*(?P<number>\d{2,5})\s*$""", re.X)


@dataclass
class LineNumber:
    raw: str
    size_in: float
    service: str
    service_name: str
    number: str
    area: str = ""
    spec: str = ""
    insulation: str = ""

    @property
    def key(self) -> str:
        """Identity of the physical pipe, ignoring spec/insulation noise."""
        return f"{self.size_in:g}|{self.service}|{self.area}|{self.number}"

    @property
    def label(self) -> str:
        return f'{self.size_in:g}"-{self.service}-{self.number}'


# Piping spec codes are letter-digit-letter (P1A, D1C, A1C). OCR routinely
# confuses the digit for a letter there — the same real drawing yielded both
# "3\"-VH-16-060-P1A" and "3\"-VH-16-060-PIA", which would otherwise become two
# separate pipes. Canonicalise the digit positions.
_SPEC_FIX = {"I": "1", "L": "1", "O": "0", "S": "5"}


def _fix_spec(spec: str) -> str:
    if len(spec) < 3:
        return spec
    mid = "".join(_SPEC_FIX.get(c, c) if i > 0 and i < len(spec) - 1 else c
                  for i, c in enumerate(spec))
    return mid


def _clean(t: str) -> str:
    return (t or "").strip().upper().replace("–", "-").replace("”", '"').replace("''", '"')


def parse(text: str) -> LineNumber | None:
    t = _clean(text).rstrip("!.,;:")
    for rx in (_LINE, _LINE_SHORT):
        m = rx.match(t)
        if not m:
            continue
        g = m.groupdict()
        svc = g["service"]
        return LineNumber(
            raw=t, size_in=float(g["size"]), service=svc,
            service_name=SERVICE.get(svc, {}).get("name", "process line"),
            number=g["number"], area=g.get("area") or "",
            spec=_fix_spec(g.get("spec") or ""), insulation=g.get("insul") or "")
    return None


def is_line_number(text: str) -> bool:
    return parse(text) is not None


def to_entity(ln: LineNumber, idx: int, cx: float, cy: float, score: float,
              box_px: int = 52) -> dict:
    """A line is a first-class graph node, not an annotation on one."""
    return {
        "id": f"l{idx:04d}",
        "pidqa_class": 106,
        "category": "line",
        "name": f'{ln.size_in:g}" {ln.service_name}',
        "bbox": [int(cx - box_px), int(cy - box_px // 2),
                 int(cx + box_px), int(cy + box_px // 2)],
        "tag": ln.raw,
        "confidence": 1.0,
        "tag_confidence": round(float(score), 3),
        "tag_location": "internal",
        "source": "blueprint",
        "line": asdict(ln),
        "size_in": ln.size_in,
        "service": ln.service,
    }


# ---------------------------------------------------------------------------
# Clipped line numbers.
#
# On a real sheet OCR routinely returns only part of a line number: the drawing
# prints `1"-DC-11-266-P1A-HE` along a pipe and the reader comes back with
# `DC266`. That fragment then looks exactly like an equipment tag, and the
# prefix classifier resolves its FIRST letter — so `DC266` became a **drum**,
# `PL103` a **pump**, `VH082` a **vessel**.
#
# Measured on the dense offshore P&ID (u_pid_real_p1_52af52): 22 of the 37 items
# it called "equipment" were line-number fragments. A field technician asking
# "what is DC266?" was told, with a drawing crop to back it up, that it is a
# drum. That is the single worst failure mode this system had.
#
# The discriminator is exact and needs no model: a MULTI-letter pipe service
# code followed by digits and nothing else. Single letters are deliberately
# excluded — `P` is both "process line" and "pump", so `P-1101A` must stay a
# pump. There is no overlap between the multi-letter service codes and the
# equipment/instrument prefixes except the ones listed as ambiguous in
# `config/tag_conventions.yaml`.
# ---------------------------------------------------------------------------
_AMBIGUOUS = {c.upper() for c in (_CONV.get("ambiguous_service_codes") or [])}
_MULTI_SVC = {c for c in SERVICE if len(c) >= 2} - _AMBIGUOUS
_FRAGMENT = re.compile(r"^([A-Z]{2,3})[-\s]?0*(\d{2,6})[A-Z]?$")


def parse_fragment(text: str) -> LineNumber | None:
    """A clipped line number, e.g. `DC266` out of `1"-DC-11-266-P1A-HE`.

    Returns a LineNumber with `size_in` 0.0 — the size was on the part of the
    string OCR lost, and inventing one would be worse than admitting it.
    """
    t = _clean(text).rstrip("!.,;:")
    if parse(t) is not None:
        return None                     # a complete number is not a fragment
    m = _FRAGMENT.match(t)
    if m is None:
        return None
    svc, num = m.group(1), m.group(2)
    if svc not in _MULTI_SVC:
        return None
    return LineNumber(raw=t, size_in=0.0, service=svc,
                      service_name=SERVICE.get(svc, {}).get("name", "process line"),
                      number=num, area="", spec="", insulation="")


def is_line_fragment(text: str) -> bool:
    return parse_fragment(text) is not None


def fragment_entity(ln: LineNumber, idx: int, cx: float, cy: float, score: float,
                    box_px: int = 52) -> dict:
    """A partial line tag: a real thing on the drawing, but not fully read.

    It is flagged for review rather than dropped. Dropping it would lose a label
    the drawing really prints; keeping it as equipment was the bug.
    """
    e = to_entity(ln, idx, cx, cy, score, box_px)
    e["name"] = f"{ln.service_name} (partial tag)"
    e["needs_review"] = True
    e["partial"] = True
    e["confidence"] = 0.5
    return e
