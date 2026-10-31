"""
Off-page connectors — where a drawing hands off to another drawing.

No P&ID stands alone. A line leaves the sheet edge labelled "TO 3RD STAGE
SEPARATOR" or joins "HP FLARE HEADER", and the rest of the story is on another
sheet. On a rig, most real questions cross sheets — "what happens to the gas
after the second stage?" cannot be answered from one drawing.

Three shapes appear on real drawings (all read off
ALPHA-AA-0000-PR-PD-0030-01 and ALPHA-AA-0000-PR-FD-0002-02):

    destination   TO 3RD STAGE · FROM LP CONDENSATE · TO LP COMPRESSOR
    header        HP FLARE HEADER · CLOSED DRAINS HEADER · ATMOS. VENT HEADER
    drawing ref   FD-0002-01 · PR-PD-0030-01

The difficulty is not finding them — it is telling them apart from note prose,
which is full of the same words: "ALL DRAIN LINES TO SLOPE TO DRAIN HEADER." is
a note, "FLARE HEADER" is a connector. Connectors are short, upper-case, and
carry no sentence punctuation; notes are sentences. Every rule below exists to
draw that line.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

# a destination label: TO / FROM followed by a short upper-case place
_DEST = re.compile(r"^(TO|FROM)\s+([A-Z0-9][A-Z0-9 ./&'\-]{2,28})$")
# a named header the sheet joins
_HEADER = re.compile(r"^([A-Z0-9][A-Z0-9 ./&'\-]{2,26})\s+HEADER$")
# a reference to another drawing
_DWGREF = re.compile(r"^([A-Z]{2,3})-(\d{3,4})-(\d{1,2})$")

# words that mark note prose rather than a label
_PROSE = re.compile(
    r"\b(SHALL|MUST|SHOULD|IS|ARE|BE|SLOPE|POCKETTED|MINIMUM|MAXIMUM|LATEST|"
    r"REQUIRED|PROVIDED|SUPPLIED|INSTALLED|DELETED|NOTE|REFER|SEE|ALL|EACH|"
    r"WITH|THAN|THAT|WHICH|FOR|AND)\b")

CATEGORY_CLASS = 105          # matches blueprint.CATEGORY_CLASS["connector"]

# Generic nouns carry no identity: "HP FLARE HEADER" and "TO FLARE" name the
# same destination, and matching them on the whole string finds nothing.
_GENERIC = {"HEADER", "SYSTEM", "LINE", "LINES", "STAGE", "LP", "HP", "THE",
            "AND", "OF", "UNIT", "PACKAGE", "SIDE"}


def core_tokens(target: str) -> frozenset[str]:
    """The distinctive words in a destination, for cross-sheet matching."""
    toks = {t for t in re.split(r"[^A-Z0-9]+", target.upper()) if len(t) > 1}
    keep = toks - _GENERIC
    return frozenset(keep or toks)


def _is_junk(target: str) -> bool:
    """OCR debris masquerading as a destination ("TO WE ME M U")."""
    toks = [t for t in re.split(r"[^A-Z0-9]+", target.upper()) if t]
    if not toks:
        return True
    short = sum(1 for t in toks if len(t) <= 2)
    return short >= 2 or (len(toks) >= 3 and short >= len(toks) / 2)


@dataclass
class Connector:
    raw: str
    kind: str            # destination | header | drawing_ref
    direction: str       # out | in | both
    target: str          # normalised destination text — the join key


def _clean(t: str) -> str:
    t = (t or "").strip().upper().replace("–", "-")
    return re.sub(r"\s+", " ", t).strip(" .,;:")


def _looks_like_prose(t: str) -> bool:
    """Notes are sentences; labels are not."""
    if len(t) > 34:
        return True
    if t.count(" ") >= 5:
        return True
    # a terminal period means it was a sentence before _clean stripped it
    return bool(_PROSE.search(t))


def parse(text: str) -> Connector | None:
    raw = (text or "").strip()
    # A connector label never ends in sentence punctuation. "ATMOS. VENT HEADER"
    # keeps its internal period; "TO VALVE." is the tail of a note.
    if raw.endswith((".", ",", ";", ":")):
        return None
    t = _clean(raw)
    if not t or len(t) < 4:
        return None

    m = _DWGREF.match(t)
    if m:
        return Connector(t, "drawing_ref", "both", t)

    m = _DEST.match(t)
    if m and not _looks_like_prose(m.group(2)):
        where = _clean(m.group(2))
        if _is_junk(where):
            return None
        return Connector(t, "destination",
                         "out" if m.group(1) == "TO" else "in", where)

    m = _HEADER.match(t)
    if m and not _looks_like_prose(t):
        return Connector(t, "header", "both", t)

    return None


def is_connector(text: str) -> bool:
    return parse(text) is not None


def to_entity(c: Connector, idx: int, cx: float, cy: float, score: float,
              box_px: int = 58) -> dict:
    name = {"destination": f"off-page connector ({c.direction})",
            "header": "shared header",
            "drawing_ref": "drawing reference"}[c.kind]
    return {
        "id": f"x{idx:04d}",
        "pidqa_class": CATEGORY_CLASS,
        "category": "connector",
        "name": name,
        "bbox": [int(cx - box_px), int(cy - box_px // 2),
                 int(cx + box_px), int(cy + box_px // 2)],
        "tag": c.raw,
        "confidence": 1.0,
        "tag_confidence": round(float(score), 3),
        "tag_location": "internal",
        "source": "blueprint",
        "connector": {"kind": c.kind, "direction": c.direction, "target": c.target},
        "target": c.target,
    }


def cross_sheet_links(sheets: dict[str, list[dict]]) -> list[dict]:
    """Join sheets that reference the same destination or header.

    Two drawings that both mention "HP FLARE HEADER" are plumbed together
    through it, even though neither drawing says so. This is what makes
    "where does the flare gas go?" answerable across a drawing set.
    """
    items = [(sid, e, core_tokens(e["target"]))
             for sid, ents in sheets.items() for e in ents
             if e.get("category") == "connector" and e.get("target")]

    links, seen = [], set()
    for i in range(len(items)):
        sa, ea, ta = items[i]
        for j in range(i + 1, len(items)):
            sb, eb, tb = items[j]
            if sa == sb or not ta or not tb:
                continue
            # One destination's distinctive words contained in the other's:
            # "TO FLARE" {FLARE} matches "HP FLARE HEADER" {FLARE}, while
            # "SUPPLY HEADER" and "RETURN HEADER" share nothing and do not.
            if not (ta <= tb or tb <= ta):
                continue
            key = (sa, ea["id"], sb, eb["id"])
            if key in seen:
                continue
            seen.add(key)
            via = ea["target"] if len(ea["target"]) <= len(eb["target"]) else eb["target"]
            links.append({
                "source_sheet": sa, "source": ea["id"],
                "target_sheet": sb, "target": eb["id"],
                "type": "CONTINUES_ON", "via": via,
                "confidence": 0.85 if ta == tb else 0.7,
            })
    return links
