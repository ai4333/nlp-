"""
Grounded context assembly for any question.

The overview handler proved the point: the model was never the weak part, the
context was. "Explain this drawing" answered about one pump because one row was
all it was given. Every other question had the same disease in a milder form —
a single generated query returns a single shape of row, and a thin row makes a
thin answer.

This module assembles everything the system actually knows that bears on a
question, before any language model is involved:

    the query result            what the graph was asked and what came back
    item cards                  for every tag named in the question or returned
    definitions                 ISA-5.1 and plant terminology for the jargon used
    neighbours                  what those items connect to, and which way it flows
    sheet context               what this drawing is, so an answer can be situated
    documents                   supporting extracts from the reference set

Nothing here is generated. Every line is read off the graph or the knowledge
layer, so the model's job is reduced to phrasing — which is the job it is
actually reliable at.
"""
from __future__ import annotations
import re

from . import field as F
from ..knowledge import __init__ as _K_mod  # noqa: F401  (kept for parity)
from .. import knowledge as K


# Words that carry no search value in a question about a drawing.
_STOP = {"what", "which", "where", "does", "this", "that", "drawing", "sheet",
         "there", "here", "with", "from", "into", "have", "has", "the", "and",
         "for", "are", "is", "on", "to", "of", "it", "me", "show", "tell",
         "about", "many", "much", "any", "all", "some", "would", "could",
         "should", "will", "can", "do", "did", "how", "why", "when", "who"}

MAX_CARDS = 3
MAX_ROWS = 20


def _fmt_rows(rows, cols) -> str:
    if not rows:
        return ""
    head = " | ".join(str(c) for c in cols) if cols else ""
    body = "\n".join(
        " | ".join("" if v is None else str(v) for v in r) for r in rows[:MAX_ROWS])
    out = (f"{head}\n{'-' * max(len(head), 3)}\n{body}" if head else body)
    return out + f"\n({len(rows)} row(s) returned)"


def _tags_in(question: str, rows, known: set[str]) -> list[str]:
    """Tags named in the question, plus tags that came back in the results."""
    found: list[str] = []
    t = F.find_tag(question, known)
    if t:
        found.append(t)
    for r in (rows or [])[:MAX_ROWS]:
        for v in r:
            if not isinstance(v, str):
                continue
            cand = F.find_tag(v, known)
            if cand and cand not in found:
                found.append(cand)
            if len(found) >= MAX_CARDS:
                return found[:MAX_CARDS]
    return found[:MAX_CARDS]


def _card_lines(card: dict) -> list[str]:
    """One item, compressed to the facts that answer questions about it."""
    out = [f"- {card['tag']} — {card['name']} ({card['category']})"
           + (f", zone {card['zone']}" if card.get("zone") else "")]
    if card.get("what_it_is"):
        out.append(f"    what it is: {card['what_it_is']}")
    for key, label in (("flows_in_from", "flow comes from"),
                       ("flows_out_to", "flow goes to")):
        if card.get(key):
            out.append(f"    {label}: " + ", ".join(
                f"{x['tag']} ({x.get('confidence', 0):.2f})" if isinstance(x, dict)
                and x.get("confidence") is not None else str(x.get("tag", x))
                for x in card[key][:5]))
    if card.get("connected"):
        out.append("    connected to: " + ", ".join(
            f"{c['tag']} ({c['name']})" for c in card["connected"][:6]))
    if card.get("lines"):
        out.append("    lines on it: " + ", ".join(
            f"{l['tag']}" + (f' {l["size_in"]:g}"' if l.get("size_in") else "")
            for l in card["lines"][:6]))
    if card.get("watched_by"):
        out.append("    instruments: " + ", ".join(
            f"{i['tag']} ({i['name']})" for i in card["watched_by"][:6]))
    if card.get("relief"):
        out.append("    relief devices: " + ", ".join(
            r["tag"] for r in card["relief"][:4]))
    if card.get("hazards"):
        out.append("    hazards: " + "; ".join(
            h.get("note", "") for h in card["hazards"][:3]))
    return out


def build(question: str, kg, sheet_id: str, rows=None, cols=None,
          cypher: str = "", docs=None, image_size=None,
          known_tags: set[str] | None = None) -> tuple[str, dict]:
    """Assemble the briefing. Returns (briefing_text, evidence_dict)."""
    known = known_tags if known_tags is not None else F.sheet_tags(kg, sheet_id)
    ev: dict = {"cards": [], "definitions": [], "has_rows": bool(rows)}
    parts: list[str] = [f"Drawing: {sheet_id}"]

    if rows:
        parts.append("\nWhat the graph returned for this question:")
        parts.append(_fmt_rows(rows, cols))

    tags = _tags_in(question, rows, known)
    cards = []
    for t in tags:
        try:
            c = F.item_card(kg, sheet_id, t, size=image_size)
        except Exception:
            c = None
        if c:
            cards.append(c)
    if cards:
        parts.append("\nWhat the drawing says about the items involved:")
        for c in cards:
            parts.extend(_card_lines(c))
            ev["cards"].append({"tag": c["tag"], "name": c["name"],
                                "bbox": c.get("bbox")})

    # Terminology the question used, defined from ISA-5.1 and the plant glossary
    # rather than from the model's memory.
    seen_terms = set()
    defs = []
    for info in (K.jargon_in(question) or [])[:3]:
        key = info.get("term", "")
        if key and key not in seen_terms:
            seen_terms.add(key)
            defs.append(info)
    if defs:
        parts.append("\nTerminology used in the question:")
        for d in defs:
            parts.append(f"- {d.get('term','')} — {d.get('name','')}: "
                         f"{d.get('definition','')}")
            ev["definitions"].append({"term": d.get("term"), "name": d.get("name")})

    if docs:
        parts.append("\nSupporting extracts from the reference documents:")
        for d in docs[:3]:
            parts.append(f"- [{d['source']} p{d['page']}] {d['text'][:400]}")

    # No tag and no rows does not mean nothing is known. "What relieves to the
    # flare on this drawing?" names no tag, so no card was built, and the
    # generated query missed — yet the sheet plainly holds relief valves and a
    # flare header. Search the graph for the words the question actually used.
    if not rows and not cards:
        words = [w for w in re.findall(r"[a-z]{4,}", (question or "").lower())
                 if w not in _STOP]
        hits: list[dict] = []
        for w in words[:4]:
            try:
                df = kg.run(
                    "MATCH (s:Symbol) WHERE s.sheet_id=$s AND "
                    "(toLower(s.name) CONTAINS $w OR toLower(s.tag) CONTAINS $w) "
                    "RETURN s.tag AS tag, s.name AS nm, s.category AS ctg LIMIT 8",
                    {"s": sheet_id, "w": w})
            except Exception:
                continue
            for x in df.itertuples():
                item = {"tag": str(x.tag) or "(untagged)", "name": str(x.nm),
                        "category": str(x.ctg), "matched": w}
                if item not in hits:
                    hits.append(item)
        if hits:
            parts.append("\nItems on this drawing matching the words in the "
                         "question:")
            for h in hits[:12]:
                parts.append(f"- {h['tag']} — {h['name']} ({h['category']}) "
                             f"[matched \"{h['matched']}\"]")
            ev["keyword_hits"] = hits[:12]

    if not rows and not cards and not ev.get("keyword_hits"):
        # Nothing matched. Say what the sheet DOES hold so the answer can be
        # useful about the gap instead of a bare refusal.
        try:
            ov = F.sheet_overview(kg, sheet_id, size=image_size)
            c = ov["counts"]
            parts.append(
                f"\nNothing on this drawing matched the question. For context, "
                f"the sheet holds {c['equipment']} equipment items, "
                f"{c['instruments']} instruments, {c['valves']} valves and "
                f"{c['lines']} tagged lines.")
            named = [e["tag"] for e in ov["equipment"] if e["tag"] != "(untagged)"]
            if named:
                parts.append("Equipment present: " + ", ".join(named[:12]))
            ev["fallback_overview"] = True
        except Exception:
            pass

    if cypher:
        parts.append(f"\n(The query that was run: {cypher})")

    return "\n".join(parts), ev
