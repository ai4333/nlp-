"""
The knowledge layer — what things ARE, as opposed to what is connected to what.

The graph knows the plant. It does not know the trade. Asked "what does SDV
mean?" the system used to write `MATCH (s:Symbol) WHERE s.tag='SDV'`, find
nothing, and say "no information available" — while `config/tag_conventions.yaml`
sat in the repository with the answer in it.

Four sources are tried in order, and each is deterministic:

  1 `glossary`  — curated plant terminology, written for a newcomer
  2 `isa`       — ISA-5.1 decoding, which explains any instrument tag ever printed
  3 line numbers — service, size and spec read out of the printed line tag
  4 tag prefixes — the customer's own equipment convention

If none of them knows, this layer says so. It never guesses, because a
confidently wrong definition of "blowdown" is a hazard, not a bad answer.
"""
from __future__ import annotations
import re

from . import glossary as G
from . import isa as ISA

_QUESTION = re.compile(
    r"^\s*(?:what(?:'s| is| are| does)?|whats|who|explain|define|meaning of|"
    r"tell me about|describe)\s+(?:the\s+|a\s+|an\s+)?", re.I)
_TAIL = re.compile(r"\s*(?:mean|means|stand for|do|does|is|are|used for|for)?\s*[?.!]*\s*$", re.I)


def strip_question(text: str) -> str:
    """'What does SDV mean?' -> 'SDV'."""
    t = _QUESTION.sub("", (text or "").strip())
    t = _TAIL.sub("", t)
    return t.strip().strip('"\'')


def explain(term: str) -> dict | None:
    """Explain a term or a tag. None when nothing authoritative is known."""
    t = (term or "").strip()
    if not t:
        return None

    hit = G.lookup(t)
    if hit:
        return {"kind": "term", "name": hit["term"], "category": "", **hit}

    from ..ingest import lines as LN

    # A complete line number is unambiguous — it carries its size and spec.
    full = LN.parse(t)
    if full:
        return _line_answer(t, full, partial=False)

    # The industry convention table (SDV, PSV, BDV, ZLO …) outranks everything:
    # these are the tags people actually say out loud, and they do not follow
    # the strict standard.
    if ISA.split(t)[0] in ISA.CONVENTION:
        return _tag_answer(ISA.decode(t))

    # A clipped line number ("DC266"). Some of these also decode as ISA letters
    # — DC is both "closed drain" and "density controller" — so when both
    # readings exist, give the pipe reading and say the other is possible rather
    # than silently picking one.
    frag = LN.parse_fragment(t)
    if frag:
        out = _line_answer(t, frag, partial=True)
        alt = ISA.decode(t)
        if alt and alt.source == "isa":
            out["ambiguous"] = True
            out["alternative"] = alt.name
            out["definition"] += (f" Read strictly as an instrument tag these letters "
                                  f"would mean '{alt.name}', so confirm against the "
                                  f"drawing before acting on it.")
        return out

    d = ISA.decode(t)
    if d:
        return _tag_answer(d)

    from ..ingest.blueprint import classify
    name, cat = classify(t)
    if cat != "unknown":
        # "vessel / separator" is not a glossary key, but "separator" is — the
        # convention table writes descriptive phrases, so match on the words.
        note = G.lookup(name)
        if note is None:
            for w in re.split(r"[^a-zA-Z]+", name):
                if len(w) > 3:
                    note = G.lookup(w)
                    if note:
                        break
        return {"kind": "tag", "term": t.upper(), "name": name, "category": cat,
                "definition": ((note["definition"] + " (Identified from the tag prefix.)")
                               if note else
                               f"A {name}, identified from the tag prefix."),
                "why_it_matters": note["why_it_matters"] if note else "",
                "source": "tag convention"}
    return None


def _tag_answer(d) -> dict:
    out = {"kind": "tag", "term": d.tag, "name": d.name, "category": d.category,
           "definition": d.plain or d.name, "source": d.source,
           "code": d.code, "loop": d.loop, "variable": d.variable,
           "function": d.function, "modifier": d.modifier,
           "on_action": d.on_action}
    extra = G.lookup(d.name) or G.lookup(d.code)
    if extra:
        out["why_it_matters"] = extra["why_it_matters"]
    return out


def _line_answer(t: str, ln, partial: bool) -> dict:
    size = f'{ln.size_in:g}" ' if ln.size_in else ""
    note = ("" if not partial else
            " The size and specification were not readable on the drawing, so this "
            "tag is only partially known — treat it as unconfirmed.")
    return {"kind": "line", "term": t.upper(), "name": f"{size}{ln.service_name}",
            "category": "line",
            "definition": (f"A pipe line number: {size}line carrying "
                           f"{ln.service_name}"
                           + (f", area {ln.area}" if ln.area else "")
                           + (f", piping spec {ln.spec}" if ln.spec else "")
                           + (", insulated or heat traced" if ln.insulation else "")
                           + f", line sequence {ln.number}.{note}"),
            "why_it_matters": G.GLOSSARY["line number"][1],
            "source": "line number", "partial": partial}


def jargon_in(text: str) -> list[dict]:
    """Glossary terms mentioned in a sentence — used to gloss an answer."""
    return G.search(text)
