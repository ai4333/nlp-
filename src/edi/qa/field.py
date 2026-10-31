"""
Field mode — answering the questions the person on the plant actually asks.

The benchmark this project is scored against asks "how many class 5 symbols are
on this drawing". Nobody on a rig has ever asked that. Run the questions a new
technician asks against the system as it stood and every one of them failed:

    "What does SDV mean?"                  -> "no information available"
    "What is connected to V-1101A?"        -> two items, both misread line labels
    "Which valves isolate P-1101A?"        -> no rows
    "What happens if I close SDV-11014?"   -> a fluent, invented answer

The last one is the reason this module exists. The graph returned a single row
containing nothing but the tag, and the language model wrote a plausible
paragraph about isolating the system. On a rig that is not a bad answer, it is a
hazard.

So field questions are routed BEFORE the text-to-Cypher path, to code that can
only say what it can prove:

    definition      -> the knowledge layer (ISA-5.1, glossary, tag conventions)
    "what is this"  -> an item card assembled from the graph, with its evidence
    isolation       -> the isolation planner, which also reports what it cannot close
    anything else   -> the existing Cypher path, but refusing to phrase an answer
                       when the query came back with nothing to phrase

Every answer states what it does not know. That is the whole point.
"""
from __future__ import annotations
import re

from .. import knowledge as K
from ..knowledge import glossary as G
from ..graph import isolation as ISO

TAG_RX = re.compile(r"\b([A-Z]{1,5}[-\s]?\d{2,6}(?:[A-Z](?:/[A-Z])*)?)\b")
LINE_RX = re.compile(r'\b\d{1,2}"?\s*-\s*[A-Z]{1,3}\s*-\s*\d+', re.I)

# "shut down" was in here and it swallowed "which instruments shut down the
# compressor" — an operator's question about trips, not a request to isolate.
# Isolation intent has to be explicit.
_ISOLATE = re.compile(
    r"\b(isolat\w*|lock ?out|loto|"
    r"valves? (?:do i|to|should i|would i|must i)? ?(?:close|shut)|"
    r"(?:which|what) valves? .*(?:close|shut|isolat)|"
    r"close .* to work|take (?:it )?out of service|de[- ]?pressuris\w*|"
    r"positive isolation|work on (?:it|this)|permit)\b", re.I)
# where does it go / where does it come from — the operator's question
_FLOW = re.compile(
    r"\b(where does .* (?:go|come from|end up)|where.*(?:goes|flows)|"
    r"upstream|downstream|what feeds|feeds? (?:into|from)|"
    r"flow (?:from|to|path|direction)|which way)\b", re.I)
_DEFINE = re.compile(
    r"\b(what (?:does|do) .* (?:mean|stand for)|what is a |what is an |"
    r"what'?s a |what'?s an |meaning of|define |explain (?:what )?a |"
    r"what does .* do)\b", re.I)
_WHATIS = re.compile(r"\b(what(?:'s| is| are)|which|tell me about|describe)\b", re.I)
# A bare identification request: "what is X", "tell me about X", "where is X".
# Nothing else in the sentence — no property being asked for, no consequence.
# The trailing token must be TAG-SHAPED — no spaces. Allowing whitespace in it
# meant "what is the design pressure of V-100" (24 chars, all letters and
# spaces) matched as a bare identification and returned a card instead of an
# answer. Whether it matched depended on how long the tag happened to be.
_BARE_ID = re.compile(
    r"^\W*(?:what(?:'s| is| are)|tell me about|describe|show me|where(?:'s| is)?|"
    r"locate|find)\s+(?:the\s+)?[A-Za-z0-9][A-Za-z0-9\-/.]{1,23}\s*\??\W*$", re.I)
_WHERE = re.compile(r"\b(where(?:'s| is| are)?|show me|locate|find)\b", re.I)
_DANGER = re.compile(r"\b(danger\w*|hazard\w*|safe|risk|flammable|toxic|hot|"
                     r"what'?s in|contents?|carry\w*)\b", re.I)
# Counting, listing and comparing are what the Cypher path is good at, and it
# stays with the Cypher path even when a tag is mentioned.
_QUANT = re.compile(r"\b(how many|count|number of|list (?:all|every)|total|"
                    r"larger than|bigger than|greater than|at least|or larger|"
                    r"more than|most|fewest|average)\b", re.I)

# "Explain the whole drawing." This was the biggest hole in the system: the
# question has no tag in it, so it fell through to Cypher, which wrote a query
# that returned ONE row, and the model then faithfully described one pump as
# though it were the whole sheet. The drawing as a whole is the single most
# natural thing to ask about and it had no handler at all.
_OVERVIEW = re.compile(
    r"\b(?:"
    r"(?:explain|describe|summar\w*|walk me through|tell me about|overview of|"
    r"what(?:'s| is)? (?:on|in|going on)|break ?down|read)\s+"
    r"(?:me\s+)?(?:the\s+|this\s+|that\s+|it\s+)?"
    r"(?:whole|entire|full|complete|overall|all)?\s*"
    r"(?:engineering\s+)?(?:drawing|sheet|p&?id|diagram|document|process|plant|"
    r"system|thing|everything)"
    r"|what (?:am i|are we) looking at"
    r"|what (?:is|does) this (?:drawing|sheet|p&?id|diagram)\b"
    r"|explain (?:everything|it all|the lot)"
    r"|give me (?:an? )?(?:overview|summary|rundown)"
    r"|what(?:'s| is) (?:on |in )?this (?:drawing|sheet|p&?id|diagram)"
    r")\b", re.I)


def find_tag(text: str, known: set[str] | None = None) -> str | None:
    """The equipment tag a question is about."""
    up = (text or "").upper()
    cands = [m.group(1).strip() for m in TAG_RX.finditer(up)]
    if known:
        for c in cands:
            if c in known:
                return c
        for c in cands:                      # tolerate a missing dash
            alt = re.sub(r"[-\s]", "", c)
            for k in known:
                if re.sub(r"[-\s]", "", k) == alt:
                    return k
    return cands[0] if cands else None


def sheet_tags(kg, sheet_id: str) -> set[str]:
    try:
        df = kg.run("MATCH (s:Symbol) WHERE s.sheet_id=$s AND s.tag<>'' "
                    "RETURN DISTINCT toUpper(s.tag) AS t", {"s": sheet_id})
        return set(df["t"].tolist())
    except Exception:
        return set()


# ───────────────────────────────────────────────────────────── the item card
def zone_of(bbox, width: int, height: int) -> str:
    """Approximate drawing zone, the way a border is lettered: 'C-4'."""
    if not width or not height:
        return ""
    cx = (bbox[0] + bbox[2]) / 2 / width
    cy = (bbox[1] + bbox[3]) / 2 / height
    col = min(8, max(1, int(cx * 8) + 1))
    row = "ABCDEF"[min(5, max(0, int(cy * 6)))]
    return f"{row}-{col}"


def item_card(kg, sheet_id: str, tag: str, size=None) -> dict | None:
    """Everything the drawing says about one item, in plain language."""
    t = (tag or "").strip().upper()
    df = kg.run(
        "MATCH (s:Symbol) WHERE s.sheet_id=$s AND toUpper(s.tag)=$t "
        "RETURN s.entity_id AS eid, s.tag AS tag, s.name AS nm, s.category AS ctg, "
        "s.size_in AS sz, s.x0 AS x0, s.y0 AS y0, s.x1 AS x1, s.y1 AS y1, "
        "s.tag_confidence AS tc, s.needs_review AS nr LIMIT 1",
        {"s": sheet_id, "t": t})
    if not len(df):
        return None
    r = df.iloc[0]
    bbox = [int(r.x0), int(r.y0), int(r.x1), int(r.y1)]

    known = K.explain(str(r.tag)) or {}
    card = {
        "tag": str(r.tag),
        "name": known.get("name") or str(r.nm),
        "category": str(r.ctg),
        "what_it_is": known.get("definition", ""),
        "why_it_matters": known.get("why_it_matters", ""),
        "source_of_definition": known.get("source", ""),
        "ambiguous": bool(known.get("ambiguous")),
        "bbox": bbox,
        "zone": zone_of(bbox, size[0], size[1]) if size else "",
        "tag_confidence": float(r.tc or 0.0),
        "needs_review": bool(r.nr),
        "sheet": sheet_id,
        "unknowns": [],
        "connected": [], "watched_by": [], "controls": [],
        "lines": [], "relief": [], "hazards": [],
        "flows_in_from": [], "flows_out_to": [],
    }

    nb = kg.run(
        "MATCH (a:Symbol)-[:CONNECTED_TO]-(b:Symbol) "
        "WHERE a.entity_id=$e AND b.sheet_id=$s "
        "RETURN DISTINCT b.tag AS tag, b.name AS nm, b.category AS ctg, "
        "b.size_in AS sz", {"e": str(r.eid), "s": sheet_id})
    for x in nb.itertuples():
        # The graph stores the name that ingestion assigned. Re-reading the tag
        # through the knowledge layer means a sheet ingested before a
        # classification fix still reads correctly here.
        kn = K.explain(str(x.tag)) if str(x.tag) else None
        item = {"tag": str(x.tag) or "(untagged)",
                "name": (kn.get("name") if kn else None) or str(x.nm),
                "category": (kn.get("category") if kn else None) or str(x.ctg)}
        if item["category"] == "line":
            svc = _service_of(str(x.tag))
            item["size_in"] = float(x.sz or 0)
            item["service"] = svc
            hz = G.hazard(svc) if svc else None
            if hz:
                item["hazard"] = hz
                if hz not in card["hazards"]:
                    card["hazards"].append(hz)
            card["lines"].append(item)
        elif _is_relief(str(x.tag), str(x.nm)):
            card["relief"].append(item)
        else:
            card["connected"].append(item)

    # Direction is the operator's question: where does this come from and where
    # does it go. It carries the confidence of the rule that decided it, because
    # some of those rules are strong (a connector that says TO) and one of them
    # is a layout convention.
    for rel, key in (("FEEDS_TO", "flows_out_to"),):
        up = kg.run(f"MATCH (a:Symbol)-[r:{rel}]->(b:Symbol) WHERE a.entity_id=$e "
                    f"RETURN DISTINCT b.tag AS tag, b.name AS nm, "
                    f"r.confidence AS cf, r.evidence AS ev", {"e": str(r.eid)})
        card[key] = [{"tag": str(x.tag) or "(untagged)", "name": str(x.nm),
                      "confidence": float(x.cf or 0), "because": str(x.ev)}
                     for x in up.itertuples()]
    dn = kg.run("MATCH (a:Symbol)-[r:FEEDS_TO]->(b:Symbol) WHERE b.entity_id=$e "
                "RETURN DISTINCT a.tag AS tag, a.name AS nm, r.confidence AS cf, "
                "r.evidence AS ev", {"e": str(r.eid)})
    card["flows_in_from"] = [{"tag": str(x.tag) or "(untagged)", "name": str(x.nm),
                              "confidence": float(x.cf or 0), "because": str(x.ev)}
                             for x in dn.itertuples()]

    ins = kg.run(
        "MATCH (i:Symbol)-[:CONTROLS]->(x:Symbol) WHERE x.entity_id=$e "
        "RETURN DISTINCT i.tag AS tag, i.name AS nm", {"e": str(r.eid)})
    for x in ins.itertuples():
        d = K.explain(str(x.tag)) or {}
        card["watched_by"].append({"tag": str(x.tag), "name": d.get("name") or str(x.nm),
                                   "what_it_does": d.get("definition", "")})

    drv = kg.run(
        "MATCH (x:Symbol)-[:CONTROLS]->(v:Symbol) WHERE x.entity_id=$e "
        "RETURN DISTINCT v.tag AS tag, v.name AS nm", {"e": str(r.eid)})
    card["controls"] = [{"tag": str(x.tag), "name": str(x.nm)} for x in drv.itertuples()]

    # be explicit about what the drawing does NOT tell you
    if not card["lines"]:
        card["unknowns"].append("No line numbers were read on this item, so the size "
                                "and contents of its connections are not known here.")
    if not card["watched_by"]:
        card["unknowns"].append("No instrument was found associated with this item on "
                                "this sheet.")
    if not card["flows_in_from"] and not card["flows_out_to"]:
        card["unknowns"].append("The drawing did not give enough evidence to say which "
                                "way the fluid flows through this item.")
    weak = [f for f in card["flows_in_from"] + card["flows_out_to"]
            if f["confidence"] < 0.6]
    if weak:
        card["unknowns"].append(
            "Some flow directions here rest on the layout convention that a P&ID "
            "reads left to right, not on anything the drawing states. They are "
            "marked with their confidence — confirm them before acting.")
    if card["needs_review"] or card["tag_confidence"] < 0.7:
        card["unknowns"].append("The tag was read with low confidence — confirm it "
                                "against the drawing before relying on it.")
    if card["ambiguous"]:
        card["unknowns"].append("These letters have more than one standard reading; "
                                "the alternative is noted above.")
    card["unknowns"].append("This is what one drawing shows. Anything that leaves the "
                            "sheet edge continues on another drawing.")
    return card


def _service_of(line_tag: str) -> str:
    from ..ingest import lines as LN
    ln = LN.parse(line_tag) or LN.parse_fragment(line_tag)
    return ln.service if ln else ""


def _is_relief(tag: str, name: str) -> bool:
    head = (tag or "").upper().split("-")[0].rstrip("0123456789")
    return head in ("PSV", "PRV", "RV", "ERV", "RD", "BDV") or \
        any(k in (name or "").lower() for k in ("relief", "safety valve", "blowdown"))


def card_to_text(card: dict) -> str:
    """The card as the answer a person reads."""
    L = [f"**{card['tag']} — {card['name']}**"]
    if card["what_it_is"]:
        L.append(card["what_it_is"])
    if card["why_it_matters"]:
        L.append(f"*On the job:* {card['why_it_matters']}")
    if card["zone"]:
        L.append(f"*Where on the sheet:* zone {card['zone']} of {card['sheet']}.")
    if card["lines"]:
        L.append("*Connected lines:* " + ", ".join(
            f"{l['tag']}" + (f" ({l['size_in']:g}\")" if l.get("size_in") else "")
            for l in card["lines"][:8]))
    if card["flows_in_from"]:
        L.append("*Flows in from:* " + ", ".join(
            f"{c['tag']} ({c['confidence']:.2f})" for c in card["flows_in_from"][:8]))
    if card["flows_out_to"]:
        L.append("*Flows out to:* " + ", ".join(
            f"{c['tag']} ({c['confidence']:.2f})" for c in card["flows_out_to"][:8]))
    if card["connected"]:
        L.append("*Connected to:* " + ", ".join(
            f"{c['tag']} ({c['name']})" for c in card["connected"][:8]))
    if card["watched_by"]:
        L.append("*Instruments on it:* " + ", ".join(
            f"{i['tag']} — {i['name']}" for i in card["watched_by"][:8]))
    if card["relief"]:
        L.append("*Relief / blowdown:* " + ", ".join(
            f"{x['tag']} ({x['name']})" for x in card["relief"]))
    for h in card["hazards"][:3]:
        L.append(f"*Hazard ({h['class']}):* {h['note']}")
    if card["unknowns"]:
        L.append("*What this does not tell you:* " + " ".join(card["unknowns"]))
    return "\n\n".join(L)


def _flow_confidence(card: dict) -> float:
    fl = card["flows_in_from"] + card["flows_out_to"]
    return round(min(f["confidence"] for f in fl), 2) if fl else 0.0


def flow_to_text(card: dict) -> str:
    """Direction, with the reason and the confidence for each leg.

    Some of these rules are strong (a connector printed with the word TO on it)
    and one of them is only the convention that drawings read left to right.
    Presenting them as equally certain would be the same mistake as inventing
    them, so each leg carries the rule that decided it.
    """
    L = [f"**Flow through {card['tag']} — {card['name']}**"]
    if card["flows_in_from"]:
        L.append("*Comes in from:*\n" + "\n".join(
            f"- **{f['tag']}**" + (f" — {f['name']}" if f["name"] else "")
            + f"  ({f['confidence']:.2f}, {f['because']})"
            for f in sorted(card["flows_in_from"],
                            key=lambda x: -x["confidence"])[:8]))
    if card["flows_out_to"]:
        L.append("*Goes out to:*\n" + "\n".join(
            f"- **{f['tag']}**" + (f" — {f['name']}" if f["name"] else "")
            + f"  ({f['confidence']:.2f}, {f['because']})"
            for f in sorted(card["flows_out_to"],
                            key=lambda x: -x["confidence"])[:8]))
    if not card["flows_in_from"] and not card["flows_out_to"]:
        L.append("The drawing does not give enough evidence to say which way the "
                 "fluid moves through this item. It is connected to "
                 + ", ".join(c["tag"] for c in (card["connected"] + card["lines"])[:6])
                 + ", but nothing on the sheet states a direction — no arrow, no "
                   "TO/FROM connector, no relief path, no service code that implies "
                   "one. I am not going to guess at it.")
    weak = [f for f in card["flows_in_from"] + card["flows_out_to"]
            if f["confidence"] <= 0.55]
    if weak:
        L.append("> The legs at 0.50 rest on the convention that a P&ID reads left "
                 "to right, not on anything the drawing states. Confirm them.")
    L.append("> Direction here is topological. It is what the drawing implies, not "
             "a hydraulic calculation.")
    return "\n\n".join(L)


def plan_to_text(p: ISO.Plan) -> str:
    L = [f"**Isolating {p.target} — {p.target_name}**"]
    if p.isolation_valves:
        L.append("*Close these, working outwards from the item:*\n" + "\n".join(
            f"- **{v.tag}**" + (f" — {v.name}" if v.name else "")
            + (f"  (reached via {' → '.join(v.via)})" if v.via else "")
            for v in p.isolation_valves))
    else:
        L.append("*No block valve was found on any branch on this sheet.*")
    if p.bleeds:
        L.append("*Bleed / drain points inside the isolation:* " +
                 ", ".join(b.tag for b in p.bleeds))
    if p.relief:
        L.append("*Relief and blowdown — do NOT block these in:* " +
                 ", ".join(f"{r.tag}" for r in p.relief))
    if p.open_branches:
        L.append("*Branches with no block valve on this sheet:* " + ", ".join(
            f"{b.tag}" for b in p.open_branches[:10]))
    if p.notes:
        L.append("\n".join(f"> {n}" for n in p.notes))
    return "\n\n".join(L)


# ─────────────────────────────────────────────────────────── the router
def sheet_overview(kg, sheet_id: str, size=None) -> dict:
    """Everything on one sheet, gathered for a whole-drawing explanation.

    The Cypher path could never answer "explain this drawing" because a single
    generated query returns one shape of row. An overview needs many shapes at
    once — the equipment list, the instruments, the lines and their services,
    the major flow paths, the control loops — so they are collected here in one
    pass and handed to the model together.
    """
    ov = {"sheet": sheet_id, "equipment": [], "instruments": [], "valves": [],
          "lines": [], "flow_paths": [], "loops": [], "hazards": [],
          "counts": {}, "systems": []}

    rows = kg.run(
        "MATCH (s:Symbol) WHERE s.sheet_id=$s "
        "RETURN s.tag AS tag, s.name AS nm, s.category AS ctg, s.size_in AS sz, "
        "s.x0 AS x0, s.y0 AS y0, s.x1 AS x1, s.y1 AS y1 "
        "ORDER BY s.x0", {"s": sheet_id})

    for x in rows.itertuples():
        tag = str(x.tag or "").strip()
        kn = K.explain(tag) if tag else None
        name = (kn.get("name") if kn else None) or str(x.nm or "")
        ctg = (kn.get("category") if kn else None) or str(x.ctg or "")
        bbox = [int(x.x0), int(x.y0), int(x.x1), int(x.y1)]
        item = {"tag": tag or "(untagged)", "name": name, "category": ctg,
                "zone": zone_of(bbox, size[0], size[1]) if size else ""}
        if ctg == "line":
            svc = _service_of(tag)
            item["size_in"] = float(x.sz or 0)
            item["service"] = svc
            hz = G.hazard(svc) if svc else None
            if hz and hz not in ov["hazards"]:
                ov["hazards"].append(hz)
            ov["lines"].append(item)
        elif ctg == "instrument":
            ov["instruments"].append(item)
        elif ctg == "valve" or _is_relief(tag, name):
            ov["valves"].append(item)
        elif ctg == "equipment":
            ov["equipment"].append(item)

    # The spine of the process: the longest directed chains the drawing states.
    fl = kg.run(
        "MATCH (a:Symbol)-[r:FEEDS_TO]->(b:Symbol) "
        "WHERE a.sheet_id=$s AND b.sheet_id=$s AND a.tag <> '' AND b.tag <> '' "
        "RETURN a.tag AS src, b.tag AS dst, r.confidence AS cf "
        "ORDER BY r.confidence DESC LIMIT 60", {"s": sheet_id})
    ov["flow_paths"] = [{"from": str(x.src), "to": str(x.dst),
                         "confidence": round(float(x.cf or 0), 2)}
                        for x in fl.itertuples()]

    lp = kg.run(
        "MATCH (i:Symbol)-[:CONTROLS]->(t:Symbol) "
        "WHERE i.sheet_id=$s AND i.tag <> '' "
        "RETURN DISTINCT i.tag AS inst, t.tag AS target LIMIT 40", {"s": sheet_id})
    ov["loops"] = [{"instrument": str(x.inst), "acts_on": str(x.target) or "(untagged)"}
                   for x in lp.itertuples()]

    ov["counts"] = {"equipment": len(ov["equipment"]),
                    "instruments": len(ov["instruments"]),
                    "valves": len(ov["valves"]),
                    "lines": len(ov["lines"]),
                    "directed_connections": len(ov["flow_paths"]),
                    "control_loops": len(ov["loops"])}
    return ov


def overview_to_text(ov: dict) -> str:
    """A grounded, deterministic briefing of the whole sheet.

    This is the fallback when no language model is reachable, and it is also the
    context the model narrates from. Every line here is read off the graph.
    """
    c = ov["counts"]
    out = [f"**This drawing — {ov['sheet']}**", "",
           f"It carries **{c['equipment']} equipment items**, "
           f"**{c['instruments']} instruments**, **{c['valves']} valves** and "
           f"**{c['lines']} identified lines**, with "
           f"{c['directed_connections']} connections where the direction of flow "
           f"could be established and {c['control_loops']} control links."]

    def _blk(title, items, fmt, limit=14):
        if not items:
            return
        out.append("")
        out.append(f"**{title}**")
        for i in items[:limit]:
            out.append(f"- {fmt(i)}")
        if len(items) > limit:
            out.append(f"- …and {len(items) - limit} more")

    _blk("Equipment", ov["equipment"],
         lambda i: f"`{i['tag']}` — {i['name']}"
                   + (f" (zone {i['zone']})" if i.get("zone") else ""))
    _blk("Valves", ov["valves"], lambda i: f"`{i['tag']}` — {i['name']}", limit=10)
    _blk("Instruments", ov["instruments"],
         lambda i: f"`{i['tag']}` — {i['name']}", limit=10)

    # Lines are summarised by service rather than listed. Forty-seven individual
    # line numbers pushed the briefing long enough that the model wrote an essay
    # and ran out of budget mid-sentence; the services are what actually tell
    # you what the plant handles.
    named = [l for l in ov["lines"] if l["tag"] != "(untagged)"]
    if named:
        by_svc: dict[str, int] = {}
        for l in named:
            by_svc[l.get("service") or "unmarked"] = \
                by_svc.get(l.get("service") or "unmarked", 0) + 1
        sizes = sorted({l["size_in"] for l in named if l.get("size_in")})
        out.append("")
        out.append(f"**Piping** — {len(named)} tagged lines, by service: "
                   + ", ".join(f"{k} ×{v}" for k, v in
                               sorted(by_svc.items(), key=lambda x: -x[1])[:8]))
        if sizes:
            out.append(f"Line sizes present: "
                       + ", ".join(f'{x:g}"' for x in sizes[:10]))

    if ov["flow_paths"]:
        out.append("")
        out.append("**How it flows** (highest-confidence routes)")
        for f in ov["flow_paths"][:10]:
            out.append(f"- {f['from']} → {f['to']}  ({f['confidence']:.2f})")
        if len(ov["flow_paths"]) > 10:
            out.append(f"- …and {len(ov['flow_paths']) - 10} more")

    if ov["loops"]:
        out.append("")
        out.append("**Control**")
        for l in ov["loops"][:8]:
            out.append(f"- {l['instrument']} acts on {l['acts_on']}")

    if ov["hazards"]:
        out.append("")
        out.append("**Hazards present on this sheet**")
        for h in ov["hazards"][:8]:
            out.append(f"- {h}")

    out.append("")
    out.append("*This is what one drawing shows. Anything crossing the sheet "
               "edge continues on another drawing, and items the reader could "
               "not tag are not listed above.*")
    return "\n".join(out)


def route(question: str, kg, sheet_id: str, known_tags: set[str] | None = None) -> str:
    """Which handler should take this question. '' means fall through to Cypher."""
    q = question or ""
    tag = find_tag(q, known_tags)
    # A whole-drawing question wins even though it also matches "what is" —
    # but only when it does not name a specific item, so "explain V-1101A"
    # still goes to the item card.
    if _OVERVIEW.search(q) and not tag:
        return "overview"
    if _ISOLATE.search(q):
        return "isolation" if tag else "isolation_no_tag"
    if _QUANT.search(q):
        return ""                        # counting and listing: Cypher does it better
    if tag and _FLOW.search(q):
        return "flow"
    if _DEFINE.search(q) and not tag:
        return "definition"
    if tag and (_WHATIS.search(q) or _WHERE.search(q) or _DANGER.search(q)):
        # "What is V-1101A" is answered BY the card — it is an identification
        # request and the card is exactly that, instantly. But "what is the
        # design pressure of V-1101A" is a different question that merely
        # contains a tag, and returning the card for it answered a question
        # nobody asked. Anything beyond bare identification becomes a briefing
        # for the model to answer from.
        if _BARE_ID.search(q):
            return "item"
        return "item_question"
    if _DEFINE.search(q):
        return "definition"
    # A question about an item that is really on this sheet, which none of the
    # patterns above claimed — "what should I check before I open BDV-11350",
    # "is P-1101A running". The item card is grounded and complete; the Cypher
    # path on these produced a single uninformative row and an invented answer.
    if tag and known_tags and tag in known_tags:
        # Bare identification is the card; anything else is a real question
        # that happens to name an item, and gets answered from the card.
        return "item" if _BARE_ID.search(q) else "item_question"
    return ""


def answer(question: str, kg, sheet_id: str, image_size=None,
           known_tags: set[str] | None = None) -> dict | None:
    """Answer a field question, or None to let the Cypher path handle it."""
    known_tags = known_tags if known_tags is not None else sheet_tags(kg, sheet_id)
    kind = route(question, kg, sheet_id, known_tags)
    if not kind:
        return None
    tag = find_tag(question, known_tags)

    if kind == "item_question":
        # The card is the grounding, not the answer. The caller narrates it
        # against the actual question asked.
        card = item_card(kg, sheet_id, tag, size=image_size)
        if card is None:
            return None                  # let the query path try
        # The formatted card carries a long tail of caveats meant for a human
        # reader. As model input that is dead weight — it lengthened the prompt
        # and the model echoed the caveats back instead of answering. The
        # compact fact list is what the model needs.
        from .context import _card_lines
        brief = "\n".join(_card_lines(card))
        return {"handler": "item_question", "answer": brief,
                "evidence": {"card": card, "briefing": brief,
                             "regions": [{"tag": card["tag"],
                                          "bbox": card["bbox"]}]},
                "confidence": 0.75}

    if kind == "overview":
        ov = sheet_overview(kg, sheet_id, size=image_size)
        if not any(ov["counts"].values()):
            return {"handler": "overview", "answer":
                    "Nothing has been read off this drawing yet, so there is "
                    "nothing for me to describe.",
                    "evidence": {}, "confidence": 0.0}
        regions = [{"tag": e["tag"], "bbox": [0, 0, 0, 0]}
                   for e in ov["equipment"][:6] if e["tag"] != "(untagged)"]
        brief = overview_to_text(ov)
        return {"handler": "overview", "answer": brief,
                "evidence": {"overview": ov, "briefing": brief, "regions": []},
                "confidence": 0.85}

    if kind == "definition":
        term = K.strip_question(question)
        info = K.explain(term) or (K.explain(tag) if tag else None)
        if not info:
            terms = K.jargon_in(question)
            info = terms[0] if terms else None
        if not info:
            return {"handler": "definition", "answer":
                    f"I do not have an authoritative definition for “{term}”. "
                    f"I will not guess at plant terminology — ask the area authority "
                    f"or check the project legend sheet.",
                    "evidence": {}, "confidence": 0.0}
        body = [f"**{info.get('term', term)} — {info.get('name','')}**".rstrip(" —"),
                info["definition"]]
        if info.get("why_it_matters"):
            body.append(f"*On the job:* {info['why_it_matters']}")
        body.append(f"*Source:* {info['source']}.")
        return {"handler": "definition", "answer": "\n\n".join(body),
                "evidence": {"knowledge": info}, "confidence": 0.95}

    if kind in ("isolation", "isolation_no_tag"):
        if kind == "isolation_no_tag":
            return {"handler": "isolation", "answer":
                    "Tell me which item you are isolating — a tag such as P-1101A — "
                    "and I will trace the block valves around it from the drawing.",
                    "evidence": {}, "confidence": 0.0}
        p = ISO.plan(kg, sheet_id, tag)
        if p is None:
            return {"handler": "isolation", "answer":
                    f"{tag} is not on this drawing, so I cannot trace its isolation. "
                    f"Check the tag, or select the sheet it appears on.",
                    "evidence": {}, "confidence": 0.0}
        return {"handler": "isolation", "answer": plan_to_text(p),
                "evidence": {"isolation": p.dict()},
                "confidence": 0.8 if p.isolation_valves else 0.3}

    if kind == "flow":
        card = item_card(kg, sheet_id, tag, size=image_size)
        if card is None:
            return {"handler": "flow", "answer":
                    f"{tag} is not on this drawing, so I cannot trace its flow.",
                    "evidence": {}, "confidence": 0.0}
        return {"handler": "flow", "answer": flow_to_text(card),
                "evidence": {"card": card,
                             "regions": [{"tag": card["tag"], "bbox": card["bbox"]}]},
                "confidence": _flow_confidence(card)}

    if kind == "item":
        card = item_card(kg, sheet_id, tag, size=image_size)
        if card is None:
            info = K.explain(tag)
            if info:
                return {"handler": "definition", "answer":
                        f"**{tag}** is not on this drawing. From its tag alone it would "
                        f"be a {info.get('name','')}. {info['definition']}",
                        "evidence": {"knowledge": info}, "confidence": 0.4}
            return {"handler": "item", "answer":
                    f"{tag} is not on this drawing and its tag does not decode to "
                    f"anything I know. Check the tag or the sheet.",
                    "evidence": {}, "confidence": 0.0}
        return {"handler": "item", "answer": card_to_text(card),
                "evidence": {"card": card, "regions": [{"tag": card["tag"],
                                                        "bbox": card["bbox"]}]},
                "confidence": 0.85 if not card["needs_review"] else 0.5}
    return None
