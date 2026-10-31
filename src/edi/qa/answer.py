"""
Answer assembly: query -> Cypher -> graph -> evidence -> natural-language answer.

The defining requirement is that an answer is never bare text.  Every response
carries the graph rows it came from, the cropped drawing regions that show those
symbols, the Cypher that was run, and a confidence derived from the underlying
detection/OCR confidences.  An engineer has to be able to check the machine.
"""
from __future__ import annotations
import json, re, time
from pathlib import Path
import cv2
import numpy as np
import ollama

from ..config import load

CFG = load()


def _entities_from_result(kg, cypher: str, sheet_id: str, limit: int = 12) -> list[dict]:
    """Re-run the query shape asking for the symbols themselves, so we can show
    the reader which boxes on the drawing produced the answer."""
    vars_ = list(dict.fromkeys(re.findall(r"\(\s*(\w+)\s*:\s*Symbol", cypher)))
    if not vars_:
        return []
    body = cypher[:re.search(r"\bRETURN\b", cypher, re.I).start()] if re.search(r"\bRETURN\b", cypher, re.I) else cypher
    ret = ", ".join(f"{v}.entity_id AS {v}_id, {v}.class AS {v}_class, "
                    f"{v}.tag AS {v}_tag, {v}.x0 AS {v}_x0, {v}.y0 AS {v}_y0, "
                    f"{v}.x1 AS {v}_x1, {v}.y1 AS {v}_y1" for v in vars_)
    try:
        df = kg.run(kg.scoped(f"{body} RETURN {ret} LIMIT {limit}", sheet_id))
    except Exception:
        return []
    out, seen = [], set()
    for _, row in df.iterrows():
        for v in vars_:
            eid = row.get(f"{v}_id")
            if eid is None or eid in seen:
                continue
            seen.add(eid)
            out.append({"entity_id": eid, "class": int(row[f"{v}_class"]),
                        "tag": row[f"{v}_tag"],
                        "bbox": [int(row[f"{v}_x0"]), int(row[f"{v}_y0"]),
                                 int(row[f"{v}_x1"]), int(row[f"{v}_y1"])]})
    return out


def save_crops(image: np.ndarray, entities: list[dict], sheet_id: str,
               pad: int = 90, max_n: int = 6) -> list[dict]:
    out_dir = CFG.get_path("paths.interim") / sheet_id / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = image.shape[:2]
    regions = []
    for e in entities[:max_n]:
        x0, y0, x1, y1 = e["bbox"]
        a, b = max(0, x0 - pad), max(0, y0 - pad)
        c, d = min(W, x1 + pad), min(H, y1 + pad)
        crop = image[b:d, a:c]
        if crop.size == 0:
            continue
        vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop.copy()
        cv2.rectangle(vis, (x0 - a, y0 - b), (x1 - a, y1 - b), (0, 90, 220), 2)
        p = out_dir / f"{e['entity_id'].replace(':', '_')}.png"
        cv2.imwrite(str(p), vis)
        regions.append({"entity_id": e["entity_id"], "tag": e["tag"],
                        "class": e["class"], "bbox": e["bbox"], "crop": str(p)})
    return regions


def reasoning_path(kg, entities: list[dict], sheet_id: str) -> list[str]:
    """A short human-readable traversal between the first two answer entities."""
    if len(entities) < 2:
        return [e.get("tag") or f"class {e['class']}" for e in entities]
    a, b = entities[0], entities[1]
    try:
        df = kg.run(
            "MATCH p = (u:Symbol)-[:CONNECTED_TO* SHORTEST 1..3]-(v:Symbol) "
            "WHERE u.entity_id=$a AND v.entity_id=$b RETURN nodes(p) AS ns LIMIT 1",
            {"a": a["entity_id"], "b": b["entity_id"]})
        if len(df):
            names, seen_n = [], set()
            for n in df["ns"].iloc[0]:
                label = n.get("tag") or f"class {n.get('class')}"
                if label in seen_n:      # SHORTEST still permits repeats on cycles
                    continue
                seen_n.add(label); names.append(label)
            path = []
            for i, n in enumerate(names):
                path.append(n)
                if i < len(names) - 1:
                    path.append("connected_to")
            return path
    except Exception:
        pass
    return [a.get("tag") or f"class {a['class']}", "connected_to",
            b.get("tag") or f"class {b['class']}"]


def _render(rows, cols) -> str:
    """Render query output as a labelled table.

    Passing bare nested lists made the model report "no connections found" even
    when two rows were present; naming the columns fixed it.
    """
    if not rows:
        return "(no rows returned)"
    head = " | ".join(str(c) for c in cols) if cols else ""
    body = "\n".join(" | ".join("" if v is None else str(v) for v in r) for r in rows)
    return (f"{head}\n{'-' * max(len(head), 3)}\n{body}" if head else body) + \
           f"\n({len(rows)} row(s))"


def phrase(question: str, rows, cols=None, docs=None,
           model: str | None = None, host: str | None = None) -> str:
    client = ollama.Client(host=host or CFG.dotted("llm.host"))
    r = client.chat(model=model or CFG.dotted("llm.answer_model", "qwen3:8b"), think=False,
                    keep_alive="30m",
                    options={"temperature": 0.0, "num_predict": 220},
                    messages=[
                        {"role": "system", "content":
                         "You answer questions about engineering drawings from retrieved graph "
                         "results. Be concise and factual. State only what the results support. "
                         "Never invent tags or counts."},
                        {"role": "user", "content":
                         f"Question: {question}\n\nGraph query results:\n{_render(rows, cols)}\n"
                         + (("\nSupporting extracts from the reference documents "
                             "(cite the source only if you actually use it):\n"
                             + "\n".join(f"[{d['source']} p{d['page']}] {d['text'][:420]}"
                                         for d in docs[:3]) + "\n") if docs else "")
                         + "\nEvery row above is a real result. If rows are present, the answer "
                           "is NOT empty. The graph is authoritative for counts and connections; "
                           "documents may add specification detail but must never override the "
                           "graph. Answer in one or two sentences."},
                    ])
    return r["message"]["content"].strip()


def narrate(question: str, briefing: str, model: str | None = None,
            host: str | None = None) -> str:
    """Turn a grounded briefing into a conversational explanation.

    The model is given the FULL sheet briefing and told to reorganise it, not to
    add to it. This is the difference between "explain this drawing" answering
    about one pump — which is what happened when a single Cypher row was all the
    context there was — and answering about the drawing.
    """
    return "".join(narrate_stream(question, briefing, model=model, host=host))


# The system prompt is shared by the streaming and non-streaming paths.
# "what each one does" is deliberately hedged: a first tag letter tells you a
# vessel is a vessel, it does NOT tell you what that vessel is for. Letting the
# model fill that in produced "V-1101A ... receives condensate", which the
# drawing never said. On a rig an invented duty is a hazard, not a nicety.
# Answering a specific question from an assembled briefing. The overview prompt
# writes a tour of a drawing; this one answers what was actually asked, and is
# far stricter about the difference between "the drawing does not show this" and
# "here is a plausible-sounding paragraph".
_ANSWER_SYS = (
    "You are a process engineer answering a colleague's question about an "
    "engineering drawing. You are given a briefing containing everything the "
    "system knows that bears on the question — query results, item details, "
    "terminology and, where relevant, document extracts.\n"
    "Answer the question directly and conversationally, in a few short "
    "paragraphs. Lead with the answer, then the supporting detail.\n"
    "STRICT RULES:\n"
    "- Use ONLY tags, names, counts and numbers that appear in the briefing.\n"
    "- Never invent a tag, a count, a pressure, a temperature or an item.\n"
    "- Describe an item by WHAT IT IS. Do not invent what it is FOR or what it "
    "contains unless the briefing says so.\n"
    "- If the briefing does not answer the question, say plainly that the "
    "drawing does not show it, and then say what IS there that is related. "
    "Never pad a gap with generic engineering advice.\n"
    "- Do not mention the query, the graph, or that you were given a briefing. "
    "Speak about the drawing.\n"
    "- Match the length to the question. A count or a single fact is ONE "
    "sentence — give it and stop. Never restate the briefing back, never pad "
    "with what the drawing does not say, and never add a closing paragraph "
    "listing absent details. Only a genuinely open question earns 2-3 short "
    "paragraphs, and never more than 150 words.\n"
    "- Always finish your final sentence.")


_NARRATE_SYS = (
    "You are a process engineer explaining a P&ID to a colleague. You are given "
    "a complete, factual briefing extracted from the drawing. Explain it in "
    "clear, flowing prose, as you would out loud: what kind of process this "
    "appears to be, the main equipment, how the fluid travels through it, how "
    "it is controlled, and which items are safety-critical.\n"
    "STRICT RULES:\n"
    "- Use ONLY tags, names, counts and numbers that appear in the briefing.\n"
    "- Never invent a tag, a count, a pressure, a temperature or an item.\n"
    "- Describe an item by WHAT IT IS (its type). Do NOT invent what it is FOR, "
    "what it contains, or what it is connected to unless the briefing says so.\n"
    "- Do not add safety advice that is not in the briefing.\n"
    "- If something is not in the briefing, say it is not shown rather than "
    "guessing.\n"
    "LENGTH: be complete but tight — about 350-450 words, four short sections "
    "(what this is, the equipment, how it flows, control and safety). Do not "
    "list every item; describe the picture. ALWAYS finish your final sentence.")


# An empty briefing must never reach the model.
#
# When the workbench read the briefing from the wrong key it got "", asked the
# model to "explain this drawing", and the model invented an entire distillation
# column — HX-101, T-101, CV-101, a 1.2 m by 18 m column — none of it on any
# drawing. Fluent, confident, and completely fabricated. Any caller that loses
# its grounding must get a refusal, not a story.
_NO_GROUND = ("I do not have the drawing's details to answer from, so I am not "
              "going to guess. Reload the sheet and ask again.")


def _ungrounded(briefing: str) -> bool:
    return len((briefing or "").strip()) < 40


def narrate_stream(question: str, briefing: str, model: str | None = None,
                   host: str | None = None):
    """Yield the explanation token by token.

    Streaming is what makes a 40-second local model feel usable: the reader
    watches it write instead of staring at a spinner. Same grounding rules.
    """
    if _ungrounded(briefing):
        yield _NO_GROUND
        return
    client = ollama.Client(host=host or CFG.dotted("llm.host"))
    stream = client.chat(
        model=model or CFG.dotted("llm.answer_model", "qwen3:8b"),
        think=False, stream=True, keep_alive="30m",
        options={"temperature": 0.1, "num_predict": 1100},
        messages=[
            {"role": "system", "content": _NARRATE_SYS},
            {"role": "user", "content":
             f"Question: {question}\n\nBriefing extracted from the "
             f"drawing:\n\n{briefing}\n\nExplain this drawing."},
        ])
    for chunk in stream:
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            yield piece


# A result that is already an answer.
#
# "How many valves are on this drawing?" returns a single number. Handing that
# to a language model to phrase costs seconds and invites invention — measured:
# for a bare count of 16 instruments the model volunteered "the drawing includes
# various items such as valves, sensors, and controllers", none of which was in
# the briefing. A deterministic result deserves a deterministic sentence.
_COUNT_Q = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b", re.I)
_NOUN = re.compile(r"how many\s+([a-z][a-z\s\-]{2,40}?)\s*(?:are|is|do|does|on|in|"
                   r"connected|present|there|\?|$)", re.I)


def direct_answer(question: str, rows, cols) -> str:
    """Answer without a model when the rows already say it, else ''."""
    if not rows or len(rows) != 1 or len(rows[0]) != 1:
        return ""
    v = rows[0][0]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return ""
    if not _COUNT_Q.search(question or ""):
        return ""
    n = int(v)
    m = _NOUN.search(question or "")
    noun = (m.group(1).strip() if m else "items").rstrip("s") if m else "items"
    if m:
        noun = m.group(1).strip()
        if n != 1 and not noun.endswith("s"):
            noun += "s"
        elif n == 1 and noun.endswith("s"):
            noun = noun[:-1]
    return (f"There {'is' if n == 1 else 'are'} {n} {noun} on this drawing."
            if m else f"{n}.")


def answer_stream(question: str, briefing: str, model: str | None = None,
                  host: str | None = None):
    """Yield a conversational answer to any question from its briefing."""
    if _ungrounded(briefing):
        yield _NO_GROUND
        return
    client = ollama.Client(host=host or CFG.dotted("llm.host"))
    stream = client.chat(
        model=model or CFG.dotted("llm.answer_model", "qwen3:8b"),
        think=False, stream=True, keep_alive="30m",
        options={"temperature": 0.1, "num_predict": 420},
        messages=[
            {"role": "system", "content": _ANSWER_SYS},
            {"role": "user", "content":
             f"Question: {question}\n\nBriefing:\n\n{briefing}\n\n"
             f"Answer the question."},
        ])
    for chunk in stream:
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            yield piece


# Rows that say nothing. A query that returns only the tag you asked about has
# not answered anything — but handed to a language model it produces a fluent
# paragraph anyway. Measured on a real drawing: "what happens if I close
# SDV-11014?" returned one row containing the string 'SDV-11014' and the model
# wrote a confident description of what closing it would do. Nothing in the
# graph supported a word of it.
def _rows_are_informative(question: str, rows, cols) -> bool:
    if not rows:
        return False
    flat = [str(v).strip().upper() for r in rows for v in r if v is not None]
    if not flat:
        return False
    q = (question or "").upper()
    # every returned value is just an identifier that was already in the question
    return not all(v in q for v in flat)


def ask(question: str, kg, generator, sheet_id: str, image: np.ndarray | None = None,
        exclude_sheets: set | None = None, with_crops: bool = True,
        docstore=None, history: list[dict] | None = None,
        field_mode: bool = True, stream_overview: bool = False,
        stream_answer: bool = False) -> dict:
    t0 = time.time()

    # Field questions ("what is this", "how do I isolate it", "what does SDV
    # mean") are answered from the graph and the knowledge layer directly. They
    # are the questions people actually ask, they are not expressible as the
    # benchmark's query shapes, and they must never be improvised.
    if field_mode:
        from . import field as FIELD
        try:
            fr = FIELD.answer(question, kg, sheet_id,
                              image_size=(image.shape[1], image.shape[0])
                              if image is not None else None)
        except Exception:
            fr = None
        if fr is not None:
            # A whole-drawing briefing is complete but long and list-shaped.
            # The model rewrites it as an explanation. If the model is not
            # reachable the deterministic briefing still stands on its own, so
            # this can never leave the question unanswered.
            # A tag question whose card is grounding, not an answer: narrate
            # it against what was actually asked.
            if fr.get("handler") == "item_question":
                brief = fr["evidence"].get("briefing", fr["answer"])
                if stream_answer:
                    fr["answer"] = ""
                else:
                    try:
                        fr["answer"] = "".join(answer_stream(question, brief))
                    except Exception:
                        pass
            if fr.get("handler") == "overview":
                # Keep the grounded briefing: the UI streams the narration
                # itself so the reader watches it write, and falls back to
                # the briefing verbatim if the model is unreachable.
                fr["evidence"]["briefing"] = fr["answer"]
                if stream_overview:
                    fr["answer"] = ""          # the caller will stream it
                else:
                    try:
                        fr["answer"] = narrate(question, fr["answer"])
                    except Exception:
                        pass
            regions = fr["evidence"].get("regions") or []
            if image is not None and with_crops and regions:
                regions = save_crops(image, [{"entity_id": f"{sheet_id}:{r['tag']}",
                                              "tag": r["tag"], "class": 0,
                                              "bbox": r["bbox"]} for r in regions],
                                     sheet_id)
            return {"question": question, "answer": fr["answer"], "cypher": "",
                    "n_attempts": 0, "columns": [], "rows": [],
                    "reasoning_path": [], "handler": fr["handler"],
                    # The briefing MUST be surfaced at the top level. It used
                    # to live only under "field", so the workbench read
                    # evidence["briefing"], got "", and streamed a narration of
                    # NOTHING — the model duly invented a distillation column
                    # with HX-101, T-101 and CV-101, none of which are on any
                    # drawing. A caller that streams must never be handed an
                    # empty briefing.
                    "evidence": {"regions": regions,
                                 "graph": {"nodes": []},
                                 "documents": [],
                                 "briefing": fr["evidence"].get("briefing", ""),
                                 "field": fr["evidence"]},
                    "confidence": fr["confidence"], "error": "",
                    "latency_ms": int((time.time() - t0) * 1000)}
    gen = generator.generate(question, kg=kg, sheet_id=sheet_id,
                             exclude_sheets=exclude_sheets, history=history)
    cy = gen["cypher"]

    try:
        df = kg.run(kg.scoped(cy, sheet_id))
        rows = df.values.tolist()[:20]
        # Untagged symbols come back as '' and the model reads that as "nothing
        # found".  Label them by class name so the answer matches the evidence.
        rows = [[("(untagged)" if (isinstance(v, str) and not v.strip()) else v) for v in r]
                for r in rows]
        cols = list(df.columns)
        err = ""
    except Exception as e:
        rows, cols, err = [], [], f"{type(e).__name__}: {e}"[:200]

    # Third evidence leg: reference documents. Retrieval is lexical, so exact
    # identifiers in the question ("P-101") match the spec text that mentions them.
    docs = []
    if docstore is not None:
        try:
            docs = docstore.search(question, k=CFG.dotted("retrieval.top_k", 4))
        except Exception:
            docs = []

    ents = _entities_from_result(kg, cy, sheet_id) if not err else []
    regions = save_crops(image, ents, sheet_id) if (image is not None and with_crops and ents) else []
    path = reasoning_path(kg, ents, sheet_id) if ents else []

    # Every question gets the same treatment the overview handler gets: assemble
    # everything the system knows that bears on it, then let the model phrase
    # that. Previously a thin or empty result went straight to a bare refusal —
    # "the drawing does not contain an answer to that" — even when the item was
    # right there in the graph and the query had simply asked the wrong shape.
    from . import context as CTX
    briefing, ctx_ev = CTX.build(question, kg, sheet_id, rows=rows, cols=cols,
                                 cypher=cy if not err else "", docs=docs,
                                 image_size=((image.shape[1], image.shape[0])
                                             if image is not None else None))
    direct = direct_answer(question, rows, cols)
    informative = _rows_are_informative(question, rows, cols)
    grounded = (informative or bool(ctx_ev.get("cards")) or bool(docs)
                or bool(ctx_ev.get("keyword_hits")))

    if direct:
        answer = direct                  # deterministic: no model, no latency
    elif stream_answer and grounded:
        answer = ""                      # the caller streams it from `briefing`
    elif grounded:
        try:
            answer = "".join(answer_stream(question, briefing))
        except Exception:
            answer = phrase(question, rows, cols, docs) if informative else (
                "The drawing does not show an answer to that.")
    else:
        answer = ("The drawing, as it has been read, does not show an answer to "
                  "that. It may be on a connected sheet, or the item may not have "
                  "been extracted from this one — I am not going to guess.")
        if err:
            answer = ("I could not answer that — the query I generated for it "
                      "failed. Try rephrasing it.")

    conf = round(float(np.mean([0.9] + [1.0] * min(len(ents), 3))), 3) if ents else 0.3
    return {
        "question": question,
        "answer": answer,
        "handler": "cypher",
        "cypher": cy,
        "n_attempts": gen["n_attempts"],
        "columns": cols,
        "rows": rows,
        "reasoning_path": path,
        "evidence": {"regions": regions,
                     "graph": {"nodes": [e["entity_id"] for e in ents]},
                     "briefing": briefing,
                     "grounded": grounded,
                     "cards": ctx_ev.get("cards", []),
                     "documents": [{"source": d["source"], "page": d["page"],
                                    "score": d["score"], "excerpt": d["text"][:300]}
                                   for d in docs]},
        "confidence": conf,
        "error": err,
        "latency_ms": int((time.time() - t0) * 1000),
    }
