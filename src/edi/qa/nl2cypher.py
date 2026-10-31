"""
Natural language -> Cypher, using a local LLM (qwen3:8b via Ollama).

Three things make this work without a paid model:
  1. A schema prompt that includes the symbol ontology, so the model can map
     "ball valve" -> class 5 instead of guessing.
  2. Few-shot examples selected per question (see retrieval/fewshot.py), mixing
     the nearest matches with one exemplar of each question type so all four
     PIDQA patterns are always demonstrated.
  3. A validate-and-repair loop: the generated query is executed, and if the
     engine rejects it the error text is fed back for one more attempt.  Cheap,
     and it converts most syntax failures into successes.
"""
from __future__ import annotations
import re
from pathlib import Path
import yaml
import ollama

from ..config import load

CFG = load()
_ONT = yaml.safe_load(open(Path(__file__).resolve().parents[3] / "config" / "symbol_ontology.yaml"))
CLASSES = _ONT["classes"]

_FENCE = re.compile(r"```(?:cypher|sql)?\s*(.*?)```", re.S | re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def schema_prompt(sample_tags: list[str] | None = None,
                  sample_names: list[str] | None = None) -> str:
    """Describe the schema, grounded in what this sheet actually contains.

    Two kinds of sheet exist. Trained-detector sheets use the synthetic 1..32
    symbol ontology. Blueprint-mode sheets (a real customer drawing, no trained
    detector) have no such legend — items are identified by tag prefix, so the
    model must filter on `name`, not on a class number. Passing the sheet's real
    vocabulary is what stops it inventing `WHERE s.class = 22` on a drawing that
    has no class 22.
    """
    if sample_names:
        lines = [f"  {n}" for n in sample_names]
    else:
        lines = [f"  class {c}: {v['name']} ({v['category']})"
                 for c, v in sorted(CLASSES.items())]
    tagnote = ""
    if sample_tags:
        # Grounding the prompt in tags that actually exist stops the model
        # splitting a printed tag: it produced `tag = '101'` for "SDL 101"
        # until real examples were shown.
        tagnote = ("\n  A tag is stored as the COMPLETE printed string, including any prefix\n"
                   "  and the space. Real tags in this drawing: "
                   + ", ".join(f"'{t}'" for t in sample_tags[:12]) + "\n")
    return (
        "GRAPH SCHEMA\n"
        "  (:Symbol {entity_id:STRING, class:INT64, name:STRING, tag:STRING,\n"
        "            category:STRING, size_in:DOUBLE, sheet_id:STRING,\n"
        "            confidence:DOUBLE})\n"
        "  category is one of: equipment · valve · instrument · line ·\n"
        "    annotation · connector · fitting · actuator\n"
        "  size_in is the nominal pipe size in inches (lines only; 0 otherwise)\n"
        "  (:Symbol)-[:CONNECTED_TO]-(:Symbol)   // undirected; write it with no arrow\n"
        "  (:Symbol)-[:CONTROLS]->(:Symbol)      // DIRECTED: instrument -> valve\n"
        "  (:Symbol)-[:FEEDS_TO]->(:Symbol)      // DIRECTED: upstream -> downstream\n"
        "  (:Symbol)-[:CONTINUES_ON]-(:Symbol)   // off-page: this drawing hands\n"
        "                                        // off to another drawing\n\n"
        + ("ITEM TYPES ON THIS DRAWING — filter with s.name, NOT s.class\n"
           if sample_names else "SYMBOL CLASSES (class numbers are 1..32)\n")
        + "\n".join(lines) + "\n\n"
        "RULES\n"
        "  - Return ONLY a Cypher query. No prose, no markdown fence, no explanation.\n"
        "  - Counting questions must RETURN count(...).\n"
        "  - Yes/no connectivity questions must RETURN a boolean, e.g. COUNT(s2) > 0.\n"
        "  - Tag questions must RETURN s.tag.\n"
        "  - String literals MUST be quoted: s.tag STARTS WITH 'ZLO'.\n"
        "  - Always traverse CONNECTED_TO undirected: -[:CONNECTED_TO]- (never -> or <-).\n"
        "  - CONTROLS and FEEDS_TO ARE directed: use -[:CONTROLS]-> with the arrow.\n"
        "  - Never split a tag: match the whole string, e.g. s.tag = 'SDL 101'.\n"
        "  - To select a KIND of item use category, never the name:\n"
        "      right:  WHERE s.category = 'instrument'\n"
        "      wrong:  WHERE s.name CONTAINS 'instrument'   (names say\n"
        "              'restriction orifice', never the word 'instrument')\n"
        "  - Item NAMES are descriptive phrases ('vessel / separator', 'pig\n"
        "    launcher / receiver'), so ALWAYS match them with CONTAINS and a\n"
        "    single distinctive word - never with '='.\n"
        "      right:  WHERE s.name CONTAINS 'vessel'\n"
        "      wrong:  WHERE s.name = 'vessel'\n"
        "  - Express neighbours as a plain pattern, NOT a comprehension:\n"
        "      MATCH (a:Symbol)-[:CONNECTED_TO]-(b:Symbol) WHERE a.tag='X' RETURN b.tag\n"
        "  - FORBIDDEN: list comprehensions [x IN ...], pattern comprehensions\n"
        "    [ (a)-[:R]-(b) | ... ], CASE, EXISTS{...}, subqueries, APOC.\n"
        + tagnote
    )


def extract_cypher(text: str) -> str:
    t = _THINK.sub("", text or "").strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1)
    t = t.strip().strip("`").strip()
    m = re.search(r"\b(MATCH|OPTIONAL MATCH)\b", t, re.I)
    if m:
        t = t[m.start():]
    t = re.split(r"\n\s*\n", t)[0]
    return " ".join(t.split()).rstrip(";")


def _pick_examples(index, question: str, k: int, exclude_sheets: set | None):
    near = index.query(question, k=k, exclude_sheets=exclude_sheets)
    seen_types = {e["qtype"] for e in near}
    for qt in ("simple_counting", "spatial_counting", "spatial_connection", "value_based"):
        if qt not in seen_types:
            for e in index.examples:
                if e["qtype"] == qt and (not exclude_sheets or e.get("sheet") not in exclude_sheets):
                    near.append({**e, "score": 0.0}); break
    return near


_WORDS = re.compile(r"[A-Za-z][A-Za-z/\-]{3,}")


def _fallback(question: str) -> str:
    """A simple, always-valid query derived from the question's own words.

    When generation and repair both fail, returning something honest and
    relevant is better than surfacing a binder exception to a user.
    """
    q = question.lower()
    tag = re.search(r"\b([A-Z]{1,4}[-\s]?[0-9]{2,5}(?:[A-Z](?:/[A-Z])*)?)\b", question)
    if tag:
        t = tag.group(1).strip()
        if any(w in q for w in ("connect", "linked", "attached", "feeds", "downstream", "upstream")):
            return (f"MATCH (a:Symbol)-[:CONNECTED_TO]-(b:Symbol) "
                    f"WHERE a.tag = '{t}' RETURN b.tag, b.name")
        return f"MATCH (s:Symbol) WHERE s.tag CONTAINS '{t}' RETURN s.tag, s.name"
    stop = {"what", "which", "list", "show", "many", "have", "does", "this", "that",
            "drawing", "sheet", "there", "connected", "with", "from", "into", "about"}
    words = [w.lower() for w in _WORDS.findall(question) if w.lower() not in stop]
    if words:
        w = max(words, key=len)
        if "how many" in q or "count" in q:
            return f"MATCH (s:Symbol) WHERE s.name CONTAINS '{w}' RETURN count(s)"
        return f"MATCH (s:Symbol) WHERE s.name CONTAINS '{w}' RETURN s.tag, s.name"
    return "MATCH (s:Symbol) RETURN s.tag, s.name LIMIT 25"


class CypherGenerator:
    def __init__(self, index, model: str | None = None, host: str | None = None):
        self.index = index
        self.model = model or CFG.dotted("llm.cypher_model", "qwen3:8b")
        self.client = ollama.Client(host=host or CFG.dotted("llm.host"))
        self.k = CFG.dotted("retrieval.fewshot_k", 6)

    def _chat(self, messages, n_predict=200) -> str:
        r = self.client.chat(model=self.model, think=False,
                             options={"temperature": CFG.dotted("llm.temperature", 0.0),
                                      "num_predict": n_predict,
                                      "num_ctx": CFG.dotted("llm.num_ctx", 8192)},
                             messages=messages)
        return r["message"]["content"]

    def _sample_names(self, kg, sheet_id) -> list[str]:
        if kg is None:
            return []
        try:
            df = kg.run("MATCH (s:Symbol) WHERE s.sheet_id=$s "
                        "RETURN DISTINCT s.name AS n, count(s) AS k "
                        "ORDER BY k DESC LIMIT 20", {"s": str(sheet_id)})
            names = [str(n) for n in df["n"].tolist()]
            # only override the synthetic ontology when this really is a
            # blueprint sheet (its names are not the 1..32 legend)
            legend = {v["name"] for v in CLASSES.values()}
            return names if names and not set(names) & legend else []
        except Exception:
            return []

    def _sample_tags(self, kg, sheet_id) -> list[str]:
        if kg is None:
            return []
        try:
            df = kg.run("MATCH (s:Symbol) WHERE s.sheet_id=$s AND s.tag <> '' "
                        "RETURN DISTINCT s.tag AS t LIMIT 12", {"s": str(sheet_id)})
            return [str(t) for t in df["t"].tolist()]
        except Exception:
            return []

    def generate(self, question: str, kg=None, sheet_id: str | None = None,
                 exclude_sheets: set | None = None, max_repairs: int = 2,
                 history: list[dict] | None = None) -> dict:
        ex = _pick_examples(self.index, question, self.k, exclude_sheets)
        shots = "\n\n".join(f"Q: {e['question']}\nCypher: {e['cypher']}" for e in ex)
        msgs = [
            {"role": "system", "content": "You translate questions about piping and instrumentation "
                                          "diagrams into Cypher. Output only the query."},
            {"role": "user", "content": f"{schema_prompt(self._sample_tags(kg, sheet_id), self._sample_names(kg, sheet_id))}"
                                        f"\nEXAMPLES\n{shots}\n\nQ: {question}\nCypher:"},
        ]
        if history:
            # Engineers ask follow-ups ("and what about V-1201?"). Without the
            # previous turns the model has no referent for "it" or "the other one".
            recent = "\n".join(f"Earlier — Q: {h['q']}  Cypher: {h.get('cypher','')}"
                               for h in history[-3:] if h.get("q"))
            if recent:
                msgs.insert(1, {"role": "user",
                                "content": f"CONVERSATION SO FAR (for pronouns and "
                                           f"follow-ups only):\n{recent}"})
        attempts = []
        cy = extract_cypher(self._chat(msgs))
        attempts.append(cy)

        for _ in range(max_repairs):
            if kg is None:
                break
            probe = kg.scoped(cy, sheet_id) if sheet_id else cy
            try:
                kg.run(probe)
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:300]
                # Repeating the constraints in the repair turn matters: without
                # them the model tends to reach for the same unsupported
                # construct again (pattern comprehensions, CASE, OR-in-pattern),
                # and burns both attempts.
                msgs2 = msgs + [
                    {"role": "assistant", "content": cy},
                    {"role": "user", "content":
                        f"That query failed with:\n{err}\n\n"
                        "Rewrite it as the SIMPLEST possible Cypher:\n"
                        "  - one MATCH, one WHERE, one RETURN\n"
                        "  - no list or pattern comprehensions, no CASE, no EXISTS\n"
                        "  - match names with CONTAINS and a single word\n"
                        "  - undirected: -[:CONNECTED_TO]-\n"
                        "Return only the query."},
                ]
                cy = extract_cypher(self._chat(msgs2))
                attempts.append(cy)

        # Last resort: a template beats an error message in front of a customer.
        if kg is not None:
            try:
                kg.run(kg.scoped(cy, sheet_id) if sheet_id else cy)
            except Exception:
                cy = _fallback(question)
                attempts.append(cy)

        return {"cypher": cy, "attempts": attempts, "n_attempts": len(attempts),
                "examples_used": [{"q": e["question"][:60], "score": round(e["score"], 3)} for e in ex]}
