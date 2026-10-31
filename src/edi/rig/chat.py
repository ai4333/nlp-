"""
Conversational rig design against the local LLM.

You type "build me a three stage separation train with gas compression and
flare" and get a P&ID. The model's only job is to turn that sentence into a
list of items and streams — a small, closed-vocabulary task an 8B model does
well. It never draws, never numbers, never decides what needs a relief valve.

Three things keep it reliable:

  1 · **A closed vocabulary.** The prompt carries the catalogue keys. The model
      picks from them; anything it invents is dropped by `catalog.find` rather
      than becoming a mystery box on the drawing.
  2 · **Repair, then fall back.** Invalid JSON is fed back once with the parse
      error. If it still fails, the request is matched against the standard
      packages in `templates.py` and the sheet is built from those. The chat
      never dead-ends — a customer watching a demo sees a drawing either way.
  3 · **Edit, don't regenerate.** Follow-ups ("add a second compressor",
      "route the LP gas to the flare") are given the current spec and asked for
      the whole updated spec back, so the conversation accumulates instead of
      resetting.
"""
from __future__ import annotations
import json
import re

import ollama

from ..config import load
from . import catalog as CAT
from . import templates as TPL

CFG = load()

_JSON = re.compile(r"\{.*\}", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def available_models() -> list[str]:
    """Whatever Ollama has pulled locally. Empty list means Ollama is not up."""
    try:
        r = ollama.Client(host=CFG.dotted("llm.host")).list()
        out = []
        for m in r.get("models", []):
            n = m.get("model") or m.get("name")
            if n:
                out.append(n)
        return sorted(out)
    except Exception:
        return []


def _vocab() -> str:
    lines = []
    for system, items in sorted(CAT.systems().items()):
        keys = ", ".join(sorted(i.key for i in items))
        lines.append(f"  {system}: {keys}")
    return "\n".join(lines)


SCHEMA = """Reply with ONE JSON object and nothing else:

{
  "title": "short sheet title",
  "facility": "e.g. Offshore production platform",
  "equipment": [{"type": "<catalogue key>", "label": "OPTIONAL PRINTED NAME"}],
  "streams":   [{"from": "<key or tag>", "to": "<key or tag>",
                 "size": <inches>, "service": "<service code>"}],
  "notes": ["SHORT DRAWING NOTE", "..."]
}

SERVICE CODES: P process · PL process liquid · PG process gas · FL flare ·
RL relief · WA water · IA instrument air · FG fuel gas · DC closed drain ·
DO open drain · CH chemical injection · N nitrogen · GL glycol · CW cooling water

RULES
 - "type" MUST be one of the catalogue keys listed above. Never invent one.
 - Every item you list should appear in at least one stream.
 - Flow left to right: feed first, product last. In a separation train that
   means HIGH pressure first: hp_separator, then mp_separator, then
   lp_separator — never the other way round.
 - Size the lines sensibly: gas 8-16", liquid 6-12", relief 4-8", chemical 1".
 - DO NOT add relief valves, ESD valves, check valves, instrument bubbles, tags
   or line numbers. Those are generated automatically and adding them yourself
   creates duplicates.
 - No prose, no markdown, no explanation. JSON only."""


SYSTEM = ("You are a process engineer laying out oil and gas P&IDs. "
          "You answer only with the JSON spec of the sheet.")


def _extract(text: str) -> dict | None:
    t = _THINK.sub("", text or "").strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1)
    m = _JSON.search(t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # a trailing comma or an unquoted key is the usual failure
        s = re.sub(r",\s*([}\]])", r"\1", m.group(0))
        try:
            return json.loads(s)
        except Exception:
            return None


def _example() -> str:
    t = TPL.TEMPLATES["gas_compression"]
    ex = {"title": t["title"], "facility": "Offshore production platform",
          "equipment": [{"type": e} for e in t["equipment"][:5]],
          "streams": [{"from": a, "to": b, "size": s, "service": v}
                      for a, b, s, v in t["streams"][:4]],
          "notes": t["notes"][:2]}
    return json.dumps(ex, indent=1)


class RigChat:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or CFG.dotted("llm.rig_model") or CFG.dotted("llm.answer_model", "qwen3:8b")
        self.client = ollama.Client(host=host or CFG.dotted("llm.host"))

    # ---------------------------------------------------------------- LLM
    def _chat(self, msgs, n_predict=1400) -> str:
        r = self.client.chat(
            model=self.model, think=False,
            options={"temperature": CFG.dotted("llm.temperature", 0.0),
                     "num_predict": n_predict,
                     "num_ctx": CFG.dotted("llm.num_ctx", 8192)},
            messages=msgs)
        return r["message"]["content"]

    def propose(self, message: str, current: dict | None = None,
                history: list[dict] | None = None) -> dict:
        """Return {'spec': raw dict, 'source': 'model'|'template', 'note': str}."""
        prompt = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                f"CATALOGUE — the only item types that exist:\n{_vocab()}\n\n"
                f"{SCHEMA}\n\nEXAMPLE OF A VALID REPLY\n{_example()}"},
        ]
        for h in (history or [])[-3:]:
            if h.get("q"):
                prompt.append({"role": "user", "content": h["q"]})
            if h.get("summary"):
                prompt.append({"role": "assistant", "content": h["summary"]})
        if current:
            slim = {"title": current.get("title", ""),
                    "equipment": [{"type": n["item"]} for n in current.get("nodes", [])
                                  if n.get("category") == "equipment" and not n.get("auto")],
                    "streams": [{"from": s["src"], "to": s["dst"],
                                 "size": s["size_in"], "service": s["service"]}
                                for s in current.get("streams", []) if not s.get("auto")]}
            prompt.append({"role": "user", "content":
                           "THE SHEET AS IT STANDS (edit this, return the WHOLE "
                           "updated spec):\n" + json.dumps(slim)})
        prompt.append({"role": "user", "content": f"REQUEST: {message}"})

        raw, err = None, ""
        try:
            txt = self._chat(prompt)
            raw = _extract(txt)
            if raw is None:
                prompt += [{"role": "assistant", "content": txt[:600]},
                           {"role": "user", "content":
                            "That was not valid JSON. Reply with the JSON object only — "
                            "no prose, no fence, no trailing commas."}]
                raw = _extract(self._chat(prompt))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        if raw and self._usable(raw):
            raw["prompt"] = message
            return {"spec": raw, "source": "model", "note": err}

        # fallback: the standard packages
        names = TPL.match(message) or ["production_train"]
        spec = TPL.merged(names[:2]) if len(names) > 1 else TPL.expand(names[0])
        spec["prompt"] = message
        why = ("Ollama did not answer" if err else
               "the model's reply had no usable items")
        return {"spec": spec, "source": "template",
                "note": f"{why} — built from the standard "
                        f"{', '.join(names[:2])} package(s) instead."}

    @staticmethod
    def _usable(raw: dict) -> bool:
        eq = raw.get("equipment") or []
        good = 0
        for e in eq:
            key = e.get("type") if isinstance(e, dict) else e
            if CAT.find(str(key or "")):
                good += 1
        return good >= 2

    # ------------------------------------------------------------ narration
    def describe(self, sp, result: dict) -> str:
        """A one-paragraph reply for the chat. Written locally, never required."""
        base = (f"Built **{sp.title}** — {result['n_equipment']} items of equipment, "
                f"{result['n_valves']} valves, {result['n_instruments']} instruments "
                f"and {result['n_lines']} lines, on a "
                f"{result['width']}×{result['height']} sheet. "
                f"{len(result['issues'])} design finding(s).")
        try:
            eq = ", ".join(f"{n.tag} {n.name}" for n in sp.equipment()[:12])
            txt = self._chat([
                {"role": "system", "content":
                    "You are a process engineer. Two sentences maximum, plain English, "
                    "no markdown headings, no lists."},
                {"role": "user", "content":
                    f"I generated a P&ID titled '{sp.title}' containing: {eq}. "
                    f"Describe in two sentences what this sheet shows and how the "
                    f"process flows."}], n_predict=160)
            txt = _THINK.sub("", txt).strip()
            if txt:
                return base + "\n\n" + txt
        except Exception:
            pass
        return base
