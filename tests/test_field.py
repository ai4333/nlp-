"""
Field mode: the questions a technician asks, on a graph we control.

A small hand-built sheet is loaded into a throwaway Kuzu database so these
assertions are about the logic and not about whatever a real ingest happened to
read that day.

    HV-101 ── V-100 ── HV-102 ── P-100 ── NRV-100 ── TO EXPORT
                │
              PSV-100        LT-100 (controls V-100)
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.edi.graph.kg import KG
from src.edi.graph import isolation as ISO
from src.edi.qa import field as F
from src.edi.qa import answer as A

SHEET = "t_field"

ENTITIES = [
    dict(id="e1", pidqa_class=101, category="equipment", name="vessel / separator",
         bbox=[1000, 800, 1200, 950], tag="V-100", confidence=1.0, tag_confidence=1.0),
    dict(id="e2", pidqa_class=102, category="valve", name="hand valve",
         bbox=[700, 850, 760, 900], tag="HV-101", confidence=1.0, tag_confidence=1.0),
    dict(id="e3", pidqa_class=102, category="valve", name="hand valve",
         bbox=[1400, 850, 1460, 900], tag="HV-102", confidence=1.0, tag_confidence=1.0),
    dict(id="e4", pidqa_class=102, category="valve", name="pressure safety valve",
         bbox=[1050, 500, 1110, 560], tag="PSV-100", confidence=1.0, tag_confidence=1.0),
    dict(id="e5", pidqa_class=101, category="equipment", name="pump",
         bbox=[1700, 800, 1900, 950], tag="P-100", confidence=1.0, tag_confidence=1.0),
    dict(id="e6", pidqa_class=102, category="valve", name="check valve",
         bbox=[2000, 850, 2060, 900], tag="NRV-100", confidence=1.0, tag_confidence=1.0),
    dict(id="e7", pidqa_class=103, category="instrument", name="level transmitter",
         bbox=[1000, 600, 1060, 660], tag="LT-100", confidence=1.0, tag_confidence=1.0),
    dict(id="e8", pidqa_class=106, category="line", name='8" process gas line',
         bbox=[820, 860, 960, 890], tag='8"-PG-11-001-P1B', confidence=1.0,
         tag_confidence=1.0, size_in=8.0),
    dict(id="e9", pidqa_class=105, category="connector", name="off-page connector",
         bbox=[2300, 850, 2500, 900], tag="TO EXPORT", confidence=1.0, tag_confidence=1.0),
]
RELATIONS = [
    dict(source="e2", target="e8", type="CONNECTED_TO", evidence="t", confidence=1.0),
    dict(source="e8", target="e1", type="CONNECTED_TO", evidence="t", confidence=1.0),
    dict(source="e1", target="e3", type="CONNECTED_TO", evidence="t", confidence=1.0),
    dict(source="e1", target="e4", type="CONNECTED_TO", evidence="t", confidence=1.0),
    dict(source="e3", target="e5", type="CONNECTED_TO", evidence="t", confidence=1.0),
    dict(source="e5", target="e6", type="CONNECTED_TO", evidence="t", confidence=1.0),
    dict(source="e6", target="e9", type="CONNECTED_TO", evidence="t", confidence=1.0),
    dict(source="e7", target="e1", type="CONNECTED_TO", evidence="t", confidence=1.0),
]
DERIVED = {"CONTROLS": [dict(source="e7", target="e1", type="CONTROLS",
                            evidence="t", confidence=1.0)],
           "FEEDS_TO": []}


@pytest.fixture(scope="module")
def kg(tmp_path_factory):
    p = tmp_path_factory.mktemp("kg") / "field_kg"
    g = KG(p)
    g.add_sheet(SHEET, ENTITIES, RELATIONS, DERIVED)
    yield g
    g.close()


# ─────────────────────────────────────────────────────────────── routing
@pytest.mark.parametrize("q,expected", [
    ("What does SDV mean?", "definition"),
    ("what is a knockout drum", "definition"),
    ("What is V-100?", "item"),
    ("Where is PSV-100?", "item"),
    # A question that merely NAMES an item is not a request for its card.
    # This used to return the card, which answered a question nobody asked;
    # it now becomes a briefing the model answers the real question from.
    ("What should I check before I open HV-101?", "item_question"),
    ("Which valves do I close to isolate V-100?", "isolation"),
    ("how do I lock out P-100", "isolation"),
    ("How many valves are on this drawing?", ""),
    ("Which lines are 6 inch or larger?", ""),
    ("Where does the flow from V-100 go next?", "flow"),
    ("What is upstream of P-100?", "flow"),
])
def test_routing(kg, q, expected):
    known = F.sheet_tags(kg, SHEET)
    assert F.route(q, kg, SHEET, known) == expected


def test_shut_down_is_not_an_isolation_request(kg):
    """"which instruments shut down the compressor" is an operator's question
    about trips, and it used to be swallowed by the isolation handler."""
    known = F.sheet_tags(kg, SHEET)
    assert F.route("Which instruments shut down V-100?", kg, SHEET, known) != "isolation"


def test_flow_answer_carries_the_rule_for_each_leg(kg):
    card = F.item_card(kg, SHEET, "V-100")
    txt = F.flow_to_text(card)
    assert "V-100" in txt
    assert "topological" in txt, "the answer must say what kind of claim it is making"


def test_flow_answer_refuses_when_the_drawing_says_nothing(kg):
    card = F.item_card(kg, SHEET, "V-100")
    card["flows_in_from"] = []
    card["flows_out_to"] = []
    txt = F.flow_to_text(card)
    assert "not going to guess" in txt


def test_counting_stays_on_the_cypher_path_even_with_a_tag(kg):
    known = F.sheet_tags(kg, SHEET)
    assert F.route("how many lines connect to V-100?", kg, SHEET, known) == ""


# ──────────────────────────────────────────────────────────── item card
def test_item_card_reads_like_an_answer_not_a_row(kg):
    card = F.item_card(kg, SHEET, "V-100", size=(3000, 2000))
    assert card["tag"] == "V-100"
    assert "separator" in card["what_it_is"].lower() or "vessel" in card["name"]
    assert card["why_it_matters"], "a card with no field note is not field mode"
    assert card["zone"], "a technician needs to know where on the sheet to look"
    assert card["unknowns"], "every card must state what it does not know"


def test_item_card_finds_the_hazard_in_the_pipe(kg):
    card = F.item_card(kg, SHEET, "HV-101")
    assert card["lines"], "the line on the valve was not found"
    assert any(h["class"] == "flammable" for h in card["hazards"])


def test_item_card_separates_relief_from_ordinary_connections(kg):
    card = F.item_card(kg, SHEET, "V-100")
    assert any(r["tag"] == "PSV-100" for r in card["relief"])
    assert not any(c["tag"] == "PSV-100" for c in card["connected"])


def test_item_card_lists_the_instruments(kg):
    card = F.item_card(kg, SHEET, "V-100")
    assert any(i["tag"] == "LT-100" for i in card["watched_by"])


def test_unknown_tag_is_refused_not_invented(kg):
    assert F.item_card(kg, SHEET, "V-999") is None
    r = F.answer("What is V-999?", kg, SHEET)
    assert r is not None and "not on this drawing" in r["answer"]


# ──────────────────────────────────────────────────────────── isolation
def test_isolation_finds_the_block_valves(kg):
    p = ISO.plan(kg, SHEET, "V-100")
    tags = {v.tag for v in p.isolation_valves}
    assert "HV-101" in tags and "HV-102" in tags


def test_isolation_never_offers_a_check_valve(kg):
    p = ISO.plan(kg, SHEET, "P-100")
    assert "NRV-100" not in {v.tag for v in p.isolation_valves}, \
        "a check valve will not hold for a permit"


def test_isolation_never_offers_a_relief_valve(kg):
    p = ISO.plan(kg, SHEET, "V-100")
    assert "PSV-100" not in {v.tag for v in p.isolation_valves}
    assert "PSV-100" in {r.tag for r in p.relief}


def test_isolation_reports_branches_it_could_not_close(kg):
    p = ISO.plan(kg, SHEET, "P-100")
    assert p.open_branches or p.crosses_sheet
    assert any("not a permit" in n for n in p.notes)


def test_isolation_of_an_unknown_tag_returns_nothing(kg):
    assert ISO.plan(kg, SHEET, "ZZ-999") is None


def test_isolation_says_so_when_it_finds_nothing(kg):
    p = ISO.plan(kg, SHEET, "TO EXPORT")
    assert p is not None
    assert any("No block valve" in n for n in p.notes) or p.isolation_valves


# ───────────────────────────────────────── refusing to fill the silence
def test_uninformative_rows_are_not_handed_to_the_model():
    """The exact shape that produced an invented answer on a real drawing."""
    assert not A._rows_are_informative("What happens if I close SDV-11014?",
                                       [["SDV-11014"]], ["s.tag"])
    assert not A._rows_are_informative("anything", [], [])
    assert A._rows_are_informative("What is connected to V-100?",
                                   [["HV-101", "hand valve"]], ["tag", "name"])


def test_field_answers_carry_their_confidence(kg):
    known = F.sheet_tags(kg, SHEET)
    r = F.answer("What is V-100?", kg, SHEET, known_tags=known)
    assert 0.0 < r["confidence"] <= 1.0
    r2 = F.answer("What does SDV mean?", kg, SHEET, known_tags=known)
    assert r2["confidence"] >= 0.9 and r2["handler"] == "definition"


# ── whole-drawing overview ────────────────────────────────────────────────
# "Explain the whole drawing" used to fall through to Cypher, which returned a
# single row, and the model then described one pump as if it were the sheet.
def test_overview_intent_routes_without_a_tag():
    import src.edi.qa.field as F
    for q in ["explain the whole engineering drawing, each and everything",
              "what is this drawing about",
              "explain this sheet",
              "give me an overview",
              "what does this drawing show",
              "summarize this p&id"]:
        assert F._OVERVIEW.search(q), q


def test_overview_does_not_steal_item_or_count_questions():
    import src.edi.qa.field as F
    # naming an item means the item card, not the overview
    assert F.find_tag("explain V-1101A") is not None
    # counting stays with the query engine
    assert F._QUANT.search("how many valves are on this drawing?")


def test_overview_text_is_grounded_and_complete():
    from src.edi.qa.field import overview_to_text
    ov = {"sheet": "s1",
          "equipment": [{"tag": "P-1", "name": "pump", "zone": "A-1"}],
          "instruments": [{"tag": "PI-1", "name": "pressure indicator"}],
          "valves": [{"tag": "SDV-1", "name": "shutdown valve"}],
          "lines": [{"tag": "6\"-PG-1", "name": "line", "category": "line",
                     "size_in": 6.0, "service": "PG"}],
          "flow_paths": [{"from": "P-1", "to": "V-1", "confidence": 0.9}],
          "loops": [{"instrument": "PI-1", "acts_on": "P-1"}],
          "hazards": [], "systems": [],
          "counts": {"equipment": 1, "instruments": 1, "valves": 1, "lines": 1,
                     "directed_connections": 1, "control_loops": 1}}
    txt = overview_to_text(ov)
    for must in ("P-1", "PI-1", "SDV-1", "P-1 → V-1", "PG"):
        assert must in txt, must
    # it must never present one sheet as the whole plant
    assert "continues on another drawing" in txt


# ─────────────────────────────────────────── grounded-context assembly
def test_context_includes_item_card_for_tag_in_question(kg):
    """A question naming a tag must carry that item's facts into the briefing.

    The whole failure mode this fixes: a thin query result produced a thin
    answer, because nothing else about the item was ever put in front of the
    model.
    """
    from src.edi.qa import context as CTX
    tag = next(iter(F.sheet_tags(kg, SHEET)))
    brief, ev = CTX.build(f"What is the design pressure of {tag}?", kg, SHEET)
    assert tag in brief
    assert ev["cards"], "the named item's card must be in the briefing"


def test_context_falls_back_to_keyword_search(kg):
    """A question naming no tag still gets grounded on the words it used."""
    from src.edi.qa import context as CTX
    brief, ev = CTX.build("what relieves to the flare on this drawing?",
                          kg, SHEET, rows=[], cols=[])
    assert ev.get("keyword_hits") or ev.get("fallback_overview"), \
        "an untagged question must still be grounded in something real"


def test_bare_identification_stays_instant(kg):
    """`what is X` is the card. Anything more is a question to be answered."""
    known = F.sheet_tags(kg, SHEET)
    tag = next(iter(known))
    assert F.route(f"What is {tag}?", kg, SHEET, known) == "item"
    assert F.route(f"What is the design pressure of {tag}?", kg, SHEET,
                       known) == "item_question"


# ─────────────────────────────────────── deterministic answers
@pytest.mark.parametrize("q,rows,expected", [
    ("How many valves are on this drawing?", [[8]],
     "There are 8 valves on this drawing."),
    ("How many pumps are there?", [[1]], "There is 1 pump on this drawing."),
    # not a count -> the model must handle it
    ("What is connected to V-101?", [["P-101"]], ""),
    # more than one row is not a scalar result
    ("How many valves?", [[3], [4]], ""),
])
def test_counts_are_answered_without_a_model(q, rows, expected):
    """A single numeric row is already the answer.

    Sending it to a language model cost seconds and invited invention: asked
    for a bare count of 16 instruments the model volunteered "valves, sensors
    and controllers", none of which was in the briefing.
    """
    from src.edi.qa.answer import direct_answer
    assert direct_answer(q, rows, ["count"]) == expected


# ──────────────────────────── grounding is not optional
def test_empty_briefing_never_reaches_the_model():
    """A caller that loses its grounding must get a refusal, not a story.

    The workbench once read the briefing from the wrong key, got "", and asked
    the model to "explain this drawing". It invented a whole distillation
    column — HX-101, T-101, CV-101, "1.2 m diameter, 18 m high" — none of which
    is on any drawing in the corpus. Fluent and entirely fabricated.
    """
    from src.edi.qa import answer as A
    for empty in ("", "   ", "\n\n", "short"):
        out = "".join(A.narrate_stream("Explain this whole drawing", empty))
        assert "not going to guess" in out, out[:120]
        out2 = "".join(A.answer_stream("What is on this drawing?", empty))
        assert "not going to guess" in out2, out2[:120]


def test_ask_surfaces_the_briefing_at_the_top_level(kg):
    """The workbench streams from evidence['briefing'] — it must be populated.

    It used to live only under evidence['field'], so the caller streamed from
    an empty string.
    """
    from src.edi.qa import answer as A

    class _Gen:                      # the overview never reaches the generator
        def generate(self, *a, **k):
            return {"cypher": "MATCH (s:Symbol) RETURN s.tag", "n_attempts": 0}

    res = A.ask("Explain this whole drawing", kg, _Gen(), SHEET,
                image=None, with_crops=False, stream_overview=True)
    assert res["handler"] == "overview"
    assert len(res["evidence"].get("briefing", "")) > 100, \
        "the briefing must be reachable where the UI reads it"
