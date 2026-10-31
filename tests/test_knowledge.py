"""The knowledge layer must be right, and must refuse to guess."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.edi import knowledge as K
from src.edi.knowledge import isa as ISA
from src.edi.knowledge import glossary as G
from src.edi.ingest import lines as LN
from src.edi.ingest.blueprint import classify


# ────────────────────────────────────────────────── ISA-5.1 decoding
@pytest.mark.parametrize("tag,name,category", [
    ("PSH-1101", "pressure switch, high", "instrument"),
    ("PSHH-1101", "pressure switch, high-high", "instrument"),
    ("LSLL-1102", "level switch, low-low", "instrument"),
    ("PDT-11003", "differential pressure transmitter", "instrument"),
    ("TT-1101", "temperature transmitter", "instrument"),
    ("LG-11009", "level gauge / sight glass", "instrument"),
    ("FQI-11389", "total flow indicator", "instrument"),
    ("VT-1201", "vibration transmitter", "instrument"),
])
def test_isa_decoding(tag, name, category):
    d = ISA.decode(tag)
    assert d is not None, f"{tag} did not decode"
    assert d.name == name
    assert d.category == category
    assert d.plain, "a decoded tag must come with a plain-English explanation"


@pytest.mark.parametrize("tag,name", [
    ("SDV-11014", "shutdown valve"),
    ("BDV-11350", "blowdown valve"),
    ("PSV-11004", "pressure safety valve"),
    ("ZLO-1201", "position switch, open"),
    ("TW-18025", "thermowell"),
    ("NRV-1401", "non-return (check) valve"),
])
def test_industry_conventions_beat_the_standard(tag, name):
    """SDV is a shutdown valve, not "speed ... valve". Convention wins."""
    d = ISA.decode(tag)
    assert d is not None and d.name == name
    assert d.source == "convention"


def test_limit_letters_are_read_from_the_end():
    """L is both 'light' and 'low'; only its position tells them apart."""
    assert ISA.decode("LSL-1").modifier == "low"
    assert ISA.decode("LSLL-1").modifier == "low-low"
    assert "light" not in ISA.decode("LSLL-1").name


def test_equipment_tags_are_not_instruments():
    for tag in ("P-1101A", "V-1101A", "C-1101A", "MQ-1101"):
        assert ISA.decode(tag) is None or ISA.decode(tag).category != "instrument"


# ───────────────────────────────────────────────────────── glossary
def test_glossary_entries_are_complete():
    for term, (definition, why) in G.GLOSSARY.items():
        assert definition.strip() and why.strip(), f"{term} is missing text"
        assert len(definition) > 20, f"{term} definition is too thin to be useful"


def test_aliases_resolve_to_a_real_entry():
    for alias in ("KO drum", "LOTO", "DBB", "PTW", "F&G", "PSV"):
        hit = G.lookup(alias)
        assert hit is not None, f"{alias} did not resolve"
        assert hit["definition"] and "see " not in hit["definition"].lower()[:6]


def test_every_service_code_used_by_the_builder_has_a_hazard_note():
    from src.edi.rig import catalog as CAT
    for code in CAT.PIPE_SPEC:
        assert G.hazard(code) is not None, f"no hazard note for service {code}"


# ────────────────────────────────────── line-number fragments (the DC266 bug)
@pytest.mark.parametrize("tag", ["DC266", "PL103", "VH082", "VL014", "VA004",
                                 "DC-6970", "VA005102"])
def test_clipped_line_numbers_are_lines_not_equipment(tag):
    """These were read as drums, pumps and vessels on a real offshore P&ID."""
    assert LN.is_line_fragment(tag)
    name, cat = classify(tag)
    assert cat == "line", f"{tag} classified as {cat}"


@pytest.mark.parametrize("tag", ["P-1101A", "V-1101A", "H-1101A", "C-1101A",
                                 "D-1501", "TA-11003", "PT-1101", "ST-1101",
                                 "PSV-11004", "SDV-11014"])
def test_real_equipment_tags_survive_the_fragment_filter(tag):
    assert not LN.is_line_fragment(tag)


def test_complete_line_numbers_are_not_fragments():
    assert LN.parse_fragment('12"-DC-11-185-P1A-HE') is None
    assert LN.parse('12"-DC-11-185-P1A-HE') is not None


def test_a_fragment_admits_it_does_not_know_the_size():
    fr = LN.parse_fragment("DC266")
    assert fr.size_in == 0.0
    e = LN.fragment_entity(fr, 0, 10, 10, 0.9)
    assert e["needs_review"] is True and e["category"] == "line"


# ─────────────────────────────────────────────────────── explain()
def test_explain_answers_the_question_that_used_to_fail():
    r = K.explain(K.strip_question("What does SDV mean?"))
    assert r and r["name"] == "shutdown valve"
    assert "shut" in r["definition"].lower()


def test_explain_never_invents():
    assert K.explain("QQQ-999") is None
    assert K.explain("frobnicator") is None
    assert K.explain("") is None


def test_explain_flags_an_ambiguous_reading():
    r = K.explain("DC266")
    assert r["kind"] == "line"
    assert r.get("ambiguous") is True
    assert "density controller" in r["definition"]


def test_explain_prefers_a_complete_line_number():
    r = K.explain('12"-DC-11-185-P1A-HE')
    assert r["kind"] == "line" and not r.get("ambiguous")
    assert "closed drain" in r["definition"]


def test_strip_question():
    assert K.strip_question("What does SDV mean?") == "SDV"
    assert K.strip_question("what is a knockout drum") == "knockout drum"
    assert K.strip_question("PSH-1101") == "PSH-1101"
