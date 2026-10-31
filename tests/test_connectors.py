"""Off-page connectors.

The hard part is not finding them — it is telling a connector label apart from
note prose, which uses the same words. Every string below was read off a real
offshore P&ID, including the notes that must be rejected.
"""
import pytest
from src.edi.ingest.connectors import parse, is_connector, to_entity, cross_sheet_links


@pytest.mark.parametrize("raw,kind,direction", [
    ("TO 3RD STAGE",         "destination", "out"),
    ("TO LP COMPRESSOR",     "destination", "out"),
    ("TO FLARE",             "destination", "out"),
    ("FROM LP CONDENSATE",   "destination", "in"),
    ("FROM 3RD",             "destination", "in"),
    ("HP FLARE HEADER",      "header",      "both"),
    ("CLOSED DRAINS HEADER", "header",      "both"),
    ("ATMOS. VENT HEADER",   "header",      "both"),
    ("FD-0002-01",           "drawing_ref", "both"),
])
def test_real_connectors(raw, kind, direction):
    c = parse(raw)
    assert c is not None, raw
    assert c.kind == kind and c.direction == direction


@pytest.mark.parametrize("prose", [
    "ALL DRAIN LINES TO SLOPE TO DRAIN HEADER.",
    "IS TO SLOPE TO FLARE HEADER. MINIMUM",
    "TO BE POCKETTED. DISCHARGE PIPING",
    "TO VALVE.",
    "TO LATEST INFO",
    "HEADER. MINIMUM",
    "7. ALL DRAIN LINES TO SLOPE TO DRAIN HEADER.",
])
def test_note_prose_rejected(prose):
    assert parse(prose) is None, prose
    assert not is_connector(prose)


def test_terminal_period_always_rejects():
    assert parse("TO FLARE.") is None
    assert parse("TO FLARE") is not None


def test_internal_period_survives():
    """ATMOS. is an abbreviation, not a sentence end."""
    assert parse("ATMOS. VENT HEADER") is not None


def test_direction_recorded():
    assert parse("TO FLARE").direction == "out"
    assert parse("FROM FLARE").direction == "in"


def test_target_is_the_join_key():
    """TO X on one sheet and FROM X on another must share a key."""
    assert parse("TO 3RD STAGE").target == parse("FROM 3RD STAGE").target


def test_entity_shape():
    e = to_entity(parse("HP FLARE HEADER"), 0, 400.0, 200.0, 0.95)
    assert e["category"] == "connector" and e["target"] == "HP FLARE HEADER"
    assert e["bbox"][2] > e["bbox"][0]


def test_cross_sheet_link_created_for_shared_header():
    a = [to_entity(parse("HP FLARE HEADER"), 0, 10, 10, 1.0)]
    b = [to_entity(parse("HP FLARE HEADER"), 0, 20, 20, 1.0)]
    links = cross_sheet_links({"sheet-a": a, "sheet-b": b})
    assert len(links) == 1
    assert links[0]["type"] == "CONTINUES_ON" and links[0]["via"] == "HP FLARE HEADER"


def test_to_and_from_across_sheets_link():
    a = [to_entity(parse("TO 3RD STAGE"), 0, 10, 10, 1.0)]
    b = [to_entity(parse("FROM 3RD STAGE"), 0, 20, 20, 1.0)]
    assert len(cross_sheet_links({"a": a, "b": b})) == 1


def test_no_link_when_only_one_sheet_mentions_it():
    a = [to_entity(parse("HP FLARE HEADER"), 0, 10, 10, 1.0),
         to_entity(parse("HP FLARE HEADER"), 1, 90, 90, 1.0)]
    assert cross_sheet_links({"only-sheet": a}) == []
