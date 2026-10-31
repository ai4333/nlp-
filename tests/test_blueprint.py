"""Detector-free ingestion of real drawings.

Every string here was read off a real offshore drawing, so these pin behaviour
against actual P&IDs rather than an invented convention.
"""
import pytest
from src.edi.ingest.blueprint import (classify, is_tag, normalise, _pair_bubbles,
                                      _drop_truncations, _merge_same_tag, CATEGORY_CLASS)


@pytest.mark.parametrize("tag,cat", [
    ("V-1103A/B", "equipment"), ("C-1105A/B", "equipment"),
    ("MQ-1101", "equipment"), ("PSV-201", "valve"),
    ("FIC-1042", "instrument"), ("RO-11018", "instrument"),
    ("LG-11009", "instrument"),
])
def test_prefix_drives_category(tag, cat):
    assert classify(tag)[1] == cat


def test_longest_prefix_wins():
    # PSV must not resolve as P (pump)
    assert classify("PSV-201")[0] == "pressure safety valve"
    assert classify("P-1101A")[0] == "pump"


def test_unknown_prefix_is_flagged_not_guessed():
    name, cat = classify("QQ-9999")
    assert cat == "unknown" and "unidentified" in name


@pytest.mark.parametrize("junk", ["NOTE 5", "1234", "12.5", "REV", "AB", ""])
def test_junk_rejected(junk):
    assert not is_tag(junk)


def test_bubble_pairing():
    """ISA bubbles print the code above the loop number as two OCR lines."""
    lines = [{"text": "PC", "score": .99, "sx": 100, "sy": 100},
             {"text": "1103", "score": .98, "sx": 104, "sy": 150},
             {"text": "LC", "score": .99, "sx": 300, "sy": 200},
             {"text": "1204", "score": .97, "sx": 298, "sy": 248}]
    assert sorted(b["text"] for b in _pair_bubbles(lines)) == ["LC-1204", "PC-1103"]


def test_bubble_pairing_ignores_distant_numbers():
    lines = [{"text": "PC", "score": .99, "sx": 100, "sy": 100},
             {"text": "1103", "score": .98, "sx": 900, "sy": 900}]
    assert _pair_bubbles(lines) == []


def test_bubble_number_must_be_below_not_above():
    lines = [{"text": "PC", "score": .99, "sx": 100, "sy": 200},
             {"text": "1103", "score": .98, "sx": 100, "sy": 150}]
    assert _pair_bubbles(lines) == []


def test_truncation_dropped():
    tags = [{"text": "V-1201"}, {"text": "V-120"}, {"text": "H-1104"}]
    kept = {normalise(t["text"]) for t in _drop_truncations(tags)}
    assert kept == {"V-1201", "H-1104"}


def test_same_tag_merges_to_one_item():
    """A P&ID prints a tag several times; that is one physical item, and
    counting it twice would inflate every answer."""
    ents = [{"id": "b0", "tag": "V-1104", "pidqa_class": 101, "bbox": [0, 0, 10, 10],
             "tag_confidence": .9, "category": "equipment"},
            {"id": "b1", "tag": "V-1104", "pidqa_class": 101, "bbox": [500, 500, 510, 510],
             "tag_confidence": .8, "category": "equipment"}]
    out = _merge_same_tag(ents)
    assert len(out) == 1
    assert out[0]["n_occurrences"] == 2
    assert out[0]["bbox"] == [0, 0, 10, 10]        # most confident instance wins
    assert out[0]["also_at"] == [[500, 500, 510, 510]]


def test_category_class_ids_distinct():
    assert len(set(CATEGORY_CLASS.values())) == len(CATEGORY_CLASS)
    assert all(v >= 100 for v in CATEGORY_CLASS.values())  # never collide with 1..32
