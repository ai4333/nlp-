"""Manual graph edits.

The overlay design exists so re-running ingestion cannot destroy human
corrections. These tests pin that guarantee.
"""
import copy
import pytest
from src.edi import edits as E

ENTS = [{"id": "e1", "pidqa_class": 3, "bbox": [0, 0, 10, 10], "tag": "A-1",
         "confidence": .9, "tag_confidence": .8},
        {"id": "e2", "pidqa_class": 5, "bbox": [50, 0, 60, 10], "tag": "B-2",
         "confidence": .9, "tag_confidence": .8},
        {"id": "e3", "pidqa_class": 28, "bbox": [99, 0, 110, 10], "tag": "SDL 1",
         "confidence": .9, "tag_confidence": .9}]
RELS = [{"source": "e1", "target": "e2", "type": "CONNECTED_TO", "confidence": .8}]
DER = {"CONTROLS": [{"source": "e3", "target": "e2", "type": "CONTROLS", "confidence": .7}],
       "FEEDS_TO": []}


def fresh():
    return copy.deepcopy(E.EMPTY)


def test_apply_is_pure():
    ed = E.remove_entity(fresh(), "e1")
    before = copy.deepcopy(ENTS)
    E.apply(ENTS, RELS, DER, ed)
    assert ENTS == before, "apply() must not mutate its inputs"


def test_remove_entity_also_drops_its_edges():
    ed = E.remove_entity(fresh(), "e2")
    ent, rel, der = E.apply(ENTS, RELS, DER, ed)
    assert {e["id"] for e in ent} == {"e1", "e3"}
    assert rel == []
    assert der["CONTROLS"] == [], "a dangling CONTROLS edge must go too"


def test_add_entity_then_connect():
    ed = fresh()
    nid = E.next_id(ENTS, ed)
    E.add_entity(ed, nid, 4, [200, 0, 210, 10], tag="NEW-1")
    E.add_relation(ed, "e1", nid)
    ent, rel, _ = E.apply(ENTS, RELS, DER, ed)
    assert nid in {e["id"] for e in ent}
    assert any(sorted([r["source"], r["target"]]) == sorted(["e1", nid]) for r in rel)


def test_modify_tag_and_class():
    ed = E.modify_entity(fresh(), "e1", tag="FIXED-9", pidqa_class=7)
    ent, _, _ = E.apply(ENTS, RELS, DER, ed)
    e = next(x for x in ent if x["id"] == "e1")
    assert e["tag"] == "FIXED-9" and e["pidqa_class"] == 7
    assert e["tag_confidence"] == 1.0 and e["tag_source"] == "human"


def test_remove_relation():
    ed = E.remove_relation(fresh(), "e1", "e2")
    _, rel, _ = E.apply(ENTS, RELS, DER, ed)
    assert rel == []


def test_relation_direction_does_not_matter():
    ed = E.remove_relation(fresh(), "e2", "e1")     # reversed order
    _, rel, _ = E.apply(ENTS, RELS, DER, ed)
    assert rel == []


def test_typed_relation_removal_is_type_specific():
    ed = E.remove_relation(fresh(), "e3", "e2", rtype="CONTROLS")
    _, rel, der = E.apply(ENTS, RELS, DER, ed)
    assert der["CONTROLS"] == []
    assert len(rel) == 1, "CONNECTED_TO must be untouched"


def test_self_connection_rejected():
    with pytest.raises(ValueError):
        E.add_relation(fresh(), "e1", "e1")


def test_no_duplicate_relation():
    ed = fresh()
    E.add_relation(ed, "e1", "e3"); E.add_relation(ed, "e3", "e1")
    _, rel, _ = E.apply(ENTS, RELS, DER, ed)
    pairs = [tuple(sorted([r["source"], r["target"]])) for r in rel]
    assert len(pairs) == len(set(pairs))


def test_undo_reverses_each_action():
    for setup in (lambda ed: E.add_entity(ed, "m0001", 4, [1, 1, 2, 2]),
                  lambda ed: E.remove_entity(ed, "e1"),
                  lambda ed: E.modify_entity(ed, "e1", tag="X"),
                  lambda ed: E.add_relation(ed, "e1", "e3"),
                  lambda ed: E.remove_relation(ed, "e1", "e2")):
        ed = setup(fresh())
        assert E.has_edits(ed)
        E.undo(ed)
        assert not E.has_edits(ed), "undo must return to a clean overlay"


def test_undo_on_empty_is_safe():
    E.undo(fresh())


def test_removing_a_manual_addition_drops_it_entirely():
    ed = fresh()
    E.add_entity(ed, "m0001", 4, [1, 1, 2, 2])
    E.remove_entity(ed, "m0001")
    assert ed["entities"]["added"] == []
    assert "m0001" not in ed["entities"]["removed"]


def test_next_id_avoids_collisions():
    ed = fresh()
    a = E.next_id(ENTS, ed); E.add_entity(ed, a, 4, [0, 0, 1, 1])
    b = E.next_id(ENTS, ed)
    assert a != b


def test_log_records_every_action():
    ed = fresh()
    E.add_entity(ed, "m0001", 4, [1, 1, 2, 2]); E.add_relation(ed, "e1", "m0001")
    assert len(ed["log"]) == 2
    assert all("ts" in x and "action" in x for x in ed["log"])
