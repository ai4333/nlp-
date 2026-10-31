"""
Flow direction: the rules must be right, and honest about how sure they are.

Two kinds of test here. Small hand-built topologies pin each rule to the thing
it is supposed to know. Then the generated packages are scored end to end
against their own specs — the rig builder knows the true direction of every
connection it draws, which is the only ground truth for flow that exists
anywhere in this project.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.edi.graph import flow as FL
from src.edi.rig import spec as SP, templates as TPL, build as BLD
from src.edi.rig.layout import place


def _e(i, tag, name, category, service="", direction="", box=(0, 0, 10, 10)):
    d = {"id": i, "tag": tag, "name": name, "category": category, "bbox": list(box)}
    if service:
        d["line"] = {"service": service}
    if direction:
        d["connector"] = {"direction": direction, "kind": "destination", "target": tag}
    return d


def _r(a, b):
    return {"source": a, "target": b, "type": "CONNECTED_TO"}


def _dir(edges):
    return {(e["source"], e["target"]): e for e in edges}


# ─────────────────────────────────────────────── R1 · connectors say it outright
def test_a_connector_that_says_TO_is_an_outlet():
    ents = [_e("v", "V-100", "vessel / separator", "equipment"),
            _e("c", "TO 3RD STAGE", "off-page connector (out)", "connector",
               direction="out")]
    got = _dir(FL.infer(ents, [_r("v", "c")], layout_hint=False))
    assert ("v", "c") in got
    assert got[("v", "c")]["confidence"] >= 0.9


def test_a_connector_that_says_FROM_is_an_inlet():
    ents = [_e("v", "V-100", "vessel / separator", "equipment"),
            _e("c", "FROM LP CONDENSATE", "off-page connector (in)", "connector",
               direction="in")]
    got = _dir(FL.infer(ents, [_r("v", "c")], layout_hint=False))
    assert ("c", "v") in got


# ───────────────────────────────────────────── R2 · relief flows one way only
def test_relief_flows_from_the_protected_item_to_the_flare():
    ents = [_e("v", "V-100", "vessel / separator", "equipment"),
            _e("p", "PSV-100", "pressure safety valve", "valve"),
            _e("l", '6"-RL-11-101-F1A', "relief line", "line", service="RL"),
            _e("f", "FS-100", "flare stack", "equipment")]
    got = _dir(FL.infer(ents, [_r("v", "p"), _r("p", "l"), _r("l", "f")],
                        layout_hint=False))
    assert ("v", "p") in got, "the protected vessel must be upstream of its PSV"
    assert ("p", "l") in got and ("l", "f") in got


def test_a_psv_on_a_flare_ko_drum_is_not_reversed():
    """The protected vessel is itself part of the flare system here."""
    ents = [_e("d", "D-100", "HP flare KO drum", "equipment"),
            _e("p", "PSV-100", "pressure safety valve", "valve"),
            _e("l", '6"-RL-11-101-F1A', "relief line", "line", service="RL"),
            _e("s", "FS-100", "flare stack", "equipment")]
    got = _dir(FL.infer(ents, [_r("d", "p"), _r("p", "l"), _r("l", "s")],
                        layout_hint=False))
    assert ("d", "p") in got, "the KO drum feeds its own relief valve, not the reverse"


def test_a_flare_header_runs_toward_the_stack_through_an_esd_valve():
    """An inline valve is rank 0 on its own; the rank must look through it."""
    ents = [_e("d", "D-100", "HP flare KO drum", "equipment"),
            _e("l", '24"-FL-15-100-F1A', "flare line", "line", service="FL"),
            _e("v", "SDV-100", "shutdown valve", "valve"),
            _e("s", "FS-100", "flare stack", "equipment")]
    got = _dir(FL.infer(ents, [_r("d", "l"), _r("l", "v"), _r("v", "s")],
                        layout_hint=False))
    assert ("d", "l") in got and ("l", "v") in got


# ──────────────────────────────────────────── R3 · the service carries intent
def test_a_drain_line_runs_to_the_drain_header():
    ents = [_e("v", "V-100", "vessel / separator", "equipment"),
            _e("l", '4"-DC-11-101-D1C', "closed drain line", "line", service="DC"),
            _e("h", "CLOSED DRAINS HEADER", "shared header", "connector")]
    got = _dir(FL.infer(ents, [_r("v", "l"), _r("l", "h")], layout_hint=False))
    assert ("v", "l") in got and ("l", "h") in got


# ───────────────────────────────────────── R6 · a check valve is on the discharge
def test_a_check_valve_marks_a_pump_discharge():
    ents = [_e("p", "P-100", "pump", "equipment"),
            _e("n", "NRV-100", "check valve", "valve"),
            _e("t", "TK-100", "tank", "equipment")]
    got = _dir(FL.infer(ents, [_r("p", "n"), _r("n", "t")], layout_hint=False))
    assert ("p", "n") in got


# ──────────────────────────────────────────────────── honesty and consistency
def test_no_edge_is_directed_both_ways():
    sp = SP.build(TPL.expand("production_train"))
    ents = BLD.to_entities(sp, place(sp))
    rels, _ = BLD.to_relations(sp)
    got = FL.infer(ents, rels)
    pairs = [(e["source"], e["target"]) for e in got]
    assert len(pairs) == len(set(pairs))
    for a, b in pairs:
        assert (b, a) not in set(pairs), f"{a}/{b} was directed both ways"


def test_the_layout_convention_is_marked_as_weak():
    sp = SP.build(TPL.expand("production_train"))
    ents = BLD.to_entities(sp, place(sp))
    rels, _ = BLD.to_relations(sp)
    for e in FL.infer(ents, rels):
        if "left to right" in e["evidence"]:
            assert e["confidence"] <= 0.55, \
                "a layout convention must never look as sure as a stated direction"
        if "connector says" in e["evidence"]:
            assert e["confidence"] >= 0.9


def test_nothing_is_invented_when_there_is_no_evidence():
    """Two unlabelled items stacked vertically: nothing says which way."""
    ents = [_e("a", "V-100", "vessel / separator", "equipment", box=(0, 0, 10, 10)),
            _e("b", "V-101", "vessel / separator", "equipment", box=(0, 500, 10, 510))]
    assert FL.infer(ents, [_r("a", "b")]) == []


# ────────────────────────────────────── scored against the generated ground truth
def _score(name, layout):
    sp = SP.build(TPL.expand(name))
    ents = BLD.to_entities(sp, place(sp))
    rels, _ = BLD.to_relations(sp)
    rels = [r for r in rels if r.get("evidence") != "instrument connection"]
    truth = {}
    for s in sp.streams:
        lid = f"L{s.id}"
        for a, b in ((s.src, lid), (lid, s.dst), (s.src, s.dst)):
            truth[tuple(sorted((a, b)))] = (a, b)
    got = FL.infer(ents, rels, layout_hint=layout)
    right = sum(1 for d in got
                if truth.get(tuple(sorted((d["source"], d["target"]))))
                == (d["source"], d["target"]))
    judged = sum(1 for d in got
                 if tuple(sorted((d["source"], d["target"]))) in truth)
    pairs = {tuple(sorted((r["source"], r["target"]))) for r in rels}
    return right / judged if judged else 0.0, len(got) / len(pairs)


@pytest.mark.parametrize("pkg", ["production_train", "gas_compression",
                                 "produced_water", "flare_relief", "drilling"])
def test_direction_is_mostly_right_on_a_known_sheet(pkg):
    prec, cov = _score(pkg, layout=True)
    assert prec >= 0.85, f"{pkg}: precision {prec:.3f}"
    assert cov >= 0.60, f"{pkg}: coverage {cov:.3f}"


def test_the_stated_evidence_alone_is_still_mostly_right():
    """Without the layout convention — the number that does not flatter itself."""
    precs = [_score(p, layout=False)[0] for p in
             ("production_train", "gas_compression", "produced_water")]
    assert min(precs) >= 0.80, precs
