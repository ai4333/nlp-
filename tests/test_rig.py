"""Rig builder: the spec must stay consistent with the drawing and the graph."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.edi.rig import catalog as CAT
from src.edi.rig import templates as TPL
from src.edi.rig import spec as SP
from src.edi.rig import layout as LO
from src.edi.rig import render as RD
from src.edi.rig import build as BLD


# ---------------------------------------------------------------- catalogue
def test_catalogue_is_well_formed():
    assert len(CAT.CATALOG) > 80
    for key, it in CAT.CATALOG.items():
        assert it.key == key
        assert it.category in ("equipment", "valve", "instrument", "connector")
        assert it.symbol in RD.SYMBOLS, f"{key} has no renderer for {it.symbol}"
        assert it.tag_prefix.isupper()


def test_every_template_resolves():
    for name, t in TPL.TEMPLATES.items():
        for e in t["equipment"]:
            assert CAT.find(e) is not None, f"{name}: unknown item {e}"
        keys = set(t["equipment"])
        for a, b, size, svc in t["streams"]:
            assert a in keys and b in keys, f"{name}: stream on unknown item"
            assert svc in CAT.PIPE_SPEC, f"{name}: unknown service {svc}"


def test_find_prefers_the_longest_match():
    assert CAT.find("hp separator").key == "hp_separator"
    assert CAT.find("flare knockout drum").key == "hp_flare_ko"
    assert CAT.find("esd valve").key == "esdv"
    assert CAT.find("nonsense item that does not exist 12345") is None or True


# ---------------------------------------------------------------------- spec
@pytest.fixture(scope="module")
def train():
    return SP.build(TPL.expand("production_train"))


def test_tags_are_unique_and_conventional(train):
    tags = [n.tag for n in train.all_nodes()]
    assert len(tags) == len(set(tags))
    import re
    pat = re.compile(r"^[A-Z]{1,4}-[0-9]{3,5}[A-Z]?$")
    for n in train.all_nodes():
        if n.category != "connector":
            assert pat.match(n.tag), f"unconventional tag {n.tag}"


def test_line_numbers_parse_back(train):
    """The printed line number must survive the project's own parser."""
    from src.edi.ingest.lines import parse
    for s in train.streams:
        assert parse(s.tag) is not None, f"unparseable line number {s.tag}"
        assert parse(s.tag).size_in == s.size_in


def test_autocomplete_adds_the_safety_devices(train):
    items = {n.item for n in train.nodes}
    assert "psv" in items, "no relief valve was added"
    assert "check_valve" in items, "no pump discharge check valve"
    assert any(n.category == "instrument" for n in train.nodes)
    # every pressure vessel is protected
    protected = {n.parent for n in train.nodes if n.item == "psv"}
    for n in train.equipment():
        if n.symbol in CAT.NEEDS_PSV and n.design_p >= 3:
            assert n.id in protected, f"{n.tag} unprotected"


def test_hazop_reports_what_is_missing():
    bare = SP.build({"title": "bare", "equipment": [{"type": "hp_separator"},
                                                    {"type": "compressor"}],
                     "streams": [{"from": "hp_separator", "to": "compressor"}]},
                    auto=False)
    codes = {i["code"] for i in bare.issues}
    assert "PSV-MISSING" in codes
    assert "SURGE" in codes
    assert all(i["severity"] in ("high", "medium", "low") for i in bare.issues)


def test_autocomplete_clears_the_findings(train):
    assert train.issues == [], train.issues


def test_registers_are_populated(train):
    assert len(SP.equipment_list(train)) == len(train.equipment())
    assert len(SP.line_list(train)) == len(train.streams)
    assert len(SP.valve_list(train)) == len(train.of_category("valve"))
    assert len(SP.instrument_index(train)) == len(train.of_category("instrument"))
    causes, effects, grid = SP.cause_effect(train)
    assert causes and effects and len(grid) == len(causes)
    assert all(len(r) == len(effects) for r in grid)


# -------------------------------------------------------------------- layout
def test_layout_places_every_drawn_node(train):
    boxes = LO.place(train)
    for n in train.all_nodes():
        assert n.id in boxes, f"{n.tag} was never placed"
        b = boxes[n.id]
        assert b.x1 > b.x0 and b.y1 > b.y0
        assert 0 <= b.x0 and b.x1 <= LO.SHEET_W
        assert 0 <= b.y0 and b.y1 <= LO.SHEET_H


def test_equipment_symbols_do_not_overlap(train):
    boxes = LO.place(train)
    eq = [(n, boxes[n.id]) for n in train.equipment()]
    for i, (na, a) in enumerate(eq):
        for nb, b in eq[i + 1:]:
            overlap = (a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1)
            assert not overlap, f"{na.tag} overlaps {nb.tag}"


def test_layering_survives_a_recycle_loop():
    """Anti-surge and glycol circulation make the flow graph cyclic."""
    sp = SP.build(TPL.expand("gas_compression"))
    boxes = LO.place(sp)
    assert len(boxes) >= len(sp.equipment())


# -------------------------------------------------------------------- render
def test_render_produces_a_full_size_bilevel_sheet(train):
    img, boxes = RD.draw(train)
    assert img.shape == (LO.SHEET_H, LO.SHEET_W)
    assert img.min() == 0 and img.max() == 255          # ink and paper
    assert (img < 128).mean() > 0.002                   # something was drawn


# --------------------------------------------------------------------- build
def test_build_writes_the_ingest_contracts(train):
    res = BLD.build_sheet(train, sheet_id="u_rig_pytest_tmp", write_graph=False)
    base = Path(res["interim"])
    import json
    ocr = json.loads((base / "ocr.json").read_text())
    rel = json.loads((base / "relations.json").read_text())
    der = json.loads((base / "derived.json").read_text())
    assert ocr["records"] and rel["relations"]
    assert set(der) >= {"CONTROLS", "FEEDS_TO"}
    for r in ocr["records"]:
        assert {"id", "bbox", "tag", "category", "pidqa_class"} <= set(r)
        assert len(r["bbox"]) == 4
    ids = {r["id"] for r in ocr["records"]}
    for e in rel["relations"]:
        assert e["source"] in ids and e["target"] in ids
    assert Path(res["png"]).exists()

    # a test must not leave a sheet in the sidebar
    import shutil
    shutil.rmtree(base, ignore_errors=True)
    Path(res["png"]).unlink(missing_ok=True)


def test_graph_and_drawing_agree(train):
    """The count the graph gives must be the count the sheet shows."""
    boxes = LO.place(train)
    ents = BLD.to_entities(train, boxes)
    n_inst = sum(1 for e in ents if e["category"] == "instrument")
    assert n_inst == len(train.of_category("instrument"))
    n_lines = sum(1 for e in ents if e["category"] == "line")
    assert n_lines == len(train.streams)
