"""Derived relation types: the objectives ask for controls/feeds_to, not just
connected_to."""
import numpy as np
from src.edi.graph.semantics import derive_controls, arrow_direction, category


def _sym(i, cls, x=0, y=0):
    return {"id": i, "pidqa_class": cls, "bbox": [x, y, x + 30, y + 30], "tag": i}


def test_instrument_controls_valve():
    ents = [_sym("a", 28), _sym("b", 3, 100)]        # SDL instrument, globe valve
    rels = [{"source": "a", "target": "b", "confidence": 0.8}]
    out = derive_controls(ents, rels)
    assert len(out) == 1
    assert out[0]["source"] == "a" and out[0]["target"] == "b"
    assert out[0]["type"] == "CONTROLS"


def test_direction_is_instrument_to_valve_regardless_of_edge_order():
    ents = [_sym("a", 3), _sym("b", 28, 100)]        # valve first this time
    rels = [{"source": "a", "target": "b", "confidence": 0.8}]
    out = derive_controls(ents, rels)
    assert out[0]["source"] == "b" and out[0]["target"] == "a"


def test_two_valves_do_not_control_each_other():
    ents = [_sym("a", 3), _sym("b", 4, 100)]
    rels = [{"source": "a", "target": "b", "confidence": 0.8}]
    assert derive_controls(ents, rels) == []


def test_arrow_direction_from_ink_mass():
    """A filled triangle's apex is opposite its ink centroid."""
    img = np.full((40, 40), 255, np.uint8)
    pts = np.array([[2, 5], [2, 34], [36, 20]])       # apex points +x
    cv2.fillPoly(img, [pts], 0)
    (_, _), (dx, dy) = arrow_direction(img, [0, 0, 40, 40])
    assert dx > 0 and abs(dx) > abs(dy)


def test_category_lookup():
    assert category(28) == "instrument"
    assert category(3) == "valve"


import cv2  # noqa: E402  (used above)
