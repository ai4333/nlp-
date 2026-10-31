"""The off-by-one that would silently break every benchmark answer.

YOLO labels are 0..31; PIDQA questions and Cypher are 1..32.  Empirically
confirmed: PIDQA asks for "class 31 symbols beginning with ZLO" with ground
truth 'ZLO 433', and YOLO class 30 is the bubble that reads ZLO 433.
"""
import pytest
from src.edi.config import yolo_to_pidqa, pidqa_to_yolo


def test_roundtrip():
    for y in range(32):
        assert pidqa_to_yolo(yolo_to_pidqa(y)) == y


def test_boundaries():
    assert yolo_to_pidqa(0) == 1
    assert yolo_to_pidqa(31) == 32
    assert pidqa_to_yolo(1) == 0
    assert pidqa_to_yolo(32) == 31


@pytest.mark.parametrize("bad", [-1, 32, 100])
def test_yolo_out_of_range(bad):
    with pytest.raises(ValueError):
        yolo_to_pidqa(bad)


@pytest.mark.parametrize("bad", [0, 33, -5])
def test_pidqa_out_of_range(bad):
    with pytest.raises(ValueError):
        pidqa_to_yolo(bad)


def test_zlo_anchor():
    """The empirical anchor: YOLO 30 <-> PIDQA 31 (the ZLO/ZLC bubble)."""
    assert yolo_to_pidqa(30) == 31
