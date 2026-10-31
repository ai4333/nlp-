"""Pipe line numbers — size, service, spec.

Every string below was read off a real offshore P&ID
(ALPHA-AA-0000-PR-PD-0030-01), so these pin behaviour against actual drawings
rather than an invented convention.
"""
import pytest
from src.edi.ingest.lines import parse, is_line_number, to_entity


@pytest.mark.parametrize("raw,size,svc,num", [
    ('12"-DC-11-185-P1A-HE', 12, "DC", "185"),
    ('1"-VL-16-350',          1, "VL", "350"),
    ('2"-PL-11-018-P1A-HE',   2, "PL", "018"),
    ('3"-PL-11-004-P1A-HE',   3, "PL", "004"),
    ('1"-VH-11-189-A1C',      1, "VH", "189"),
    ('2"-DC-28-083-D1C',      2, "DC", "083"),
    ('6"-LD-753',             6, "LD", "753"),
])
def test_real_line_numbers(raw, size, svc, num):
    ln = parse(raw)
    assert ln is not None, raw
    assert ln.size_in == size and ln.service == svc and ln.number == num


def test_spec_and_insulation_captured():
    ln = parse('12"-DC-11-185-P1A-HE')
    assert ln.spec == "P1A" and ln.insulation == "HE"


def test_service_name_resolved():
    assert "drain" in parse('12"-DC-11-185-P1A-HE').service_name
    assert "vent" in parse('1"-VL-16-350').service_name


def test_unknown_service_still_parses():
    ln = parse('4"-ZZ-11-004')
    assert ln is not None and ln.size_in == 4
    assert ln.service_name == "process line"      # sane default, not a crash


@pytest.mark.parametrize("raw", ["V-1103A/B", "NOTE 5", "C-1105A/B", "1234", "", "PSV-201"])
def test_not_line_numbers(raw):
    assert parse(raw) is None
    assert not is_line_number(raw)


def test_en_dash_and_curly_quote_normalised():
    assert parse('2”–PL–11–018') is not None


def test_trailing_ocr_noise_tolerated():
    assert parse('1"-VL-16-350!') is not None


def test_entity_shape():
    e = to_entity(parse('12"-DC-11-185-P1A-HE'), 0, 500.0, 300.0, 0.98)
    assert e["category"] == "line" and e["size_in"] == 12
    assert e["tag"] == '12"-DC-11-185-P1A-HE'
    assert len(e["bbox"]) == 4 and e["bbox"][2] > e["bbox"][0]


def test_half_inch_sizes():
    ln = parse('1.5"-PL-11-004')
    assert ln is not None and ln.size_in == 1.5
