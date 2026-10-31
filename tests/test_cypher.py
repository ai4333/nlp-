"""PIDQA ships Cypher with unquoted string literals - invalid in any engine."""
from src.edi.eval.run_pidqa import repair_cypher, parse_gt, _norm_text
from src.edi.qa.nl2cypher import extract_cypher


def test_quotes_unquoted_startswith():
    out = repair_cypher("MATCH (s:Symbol) WHERE s.tag STARTS WITH ZLO RETURN s.tag")
    assert "STARTS WITH 'ZLO'" in out


def test_leaves_already_quoted_alone():
    q = "MATCH (s:Symbol) WHERE s.tag STARTS WITH 'RO-10' RETURN s.tag"
    assert repair_cypher(q).count("'") == 2


def test_strips_markdown_fence_and_think():
    raw = "<think>reasoning here</think>\n```cypher\nMATCH (s:Symbol) RETURN count(s)\n```"
    assert extract_cypher(raw) == "MATCH (s:Symbol) RETURN count(s)"


def test_strips_prose_preamble():
    raw = "Here is the query:\nMATCH (s:Symbol) WHERE s.class = 3 RETURN count(s)"
    assert extract_cypher(raw).startswith("MATCH")


def test_degree_sign_normalised():
    """OCR's English charset cannot emit the degree sign."""
    assert _norm_text("INS ( 45°C )") == _norm_text("INS ( 45C )")


def test_gt_parsing():
    assert parse_gt("['ZLO 433']", "value_based") == {"ZLO 433"}
    assert parse_gt(5, "simple_counting") == 5
    assert parse_gt(True, "spatial_connection") is True
