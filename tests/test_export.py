"""Exports must be consumable by other engineering software."""
import io, json, zipfile
from src.edi.export import entities_df, relations_df, graph_json, cypher_dump, bundle

ENTS = [{"id": "e1", "pidqa_class": 3, "tag": "CD-31021", "bbox": [10, 20, 40, 60],
         "confidence": 0.94, "tag_confidence": 0.99},
        {"id": "e2", "pidqa_class": 28, "tag": "SDL 101", "bbox": [90, 20, 120, 60],
         "confidence": 0.91, "tag_confidence": 1.0}]
RELS = [{"source": "e1", "target": "e2", "type": "CONNECTED_TO", "confidence": 0.8}]


def test_entities_csv_shape():
    df = entities_df("s1", ENTS)
    assert len(df) == 2
    for col in ("sheet_id", "entity_id", "class", "name", "tag", "x0", "cx"):
        assert col in df.columns


def test_relations_resolve_both_ends():
    df = relations_df("s1", ENTS, RELS)
    assert df.iloc[0]["source_tag"] == "CD-31021"
    assert df.iloc[0]["target_tag"] == "SDL 101"


def test_graph_json_is_valid_and_1_based():
    d = json.loads(graph_json("s1", ENTS, RELS))
    assert d["schema"]["class_convention"].startswith("PIDQA")
    assert {n["class"] for n in d["nodes"]} == {3, 28}
    assert len(d["edges"]) == 1


def test_cypher_dump_quotes_are_escaped():
    ents = ENTS + [{"id": "e3", "pidqa_class": 3, "tag": "IT'S", "bbox": [0, 0, 1, 1],
                    "confidence": 1.0, "tag_confidence": 1.0}]
    out = cypher_dump("s1", ents, [])
    assert "\\'" in out


def test_bundle_contains_every_format():
    z = zipfile.ZipFile(io.BytesIO(bundle("s1", ENTS, RELS)))
    names = z.namelist()
    assert any(n.endswith("_entities.csv") for n in names)
    assert any(n.endswith("_relations.csv") for n in names)
    assert any(n.endswith("_graph.json") for n in names)
    assert any(n.endswith("_graph.cypher") for n in names)
    assert "README.txt" in names
