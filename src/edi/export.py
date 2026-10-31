"""
Data export.

"Can I get the data out?" is the first question any buyer asks, and a system
that can only answer questions inside its own UI is a demo, not a tool. These
produce plain formats other engineering software can consume.
"""
from __future__ import annotations
import io, json, zipfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from .config import load
from .graph.kg import class_name

CFG = load()


def entities_df(sheet_id: str, entities: list[dict]) -> pd.DataFrame:
    rows = []
    for e in entities:
        x0, y0, x1, y1 = e["bbox"]
        rows.append({
            "sheet_id": sheet_id, "entity_id": e["id"],
            "class": e["pidqa_class"], "name": class_name(e["pidqa_class"]),
            "tag": e.get("tag", ""),
            "detect_confidence": round(float(e.get("confidence", 1.0)), 4),
            "tag_confidence": round(float(e.get("tag_confidence", 0.0)), 4),
            "tag_source": e.get("tag_source", "ocr" if e.get("tag") else ""),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "cx": (x0 + x1) // 2, "cy": (y0 + y1) // 2,
        })
    return pd.DataFrame(rows)


def relations_df(sheet_id: str, entities: list[dict], relations: list[dict]) -> pd.DataFrame:
    by = {e["id"]: e for e in entities}
    rows = []
    for r in relations:
        a, b = by.get(r["source"]), by.get(r["target"])
        if not a or not b:
            continue
        rows.append({
            "sheet_id": sheet_id, "type": r.get("type", "CONNECTED_TO"),
            "source_id": r["source"], "source_tag": a.get("tag", ""),
            "source_class": a["pidqa_class"], "source_name": class_name(a["pidqa_class"]),
            "target_id": r["target"], "target_tag": b.get("tag", ""),
            "target_class": b["pidqa_class"], "target_name": class_name(b["pidqa_class"]),
            "confidence": round(float(r.get("confidence", 0.8)), 3),
            "evidence": r.get("evidence", ""),
        })
    return pd.DataFrame(rows)


def graph_json(sheet_id: str, entities: list[dict], relations: list[dict]) -> str:
    """Node-link JSON — the interchange format most graph tools import."""
    return json.dumps({
        "sheet_id": sheet_id,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema": {"node": "Symbol", "edge": "CONNECTED_TO",
                   "class_convention": "PIDQA 1..32"},
        "nodes": [{"id": e["id"], "class": e["pidqa_class"],
                   "name": class_name(e["pidqa_class"]), "tag": e.get("tag", ""),
                   "bbox": e["bbox"],
                   "confidence": round(float(e.get("confidence", 1.0)), 4),
                   "tag_confidence": round(float(e.get("tag_confidence", 0.0)), 4)}
                  for e in entities],
        "edges": [{"source": r["source"], "target": r["target"],
                   "type": r.get("type", "CONNECTED_TO"),
                   "confidence": round(float(r.get("confidence", 0.8)), 3)}
                  for r in relations],
    }, indent=1)


def cypher_dump(sheet_id: str, entities: list[dict], relations: list[dict]) -> str:
    """Portable Cypher so the graph can be rebuilt in Neo4j."""
    out = [f"// Engineering Drawing Intelligence — sheet {sheet_id}",
           f"// generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
           "CREATE INDEX symbol_tag IF NOT EXISTS FOR (s:Symbol) ON (s.tag);", ""]
    for e in entities:
        tag = (e.get("tag") or "").replace("'", "\\'")
        out.append(
            f"CREATE (:Symbol {{entity_id:'{sheet_id}:{e['id']}', class:{e['pidqa_class']}, "
            f"name:'{class_name(e['pidqa_class'])}', tag:'{tag}', sheet_id:'{sheet_id}'}});")
    out.append("")
    for r in relations:
        out.append(
            f"MATCH (a:Symbol {{entity_id:'{sheet_id}:{r['source']}'}}), "
            f"(b:Symbol {{entity_id:'{sheet_id}:{r['target']}'}}) "
            f"CREATE (a)-[:CONNECTED_TO {{confidence:{float(r.get('confidence',0.8)):.2f}}}]->(b);")
    return "\n".join(out)


def bundle(sheet_id: str, entities: list[dict], relations: list[dict]) -> bytes:
    """One zip with every format."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{sheet_id}_entities.csv", entities_df(sheet_id, entities).to_csv(index=False))
        z.writestr(f"{sheet_id}_relations.csv",
                   relations_df(sheet_id, entities, relations).to_csv(index=False))
        z.writestr(f"{sheet_id}_graph.json", graph_json(sheet_id, entities, relations))
        z.writestr(f"{sheet_id}_graph.cypher", cypher_dump(sheet_id, entities, relations))
        z.writestr("README.txt",
                   "Engineering Drawing Intelligence export\n"
                   "=======================================\n"
                   f"sheet: {sheet_id}\n"
                   f"entities: {len(entities)}   relations: {len(relations)}\n\n"
                   "entities.csv   one row per detected symbol, with tag and pixel bbox\n"
                   "relations.csv  one row per CONNECTED_TO edge, both ends resolved\n"
                   "graph.json     node-link JSON for graph tooling\n"
                   "graph.cypher   replayable Cypher to rebuild in Neo4j\n\n"
                   "Class numbers follow the PIDQA convention (1..32).\n")
    return buf.getvalue()


def answer_report(res: dict, sheet_id: str) -> str:
    """A markdown record of one answer, with its evidence, for pasting into a report."""
    lines = [f"# Query result — sheet {sheet_id}", "",
             f"**Question.** {res['question']}", "",
             f"**Answer.** {res['answer']}", ""]
    if res.get("reasoning_path"):
        lines += ["**Reasoning path.**", "", "`" + " ".join(res["reasoning_path"]) + "`", ""]
    lines += ["**Cypher executed.**", "", "```cypher", res["cypher"], "```", ""]
    if res.get("rows"):
        cols = res.get("columns") or []
        lines += ["**Rows returned.**", ""]
        if cols:
            lines += ["| " + " | ".join(map(str, cols)) + " |",
                      "|" + "---|" * len(cols)]
        for r in res["rows"][:25]:
            lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
        lines.append("")
    regs = res.get("evidence", {}).get("regions", [])
    if regs:
        lines += ["**Evidence regions.**", ""]
        for r in regs:
            lines.append(f"- `{r['tag'] or class_name(r['class'])}` — bbox {r['bbox']} "
                         f"— crop `{Path(r['crop']).name}`")
        lines.append("")
    lines += [f"_confidence {res.get('confidence')} · {res.get('latency_ms')} ms · "
              f"{res.get('n_attempts')} generation attempt(s)_"]
    return "\n".join(lines)
