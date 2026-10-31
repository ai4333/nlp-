"""
Knowledge graph store (Kùzu - embedded, MIT, speaks Cypher, no server).

Schema mirrors PIDQA's ground-truth queries exactly:
    (:Symbol {class:INT64, tag:STRING, ...})-[:CONNECTED_TO]-(:Symbol)
`class` is stored 1-based (PIDQA convention), never the 0-based YOLO id.

Kùzu was verified to execute all four PIDQA query shapes, including the
undirected 3-hop pattern `(a)-[:CONNECTED_TO]-(b)-[:CONNECTED_TO]-(c)` and
`STARTS WITH`.  Because Kùzu relationships are directed, every edge is inserted
in BOTH directions so undirected matching behaves like Neo4j's.
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import yaml
import kuzu

from ..config import load

CFG = load()
_ONT = yaml.safe_load(open(Path(__file__).resolve().parents[3] / "config" / "symbol_ontology.yaml"))["classes"]


def _category_of(pidqa_class: int) -> str:
    return _ONT.get(pidqa_class, {}).get("category", "")


def class_name(pidqa_class: int) -> str:
    return _ONT.get(pidqa_class, {}).get("name", f"class_{pidqa_class}")


class KG:
    def __init__(self, path: str | Path | None = None, fresh: bool = False,
                 read_only: bool = False):
        self.path = Path(path or CFG.get_path("graph.path"))
        if fresh:
            # Kuzu may persist as a directory OR a file (+ .wal sidecar);
            # rmtree alone silently no-ops on the file form and stale rows survive.
            for p in (self.path, Path(str(self.path) + ".wal"),
                      Path(str(self.path) + ".lock"), Path(str(self.path) + ".shadow")):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Kuzu is single-writer: a running ingest holds an exclusive lock, so
        # readers (the app, eval) must open read-only or they hit
        # "Could not set lock on file".
        self.read_only = read_only
        self.db = kuzu.Database(str(self.path), read_only=read_only)
        self.conn = kuzu.Connection(self.db)
        if not read_only:
            self._schema()

    # NOTE: edges are stored in a single direction only; see add_sheet().

    def _schema(self):
        try:
            self.conn.execute("""
                CREATE NODE TABLE Symbol(
                    entity_id STRING, class INT64, name STRING, tag STRING,
                    sheet_id STRING, x0 INT64, y0 INT64, x1 INT64, y1 INT64,
                    confidence DOUBLE, tag_confidence DOUBLE, needs_review BOOLEAN,
                    category STRING, size_in DOUBLE,
                    PRIMARY KEY(entity_id))""")
        except RuntimeError:
            pass
        for rel in ("CONNECTED_TO", "CONTROLS", "FEEDS_TO", "CONTINUES_ON"):
            # Separate tables per type: Kuzu requires an explicit schema, and
            # keeping them apart lets a query ask for control loops specifically
            # rather than filtering a property.
            try:
                self.conn.execute(
                    f"CREATE REL TABLE {rel}(FROM Symbol TO Symbol, "
                    "confidence DOUBLE, evidence STRING)")
            except RuntimeError:
                pass

    # ---------------- loading ----------------
    def add_sheet(self, sheet_id: str, entities: list[dict], relations: list[dict],
                  derived: dict[str, list[dict]] | None = None):
        for e in entities:
            x0, y0, x1, y1 = e["bbox"]
            self.conn.execute(
                """CREATE (:Symbol {entity_id:$i, class:$c, name:$n, tag:$t, sheet_id:$s,
                     x0:$a, y0:$b, x1:$d, y1:$f, confidence:$cf,
                     tag_confidence:$tc, needs_review:$nr,
                     category:$cat, size_in:$sz})""",
                # Blueprint-mode entities carry their own name, read from the tag
                # prefix ("compressor", "vessel / separator"). Forcing the 1..32
                # synthetic ontology here threw those away and every item came
                # back as "class_101".
                {"i": f"{sheet_id}:{e['id']}", "c": int(e["pidqa_class"]),
                 "n": e.get("name") or class_name(int(e["pidqa_class"])),
                 "t": e.get("tag", "") or "",
                 "s": str(sheet_id), "a": int(x0), "b": int(y0), "d": int(x1), "f": int(y1),
                 "cf": float(e.get("confidence", 1.0)),
                 "tc": float(e.get("tag_confidence", 0.0)),
                 "nr": bool(e.get("needs_review", False)),
                 # categorical facets: filtering on these is exact, where
                 # matching a descriptive name ("restriction orifice") for the
                 # word "instrument" silently returns nothing.
                 "cat": e.get("category") or _category_of(int(e["pidqa_class"])),
                 "sz": float(e.get("size_in") or 0.0)})
        for r in relations:
            # Insert ONCE only.  Kuzu's `-[:R]-` pattern is already undirected, so
            # storing both directions makes every undirected match fire twice and
            # silently doubles every count-based answer.
            self.conn.execute(
                """MATCH (u:Symbol),(v:Symbol)
                   WHERE u.entity_id=$a AND v.entity_id=$b
                   CREATE (u)-[:CONNECTED_TO {confidence:$c, evidence:$e}]->(v)""",
                {"a": f"{sheet_id}:{r['source']}", "b": f"{sheet_id}:{r['target']}",
                 "c": float(r.get("confidence", 0.8)), "e": r.get("evidence", "")})

        for rtype, rels in (derived or {}).items():
            if rtype not in ("CONTROLS", "FEEDS_TO", "CONTINUES_ON"):
                continue
            for r in rels:
                self.conn.execute(
                    f"""MATCH (u:Symbol),(v:Symbol)
                        WHERE u.entity_id=$a AND v.entity_id=$b
                        CREATE (u)-[:{rtype} {{confidence:$c, evidence:$e}}]->(v)""",
                    {"a": f"{sheet_id}:{r['source']}", "b": f"{sheet_id}:{r['target']}",
                     "c": float(r.get("confidence", 0.8)), "e": r.get("evidence", "")})

    # ---------------- querying ----------------
    def run(self, cypher: str, params: dict | None = None):
        res = self.conn.execute(cypher, params or {})
        return res.get_as_df()

    def scoped(self, cypher: str, sheet_id: str) -> str:
        """Constrain a sheet-agnostic query to one sheet.

        PIDQA's reference Cypher assumes a database holding exactly one drawing;
        ours holds many, so every Symbol pattern gets a sheet_id predicate.

        EXCEPT for CONTINUES_ON, which is the one relation that spans sheets by
        definition — scoping both ends of it asks for a cross-sheet link whose
        two ends are on the same sheet, which can never match. There, only the
        near side is pinned.
        """
        import re
        out = cypher
        vars_ = list(dict.fromkeys(re.findall(r"\(\s*(\w+)\s*:\s*Symbol", cypher)))
        if not vars_:
            return out
        if re.search(r"CONTINUES_ON", cypher, re.I):
            vars_ = vars_[:1]
        cond = " AND ".join(f"{v}.sheet_id = '{sheet_id}'" for v in vars_)
        if re.search(r"\bWHERE\b", out, re.I):
            out = re.sub(r"\bWHERE\b", f"WHERE {cond} AND ", out, count=1, flags=re.I)
        else:
            m = re.search(r"\bRETURN\b", out, re.I)
            out = out[:m.start()] + f"WHERE {cond} " + out[m.start():] if m else out
        return out

    def add_cross_sheet(self, links: list[dict]) -> int:
        """Join drawings that hand off to each other.

        These edges span sheets by design, so unlike everything else they are
        NOT written with a sheet prefix pair from one sheet.
        """
        n = 0
        for l in links:
            try:
                self.conn.execute(
                    """MATCH (u:Symbol),(v:Symbol)
                       WHERE u.entity_id=$a AND v.entity_id=$b
                       CREATE (u)-[:CONTINUES_ON {confidence:$c, evidence:$e}]->(v)""",
                    {"a": f"{l['source_sheet']}:{l['source']}",
                     "b": f"{l['target_sheet']}:{l['target']}",
                     "c": float(l.get("confidence", 0.75)),
                     "e": f"via {l.get('via','')}"})
                n += 1
            except Exception:
                pass
        return n

    def search_tag(self, term: str, limit: int = 40) -> list[dict]:
        """Find a tag across EVERY indexed sheet.

        On a real facility the same equipment appears on several drawings, and
        "which sheets is P-101 on?" is one of the first things anyone asks. The
        per-sheet scoping used for benchmark queries is deliberately bypassed here.
        """
        t = (term or "").strip()
        if not t:
            return []
        df = self.run(
            "MATCH (s:Symbol) WHERE s.tag <> '' AND "
            "(toUpper(s.tag) CONTAINS toUpper($t)) "
            "RETURN s.sheet_id AS sheet, s.tag AS tag, s.class AS class, "
            "s.name AS name, s.entity_id AS eid, s.x0 AS x0, s.y0 AS y0, "
            "s.x1 AS x1, s.y1 AS y1 ORDER BY s.sheet_id LIMIT $k",
            {"t": t, "k": int(limit)})
        return [{"sheet": r["sheet"], "tag": r["tag"], "class": int(r["class"]),
                 "name": r["name"], "entity_id": r["eid"],
                 "bbox": [int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])]}
                for _, r in df.iterrows()]

    def sheet_summary(self) -> list[dict]:
        # Kuzu rejects ORDER BY on the pre-aggregation expression — it must be
        # the alias. This raised "Variable s is not in scope" for anyone who
        # called it.
        df = self.run("MATCH (s:Symbol) RETURN s.sheet_id AS sheet, count(s) AS n "
                      "ORDER BY sheet")
        return [{"sheet": r["sheet"], "symbols": int(r["n"])} for _, r in df.iterrows()]

    def counts(self) -> dict:
        n = self.run("MATCH (s:Symbol) RETURN count(s) AS n")["n"][0]
        out = {"symbols": int(n)}
        for rel in ("CONNECTED_TO", "CONTROLS", "FEEDS_TO", "CONTINUES_ON"):
            try:
                out[rel.lower()] = int(self.run(
                    f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n")["n"][0])
            except Exception:
                out[rel.lower()] = 0
        out["edges_directed"] = out["connected_to"]
        return out

    def close(self):
        self.conn.close(); self.db.close()
