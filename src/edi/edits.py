"""
Manual graph edits.

An extraction pipeline is never perfect, so an engineer has to be able to fix
it: add a symbol the detector missed, delete a false positive, correct a class
or tag, draw a connection the tracer did not find.

Edits are stored as an OVERLAY (`edits.json`) rather than written back into
`ocr.json` / `relations.json`. Re-running ingestion regenerates those files, and
overwriting them would silently destroy hours of human correction the next time
anyone re-processed a sheet. The overlay is replayed on top of whatever the
pipeline currently produces, so corrections survive re-extraction, model
upgrades, and parameter changes.

Every edit is appended to a log with a timestamp and the reason, so a graph can
always be traced back to what the machine said and what a person changed.
"""
from __future__ import annotations
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .config import load

CFG = load()

EMPTY = {
    "entities": {"added": [], "removed": [], "modified": {}},
    "relations": {"added": [], "removed": []},
    "log": [],
}


def _path(sheet_id: str) -> Path:
    return CFG.get_path("paths.interim") / sheet_id / "edits.json"


def load_edits(sheet_id: str) -> dict:
    p = _path(sheet_id)
    if not p.exists():
        return deepcopy(EMPTY)
    try:
        d = json.loads(p.read_text())
    except Exception:
        return deepcopy(EMPTY)
    for k, v in EMPTY.items():
        d.setdefault(k, deepcopy(v))
    for k, v in EMPTY["entities"].items():
        d["entities"].setdefault(k, deepcopy(v))
    for k, v in EMPTY["relations"].items():
        d["relations"].setdefault(k, deepcopy(v))
    return d


def save_edits(sheet_id: str, edits: dict) -> Path:
    p = _path(sheet_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(edits, indent=1))
    return p


def _log(edits: dict, action: str, detail: str) -> None:
    edits["log"].append({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": action, "detail": detail,
    })


def has_edits(edits: dict) -> bool:
    e, r = edits["entities"], edits["relations"]
    return bool(e["added"] or e["removed"] or e["modified"]
                or r["added"] or r["removed"])


def count(edits: dict) -> dict:
    e, r = edits["entities"], edits["relations"]
    return {"added": len(e["added"]), "removed": len(e["removed"]),
            "modified": len(e["modified"]),
            "rel_added": len(r["added"]), "rel_removed": len(r["removed"])}


# ---------------------------------------------------------------- mutations
def next_id(entities: list[dict], edits: dict) -> str:
    used = {e["id"] for e in entities} | {e["id"] for e in edits["entities"]["added"]}
    n = 1
    while f"m{n:04d}" in used:
        n += 1
    return f"m{n:04d}"


def add_entity(edits: dict, entity_id: str, pidqa_class: int, bbox, tag: str = "",
               reason: str = "") -> dict:
    edits["entities"]["added"].append({
        "id": entity_id, "pidqa_class": int(pidqa_class),
        "bbox": [int(v) for v in bbox], "tag": tag or "",
        "confidence": 1.0, "tag_confidence": 1.0 if tag else 0.0,
        "tag_location": "external", "source": "human",
    })
    _log(edits, "add_entity", f"{entity_id} class {pidqa_class} tag '{tag}' {reason}".strip())
    return edits


def remove_entity(edits: dict, entity_id: str, reason: str = "") -> dict:
    if entity_id in {e["id"] for e in edits["entities"]["added"]}:
        # never manually added in the first place -> just drop the addition
        edits["entities"]["added"] = [e for e in edits["entities"]["added"]
                                      if e["id"] != entity_id]
    elif entity_id not in edits["entities"]["removed"]:
        edits["entities"]["removed"].append(entity_id)
    edits["entities"]["modified"].pop(entity_id, None)
    _log(edits, "remove_entity", f"{entity_id} {reason}".strip())
    return edits


def modify_entity(edits: dict, entity_id: str, *, tag: str | None = None,
                  pidqa_class: int | None = None, reason: str = "") -> dict:
    for e in edits["entities"]["added"]:
        if e["id"] == entity_id:          # editing a manual addition in place
            if tag is not None:
                e["tag"] = tag; e["tag_confidence"] = 1.0 if tag else 0.0
            if pidqa_class is not None:
                e["pidqa_class"] = int(pidqa_class)
            _log(edits, "modify_entity", f"{entity_id} {reason}".strip())
            return edits
    m = edits["entities"]["modified"].setdefault(entity_id, {})
    if tag is not None:
        m["tag"] = tag
    if pidqa_class is not None:
        m["pidqa_class"] = int(pidqa_class)
    _log(edits, "modify_entity", f"{entity_id} -> {m} {reason}".strip())
    return edits


def _key(a: str, b: str) -> list[str]:
    return sorted([a, b])


def add_relation(edits: dict, a: str, b: str, rtype: str = "CONNECTED_TO",
                 reason: str = "") -> dict:
    if a == b:
        raise ValueError("a symbol cannot connect to itself")
    k = _key(a, b)
    edits["relations"]["removed"] = [r for r in edits["relations"]["removed"]
                                     if sorted(r[:2]) != k or (len(r) > 2 and r[2] != rtype)]
    if not any(sorted([r["source"], r["target"]]) == k and r.get("type") == rtype
               for r in edits["relations"]["added"]):
        edits["relations"]["added"].append(
            {"source": k[0], "target": k[1], "type": rtype,
             "confidence": 1.0, "evidence": "human"})
    _log(edits, "add_relation", f"{k[0]} -[{rtype}]- {k[1]} {reason}".strip())
    return edits


def remove_relation(edits: dict, a: str, b: str, rtype: str = "CONNECTED_TO",
                    reason: str = "") -> dict:
    k = _key(a, b)
    before = len(edits["relations"]["added"])
    edits["relations"]["added"] = [
        r for r in edits["relations"]["added"]
        if not (sorted([r["source"], r["target"]]) == k and r.get("type") == rtype)]
    if len(edits["relations"]["added"]) == before:
        entry = k + [rtype]
        if entry not in edits["relations"]["removed"]:
            edits["relations"]["removed"].append(entry)
    _log(edits, "remove_relation", f"{k[0]} -[{rtype}]- {k[1]} {reason}".strip())
    return edits


def undo(edits: dict) -> dict:
    """Reverse the most recent edit."""
    if not edits["log"]:
        return edits
    last = edits["log"][-1]
    act, det = last["action"], last["detail"]
    ident = det.split()[0] if det else ""
    if act == "add_entity":
        edits["entities"]["added"] = [e for e in edits["entities"]["added"] if e["id"] != ident]
    elif act == "remove_entity":
        edits["entities"]["removed"] = [x for x in edits["entities"]["removed"] if x != ident]
    elif act == "modify_entity":
        edits["entities"]["modified"].pop(ident, None)
    elif act in ("add_relation", "remove_relation"):
        parts = det.split()
        if len(parts) >= 3:
            a, b = parts[0], parts[2]
            k = _key(a, b)
            edits["relations"]["added"] = [
                r for r in edits["relations"]["added"]
                if sorted([r["source"], r["target"]]) != k]
            edits["relations"]["removed"] = [
                r for r in edits["relations"]["removed"] if sorted(r[:2]) != k]
    edits["log"].pop()
    return edits


def clear(sheet_id: str) -> None:
    p = _path(sheet_id)
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------- replay
def apply(entities: list[dict], relations: list[dict], derived: dict | None,
          edits: dict) -> tuple[list[dict], list[dict], dict]:
    """Replay the overlay on top of pipeline output. Pure - inputs untouched."""
    ent = deepcopy(entities)
    rel = deepcopy(relations)
    der = deepcopy(derived) if derived else {"CONTROLS": [], "FEEDS_TO": []}

    removed = set(edits["entities"]["removed"])
    mods = edits["entities"]["modified"]

    ent = [e for e in ent if e["id"] not in removed]
    for e in ent:
        m = mods.get(e["id"])
        if m:
            if "tag" in m:
                e["tag"] = m["tag"]; e["tag_confidence"] = 1.0 if m["tag"] else 0.0
                e["tag_source"] = "human"
            if "pidqa_class" in m:
                e["pidqa_class"] = int(m["pidqa_class"]); e["class_source"] = "human"
            e["edited"] = True
    for e in edits["entities"]["added"]:
        if e["id"] not in removed:
            ent.append({**e, "edited": True})

    alive = {e["id"] for e in ent}
    drop = {tuple(sorted(r[:2])) + (r[2] if len(r) > 2 else "CONNECTED_TO",)
            for r in edits["relations"]["removed"]}

    def keep(r, rtype):
        k = tuple(sorted([r["source"], r["target"]])) + (rtype,)
        return (r["source"] in alive and r["target"] in alive and k not in drop)

    rel = [r for r in rel if keep(r, "CONNECTED_TO")]
    for key in ("CONTROLS", "FEEDS_TO"):
        der[key] = [r for r in der.get(key, []) if keep(r, key)]

    for r in edits["relations"]["added"]:
        if r["source"] not in alive or r["target"] not in alive:
            continue
        t = r.get("type", "CONNECTED_TO")
        target = rel if t == "CONNECTED_TO" else der.setdefault(t, [])
        if not any(sorted([x["source"], x["target"]]) == sorted([r["source"], r["target"]])
                   for x in target):
            target.append({**r, "edited": True})

    return ent, rel, der


def load_sheet_edited(sheet_id: str):
    """Pipeline output for a sheet with its edit overlay already applied."""
    base = CFG.get_path("paths.interim") / sheet_id
    ents = json.loads((base / "ocr.json").read_text())["records"]
    rels = json.loads((base / "relations.json").read_text())["relations"]
    dp = base / "derived.json"
    der = json.loads(dp.read_text()) if dp.exists() else None
    return apply(ents, rels, der, load_edits(sheet_id))
