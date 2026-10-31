"""
PIDQA evaluation harness.

PIDQA ships, for every question, both a ground-truth ANSWER and a ground-truth
CYPHER query.  That enables a three-level diagnostic that separates vision
errors from language errors:

  L1  graph quality   - run PIDQA's REFERENCE Cypher on OUR graph.
                        Wrong => the fault is detection / OCR / relations.
  L2  cypher quality  - run OUR GENERATED Cypher on a ground-truth graph.
                        Wrong => the fault is the LLM.
  L3  end-to-end      - our Cypher, our graph.

This module implements L1 and L3.
"""
from __future__ import annotations
import ast, json, re, unicodedata
from pathlib import Path
import pandas as pd

from ..config import load

CFG = load()

FILES = {
    "simple_counting":   "Simple Counting/simple_counting.csv",
    "spatial_counting":  "Spatial Counting/spatial_counting.csv",
    "spatial_connection":"Spatial Connections/spatial_connectivity.csv",
    "value_based":       "Value/value_based.csv",
}


# --------------------------------------------------------------------------
# Cypher repair
# --------------------------------------------------------------------------
_STARTSWITH = re.compile(r"(STARTS\s+WITH\s+)(?!['\"])([A-Za-z0-9_\-]+)", re.I)


def repair_cypher(q: str) -> str:
    """PIDQA's value-based queries contain unquoted string literals, e.g.
    `s.tag STARTS WITH ZLO`, which is not valid Cypher in any engine.  Quote them."""
    q = _STARTSWITH.sub(lambda m: f"{m.group(1)}'{m.group(2)}'", q)
    return q.replace("\n", " ").strip()


# --------------------------------------------------------------------------
# answer normalisation
# --------------------------------------------------------------------------
def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = s.replace("°", "")                    # OCR charset has no degree sign
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return " ".join(s.split()).upper()


def parse_gt(raw, qtype: str):
    if qtype == "value_based":
        try:
            v = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except Exception:
            v = [raw]
        return {_norm_text(x) for x in (v if isinstance(v, (list, tuple, set)) else [v])}
    if qtype == "spatial_connection":
        return bool(raw) if not isinstance(raw, str) else raw.strip().lower() == "true"
    return int(raw)


def parse_result(df: pd.DataFrame, qtype: str):
    if df is None or len(df) == 0:
        return set() if qtype == "value_based" else (False if qtype == "spatial_connection" else 0)
    col = df.columns[0]
    if qtype == "value_based":
        return {_norm_text(v) for v in df[col].tolist() if str(v).strip()}
    v = df[col].iloc[0]
    if qtype == "spatial_connection":
        return bool(v)
    try:
        return int(v)
    except Exception:
        return 0


def compare(pred, gt, qtype: str) -> bool:
    if qtype == "value_based":
        return pred == gt
    return pred == gt


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def load_questions(sheets: list[str], per_type: int | None = None) -> pd.DataFrame:
    """PIDQA questions exist only for the numbered benchmark sheets. Uploaded
    drawings ("u_pid_real_p1_…") carry no ground truth, so they are excluded
    rather than parsed as integers."""
    root = CFG.get_path("paths.pidqa")
    keep = {int(s) for s in sheets if str(s).isdigit()}
    if not keep:
        return pd.DataFrame(columns=["P&ID_number", "qtype", "Question", "GT", "Cypher"])
    frames = []
    for qtype, rel in FILES.items():
        df = pd.read_csv(root / rel)
        df = df[df["P&ID_number"].isin(keep)].copy()
        df["qtype"] = qtype
        if per_type:
            df = df.groupby("P&ID_number", group_keys=False).head(per_type)
        frames.append(df[["P&ID_number", "qtype", "Question", "GT", "Cypher"]])
    return pd.concat(frames, ignore_index=True)


def evaluate_L1(kg, sheets: list[str], per_type: int | None = None,
                verbose: bool = False) -> dict:
    """Run PIDQA's reference Cypher against our graph."""
    qs = load_questions(sheets, per_type)
    rows = []
    for _, r in qs.iterrows():
        sheet = str(r["P&ID_number"])
        qtype = r["qtype"]
        cy = kg.scoped(repair_cypher(r["Cypher"]), sheet)
        try:
            df = kg.run(cy)
            pred = parse_result(df, qtype)
            err = ""
        except Exception as e:
            pred, err = (set() if qtype == "value_based" else 0), f"{type(e).__name__}: {e}"[:120]
        gt = parse_gt(r["GT"], qtype)
        ok = compare(pred, gt, qtype) and not err
        rows.append({"sheet": sheet, "qtype": qtype, "question": r["Question"],
                     "gt": gt, "pred": pred, "ok": ok, "error": err, "cypher": cy})
        if verbose and not ok:
            print(f"  MISS [{qtype}] {r['Question'][:70]}\n      gt={gt} pred={pred} {err}")

    res = pd.DataFrame(rows)
    by_type = res.groupby("qtype")["ok"].agg(["mean", "count"]).to_dict("index")
    return {
        "level": "L1_graph_quality",
        "sheets": len(sheets),
        "n_questions": len(res),
        "overall_accuracy": round(float(res["ok"].mean()), 4) if len(res) else 0.0,
        "by_type": {k: {"accuracy": round(v["mean"], 4), "n": int(v["count"])}
                    for k, v in by_type.items()},
        "n_errors": int((res["error"] != "").sum()),
        "_frame": res,
    }


def evaluate_L3(kg, generator, sheets: list[str], per_type: int = 5,
                verbose: bool = False) -> dict:
    """End-to-end: OUR generated Cypher on OUR graph.

    Slow - one LLM call per question (plus repairs) - so this is sampled, not
    exhaustive.  Reported alongside L1 the gap between them is exactly the
    language model's contribution to the error.
    """
    qs = load_questions(sheets, per_type)
    rows = []
    for _, r in qs.iterrows():
        sheet, qtype = str(r["P&ID_number"]), r["qtype"]
        gt = parse_gt(r["GT"], qtype)
        try:
            gen = generator.generate(r["Question"], kg=kg, sheet_id=sheet,
                                     exclude_sheets={int(sheet)})
            cy = gen["cypher"]
            df = kg.run(kg.scoped(cy, sheet))
            pred, err = parse_result(df, qtype), ""
            attempts = gen["n_attempts"]
        except Exception as e:
            pred, err, cy, attempts = (set() if qtype == "value_based" else 0), \
                f"{type(e).__name__}: {e}"[:120], "", 0
        ok = compare(pred, gt, qtype) and not err
        rows.append({"sheet": sheet, "qtype": qtype, "question": r["Question"],
                     "gt": gt, "pred": pred, "ok": ok, "error": err,
                     "cypher": cy, "attempts": attempts})
        if verbose:
            print(f"  {'OK ' if ok else 'MISS'} [{qtype}] {r['Question'][:58]}", flush=True)

    res = pd.DataFrame(rows)
    by_type = res.groupby("qtype")["ok"].agg(["mean", "count"]).to_dict("index")
    executable = float((res["error"] == "").mean()) if len(res) else 0.0
    return {
        "level": "L3_end_to_end",
        "sheets": len(sheets),
        "n_questions": len(res),
        "overall_accuracy": round(float(res["ok"].mean()), 4) if len(res) else 0.0,
        "executable_query_rate": round(executable, 4),
        "by_type": {k: {"accuracy": round(v["mean"], 4), "n": int(v["count"])}
                    for k, v in by_type.items()},
        "n_errors": int((res["error"] != "").sum()),
        "_frame": res,
    }


def save_report(report: dict, name: str) -> Path:
    out = CFG.get_path("paths.reports"); out.mkdir(parents=True, exist_ok=True)
    frame = report.pop("_frame", None)
    p = out / f"{name}.json"
    p.write_text(json.dumps(report, indent=2, default=str))
    if frame is not None:
        frame.to_csv(out / f"{name}_detail.csv", index=False)
        report["_frame"] = frame
    return p
