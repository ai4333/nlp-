#!/usr/bin/env python
"""Audit every dataset the project depends on: present? complete? usable?"""
import sys, json, hashlib
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.edi.config import load

CFG = load()
R = []

def row(name, status, detail): R.append((status, name, detail))

# 1 Dataset-P&ID -----------------------------------------------------------
d = CFG.get_path("paths.dataset")
try:
    tr_i = sorted((d/"images"/"train").glob("*.jpg")); va_i = sorted((d/"images"/"val").glob("*.jpg"))
    tr_l = sorted((d/"labels"/"train").glob("*.txt")); va_l = sorted((d/"labels"/"val").glob("*.txt"))
    pair_ok = ({p.stem for p in tr_i} == {p.stem for p in tr_l} and
               {p.stem for p in va_i} == {p.stem for p in va_l})
    boxes = sum(1 for p in (tr_l+va_l) for ln in p.read_text().split("\n") if len(ln.split())==5)
    cls = Counter(int(ln.split()[0]) for p in (tr_l+va_l)
                  for ln in p.read_text().split("\n") if len(ln.split())==5)
    row("Dataset-P&ID (images+labels)", "OK" if pair_ok else "WARN",
        f"{len(tr_i)} train + {len(va_i)} val sheets · {boxes:,} boxes · "
        f"{len(cls)} classes ({min(cls)}..{max(cls)}) · image/label pairing "
        f"{'complete' if pair_ok else 'MISMATCHED'}")
except Exception as e:
    row("Dataset-P&ID", "MISSING", str(e)[:80])

# 2 PIDQA ------------------------------------------------------------------
try:
    p = CFG.get_path("paths.pidqa")
    tot = 0; parts = []
    for f in ["Simple Counting/simple_counting.csv","Spatial Counting/spatial_counting.csv",
              "Spatial Connections/spatial_connectivity.csv","Value/value_based.csv"]:
        df = pd.read_csv(p/f); tot += len(df); parts.append(f"{f.split('/')[0]}:{len(df):,}")
    sheets = sorted(pd.read_csv(p/"Value/value_based.csv")["P&ID_number"].unique())
    row("PIDQA benchmark", "OK",
        f"{tot:,} questions · sheets {min(sheets)}..{max(sheets)} ({len(sheets)}) · " + " ".join(parts))
except Exception as e:
    row("PIDQA benchmark", "MISSING", str(e)[:80])

# 3 tiles ------------------------------------------------------------------
for name, key in [("Tiles 1024 (training)", "data/tiles"), ("Tiles 512 (training)", "data/tiles512")]:
    t = Path(key)
    if (t/"data.yaml").exists():
        n_tr = len(list((t/"images"/"train").glob("*.jpg")))
        n_va = len(list((t/"images"/"val").glob("*.jpg")))
        sz = sum(f.stat().st_size for f in t.rglob("*.jpg")) / 1e9
        row(name, "OK", f"{n_tr:,} train + {n_va:,} val tiles · {sz:.1f} GB")
    else:
        row(name, "ABSENT", "not built (optional — only needed to train)")

# 4 optional large corpora -------------------------------------------------
for f, why in [("data/raw/PID2Graph.zip", "real+synthetic P&IDs with graph GT"),
               ("data/raw/PID_Dataset.zip", "real industry P&IDs")]:
    p = Path(f)
    if p.exists():
        row(Path(f).name, "PARTIAL" if p.stat().st_size < 5e9 else "OK",
            f"{p.stat().st_size/1e9:.2f} GB on disk — {why}")
    else:
        row(Path(f).name, "ABSENT", f"{why} · optional · scripts/fetch_optional_datasets.sh")

# 5 models -----------------------------------------------------------------
w = Path(CFG.get_path("detection.weights"))
row("Detector weights", "OK" if w.exists() else "ABSENT",
    f"{w}" + (f" ({w.stat().st_size/1e6:.1f} MB)" if w.exists() else " — uploads need this"))
try:
    import ollama
    ms = {m.model for m in ollama.Client(host=CFG.dotted("llm.host")).list().models}
    need = ["qwen3:8b", "qwen3-vl:8b", "bge-m3:latest"]
    have = [n for n in need if n in ms]
    row("Local models (Ollama)", "OK" if len(have) == len(need) else "WARN",
        f"{len(have)}/{len(need)} present: {', '.join(have)}")
except Exception as e:
    row("Local models (Ollama)", "FAIL", str(e)[:70])

# 6 derived artefacts ------------------------------------------------------
interim = CFG.get_path("paths.interim")
sheets = [d for d in interim.glob("*") if (d/"ocr.json").exists()]
miss_rel = [d.name for d in sheets if not (d/"relations.json").exists()]
miss_der = [d.name for d in sheets if not (d/"derived.json").exists()]
row("Ingested sheets", "OK" if not miss_rel else "WARN",
    f"{len(sheets)} sheets" +
    (f" · MISSING relations: {miss_rel}" if miss_rel else "") +
    (f" · missing derived: {len(miss_der)}" if miss_der else " · all have derived relations"))

# 7 documents --------------------------------------------------------------
try:
    from src.edi.retrieval.documents import DocumentStore
    ds = DocumentStore()
    row("Reference documents", "OK" if ds.chunks else "EMPTY",
        f"{len(ds.chunks)} chunks · {len(ds.stats())} document(s)")
except Exception as e:
    row("Reference documents", "FAIL", str(e)[:70])

ICON = {"OK": "  ok  ", "WARN": " warn ", "ABSENT": "absent", "PARTIAL": "partial",
        "MISSING": " MISS ", "EMPTY": "empty ", "FAIL": " FAIL "}
print()
for st, name, detail in R:
    print(f"[{ICON.get(st,st):6}] {name:28s} {detail}")
bad = [r for r in R if r[0] in ("MISSING", "FAIL")]
print(f"\n{len(R)-len(bad)}/{len(R)} dataset checks passed"
      + (f"  ({len(bad)} blocking)" if bad else ""))
