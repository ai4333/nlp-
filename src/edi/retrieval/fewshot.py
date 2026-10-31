"""
Few-shot example selection for NL->Cypher.

The reference implementation (PID-KnowledgeGraph-demo) used OpenAI embeddings +
FAISS to pick similar solved question/Cypher pairs.  We need the same mechanism
for free, and under an 18 GB memory ceiling.

Backend choice, measured rather than assumed: bge-m3 cost 93s to load when the
GPU was already busy, and keeping it resident spends 1.2 GB of the query-phase
budget.  PIDQA questions are highly templated ("How many symbols are categorized
as N?"), so character+word TF-IDF matches the right template essentially always
and costs nothing.  bge-m3 stays available behind the same interface for
document retrieval, where wording actually varies.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import load

CFG = load()
FILES = {
    "simple_counting":    "Simple Counting/simple_counting.csv",
    "spatial_counting":   "Spatial Counting/spatial_counting.csv",
    "spatial_connection": "Spatial Connections/spatial_connectivity.csv",
    "value_based":        "Value/value_based.csv",
}


def _clean(q: str) -> str:
    return " ".join(str(q).split())


class FewShotIndex:
    """Selects the most similar solved (question -> cypher) pairs."""

    def __init__(self, examples: list[dict]):
        self.examples = examples
        corpus = [_clean(e["question"]) for e in examples]
        self.vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                   sublinear_tf=True, min_df=1)
        self.M = self.vec.fit_transform(corpus)
        # L2-normalise so a dot product is cosine similarity
        norms = np.sqrt(self.M.multiply(self.M).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        self._norms = norms

    def query(self, question: str, k: int = 6, exclude_sheets: set | None = None) -> list[dict]:
        v = self.vec.transform([_clean(question)])
        sims = (self.M @ v.T).toarray().ravel() / self._norms
        order = np.argsort(-sims)
        out = []
        for i in order:
            e = self.examples[i]
            if exclude_sheets and e.get("sheet") in exclude_sheets:
                continue          # never leak the sheet under test
            out.append({**e, "score": float(sims[i])})
            if len(out) >= k:
                break
        return out


def build_index(exclude_sheets: set[str] | None = None,
                per_type: int = 400, seed: int = 0) -> FewShotIndex:
    """Sample solved examples from PIDQA, holding out the sheets under test."""
    root = CFG.get_path("paths.pidqa")
    rng = np.random.default_rng(seed)
    ex: list[dict] = []
    excl = {int(s) for s in (exclude_sheets or set())}
    for qtype, rel in FILES.items():
        df = pd.read_csv(root / rel)
        df = df[~df["P&ID_number"].isin(excl)]
        if len(df) > per_type:
            df = df.iloc[rng.choice(len(df), per_type, replace=False)]
        for _, r in df.iterrows():
            ex.append({"question": r["Question"], "cypher": " ".join(str(r["Cypher"]).split()),
                       "qtype": qtype, "sheet": int(r["P&ID_number"])})
    # Seed exemplars for question shapes PIDQA lacks (tag-based neighbour
    # queries, listing, degree).  Repeated so they compete on similarity.
    seed_p = Path(__file__).resolve().parents[3] / "config" / "seed_examples.yaml"
    if seed_p.exists():
        for e in yaml.safe_load(open(seed_p))["examples"]:
            ex.append({"question": e["question"], "cypher": e["cypher"],
                       "qtype": e["qtype"], "sheet": -1})
    return FewShotIndex(ex)
