"""
Reference-document retrieval — the third leg of the multimodal evidence.

The graph says what connects to what; the drawing crop proves it visually; the
specification document says what the thing actually is — pressure rating,
material, trip setpoint. An answer that cites all three is checkable; one that
cites only the graph is a claim.

Retrieval is lexical (word + character TF-IDF) rather than dense, on purpose:
the queries that matter here are dominated by exact identifiers — "P-101",
"RO-10", "ZLO 433" — and a dense embedder is actively worse at those than
character n-grams. It also keeps the query phase free of a 1.2 GB model, which
matters on an 18 GB machine (see INPUT.md §A.3).
"""
from __future__ import annotations
import json, pickle, re
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from ..config import load

CFG = load()
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    doc_id: str
    source: str
    page: int
    idx: int
    text: str


def _store_dir() -> Path:
    d = CFG.get_path("paths.index") / "documents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _split(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= CHUNK_CHARS:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf:
                out.append(buf)
            buf = (buf[-CHUNK_OVERLAP:] + "\n" + p).strip() if buf else p
            while len(buf) > CHUNK_CHARS:
                out.append(buf[:CHUNK_CHARS]); buf = buf[CHUNK_CHARS - CHUNK_OVERLAP:]
    if buf:
        out.append(buf)
    return out


def read_document(path: str | Path) -> list[tuple[int, str]]:
    """Return [(page, text)]. PDFs page-by-page so citations can name a page."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        import fitz
        doc = fitz.open(p)
        pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
        doc.close()
        return [(n, t) for n, t in pages if t.strip()]
    if p.suffix.lower() in {".txt", ".md", ".csv"}:
        return [(1, p.read_text(errors="ignore"))]
    if p.suffix.lower() == ".docx":
        try:
            import docx
            return [(1, "\n\n".join(par.text for par in docx.Document(str(p)).paragraphs))]
        except Exception as e:
            raise ValueError(f"cannot read .docx ({e}); export to PDF")
    raise ValueError(f"unsupported document type: {p.suffix}")


class DocumentStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vec_w = self.vec_c = None
        self.M = None
        self._load()

    # ---------------- persistence ----------------
    @property
    def _meta(self) -> Path: return _store_dir() / "chunks.json"
    @property
    def _model(self) -> Path: return _store_dir() / "index.pkl"

    def _load(self):
        if self._meta.exists():
            self.chunks = [Chunk(**c) for c in json.loads(self._meta.read_text())]
        if self._model.exists() and self.chunks:
            try:
                self.vec_w, self.vec_c, self.M = pickle.loads(self._model.read_bytes())
            except Exception:
                self._fit()

    def _save(self):
        self._meta.write_text(json.dumps([asdict(c) for c in self.chunks], indent=1))
        if self.M is not None:
            self._model.write_bytes(pickle.dumps((self.vec_w, self.vec_c, self.M)))

    # ---------------- indexing ----------------
    def _fit(self):
        if not self.chunks:
            self.M = None; return
        corpus = [c.text for c in self.chunks]
        self.vec_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     sublinear_tf=True, min_df=1, stop_words="english")
        self.vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     sublinear_tf=True, min_df=1, max_features=60000)
        Mw = self.vec_w.fit_transform(corpus)
        Mc = self.vec_c.fit_transform(corpus)
        self.M = hstack([Mw, Mc]).tocsr()
        norms = np.sqrt(self.M.multiply(self.M).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        self._norms = norms

    def add(self, path: str | Path) -> int:
        p = Path(path)
        doc_id = p.stem
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]   # replace
        n = 0
        for page, text in read_document(p):
            for i, ch in enumerate(_split(text)):
                if len(ch.strip()) < 40:
                    continue
                self.chunks.append(Chunk(doc_id, p.name, page, i, ch))
                n += 1
        self._fit(); self._save()
        return n

    def remove(self, doc_id: str):
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        self._fit(); self._save()

    def clear(self):
        self.chunks = []; self.M = None
        for f in (self._meta, self._model):
            if f.exists():
                f.unlink()

    # ---------------- retrieval ----------------
    def search(self, query: str, k: int = 4, min_score: float = 0.11,
               rel_cut: float = 0.55) -> list[dict]:
        """Retrieve up to k chunks.

        Two gates, because a lexical index always returns *something*: an
        absolute floor, and a relative one that drops anything far weaker than
        the best hit. Without them a pure counting question came back with eight
        irrelevant excerpts scoring 0.05-0.12, which is evidence-shaped noise.
        """
        if self.M is None or not self.chunks:
            return []
        if not hasattr(self, "_norms"):
            self._fit()
        q = hstack([self.vec_w.transform([query]), self.vec_c.transform([query])]).tocsr()
        sims = (self.M @ q.T).toarray().ravel() / self._norms
        qn = float(np.sqrt((q.multiply(q)).sum())) or 1.0
        sims = sims / qn
        order = np.argsort(-sims)
        top = float(sims[order[0]]) if len(order) else 0.0
        floor = max(min_score, top * rel_cut)
        out = []
        for i in order[:k]:
            if sims[i] < floor:
                break
            c = self.chunks[i]
            out.append({"source": c.source, "page": c.page, "score": round(float(sims[i]), 4),
                        "text": c.text, "doc_id": c.doc_id})
        return out

    def stats(self) -> dict:
        docs = {}
        for c in self.chunks:
            docs.setdefault(c.doc_id, {"source": c.source, "chunks": 0, "pages": set()})
            docs[c.doc_id]["chunks"] += 1
            docs[c.doc_id]["pages"].add(c.page)
        return {d: {"source": v["source"], "chunks": v["chunks"], "pages": len(v["pages"])}
                for d, v in docs.items()}
