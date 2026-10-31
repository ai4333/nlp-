"""Document retrieval - the third evidence leg."""
from src.edi.retrieval.documents import _split, DocumentStore, CHUNK_CHARS


def test_split_respects_chunk_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(12))
    chunks = _split(text)
    assert chunks
    assert all(len(c) <= CHUNK_CHARS + 5 for c in chunks)


def test_split_short_text_is_one_chunk():
    assert len(_split("A short specification note.")) == 1


def test_search_on_empty_store_returns_nothing():
    ds = DocumentStore.__new__(DocumentStore)
    ds.chunks, ds.M = [], None
    assert ds.search("anything") == []
