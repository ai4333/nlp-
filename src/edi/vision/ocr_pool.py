"""
Process-level parallel OCR.

PaddleOCR on Apple Silicon is CPU-only and effectively single-threaded: measured
~1000 ms/crop at OMP_NUM_THREADS of 1, 6 and 10 alike, so raising the OpenMP
thread count buys nothing. The only way to use the other cores is separate
processes, each holding its own recogniser.

The pool is created once and reused, because model load is ~8 s per worker and
would otherwise dominate. Workers are capped conservatively: each holds roughly
1 GB, and this machine has 18 GB shared with everything else.
"""
from __future__ import annotations
import os
from multiprocessing import get_context
from typing import Sequence
import numpy as np

_POOL = None
_WORKERS = 0
_ENGINE = None


def _init_worker():
    global _ENGINE
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from .ocr import OcrEngine
    _ENGINE = OcrEngine.get()


def _work(batch):
    global _ENGINE
    if _ENGINE is None:
        _init_worker()
    return _ENGINE._run(list(batch))


def default_workers() -> int:
    try:
        cpu = os.cpu_count() or 4
    except Exception:
        cpu = 4
    # leave headroom for the OS and whatever else is resident
    return max(1, min(5, cpu - 3))


def start(workers: int | None = None):
    global _POOL, _WORKERS
    n = workers if workers is not None else default_workers()
    if _POOL is not None and _WORKERS == n:
        return _POOL
    stop()
    if n <= 1:
        _WORKERS = 1
        return None
    ctx = get_context("spawn")
    _POOL = ctx.Pool(processes=n, initializer=_init_worker)
    _WORKERS = n
    return _POOL


def stop():
    global _POOL, _WORKERS
    if _POOL is not None:
        try:
            _POOL.terminate(); _POOL.join()
        except Exception:
            pass
    _POOL = None; _WORKERS = 0


def run(images: Sequence[np.ndarray], fallback_engine=None) -> list[list[dict]]:
    """OCR a list of images, spreading them over the pool. Order is preserved."""
    if not images:
        return []
    pool = _POOL or start()
    if pool is None:
        eng = fallback_engine
        if eng is None:
            from .ocr import OcrEngine
            eng = OcrEngine.get()
        return eng._run(list(images))

    n = _WORKERS
    # contiguous shards keep each worker's internal batching effective
    size = (len(images) + n - 1) // n
    shards = [images[i:i + size] for i in range(0, len(images), size)]
    out: list[list[dict]] = []
    for part in pool.map(_work, shards):
        out.extend(part)
    return out
