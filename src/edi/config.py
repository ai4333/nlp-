"""Central config loader. Every module reads settings from here, never hard-codes paths."""
from __future__ import annotations
import os, yaml
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]

class Cfg(dict):
    """dict with attribute access and dotted lookup: cfg.get_path('paths.tiles')"""
    def __getattr__(self, k):
        v = self[k]
        return Cfg(v) if isinstance(v, dict) else v

    def dotted(self, path, default=None):
        cur = self
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def get_path(self, dotted_key) -> Path:
        v = self.dotted(dotted_key)
        if v is None:
            raise KeyError(dotted_key)
        p = Path(v)
        return p if p.is_absolute() else ROOT / p


@lru_cache(maxsize=1)
def load(path: str | None = None) -> Cfg:
    p = Path(path) if path else ROOT / "config" / "pipeline.yaml"
    with open(p) as f:
        return Cfg(yaml.safe_load(f))


# ---- the single source of truth for the class-id convention ----------------
# YOLO labels are 0-based (0..31).  PIDQA questions and Cypher are 1-based (1..32).
# Mixing these silently corrupts every benchmark answer, so both directions live
# here and nowhere else.  See tests/test_class_ids.py

def yolo_to_pidqa(yolo_class: int) -> int:
    if not 0 <= yolo_class <= 31:
        raise ValueError(f"yolo class out of range 0..31: {yolo_class}")
    return yolo_class + 1

def pidqa_to_yolo(pidqa_class: int) -> int:
    if not 1 <= pidqa_class <= 32:
        raise ValueError(f"pidqa class out of range 1..32: {pidqa_class}")
    return pidqa_class - 1
