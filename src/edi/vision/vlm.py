"""
Vision-language fallback for tags OCR cannot read.

Scope is deliberately narrow.  The VLM is NOT a detector and NOT a page reader:
a 7168x4562 sheet downsamples internally to ~1000px, at which point a symbol is
6px wide and the model confabulates.  DesignQA (2024) found GPT-4-class models
fail at exactly this, and qwen3-vl:8b is far smaller.

It is used on ONE case only: a single symbol crop, a few hundred pixels wide,
where PaddleOCR returned nothing or something that fails the completeness check.
Measured, that is ~20% of value-based failures.

Cost: ~2-6s per crop versus ~0.4s for OCR, and 7 GB resident.  So it is opt-in
(`--vlm`) and runs after the OCR pass, never instead of it.
"""
from __future__ import annotations
import base64, io, re
import cv2
import numpy as np
import ollama

from ..config import load

CFG = load()

_PROMPT = (
    "This is a close-up crop from an engineering P&ID drawing.\n"
    "Read the equipment tag text printed on or beside the symbol.\n"
    "Tags look like 'SDL 101', 'ZLO 433', 'RO-10 653', 'UV-05201', 'ERV-8-20'.\n"
    "Reply with ONLY the tag text. If there is no readable tag, reply exactly: NONE"
)


def _png_b64(img: np.ndarray) -> str:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return base64.b64encode(buf.tobytes()).decode()


class VlmReader:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or CFG.dotted("llm.vlm_model", "qwen3-vl:8b")
        self.client = ollama.Client(host=host or CFG.dotted("llm.host"))

    def read_tag(self, crop: np.ndarray, upscale: float = 4.0) -> tuple[str, float]:
        big = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        try:
            r = self.client.chat(
                model=self.model, think=False,
                options={"temperature": 0.0, "num_predict": 32},
                messages=[{"role": "user", "content": _PROMPT, "images": [_png_b64(big)]}])
            txt = (r["message"]["content"] or "").strip()
        except Exception:
            return "", 0.0
        txt = re.sub(r"^[`'\"]+|[`'\"]+$", "", txt.split("\n")[0]).strip()
        if not txt or txt.upper() in {"NONE", "N/A", "NO TAG"}:
            return "", 0.0
        # a VLM read is inherently less certain than a confident OCR read
        return txt, 0.55

    def unload(self):
        """Evict the model so the query phase gets its memory back."""
        try:
            self.client.generate(model=self.model, prompt="", keep_alive=0)
        except Exception:
            pass


def repair(image: np.ndarray, records: list[dict], score_fn,
           threshold: float = 0.4, max_items: int = 40) -> int:
    """Re-read low-confidence tags with the VLM. Returns how many were changed."""
    reader = VlmReader()
    H, W = image.shape[:2]
    fixed = 0
    targets = [r for r in records
               if r.get("tag_location") != "none" and score_fn(r.get("tag", "")) < threshold]
    for r in targets[:max_items]:
        x0, y0, x1, y1 = r["bbox"]
        pad = 12 if r.get("tag_location") == "internal" else 110
        crop = image[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
        if crop.size == 0:
            continue
        txt, conf = reader.read_tag(crop)
        if txt and score_fn(txt) > score_fn(r.get("tag", "")):
            r["tag"], r["tag_confidence"], r["tag_source"] = txt, conf, "vlm"
            fixed += 1
    reader.unload()
    return fixed
