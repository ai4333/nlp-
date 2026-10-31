"""
OCR for engineering drawings.

Two facts from inspecting the corpus (scratch/atlas/*.png) drive this design:

1. Instrument symbols (PIDQA classes 26-32) print their tag INSIDE the symbol
   box - e.g. a bubble reading "SDL" / "101".  For those, the tag is whatever
   OCR finds within the box; no spatial matching is needed at all.
2. Valves and fittings carry their tag OUTSIDE, floating nearby, so those need a
   padded search region and a nearest-text rule.

The ontology's `tag_location` field decides which path a class takes, and classes
marked `none` are skipped entirely - roughly 15% of all instances, most of them
flanges, which never carry a tag and would otherwise dominate the OCR bill.

Performance note: the default PaddleOCR pipeline also runs document-orientation
classification and unwarping.  Both are pointless on a small upright crop from a
vector-rendered sheet and cost ~6x.  Disabling them took a crop from 2459ms to
398ms (measured).
"""
from __future__ import annotations
import json, os, re
from pathlib import Path
import cv2
import numpy as np
import yaml

from ..config import load

CFG = load()
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

_ONTOLOGY = yaml.safe_load(open(Path(__file__).resolve().parents[3] / "config" / "symbol_ontology.yaml"))
CLASSES = _ONTOLOGY["classes"]
TAG_PREFIXES = _ONTOLOGY.get("tag_prefixes", {})

UPSCALE = 3          # tiny glyphs; cubic upscale materially improves recognition
BATCH = 48

# Fast mode skips the multi-scale retry ladders. Roughly 2-3x faster
# (measured: ~55s vs ~150s per sheet) at a real accuracy cost, for live demos
# and first-pass triage rather than production extraction.
FAST_MODE = False

# Process-level parallelism. Paddle is single-threaded on this platform, so the
# only way to use the other cores is separate processes. Measured 1.93x with 3
# workers on 40 crops, byte-identical output. 0 = serial.
WORKERS = 0


def set_workers(n: int) -> None:
    global WORKERS
    WORKERS = max(0, int(n))


def set_fast_mode(on: bool) -> None:
    global FAST_MODE
    FAST_MODE = bool(on)


def tag_location(pidqa_class: int) -> str:
    return CLASSES.get(pidqa_class, {}).get("tag_location", "external")


class OcrEngine:
    _inst = None

    def __init__(self):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            lang=CFG.dotted("ocr.lang", "en"), device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=CFG.dotted("ocr.use_angle_cls", True),
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        )
        self.min_conf = CFG.dotted("ocr.min_confidence", 0.6)

    @classmethod
    def get(cls) -> "OcrEngine":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def _run(self, images: list[np.ndarray]) -> list[list[dict]]:
        if not images:
            return []
        out: list[list[dict]] = []
        for s in range(0, len(images), BATCH):
            for r in self.ocr.predict(images[s:s + BATCH]):
                texts = r.get("rec_texts", []) or []
                scores = r.get("rec_scores", []) or []
                polys = r.get("rec_polys", r.get("dt_polys", [])) or []
                lines = []
                for i, t in enumerate(texts):
                    sc = float(scores[i]) if i < len(scores) else 0.0
                    if not t or sc < self.min_conf:
                        continue
                    if i < len(polys):
                        p = np.asarray(polys[i]).reshape(-1, 2)
                        box = [float(p[:, 0].min()), float(p[:, 1].min()),
                               float(p[:, 0].max()), float(p[:, 1].max())]
                    else:
                        box = [0.0, 0.0, 0.0, 0.0]
                    lines.append({"text": t.strip(), "score": round(sc, 3), "box": box})
                out.append(lines)
        return out


def _run_batch(eng, images):
    """Run a batch serially, or across the worker pool when one is configured."""
    if WORKERS and len(images) >= 8:
        from . import ocr_pool
        ocr_pool.start(WORKERS)
        return ocr_pool.run(images, fallback_engine=eng)
    return eng._run(images)


def _prep(img: np.ndarray, x0, y0, x1, y1, pad: int, upscale: float = UPSCALE) -> tuple[np.ndarray, int, int]:
    H, W = img.shape[:2]
    a, b = max(0, x0 - pad), max(0, y0 - pad)
    c, d = min(W, x1 + pad), min(H, y1 + pad)
    crop = img[b:d, a:c]
    if crop.size == 0:
        crop = np.full((8, 8), 255, np.uint8)
    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop
    crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return crop, a, b


def _assemble(lines: list[dict]) -> tuple[str, float]:
    """Join the text lines found inside a bubble, top-to-bottom, into one tag."""
    if not lines:
        return "", 0.0
    ordered = sorted(lines, key=lambda l: l["box"][1])
    txt = " ".join(l["text"] for l in ordered).strip()
    conf = float(np.mean([l["score"] for l in ordered]))
    return txt, round(conf, 3)



# --- tag quality scoring & multi-scale retry --------------------------------
# Internal bubble tags follow tight patterns (e.g. "SDL 101", "ZLO 433",
# "RO-10 653").  A single crop/scale sometimes misses them entirely - measured:
# the ZLO 433 bubble on sheet 0 reads nothing at pad=4/3x but reads correctly at
# pad=8/4x.  So we score each attempt and escalate only for the ones that look
# wrong, keeping the fast path a single batch.

_RETRY_LADDER = [(8, 4.0), (2, 5.0), (12, 3.5)]   # (pad_px, upscale)

# External tags in this corpus look like "UV-05201", "IJ-43075", "ERV-8-20":
# a letter prefix, then either one long digit run or dash-separated groups.
# Truncation is the dominant OCR failure (measured: "IJ 43075" read as
# "IJ 4307", "ST 47622" as "ST 476"), so completeness is scored explicitly and
# anything that looks clipped is re-read on a wider crop.
# Corpus-specific: every simple-form tag observed in Dataset-P&ID has a
# 2-letter prefix and exactly FIVE digits (UV-05201, IJ-43075, ST-47622).
# Accepting 4 digits scored the truncated "IJ 4307" as complete, so the count
# is pinned. Widen this for drawings with a different tagging convention.
_EXT_FULL = re.compile(r"^[A-Z]{1,4}[-\s]?\d{5}$", re.I)
_EXT_SHORT = re.compile(r"^[A-Z]{1,4}[-\s]?\d{4}$", re.I)
_EXT_GROUPED = re.compile(r"^[A-Z]{1,4}[-\s]\d{1,2}[-\s]\d{1,2}$", re.I)
_EXT_PARENS = re.compile(r"^[A-Z]{2,4}\s*\(.*\)$", re.I)
_EXT_RETRY = [(150, 3.0), (220, 2.5)]     # (pad_px, upscale)
# Hard ceiling on a retry crop's longest side. Without it a symbol on a dense
# sheet gets a ~500px crop upscaled 2.5x to ~1250px, and OCR cost explodes:
# one 162-symbol sheet took 1004s versus ~150s typical. The upscale is reduced
# to fit rather than the search window, so the tag stays inside the crop.
_RETRY_MAX_SIDE = 700


def _score_external(text: str) -> float:
    """0..1 confidence that an external tag is COMPLETE (not clipped)."""
    if not text:
        return 0.0
    t = text.strip().rstrip("!.,;:").upper()
    if _EXT_FULL.match(t) or _EXT_GROUPED.match(t) or _EXT_PARENS.match(t):
        return 1.0
    if re.match(r"^[A-Z]{1,4}$", t):          # prefix only - clearly truncated
        return 0.15
    if _EXT_SHORT.match(t):                          # one digit short - suspect
        return 0.50
    if re.match(r"^[A-Z]{1,4}[-\s]?\d{1,3}$", t):   # clearly too few digits
        return 0.35
    return 0.55


def _nearest_outside(lines, box, ax, ay, upscale) -> tuple[str, float]:
    cands = []
    cx, cy = (box[0] + box[2]) / 2 - ax, (box[1] + box[3]) / 2 - ay
    for l in lines:
        lx = (l["box"][0] + l["box"][2]) / 2 / upscale
        ly = (l["box"][1] + l["box"][3]) / 2 / upscale
        if (box[0] - ax <= lx <= box[2] - ax) and (box[1] - ay <= ly <= box[3] - ay):
            continue
        cands.append((np.hypot(lx - cx, ly - cy), l))
    cands.sort(key=lambda t: t[0])
    return (cands[0][1]["text"], cands[0][1]["score"]) if cands else ("", 0.0)


def _score_tag(text: str, pidqa_class: int) -> float:
    """0..1 confidence that `text` is a well-formed tag for this class."""
    if not text:
        return 0.0
    t = text.strip()
    score = 0.35
    prefixes = TAG_PREFIXES.get(pidqa_class) or []
    if prefixes:
        up = t.upper().replace(" ", "")
        if any(up.startswith(p.upper().replace("-", "").replace(" ", "").replace("_", ""))
               or t.upper().startswith(p.upper()) for p in prefixes):
            score += 0.4
    # a 3-digit group is the usual instrument serial
    if re.search(r"\b\d{3}\b", t):
        score += 0.25
    if len(t) < 2:
        score = min(score, 0.2)
    return min(score, 1.0)


def _retry_internal(image, det, engine) -> tuple[str, float, float]:
    """Re-read one internal tag across the escalation ladder; return the best."""
    x0, y0, x1, y1 = det["bbox"]
    best = ("", 0.0, 0.0)
    for pad, up in _RETRY_LADDER:
        crop, _, _ = _prep(image, x0, y0, x1, y1, pad, upscale=up)
        lines = engine._run([crop])[0]
        text, conf = _assemble(lines)
        q = _score_tag(text, det["pidqa_class"])
        if q > best[2]:
            best = (text, conf, q)
        if q >= 0.9:
            break
    return best



# ---------------------------------------------------------------------------
# Rotated text + global assignment
#
# Inspecting the failures (scripts/diag_ocr.py) showed two dominant causes, and
# neither was the recogniser being weak:
#
#   1. Tags printed VERTICALLY along pipe runs. PaddleOCR reads them partially -
#      "OP-44323" comes back as "-44323", "GH-20054" as "-20054" - losing the
#      letter prefix, which is exactly the part that identifies the equipment.
#      Rotating the crop so the text is horizontal fixes it outright.
#
#   2. Nearest-text stealing a NEIGHBOUR'S tag: symbol IJ-07591 was labelled
#      "ERV-7-11", and CS-93 was labelled "627" from an adjacent instrument
#      bubble. Choosing per symbol independently lets two symbols claim the same
#      text, and lets a symbol claim text that plainly belongs to another.
#      Assignment has to be global.
# ---------------------------------------------------------------------------

# Measured on sheet 0 (48 ground-truth tags):
#   0/+90/-90  187.6s  recall 0.938
#   0/-90      127.1s  recall 0.938   <- chosen
#   0/+90      126.9s  recall 0.938
#   0/±90 @2x  110.5s  recall 0.896
# One rotation is enough: PaddleOCR's textline-orientation classifier already
# resolves the 180 degree flip, so the second rotation buys nothing but time.
ORIENTATIONS = (0, -90)
ROTATED = (-90,)


def _rotate(img: np.ndarray, deg: int):
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == -90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _unrotate_pt(x: float, y: float, deg: int, h: int, w: int):
    """Map a point in the rotated image back to the original crop.
    h, w are the ORIGINAL crop dimensions. Verified against cv2.rotate."""
    if deg == 90:
        return (y, h - 1 - x)
    if deg == -90:
        return (w - 1 - y, x)
    return (x, y)


def _collect_rotated(eng, crops: list[tuple], upscale: float,
                     orientations=None) -> list[dict]:
    """OCR every (crop, origin) at all orientations in ONE batched pass.

    Running the three orientations per symbol meant ~60 tiny batches of 3 per
    sheet, which is the worst case for a batched recogniser: measured 188-249s
    per sheet. Flattening to a single stream of images cuts that substantially
    at identical accuracy.
    """
    images, meta = [], []
    for crop, ax, ay in crops:
        big = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        bh, bw = big.shape[:2]
        for deg in (orientations or ORIENTATIONS):
            images.append(_rotate(big, deg))
            meta.append((deg, bh, bw, ax, ay))

    out = []
    for (deg, bh, bw, ax, ay), lines in zip(meta, _run_batch(eng, images)):
        for l in lines:
            t = l["text"].strip()
            if not t:
                continue
            bx0, by0, bx1, by1 = l["box"]
            ux, uy = _unrotate_pt((bx0 + bx1) / 2, (by0 + by1) / 2, deg, bh, bw)
            out.append({"text": t, "score": l["score"], "deg": deg,
                        "sx": ax + ux / upscale, "sy": ay + uy / upscale})
    return out


def _dedupe(cands: list[dict], tol: float = 26.0) -> list[dict]:
    """Collapse the same physical text found by several orientations/ROIs.
    Keeps the variant with the best completeness score - which is how the
    rotated read wins over the clipped horizontal one."""
    kept: list[dict] = []
    for c in sorted(cands, key=lambda c: (-_score_external(c["text"]), -c["score"])):
        if any(abs(c["sx"] - k["sx"]) < tol and abs(c["sy"] - k["sy"]) < tol for k in kept):
            continue
        kept.append(c)
    return kept


def _assign(symbols: list[dict], texts: list[dict], max_dist: float) -> dict[str, dict]:
    """Globally match text lines to symbols: one tag per symbol, one symbol per tag.

    Cost is centre distance, discounted when the text sits on the symbol's own
    row or column - engineering tags are set square to the item they label, so
    axis alignment is strong evidence of ownership.
    """
    pairs = []
    for s in symbols:
        x0, y0, x1, y1 = s["bbox"]
        sx, sy = (x0 + x1) / 2, (y0 + y1) / 2
        for t in texts:
            dx, dy = abs(t["sx"] - sx), abs(t["sy"] - sy)
            d = float(np.hypot(dx, dy))
            if d > max_dist:
                continue
            aligned = (dx < (x1 - x0) / 2 + 30) or (dy < (y1 - y0) / 2 + 30)
            cost = d * (0.55 if aligned else 1.0)
            cost *= (1.0 if _score_external(t["text"]) >= 0.9 else 1.5)
            pairs.append((cost, s["id"], id(t), t))
    pairs.sort(key=lambda p: p[0])
    out, used_sym, used_txt = {}, set(), set()
    for cost, sid, tid, t in pairs:
        if sid in used_sym or tid in used_txt:
            continue
        out[sid] = t
        used_sym.add(sid); used_txt.add(tid)
    return out


def ocr_symbols(image: np.ndarray, detections: list[dict],
                engine: OcrEngine | None = None) -> list[dict]:
    """Attach a `tag` to every detection whose class is normally tagged.

    Internal tags (instrument bubbles) are read straight out of the symbol box.
    External tags go through a two-phase pass: gather candidates from every
    symbol's neighbourhood at three orientations, then assign them globally.
    """
    eng = engine or OcrEngine.get()
    pad_ext = CFG.dotted("ocr.roi_pad_px", 85)
    out: list[dict] = []

    internal = [d for d in detections if tag_location(d["pidqa_class"]) == "internal"]
    external = [d for d in detections if tag_location(d["pidqa_class"]) == "external"]
    untagged = [d for d in detections if tag_location(d["pidqa_class"]) == "none"]

    # ---- internal: the tag is printed inside the box ----------------------
    if internal:
        crops = []
        for d in internal:
            c, _, _ = _prep(image, *d["bbox"], 4)
            crops.append(c)
        for d, lines in zip(internal, _run_batch(eng, crops)):
            text, conf = _assemble(lines)
            if not FAST_MODE and _score_tag(text, d["pidqa_class"]) < 0.75:
                r_text, r_conf, r_q = _retry_internal(image, d, eng)
                if r_q > _score_tag(text, d["pidqa_class"]):
                    text, conf = r_text, r_conf
            out.append({**d, "tag": text, "tag_confidence": conf,
                        "tag_location": "internal", "ocr_lines": len(lines)})

    # ---- external: gather candidates, then assign globally ----------------
    if external:
        H, W = image.shape[:2]
        pad = pad_ext if FAST_MODE else pad_ext + 45
        crops = []
        for d in external:
            x0, y0, x1, y1 = d["bbox"]
            ax, ay = max(0, x0 - pad), max(0, y0 - pad)
            crop = image[ay:min(H, y1 + pad), ax:min(W, x1 + pad)]
            if crop.size == 0:
                continue
            crops.append((cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop,
                          ax, ay))
        ups = UPSCALE if not FAST_MODE else 2.0

        # An adaptive "upright first, rotate only the weak ones" pass was tried:
        # 17% faster but recall on sheet 2 fell 0.933 -> 0.867, because a wrong
        # horizontal read scores as complete and the symbol is never re-examined.
        # Both orientations, always.
        texts = _dedupe(_collect_rotated(eng, crops, ups))
        max_d = CFG.dotted("entity_resolution.max_tag_distance_px", 150) + pad_ext
        chosen = _assign(external, texts, max_d)

        # A wider-crop second look was tried for symbols with a missing or clipped
        # tag. It doubled runtime (127s -> 272s) and made recall WORSE on sheet 2
        # (0.933 -> 0.900): the extra candidates give the global assignment more
        # chances to hand a symbol text that belongs to its neighbour.
        chosen = _assign(external, texts, max_d)
        for d in external:
            t = chosen.get(d["id"])
            txt = (t["text"].strip().rstrip("!.,;:") if t else "")
            out.append({**d, "tag": txt,
                        "tag_confidence": round(float(t["score"]), 3) if t else 0.0,
                        "tag_location": "external",
                        "tag_orientation": (t["deg"] if t else 0),
                        "ocr_lines": len(texts)})

    for d in untagged:
        out.append({**d, "tag": "", "tag_confidence": 0.0,
                    "tag_location": "none", "ocr_lines": 0})

    return sorted(out, key=lambda d: d["id"])


def save(records: list[dict], sheet_id: str) -> Path:
    out = CFG.get_path("paths.interim") / sheet_id
    out.mkdir(parents=True, exist_ok=True)
    p = out / "ocr.json"
    p.write_text(json.dumps({"sheet_id": sheet_id, "n": len(records),
                             "records": records}, indent=1))
    return p
