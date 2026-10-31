"""
Turn an arbitrary uploaded file into normalised sheets the pipeline can process.

Accepts PDF (vector or scanned), PNG, JPG, TIFF, BMP. PDFs are rasterised with
PyMuPDF at a DPI chosen so the long edge lands in a workable range: too low and
the ~33x62px symbols vanish, too high and OCR cost explodes for no gain.
"""
from __future__ import annotations
import hashlib, shutil
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from ..config import load

CFG = load()
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TARGET_LONG_EDGE = 7000          # matches the corpus the detector was trained on
MAX_LONG_EDGE = 12000
MIN_LONG_EDGE = 2500


@dataclass
class Sheet:
    sheet_id: str
    path: Path
    page: int
    width: int
    height: int
    source_name: str
    warnings: list[str]


def _store_dir() -> Path:
    d = CFG.get_path("paths.interim").parent / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sheet_id(stem: str, page: int, digest: str) -> str:
    base = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:40]
    return f"u_{base}_p{page}_{digest[:6]}"


def _normalise(img: np.ndarray) -> tuple[np.ndarray, list[str]]:
    warn: list[str] = []
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge < MIN_LONG_EDGE:
        warn.append(
            f"Low resolution ({w}x{h}). Symbols may fall below ~15px and detection "
            f"will degrade. {MIN_LONG_EDGE}px+ on the long edge is recommended.")
    if long_edge > MAX_LONG_EDGE:
        s = MAX_LONG_EDGE / long_edge
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        warn.append(f"Downscaled from {w}x{h} to fit {MAX_LONG_EDGE}px.")
    return img, warn


def from_pdf(path: Path, dpi: int | None = None, max_pages: int = 20) -> list[Sheet]:
    import fitz
    doc = fitz.open(path)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    out: list[Sheet] = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        rect = page.rect
        # choose DPI so the long edge lands near TARGET_LONG_EDGE
        long_pt = max(rect.width, rect.height) or 1
        use_dpi = dpi or int(min(600, max(150, TARGET_LONG_EDGE / long_pt * 72)))
        pix = page.get_pixmap(dpi=use_dpi, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        img, warn = _normalise(img)
        sid = _sheet_id(path.stem, i + 1, digest)
        dst = _store_dir() / f"{sid}.png"
        cv2.imwrite(str(dst), img)
        out.append(Sheet(sid, dst, i + 1, img.shape[1], img.shape[0], path.name,
                         warn + [f"rasterised at {use_dpi} DPI"]))
    doc.close()
    return out


def from_image(path: Path) -> list[Sheet]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not read image: {path.name}")
    img, warn = _normalise(img)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    sid = _sheet_id(path.stem, 1, digest)
    dst = _store_dir() / f"{sid}.png"
    cv2.imwrite(str(dst), img)
    return [Sheet(sid, dst, 1, img.shape[1], img.shape[0], path.name, warn)]


def acquire(path: str | Path, dpi: int | None = None) -> list[Sheet]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".pdf":
        return from_pdf(p, dpi)
    if p.suffix.lower() in IMAGE_EXT:
        return from_image(p)
    raise ValueError(f"unsupported file type: {p.suffix}")


def save_upload(data: bytes, filename: str) -> Path:
    tmp = _store_dir() / "_incoming"
    tmp.mkdir(parents=True, exist_ok=True)
    dst = tmp / filename
    dst.write_bytes(data)
    return dst


def load_any(sheet_id: str) -> np.ndarray:
    """Load a sheet image whether it came from the dataset or an upload."""
    up = _store_dir() / f"{sheet_id}.png"
    if up.exists():
        return cv2.imread(str(up), cv2.IMREAD_GRAYSCALE)
    from ..vision.detect import load_sheet
    return load_sheet(sheet_id)[0]
