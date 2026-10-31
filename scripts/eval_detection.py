#!/usr/bin/env python
"""
Full-sheet detection accuracy.

Ultralytics reports mAP on TILES. What matters here is accuracy on the whole
sheet after tiles are merged and de-duplicated, because that is what feeds the
graph. A model can look good per-tile and still double-count across seams.
"""
import sys, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.edi.config import load
from src.edi.vision.detect import YoloDetector, GroundTruthDetector, load_sheet

CFG = load()


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match(pred, gt, thr=0.5):
    """Greedy class-aware matching by descending confidence."""
    tp = fp = 0
    used = set()
    order = sorted(range(len(pred)), key=lambda i: -pred[i]["confidence"])
    gtb = np.array([g["bbox"] for g in gt], dtype=float) if gt else np.zeros((0, 4))
    gtc = np.array([g["pidqa_class"] for g in gt]) if gt else np.zeros(0)
    for i in order:
        p = pred[i]
        if len(gt) == 0:
            fp += 1; continue
        ious = iou_matrix(np.array([p["bbox"]], dtype=float), gtb)[0]
        ious = np.where(gtc == p["pidqa_class"], ious, 0.0)
        for j in np.argsort(-ious):
            if ious[j] < thr:
                fp += 1; break
            if j not in used:
                used.add(int(j)); tp += 1; break
        else:
            fp += 1
    return tp, fp, len(gt) - tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/yolo_pid.pt")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--split", default="val")
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    root = CFG.get_path("paths.dataset") / "images" / a.split
    sheets = [p.stem for p in sorted(root.glob("*.jpg"), key=lambda p: int(p.stem))][:a.n]
    det = YoloDetector(weights=a.weights)
    gtd = GroundTruthDetector(split=a.split)

    TP = FP = FN = 0
    per_class = {}
    rows = []
    for i, s in enumerate(sheets, 1):
        img, _ = load_sheet(s, a.split)
        pred = det.detect(img, s)
        gt = gtd.detect(img, s)
        tp, fp, fn = match(pred, gt, a.iou)
        TP += tp; FP += fp; FN += fn
        for g in gt:
            per_class.setdefault(g["pidqa_class"], [0, 0])[1] += 1
        for p in pred:
            per_class.setdefault(p["pidqa_class"], [0, 0])[0] += 1
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        rows.append({"sheet": s, "gt": len(gt), "pred": len(pred),
                     "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(prec, 4), "recall": round(rec, 4)})
        print(f"  [{i}/{len(sheets)}] sheet {s:>4}: gt {len(gt):3d}  pred {len(pred):3d}  "
              f"P {prec:.3f}  R {rec:.3f}", flush=True)

    P = TP / max(TP + FP, 1); R = TP / max(TP + FN, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    rep = {"weights": a.weights, "sheets": len(sheets), "iou": a.iou,
           "tp": TP, "fp": FP, "fn": FN,
           "precision": round(P, 4), "recall": round(R, 4), "f1": round(F1, 4),
           "per_sheet": rows}
    out = CFG.get_path("paths.reports"); out.mkdir(parents=True, exist_ok=True)
    (out / "detection.json").write_text(json.dumps(rep, indent=2))
    print(f"\n=== full-sheet detection @IoU{a.iou} ===")
    print(f"  precision {P:.4f}   recall {R:.4f}   F1 {F1:.4f}")
    print(f"  TP {TP}  FP {FP}  FN {FN}")
    print(f"report -> {out/'detection.json'}")


if __name__ == "__main__":
    main()
