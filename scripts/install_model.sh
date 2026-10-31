#!/usr/bin/env bash
# Install a detector trained elsewhere, then re-measure the whole system.
#
#   bash scripts/install_model.sh ~/Downloads/best.pt
#
# Everything after this is automatic: it scores the model on full sheets,
# re-reads every drawing with it, rebuilds the graph and re-runs the benchmark.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

SRC="${1:-}"
if [ -z "$SRC" ] || [ ! -f "$SRC" ]; then
  echo "usage: bash scripts/install_model.sh /path/to/best.pt"
  echo
  echo "That is the file your teammate sends back from training_bundle/runs/pid/weights/."
  exit 1
fi

SHEETS=${SHEETS:-0,2,5,6,9,11,14,15,17,19,22,31,35,37,41}

echo "── 1/5  install the model ───────────────────────────────"
mkdir -p models
cp "$SRC" models/yolo_pid.pt
python - <<'PY'
from src.edi.vision.detect import YoloDetector
d = YoloDetector()
i = d.describe()
print(f"    {i['weights']}  ·  trained at {i['imgsz']}px  ·  tiling {i['tile']}/{i['stride']}")
print(f"    classes: {len(d.model.names)}")
PY

echo
echo "── 2/5  how good is it on whole sheets? ─────────────────"
python scripts/eval_detection.py --n 12 --split val

echo
echo "── 3/5  re-read every drawing with the real detector ────"
python scripts/ingest.py --sheets "$SHEETS" --detector yolo

echo
echo "── 4/5  rebuild the knowledge graph ─────────────────────"
python scripts/build_graph.py --fresh

echo
echo "── 5/5  benchmark ───────────────────────────────────────"
python scripts/evaluate.py --per-type 32

echo
echo "Done. Restart the app to see it:"
echo "  python -m streamlit run src/edi/app/streamlit_app.py --server.port 8520"
