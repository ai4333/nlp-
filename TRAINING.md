# Training the detector on another machine

The pipeline runs fine without this — `--detector gt` replays dataset labels so
every downstream stage can be developed and measured. This is only needed to
detect symbols on **new** drawings.

## What to copy

```
data/tiles512/          # 57,761 pre-cut 512px tiles + labels + data.yaml
scripts/train_yolo.py
```
`data/tiles512` is the whole dataset already tiled — nothing else is required.
If you'd rather rebuild it from the raw sheets:
```bash
EDI_CONFIG=config/pipeline_512.yaml python scripts/build_tiles.py
```

## Run it

```bash
pip install ultralytics
python scripts/train_yolo.py \
    --data data/tiles512/data.yaml \
    --model yolov8s.pt --imgsz 512 --batch 32 \
    --fraction 1.0 --epochs 60 --name pid_full
```

On a CUDA GPU drop `--no-val`; per-epoch validation only breaks on Apple MPS.
Change `device="mps"` to `device=0` in `scripts/train_yolo.py`.

## Two things that will bite you

**1 · Per-epoch validation crashes on Apple Silicon.** Not on CUDA. The symptom
is the process vanishing at the end of epoch 1 with a leaked-semaphore warning
and no traceback. Workaround on a Mac: `--no-val`, then score separately.

**2 · Memory.** On this 18 GB MacBook, yolov8s @640 batch 16 needed 7.4 GB and
thrashed swap — 129 million pageins, throughput fell to 3 iterations/minute.
yolov8n @512 batch 16 fits in 2.5 GB and runs at **1.78 it/s (≈2 min/epoch)**.
On a machine with a real GPU, use yolov8s or yolov8m and a larger batch.

## Bring the model back

```bash
cp runs/pid_full/weights/best.pt models/yolo_pid.pt
bash scripts/rebuild_all.sh          # score, re-ingest, rebuild graph, benchmark
```

`rebuild_all.sh` measures **full-sheet** precision/recall/F1 after tiles are
merged — not per-tile mAP. A model can look good per-tile and still double-count
across seams, and it is the merged result that feeds the graph.

## Expect the benchmark to drop

Every number in `docs/RESULTS.md` was measured with ground-truth detection. That
is the ceiling, not the end-to-end score. When the real detector goes in,
accuracy falls by however much detection misses — that is the honest figure.
