# Detector training — run this on a gaming laptop

You are training one thing: a **YOLO object detector** that finds 32 kinds of
engineering symbol on a P&ID. Nothing else. When it finishes you send back a
single file, `best.pt`, about 22 MB.

Everything downloads itself. You need ~4 GB of disk and an NVIDIA GPU.
No knowledge of the project is required.

---

## 1 · Setup (once, ~3 minutes)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2 · Get the data (~10 minutes, 1.3 GB)

```bash
python fetch_data.py
```
Downloads 500 P&ID drawings with their labelled symbols.

## 3 · Cut them into tiles (~2 minutes)

```bash
python build_tiles.py
```

A drawing is 7168 × 4562 pixels and a symbol is about 33 × 62 pixels. Shrinking
the whole sheet to fit the detector would make each symbol six pixels across, so
the sheet is cut into overlapping tiles instead. This step makes about 46,000 of
them.

## 4 · Find out how long it will take (~1 min)

```bash
python estimate.py
```

Runs about 40 real training steps on your GPU and prints your own number,
rather than a guess from someone else's hardware.

## 5 · Train

```bash
python train.py
```

Defaults: YOLOv8s, 512 px, 60 epochs, batch 32 — tuned for an 8 GB GPU.

| Your GPU | Command | Rough time, 60 epochs |
|---|---|---|
| 6 GB · 3050 / 3060 laptop | `python train.py --batch 16` | 5–8 h |
| 8–12 GB · 3070 / 4060 / 4070 | `python train.py` | 3–5 h |
| 16 GB+ · 4080 / 4090 | `python train.py --model yolov8m.pt --batch 48` | 1.5–3 h |
| No GPU | `python train.py --device cpu --epochs 10` | overnight+ |

Those are estimates — **`python estimate.py` prints your real number.**

It shows progress every epoch, saves every 5th, and can be stopped and resumed
(`python train.py --resume`) at any time. **You do not have to run all 60
epochs** — accuracy is usually good by 25–30. If you are short on time, stop it
and send what you have along with the mAP number.

## 6 · Check it worked

```bash
python verify.py
```

Prints mAP50. **Above 0.85 is good, above 0.90 is what we want.**

## 7 · Send back

Zip and return **just these two files**:

```
runs/pid/weights/best.pt        ← the model (~22 MB)
runs/pid/results.csv            ← the training curve
```

Nothing else. Please do not send the dataset back.

---

## If something breaks

**"CUDA out of memory"** — lower the batch: `python train.py --batch 8`

**"No CUDA GPUs are available"** — the GPU build of PyTorch is not installed:
```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Training vanishes with no error at the end of epoch 1** — a known Apple-Silicon
fault, not a Windows/NVIDIA one. If it somehow happens: `python train.py --no-val`

**It is very slow** — check the GPU is actually being used; the log prints the
device on the first line. CPU training is roughly 20× slower.

Any other problem: send the last 30 lines of the output.
