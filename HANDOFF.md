# Training handoff — send it out, get it back

Two sides. **A** is what your teammate does on the gaming laptop. **B** is what
you do here when they send it back. Nobody needs to understand the project.

---

# A · On the gaming laptop

Send them **`TRAINING_BUNDLE.zip`** (8.4 KB — WhatsApp, email, Drive, anything).
It downloads its own data, so nothing large is being sent.

They need: **an NVIDIA GPU**, Python 3.9+, ~5 GB free disk.

### The five commands

```bash
# 0 · unzip it and go into the folder
cd training_bundle

# 1 · setup                                        (~3 min)
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Mac / Linux
pip install -r requirements.txt

# 2 · download the drawings                        (~10 min, 1.3 GB)
python fetch_data.py

# 3 · cut them into tiles                          (~2 min)
python build_tiles.py

# 4 · how long will this take on MY machine?       (~1 min)
python estimate.py

# 5 · train                                        (see below)
python train.py
```

### How long step 5 takes

`estimate.py` prints their real number. Roughly, for 60 epochs:

| Their GPU | Command | Time |
|---|---|---|
| 6 GB · 3050 / 3060 laptop | `python train.py --batch 16` | 5–8 h |
| 8–12 GB · 3070 / 4060 / 4070 | `python train.py` | 3–5 h |
| 16 GB+ · 4080 / 4090 | `python train.py --model yolov8m.pt --batch 48` | 1.5–3 h |
| No GPU | `python train.py --device cpu --epochs 10` | overnight+ |

**They do not have to finish all 60 epochs.** It saves every 5th epoch, resumes
with `python train.py --resume`, and accuracy is usually good by 25–30. If they
are short on time: stop it, run step 6, send what exists.

### 6 · Check and send back

```bash
python verify.py
```

Prints the mAP50 score. **Above 0.85 is good, above 0.90 is what we want.**

Then they send back **exactly two files**:

```
training_bundle/runs/pid/weights/best.pt      ← the model, ~22 MB
training_bundle/runs/pid/results.csv          ← the training curve, tiny
```

Not the dataset. Not the tiles. Just those two.

---

# B · Back on your laptop

Put `best.pt` anywhere — Downloads is fine. Then **one command**:

```bash
cd /Users/nikhilsridhara/nlp
bash scripts/install_model.sh ~/Downloads/best.pt
```

That does all of it:

1. installs the model and reports what size it was trained at
2. scores it on whole sheets — precision, recall, F1
3. **re-reads all 15 benchmark drawings with the real detector**
4. rebuilds the knowledge graph
5. re-runs the 2,176-question benchmark

Takes about 40 minutes, mostly OCR. Then:

```bash
source .venv/bin/activate
python -m streamlit run src/edi/app/streamlit_app.py --server.port 8520
```

### What changes once it is installed

**Before:** blueprint mode only — it reads tags and traces pipes, so an item
without a printed tag is invisible.

**After:** it also *sees the symbols*. Untagged valves and fittings appear,
symbol counts become real, and **uploading a brand-new drawing works** — the
Detector engine in the sidebar unlocks.

### Expect the benchmark number to move

Today's **0.8612** was measured with ground-truth detection — a ceiling, not an
end-to-end score. With a real detector it will drop to whatever detection
actually achieves. **That lower number is the honest one**, and it is the one to
quote. If it comes back below ~0.75, the fix is more epochs or a bigger model,
not a change to the pipeline.

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | `python train.py --batch 8` |
| `No CUDA GPUs are available` | CPU-only PyTorch installed — `pip uninstall -y torch torchvision` then `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| Training vanishes at the end of epoch 1, no error | Apple-Silicon fault only; will not happen on NVIDIA. If it does: `python train.py --no-val` |
| Very slow | The first log line names the device. If it says `cpu`, see above |
| `install_model.sh` says usage | Give it the real path to `best.pt` |

Anything else: send the last 30 lines of output.

## A detail that is already handled

The bundle trains at 512 px. The detector reads the size **out of the model file**
and tiles to match, so if your teammate changes `--imgsz` it still works. No
config editing on either side.
