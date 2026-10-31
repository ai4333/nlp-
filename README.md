# Engineering Drawing Intelligence

> **New here, or picking this up after a break? Read [PROJECT.md](PROJECT.md).**
> It carries the full context: what this is, why it exists, exactly where it
> stands, and the five things that will bite you. [ARCHITECTURE.md](ARCHITECTURE.md)
> explains the code.

Ask questions in plain English about P&IDs and get answers **with the drawing
region, the graph path, and the confidence attached**. Runs fully offline on an
18 GB MacBook Pro. No paid API.

```
drawing.pdf ──► tile ──► detect ──► OCR ──► resolve ──► trace pipes ──► knowledge graph
                                                                              │
                          "What is connected to SDL 101?" ──► Cypher ──────────┘
                                                                              │
                          answer + drawing crop + reasoning path ◄─────────────┘
```

**Measured end to end, with the trained detector: 0.8038** on 2,176 PIDQA
questions (17 sheets, 0 query errors). Detection full-sheet **F1 0.992**.
Full numbers, every bug found, and the honest limitations:
**[docs/RESULTS.md](docs/RESULTS.md)**.

---

## Quick start

### Prerequisites
* Python 3.10 - 3.13
* [Ollama](https://ollama.ai) with `qwen3:8b` (optional, for natural language Cypher generation):
  ```bash
  ollama pull qwen3:8b
  ```

### Installation
```bash
git clone https://github.com/ai4333/nlp-.git
cd nlp-
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the App
```bash
python -m streamlit run src/edi/app/streamlit_app.py --server.port 8520
```
Then visit **http://localhost:8520**

### Ingesting Drawings
**1 · Ingest a real drawing** (Blueprint mode — zero-shot, no training needed):
```bash
python scripts/ingest_blueprint.py path/to/drawing.pdf --graph
```

**2 · Build/Rebuild the knowledge graph:**
```bash
python scripts/build_graph.py --fresh
```

**3 · Run test suite & health check:**
```bash
python scripts/healthcheck.py
pytest tests/ -q
```

---

## What's in the app

| Screen | Wireframe | What it does |
|---|---|---|
| **Workbench** | 1b + 1a | Real canvas — wheel-zoom, drag-pan, click any symbol for an inspector card. Overlays for symbols, tags, `connected_to`, `controls` and `feeds_to`. Ask bar with answer, evidence chips and reasoning path |
| **Knowledge graph** | 1c | Node-link view laid out on the drawing's real geometry. Click to trace, filter by category, NL⇄Cypher toggle |
| **Review queue** | 1f | Low-confidence tags with crop, OCR read, expected prefix and a correction box that writes back into the graph |
| **Evaluation** | 1g | PIDQA accuracy by question type, L1 and L3 scores, failure-case table |
| **Field mode** | — | For the technician, not the engineer. Ask in plain English or type a tag: what it is, what it means on the job, where on the sheet, what is in the pipes and the hazard that implies, what to close to isolate it — and what the drawing does *not* tell you. No model output anywhere on the screen. See [docs/REALITY_CHECK.md](docs/REALITY_CHECK.md) |
| **Rig builder** | — | Describe a piece of an oil rig in plain English; a local model turns it into a spec and the system draws a real 7000×4600 P&ID, checks it, loads it into the graph and exports the registers. See [docs/RIG_BUILDER.md](docs/RIG_BUILDER.md) |
| **Edit graph** | — | Add a missed symbol, delete a false positive, correct a class or tag, draw or remove a connection. Stored as an overlay, so re-extraction never destroys corrections |
| **Upload → Extract** | 1d → 1e | PDF/PNG/TIFF in, live six-stage progress tracker |

Also in the sidebar: **cross-drawing tag search**, **reference-document indexing**,
and **export** (entities CSV, relations CSV, node-link JSON, replayable Cypher).

## Evidence is tri-modal

Every answer carries three kinds of proof, which is the point of the system:

- **Graph** — the reasoning path and the Cypher that produced it
- **Drawing** — cropped regions with the box drawn on the symbol
- **Documents** — indexed specification text, cited by source and page

## Relation types

Pipe tracing yields `CONNECTED_TO`. Two more are derived and recorded with the
rule that produced them:

- **`CONTROLS`** — an instrument adjacent to a valve is a control loop, not just
  a pipe neighbour. Directed instrument → valve.
- **`FEEDS_TO`** — flow direction read off the drawing's own arrows. A filled
  triangle carries its ink at the base, so the apex is opposite the ink centroid.

`CONTAINS` is not derived: nothing in this corpus expresses containment, and
inventing it would be worse than leaving it out.

---

## The stack (all free)

| Role | Choice | Why |
|---|---|---|
| Detection | YOLOv8s @512px (Ultralytics 8.4.128, AGPL-3.0) | Trained, F1 0.992. Retraining: [HANDOFF.md](HANDOFF.md) |
| OCR | PaddleOCR PP-OCRv5 mobile (Apache-2.0) | Two orientations + global tag assignment |
| LLM | `qwen3:8b` via Ollama (Apache-2.0) | NL→Cypher + answer phrasing |
| VLM | `qwen3-vl:8b` (Apache-2.0) | Hard OCR crops only — never whole sheets |
| Few-shot | scikit-learn TF-IDF | 0.2s vs 93s for bge-m3, and better on templated questions |
| Graph | **Kùzu** (MIT, embedded) | Speaks Cypher, no Docker, no JVM, ~50 MB vs Neo4j's ~2 GB |

---

## Three things that decide whether this works

**1 · Tiling is mandatory.** Sheets are 7168×4562 px and symbols are ~33×62 px.
Resize the sheet to 1280 and a symbol becomes 6×11 px — unrecoverable. We tile
at 1024 with stride 768 (54 tiles/sheet), detect per tile, map back, and run a
global NMS to remove the duplicates that overlap creates.

**2 · The class ids are off by one.** YOLO labels are `0..31`; PIDQA questions
and Cypher are `1..32`. Both conversions live in `src/edi/config.py` and nowhere
else, guarded by `tests/test_class_ids.py`. Get this wrong and all 64,000
benchmark queries return confidently wrong answers.

**3 · Tags are printed vertically.** 15 of 19 inspected OCR failures were text
set along a pipe run, coming back as `-44323` instead of `OP-44323` — losing
exactly the part that identifies the equipment. Reading each crop at two
orientations and assigning tags to symbols *globally* (one tag per symbol, one
symbol per tag) took tag recall from ~0.70 to **0.938**.

**4 · A pipe run is a chain, not a clique.** Relation extraction erases symbols
and text from the ink mask, recovers horizontal and vertical runs separately
(so crossing pipes don't fuse), joins them only at T-junctions, and reduces each
net to its minimum spanning tree. Emitting all pairs within a net over-counted
on 10 of 10 failing questions; the MST fix moved spatial counting from 0.688 to
0.750 and value-based from 0.562 to 0.719.

---

## Health checks

```bash
python scripts/dataset_report.py    # datasets present and complete?
python scripts/healthcheck.py       # all 19 subsystems live?
python -m pytest tests/ -q          # 224 tests
```

## Layout

```
config/     pipeline.yaml · symbol_ontology.yaml · seed_examples.yaml
src/edi/
  ingest/   tiling.py
  vision/   detect.py  ocr.py
  graph/    relations.py  kg.py
  retrieval/fewshot.py
  qa/       nl2cypher.py  answer.py
  eval/     run_pidqa.py
  app/      streamlit_app.py
scripts/    build_tiles.py  train_yolo.py  ingest.py  build_graph.py
            evaluate.py  symbol_atlas.py  fetch_optional_datasets.sh
PROJECT.md  ARCHITECTURE.md  TRAINING.md
docs/       INPUT.md · OUTPUT.md · RESULTS.md · DEMO.md
tests/      23 tests — pytest tests/ -q
```

---

## Detector modes

`--detector gt` replays the dataset's own labels as detections. It is not a
stub: it isolates failures. If an answer is wrong under ground-truth detection,
the bug is in relations or the LLM, never the detector — and it gives the
"perfect detection" ceiling for the ablation table.

`--detector yolo` uses the trained model once `models/yolo_pid.pt` exists:

```bash
python scripts/train_yolo.py --epochs 30          # ~25 min/epoch on M3 Pro MPS
cp runs/pid_s640/weights/best.pt models/yolo_pid.pt
python scripts/ingest.py --n 14 --detector yolo
```

---

## Speed

PaddleOCR here is CPU-only and single-threaded — raising `OMP_NUM_THREADS` from
1 to 10 changed nothing (~1000 ms/crop either way). Parallelism has to be
process-level: three worker processes give **1.93× with byte-identical output**.
Set `ocr.workers` in `config/pipeline.yaml`; `Fast mode` in the sidebar skips the
multi-scale retries for roughly another 2×, at a real accuracy cost.

## Memory

18 GB unified is shared by CPU, GPU, and everything else. Vision and language
models must not be resident together — verified the hard way: running Ollama
during YOLO training timed out at 2 minutes.

- **Ingestion phase**: YOLO + PaddleOCR + (optional VLM) ≈ 13.5 GB
- **Query phase**: `qwen3:8b` + Kùzu ≈ 8 GB

Suspend training rather than killing it if you need the GPU:
```bash
kill -STOP $(pgrep -f train_yolo.py)    # resume with -CONT
```

Kùzu is single-writer with an exclusive file lock, which is why ingest writes
JSON and `build_graph.py` is a separate, short-lived step.
