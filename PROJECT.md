# PROJECT — Engineering Drawing Intelligence

**Read this file first.** It is the single source of context: what this is, why
it exists, exactly where it stands, and how to pick it up cold. If a
conversation is lost, this file replaces the explanation.

Last updated: 2026-08-21

---

## 1 · What this is

A system that turns a P&ID (piping & instrumentation diagram) from *pixels* into
something you can **ask questions of in plain English**, with the answer showing
the drawing region, the graph path, and the query that produced it.

```
drawing.pdf → tile → detect → OCR → resolve → trace pipes → knowledge graph
                                                                    │
              "What is connected to SDL 101?" → Cypher ─────────────┘
                                                                    │
              answer + drawing crop + reasoning path ◄──────────────┘
```

Everything runs **offline on an 18 GB MacBook Pro**. No paid API, no cloud.

## 2 · Why it exists

An engineering drawing is the record of how a physical plant is built — every
pump, valve, tank, line and tag. But it is stored as pixels, so all that
structure is only readable by a person squinting at a sheet.

Today, answering "which valves isolate the line off P-101?" means: find the
right sheet among hundreds, zoom in, **trace the pipe by eye** across a
7000-pixel-wide drawing, note the tags, then cross-reference a separate spec PDF.
Every time. On an offshore rig with hundreds of sheets that does not scale, and
being wrong about which valve isolates which line is a safety problem.

**One line: drawings hold structured knowledge in an unqueryable form. This makes
them queryable.**

### Why it isn't already solved

Three facts, each of which dictates a piece of the architecture:

1. **Symbols are tiny** — ~33×62 px on a 7168×4562 sheet. Resize the sheet to fit
   a detector and a symbol becomes 6 px. So the sheet must be **tiled**.
2. **Meaning is topological, not visual** — "what connects to P-101" is a graph
   traversal, not a similarity search. Hence a **knowledge graph**, not a vector
   store.
3. **An unsourced answer is useless here** — no engineer acts on a black box.
   Hence **evidence**: graph path + drawing crop + document citation.

### The academic contribution

| Prior work | Has | Missing |
|---|---|---|
| PIDQA (2025) | graph QA over P&IDs | no visual evidence; graph built offline |
| Talking like P&IDs (2025) | GraphRAG | starts from DEXPI data, **not images** |
| BLUEPRINT (2026) | multimodal retrieval | **no graph, no multi-hop** |
| Bayer et al. (2026) | YOLO+OCR+ontology KG | SPARQL validation, **not NL QA** |
| MMGraphRAG (2025) | multimodal KG | general domain, not drawings |

**Nobody has done pixels → graph → evidence-grounded answer end to end.** That,
plus running entirely free and offline, is the claim.

Team: Manoj D · Mohammed Rayyan Ul Asgar · Mohammed Umar Farooq · Mohith D ·
Nikhil Sridhara. Guide: Prof. Khushboo Pandey.

---

## 3 · Where it stands right now

**End-to-end benchmark: 0.8038** on 2,176 PIDQA questions, 17 sheets,
**0 query errors** — measured with the trained detector, start to finish.

| Question type | end-to-end | perfect-detection ceiling |
|---|---|---|
| Simple counting | **0.9908** | 1.0000 |
| Spatial connection | 0.8879 | 0.9044 |
| Spatial counting | 0.7923 | 0.7941 |
| Value / tags | 0.5441 | 0.7463 |
| **Overall** | **0.8038** | 0.8612 |

**Detection: full-sheet F1 0.992** (precision 0.991, recall 0.994) over 2,311
symbols on 20 unseen sheets.
| End-to-end (L3, sampled) | 0.8542 · **executable query rate 1.000** |

Graph: **1,948 symbols · 2,039 `connected_to` · 495 `controls` · 304 `feeds_to`**

> **Quote 0.8038.** It is the real figure with the trained detector doing the
> seeing. The 0.8612 column is the same pipeline fed ground-truth boxes — an
> ablation showing the ceiling, never the result.
>
> **Value/tags fell hardest** (0.746 → 0.544) when real detection went in. The
> detector's boxes differ slightly from ground-truth boxes, and OCR crops its
> search region from those boxes, so tag reads degrade. Tuning the OCR ROI
> padding against real detections is the biggest fix still available.

### Blueprint mode — works on ANY real drawing today, no training

Reads the drawing's own tags, **pipe line numbers** and routing instead of
recognising symbols. Verified on two real offshore sheets:

| Sheet | Extracted |
|---|---|
| Alpha Field gas treatment (PFD) | 15 equipment, all correct · 10 connections |
| LP compressor suction cooler (**dense P&ID**) | **81 items** — 37 equipment, 21 pipe lines with sizes, 10 instruments, 7 valves · 93 connections · 14 control loops |

Answers verified on the real P&ID: *"how many instruments"* → 10 ✓ · *"which
lines are 4 inch or larger"* → 8", 6", 4" ✓ · *"how many valves"* → 7 ✓ ·
*"what is the largest line"* → 8" process gas ✓.
See [docs/BLUEPRINT_MODE.md](docs/BLUEPRINT_MODE.md).

This is what you demo with. The trained detector adds untagged symbols on top.

### What works

- Tiling (1024/stride 768 → 54 tiles/sheet)
- Detection interface (ground-truth backend live; YOLO backend ready, untrained)
- OCR: two orientations + global tag assignment — **tag recall 0.938**
- Entity resolution, review queue with human correction
- Relation extraction (tuned by sweep against the benchmark)
- Semantic relations: `CONTROLS`, `FEEDS_TO`
- Knowledge graph (Kùzu, embedded, speaks Cypher)
- NL→Cypher with validate-and-repair, few-shot selection
- Tri-modal evidence: graph path + drawing crops + document citations
- Reference document indexing and retrieval
- **Manual graph editing directly on the drawing** — click to select, drag a box
  to add a symbol, click two symbols to connect, click to delete. Coordinates
  come back in real sheet pixels at any zoom level.
- **Trained symbol detector** — YOLOv8s @512px, F1 0.992, at `models/yolo_pid.pt`
- **Blueprint mode** — detector-free ingestion, for a customer's first drawing
  before any model is trained on their legend
- **Pipe line numbers** — size, service and spec parsed from `12"-DC-11-185-P1A-HE`
- **Off-page connectors** — `TO 3RD STAGE`, `HP FLARE HEADER`, `FD-0002-01`,
  with `CONTINUES_ON` edges joining drawings that hand off to each other
- **Categorical graph facets** — `category` and `size_in`, so "how many
  instruments" and "lines 4 inch or larger" are exact queries, not string guesses
- **Conversational chat** — multi-turn, follow-up questions, exportable
- **Rig builder** — describe a piece of an oil rig in plain English, a local
  model turns it into a spec, and the system draws a real 7000x4600 P&ID from
  it: 105 item types, 11 standard packages, ISA tagging, automatic relief and
  ESD valves, instrument loops, EPC line numbers, a HAZOP-style design review,
  and the five registers (equipment, line, valve, instrument, cause & effect).
  The drawing and the graph come from the same spec, so they cannot disagree.
  See [docs/RIG_BUILDER.md](docs/RIG_BUILDER.md).
- **Flow direction on real drawings** — `FEEDS_TO` used to be empty in blueprint
  mode, so nothing about upstream, downstream or "where does this go" could be
  answered. Direction is now read out of the drawing's own labelling (off-page
  connectors that say TO/FROM, relief paths, service codes, check valves on
  machine discharges) and propagated by conservation. Every edge carries the
  rule that decided it and a confidence. Scored against the rig builder's specs:
  precision 0.884 on stated evidence, 0.934 including the left-to-right
  convention. See [docs/FLOW_DIRECTION.md](docs/FLOW_DIRECTION.md).
- **Field mode** — the screen for the technician who does not read P&IDs. Plain
  English item cards (what it is, what it means on the job, where on the sheet,
  what is in the pipes and the hazard that implies, what watches it, what it
  does NOT tell you), ISA-5.1 tag decoding, a 62-term plant glossary, and
  isolation planning that refuses to offer a check valve as a block valve and
  reports the branches it could not close. Nothing on it is model-generated.
- Cross-drawing tag search
- Export: entities CSV, relations CSV, node-link JSON, replayable Cypher
- Streamlit app, 7 screens, localhost
- 224 tests, 19/19 health check

### What does NOT work yet

| Gap | Impact | Effort |
|---|---|---|
| Real rig symbols | Detector learns 32 synthetic shapes; a real legend differs | Retrain on ~200 of their drawings |
| Mechanical / structural drawings | No training data, no benchmark | Not started |
| `CONTAINS` relation | Nothing in this corpus expresses containment | Deliberately not faked |

---

## 4 · How to pick this up cold

```bash
cd /Users/nikhilsridhara/nlp
source .venv/bin/activate

python scripts/dataset_report.py     # are the datasets present and complete?
python scripts/healthcheck.py        # do all 19 subsystems work?
python -m pytest tests/ -q           # 224 tests

python -m streamlit run src/edi/app/streamlit_app.py --server.port 8520
```

Then open **http://localhost:8520**.

### Rebuild everything from scratch

```bash
python scripts/ingest.py --sheets 0,2,5,6,9 --detector gt   # ~130 s/sheet (OCR-bound)
python scripts/redo_relations.py                            # fast; retunes topology only
python scripts/build_graph.py --fresh                       # loads graph + applies edits
python scripts/evaluate.py --per-type 32                    # benchmark
```

---

## 5 · The five things that will bite you

**1 · Class ids are off by one.** YOLO labels are `0..31`; PIDQA questions and
Cypher are `1..32`. Both conversions live in `src/edi/config.py` and nowhere
else, guarded by `tests/test_class_ids.py`. Empirically confirmed: PIDQA asks for
"class 31 tags starting with ZLO" → `ZLO 433`, and YOLO class **30** is the bubble
reading exactly `ZLO 433`.

**2 · `snap` must exceed `symbol_erase_pad_px`.** Relation extraction erases
symbols N px before tracing pipes, so a smaller snap attaches nothing and the
graph comes out **completely empty**. An early sweep produced a wall of zeros
before this was spotted.

**3 · Kùzu is single-writer.** Many readers coexist; one writer locks everyone
out. So: ingest writes JSON only, `build_graph.py` holds the lock for seconds,
and the app opens read-only and takes a writer only when writing.

**4 · Vision and language models cannot be co-resident in 18 GB.** Verified: with
YOLO training holding 7.4 GB, an Ollama call timed out at 120 s; suspending
training gave a 5.4 s response. Never run training and the app together.

**5 · Per-epoch validation crashes YOLO on Apple MPS.** The process vanishes at
the end of epoch 1 with a leaked-semaphore warning and no traceback. Use
`--no-val` on a Mac and score separately.

---

## 6 · Document map

| File | What it holds |
|---|---|
| **PROJECT.md** | ← you are here. Context, goals, status, gotchas |
| `ARCHITECTURE.md` | Code map, data flow, why each design decision was made |
| `README.md` | Quick start and what's in the app |
| `docs/RESULTS.md` | Every measurement, every tuning decision, negative results |
| `docs/DEMO.md` | 6-minute buyer walkthrough and the questions you'll be asked |
| `docs/INPUT.md` | Original plan: hardware, datasets, models, licences, schemas |
| `docs/OUTPUT.md` | Original plan: system design, screens, milestones, work split |
| `docs/BLUEPRINT_MODE.md` | Detector-free ingestion — how the real-drawing demo works |
| `docs/REAL_DRAWING_TEST.md` | Evidence: what transfers to a real offshore sheet |
| `TRAINING.md` | Detector training handoff for another machine |

---

## 7 · What to do next

In the order I would do it:

1. **OCR ROI padding tuned against real detector boxes** — recovers most of the
   value/tags drop; the biggest single win available
2. **Retrain on real rig symbols** — the thing that makes it work on their data
3. UI/UX polish
4. Spatial-counting accuracy (0.794 — the weakest topological metric)
5. Region classification — stream tables and title blocks are still read as text
