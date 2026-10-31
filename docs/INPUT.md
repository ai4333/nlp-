# INPUT SPECIFICATION
### Engineering Drawing Intelligence using Knowledge Graphs and Multimodal RAG
**Team:** Manoj D · Mohammed Rayyan Ul Asgar · Mohammed Umar Farooq · Mohith D · Nikhil Sridhara
**Guide:** Prof. Khushboo Pandey · School of Computer Science and Engineering
**Document status:** planning baseline, written 2026-08-21. Every model listed is free and runs offline on the target laptop.

---

## 0. How to read this document

This file lists **everything that goes INTO the system** — hardware, datasets, models, libraries, file formats, schemas and configuration. Its companion, `OUTPUT.md`, describes **what gets built and what comes out**.

Three rules govern every choice below:

1. **Free.** No paid API, no subscription, no commercial licence. Everything is open-weight or open-source.
2. **Runs locally** on an Apple Silicon MacBook Pro with 18 GB unified memory, offline.
3. **Verified, not assumed.** Numbers marked ✅ were measured on the actual machine/data. Numbers marked ~ are estimates and are labelled as such.

---

# PART A — HARDWARE ENVELOPE

## A.1 The actual machine (measured ✅)

| Property | Value |
|---|---|
| Model | MacBook Pro `Mac15,6` |
| Chip | **Apple M3 Pro** — 11 cores (5 performance + 6 efficiency) |
| GPU | Integrated Apple GPU (14-core), Metal / MPS backend |
| Memory | **18 GB unified** (shared between CPU and GPU) |
| Free disk | **158 GB** available |
| OS | macOS (Darwin 25.5.0) |
| CUDA | **None.** No NVIDIA GPU. `paddle.is_compiled_with_cuda() == False` ✅ |
| PyTorch MPS | **Available** ✅ (`torch 2.9.0`, `torch.backends.mps.is_available() == True`) |

> **The single most important consequence:** unified memory means the GPU and CPU share the same 18 GB. There is no separate 8 GB VRAM pool. Everything — macOS, browser, Neo4j's JVM, PyTorch tensors and every loaded language model — competes for the same 18 GB.

## A.2 Memory budget — what actually fits

Assume macOS + editor + browser consume ~5 GB. That leaves **~13 GB of working headroom**.

| Component | Resident size | Notes |
|---|---|---|
| macOS + apps baseline | ~5.0 GB | Not negotiable |
| YOLO detector (inference) | ~0.5 GB | Small model, MPS |
| PaddleOCR (det + rec) | ~1.0 GB | CPU-only on Mac ✅ |
| `qwen3-vl:8b` (VLM, Q4) | ~6.0 GB | + ~1 GB KV cache |
| `qwen3:8b` (LLM, Q4) | ~5.2 GB | |
| `bge-m3` (embeddings) | ~1.2 GB | |
| Neo4j Community (JVM) | ~1.5–2.0 GB | Configurable heap |

**Naively loading everything at once = ~16–17 GB → the machine will swap and crawl.**

### A.3 The fix: a two-phase memory plan

The pipeline is deliberately split so the vision models and the language models **never need to be resident at the same time**.

**Phase 1 — Ingestion (offline, batch, run once per drawing):**
```
macOS 5.0 + YOLO 0.5 + PaddleOCR 1.0 + qwen3-vl:8b 7.0  ≈ 13.5 GB   → fits
```
Neo4j is stopped, the text LLM is unloaded.

**Phase 2 — Query (online, interactive, what the user touches):**
```
macOS 5.0 + qwen3:8b 5.2 + bge-m3 1.2 + Neo4j 2.0 + YOLO 0.5 ≈ 13.9 GB → fits
```
The VLM is unloaded. Vision work is already done and stored on disk.

Enforce this with Ollama's `keep_alive` parameter — set `keep_alive: 0` on the last call of a phase to evict the model immediately:
```bash
curl http://localhost:11434/api/generate -d '{"model":"qwen3-vl:8b","keep_alive":0}'
```

**This staging is a hard design requirement, not an optimisation.** Any design that needs the VLM live during interactive Q&A will not fit in 18 GB.

## A.4 Where training happens

| Task | Where | Why |
|---|---|---|
| YOLO **inference** | Local, MPS | Fast enough (~1–3 s/sheet estimated) |
| YOLO **training** (full 500-sheet set) | **Google Colab free T4** | MPS training is roughly 5–10× slower than a T4 and lacks several fused kernels. This is not a limitation of the plan — the project synopsis already anticipates it. |
| OCR | Local CPU | No training required (pretrained) |
| LLM / VLM | Local inference only | No fine-tuning needed anywhere in this project |

Train on Colab → download `best.pt` (~6–22 MB) → run everything else locally. Nothing about the system depends on Colab at query time.

---

# PART B — RUNTIME INPUTS (what a user feeds the finished system)

## B.1 Primary input: the drawing

| Field | Specification |
|---|---|
| Accepted formats | `.pdf` (vector or scanned), `.png`, `.jpg`, `.tiff`, `.bmp` |
| Multi-page | Yes — a PDF is split into one *sheet* per page |
| Expected resolution | 200–400 DPI, or ≥ 4000 px on the long edge |
| **Minimum usable** | ~2500 px long edge. Below this, symbols fall under ~15 px and detection degrades sharply |
| Drawing types | P&ID (primary), mechanical assembly, structural/bridge GA |
| Colour | Colour or greyscale; converted to greyscale internally |
| Typical size | 7168 × 4562 px ✅ (measured on the training corpus) |

**Rendering PDFs:** rasterise at **300 DPI** using PyMuPDF (`fitz`). Do not accept the default 72 DPI — it destroys small symbols.

## B.2 Secondary input: reference documents (optional)

Equipment datasheets, specification documents, engineering standards, valve schedules. Formats: PDF, DOCX, TXT, CSV. These populate the **document** side of the multimodal retrieval index so the system can cite "spec 4.2" alongside a drawing region.

## B.3 Tertiary input: the question

Free-text natural language, English. Four question families are supported and are directly benchmarkable (they mirror the PIDQA taxonomy, §C.3):

| Family | Example |
|---|---|
| Simple counting | "How many control valves are on this sheet?" |
| Spatial counting | "How many class-28 symbols are directly connected to class-32 symbols?" |
| Spatial connection | "What valves work with the pipeline connected to P-101?" |
| Value / tag lookup | "Give me the tags of the flow transmitters beginning with ZLO" |

## B.4 Input contract (the JSON the API receives)

```json
{
  "sheet_id": "unit100-pid-r4-p1",
  "source_file": "data/uploads/unit100-pid-r4.pdf",
  "page": 1,
  "drawing_type": "pid",
  "render_dpi": 300,
  "reference_docs": ["specs/valve_schedule.pdf"],
  "question": "What valves work with the pipeline connected to P-101?"
}
```

---

# PART C — DATASETS

## C.1 Status summary

| # | Dataset | Gives us | Size | Licence | Status |
|---|---|---|---|---|---|
| 1 | **Dataset-P&ID** (`hamzas/digitize-pid-yolo`) | 500 P&ID sheets + YOLO symbol boxes | ~2.0 GB | Research use | ✅ **Downloading now** |
| 2 | **PIDQA** | 64,000 QA pairs + ground-truth Cypher | 14 MB | See repo LICENSE | ✅ **Downloaded** |
| 3 | PID-KnowledgeGraph-demo | Reference Streamlit implementation | 2 MB | — | ✅ **Downloaded** |
| 4 | **PID2Graph** | Real + synthetic P&IDs **with graph ground truth** | 9.3 GB | CC-BY-SA-4.0 | ⏳ Overnight job |
| 5 | PID_dataset (Zenodo) | Real industry / webscraped P&IDs | 6.7 GB | CC-BY-4.0 | ⏳ Overnight job |
| 6 | DocVQA | Document-VQA benchmark (comparison baseline) | ~10 GB | Research use | Optional |
| 7 | DesignQA | Engineering-documentation MLLM benchmark | varies | MIT | Optional |

Datasets 4 and 5 live on Zenodo, which **measured at 31 KB/s ✅ → ~80 hours ETA**. They are therefore not on the critical path. A resumable fetch script is ready:

```bash
bash scripts/fetch_optional_datasets.sh
```
It uses `curl -C -` with infinite retry, so it survives sleep/disconnect and resumes where it stopped. Run it overnight. **Phase 1 of the project does not need them.**

---

## C.2 Dataset-P&ID — the core training corpus

The 500 synthetic P&IDs from the *Digitize-PID* paper (arXiv:2109.03794), republished in YOLO format.

| Property | Value (measured ✅) |
|---|---|
| Sheets | **500** — 400 train / 100 val |
| Image size | **7168 × 4562 px**, RGB JPEG |
| Annotation format | YOLO `.txt` — `class cx cy w h`, normalised |
| Symbol classes | **32** (YOLO ids `0`–`31`) |
| Symbols per sheet | ~89 (sheet 0 had 89 boxes across 30 of 32 classes) |
| Total instances | ~44,500 |
| Layout | `DigitizePID_Dataset/{images,labels}/{train,val}/` |

### ⚠️ C.2.1 The resolution problem — this drives the whole vision design

A real bounding box from sheet 0:
```
20  0.23514229  0.17496163  0.00460379  0.01359351
```
Converted to pixels: **33 px wide × 62 px tall.**

| If you resize the full sheet to… | A symbol becomes… | Verdict |
|---|---|---|
| 640 px | **3 × 6 px** | Impossible |
| 1280 px | **6 × 11 px** | Impossible |
| Native 7168 px, **tiled at 1024** | **33 × 62 px** | ✅ Workable |

**Conclusion: the sheet must be tiled. Never feed a whole P&ID to a detector or a VLM.** This is the number-one reason engineering-drawing projects fail, and it is why the plan budgets real effort for the tiling layer.

### C.2.2 Tiling parameters (the recommended default)

| Parameter | Value | Derivation |
|---|---|---|
| Tile size | **1024 × 1024** | Symbol occupies 3–6 % of tile — good for YOLO |
| Stride | **768** (25 % overlap) | Overlap prevents symbols being cut at tile seams |
| Tiles per sheet | **54** (9 cols × 6 rows) | `ceil((7168−1024)/768)+1 = 9`, `ceil((4562−1024)/768)+1 = 6` |
| Total training tiles | **~27,000** | 500 × 54 |
| Merge strategy | Global NMS, IoU 0.5, after mapping tile coords → sheet coords | Removes duplicates from overlap |

A 640-px tiling (stride 512) is the alternative: 126 tiles/sheet, ~63,000 tiles — faster per tile, more tiles, more seam duplicates. Start with 1024.

### ⚠️ C.2.3 The off-by-one class trap

- **YOLO labels use classes `0`–`31`.** ✅ (verified in the label files)
- **PIDQA questions and Cypher use classes `1`–`32`.** ✅ (per PIDQA README: *"Symbols are named as numbers ranging from 1 to 32"*)

Therefore:
```python
pidqa_class = yolo_class + 1
```
Every graph node must be written with the **PIDQA (1-based)** class, or 100 % of the benchmark Cypher queries will silently return wrong answers. Put this conversion in exactly one function and unit-test it.

### C.2.4 The semantic-naming gap (a real limitation to plan for)

Dataset-P&ID symbols are **numbers**, not names. The corpus knows "class 17", not "centrifugal pump". But the project synopsis and the wireframe both promise answers like *"P-101 → connected_to → L-302 → controlled_by → V-203"*.

Bridging that gap needs a **class→name ontology file** — a 32-row mapping authored once by hand:

```yaml
# config/symbol_ontology.yaml
1:  {name: "ball valve",        category: valve,      connects: [line]}
17: {name: "centrifugal pump",  category: equipment,  connects: [line]}
32: {name: "flow transmitter",  category: instrument, connects: [equipment, line]}
```

Source the names from Figure 1 of the Digitize-PID paper (the 32-symbol legend) and from ISA-5.1 instrumentation conventions. **This is a half-day task for one team member and it unlocks every human-readable answer the demo shows.** Treat it as a Week-2 deliverable, not an afterthought.

---

## C.3 PIDQA — the evaluation corpus (downloaded ✅)

64,000 question–answer pairs over the **same 500 sheets**. This alignment is the most valuable thing in the whole project: detection ground truth, graph ground truth and QA ground truth all refer to identical images.

| File | Rows | Columns |
|---|---|---|
| `Simple Counting/simple_counting.csv` | 48,001 lines | `P&ID_number, Type, Symbol_Class, Q_id, Question, GT, Cypher` |
| `Spatial Counting/spatial_counting.csv` | 48,001 | `P&ID_number, Type, Symbol_XX, Symbol_YY, Q_id, Question, GT, Cypher` |
| `Spatial Connections/spatial_connectivity.csv` | 48,001 | `P&ID_number, Type, Symbol_XX, Symbol_YY, Symbol_ZZ, Q_id, Question, GT, Cypher` |
| `Value/value_based.csv` | 48,001 | `P&ID_number, Type, Symbol_Class, Prefix, Q_id, Question, GT, Cypher` |

*(Rows exceed 16,000 per file because the embedded Cypher strings contain newlines — parse with a proper CSV reader, never `split(',')`.)*

**Every row carries a ground-truth Cypher query.** That gives three free capabilities:

1. **An exact-match QA metric** — compare our answer to `GT`.
2. **A NL→Cypher training/eval signal** — compare our generated Cypher to the reference.
3. **The authoritative graph schema**, read straight off the reference queries:

```cypher
MATCH (s1:Symbol)-[:CONNECTED_TO]-(s2:Symbol)-[:CONNECTED_TO]-(s3:Symbol)
WHERE s1.class = 21 AND s2.class = 17 AND s3.class = 25
```
```cypher
MATCH (s:Symbol) WHERE s.class = 31 AND s.tag STARTS WITH ZLO
```

→ Node label `Symbol`, properties `class` (int) and `tag` (string), relationship `CONNECTED_TO` (**undirected in the queries** — note the `-[:CONNECTED_TO]-` with no arrow). Build the graph to match this schema exactly and the 64,000 reference queries become an instant regression suite.

---

## C.4 Reference implementation (downloaded ✅)

`mgupta70/PID-KnowledgeGraph-demo` — the Streamlit demo behind the PIDQA paper. Reading it saved us a lot of guesswork. Its actual approach:

- `ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)` generates Cypher
- `OpenAIEmbeddings()` + FAISS power a **semantic few-shot example selector** — it retrieves the most similar solved question/Cypher pairs and injects them into the prompt
- The generated Cypher runs against Neo4j; results are fed back to the LLM to phrase a human answer
- The knowledge graph itself is **loaded from a pre-built pickle** — the demo does not build it

**What this tells us about our contribution.** The demo is a thin query layer over a graph someone else made. The two things it does *not* do are exactly our project's substance:
1. **Building the graph from pixels** (detection → OCR → entity resolution → relation extraction).
2. **Multimodal retrieval** — it retrieves nothing visual and cites no evidence.

And its two paid dependencies (GPT-3.5, OpenAI embeddings) are what we replace with free local models. Our few-shot selector uses `bge-m3` + FAISS instead; identical mechanism, zero cost.

---

# PART D — MODELS (all free, all local)

## D.1 Master model table

| Role | Model | Size on disk | Resident RAM | Licence | Runtime |
|---|---|---|---|---|---|
| Symbol detection | **YOLOv8s** (or v11s) | 22 MB | ~0.5 GB | AGPL-3.0 ⚠️ | Ultralytics + MPS |
| Primary OCR | **PaddleOCR 3.4** (PP-OCRv5) | ~200 MB | ~1.0 GB | Apache-2.0 | CPU ✅ |
| Fallback OCR | **EasyOCR** or **Tesseract 5** | ~100 MB | ~0.6 GB | Apache-2.0 / Apache-2.0 | CPU |
| Vision-language | **`qwen3-vl:8b`** | ~6.0 GB | ~7.0 GB | Apache-2.0 | Ollama |
| Text LLM (Cypher + answers) | **`qwen3:8b`** | ~5.2 GB | ~5.2 GB | Apache-2.0 | Ollama |
| Larger LLM (optional) | `qwen2.5:14b` | 9.0 GB | ~9.0 GB | Apache-2.0 | Ollama — already installed ✅ |
| Text embeddings | **`bge-m3`** | ~1.2 GB | ~1.2 GB | MIT | Ollama |
| Image embeddings | **SigLIP-base-patch16-224** | ~0.8 GB | ~0.9 GB | Apache-2.0 | transformers + MPS |
| Reranker (optional) | `bge-reranker-v2-m3` | ~1.1 GB | ~1.2 GB | Apache-2.0 | transformers |

**Already on the machine ✅:** `qwen2.5:14b`, `llama3.2-vision:11b`, `mistral`, `moondream`, `llama3.1:8b`, `deepseek-coder-v2:16b`.
**Pulling now:** `qwen3-vl:8b`, `qwen3:8b`, `bge-m3`.

## D.2 Why each model — and what it must NOT be used for

### Detection — YOLOv8s / YOLOv11s
Fine-tuned on the 27,000 tiles. Small, fast on MPS, and the literature backs it: Bayer et al. (2026) reached **mAP@50 of 97.4 %** on bridge drawings with YOLOv8/v11, and Joshua et al. (2024) used YOLOv7 for the same job.

> ⚠️ **Licence caution.** Ultralytics YOLO is **AGPL-3.0**. For an academic project that publishes its source, this is fine — just cite it and keep the repo public. If this ever became a closed commercial product, AGPL would force disclosure. Free alternatives without that constraint: **RT-DETR** (Apache-2.0) or **YOLOX** (Apache-2.0). Note it in the report; don't let it surprise you later.

### OCR — PaddleOCR (primary)
Already installed ✅ (`paddleocr 3.4.0`, `paddlepaddle 3.3.0`). Handles **rotated and vertical text**, which matters enormously: P&ID tags are routinely printed at 90° along vertical pipe runs, and a detector that only reads horizontal text will miss a large fraction of them. Set `use_angle_cls=True`.

> **Known constraint (measured ✅):** PaddlePaddle on this Mac is **CPU-only** — `is_compiled_with_cuda() == False`, `get_device() == 'cpu'`. OCR will be the slowest stage. Mitigate by running OCR **only inside detected regions plus text-dense strips**, never over all 54 raw tiles blindly.

### Vision-language — `qwen3-vl:8b`
Qwen3-VL is the strongest open document-vision family that fits in this memory budget (Qwen3-VL-8B is reported at ~96 DocVQA). Apache-2.0, so no licence friction.

> ⚠️ **Do not use the VLM as the detector.** It is used for three narrow jobs only:
> 1. Re-reading OCR failures on a cropped region (a 300×300 crop, not a sheet).
> 2. Describing a region for the evidence panel.
> 3. Generalising to *unseen* drawing types (mechanical, structural) where no YOLO model is trained.
>
> Feeding a 7168×4562 sheet to an 8B VLM produces confident nonsense — it downsamples internally to ~1000 px, at which point symbols are 6 px wide (§C.2.1). The literature agrees: **DesignQA (2024) found GPT-4-class models fail at exactly this task**, and that is a much larger model than anything that fits here.

### Text LLM — `qwen3:8b`
Two jobs: **NL→Cypher generation** and **answer phrasing from retrieved evidence**. Qwen3 is markedly better at structured/code output than Llama 3.1 8B, which is what Cypher generation needs. `qwen2.5:14b` (already installed) is the quality fallback if 8B's Cypher accuracy disappoints — at the cost of ~4 GB more RAM and slower tokens.

Run at `temperature=0.0` for Cypher — determinism matters more than creativity, and the reference implementation does the same.

### Embeddings — `bge-m3` (text) + SigLIP (image)
`bge-m3` powers the few-shot Cypher example selector and the document retrieval index. SigLIP embeds drawing-region crops so the system can retrieve *visual* evidence — the piece the reference demo lacks entirely.

## D.3 Install commands

```bash
ollama pull qwen3-vl:8b
ollama pull qwen3:8b
ollama pull bge-m3
```

---

# PART E — SOFTWARE STACK

## E.1 Already installed ✅ (verified on this machine)

| Package | Version |
|---|---|
| Python | 3.13.5 |
| torch | 2.9.0 (MPS ✅) |
| torchvision | 0.22.1 |
| ultralytics | 8.3.155 |
| paddleocr | 3.4.0 |
| paddlepaddle | 3.3.0 (CPU) |
| transformers | 4.57.1 |
| sentence-transformers | 5.4.0 |
| opencv-python | 4.11.0.86 |
| huggingface-hub | 0.36.0 |
| Ollama | installed, server running ✅ |

## E.2 To install

```bash
pip install \
  neo4j==5.27.0 \
  langchain==0.3.* langchain-community==0.3.* langchain-ollama \
  faiss-cpu \
  chromadb \
  pymupdf \
  networkx \
  streamlit \
  rapidfuzz \
  pandas pyyaml pydantic \
  scikit-learn \
  shapely \
  pytest
```

> ⚠️ **Python 3.13 compatibility.** Some of these (notably `faiss-cpu` and `chromadb`) have historically lagged on the newest CPython. If a wheel fails to build, create a **Python 3.11 virtual environment** for the project rather than fighting it:
> ```bash
> brew install python@3.11 && python3.11 -m venv .venv && source .venv/bin/activate
> ```
> This is the single most likely setup snag. Budget an hour for it.

## E.3 Graph database

**Neo4j Community Edition 5.x** — free (GPLv3), speaks the exact Cypher dialect PIDQA's ground truth is written in.

```bash
docker run -d --name pidkg \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_server_memory_heap_max__size=1G \
  neo4j:5-community
```
Cap the heap at 1 GB — the default will try to claim a large share of the 18 GB.

**Fallback if the JVM is too heavy:** **Kùzu** (MIT, embedded, no server, speaks Cypher) — `pip install kuzu`. It costs ~50 MB instead of ~2 GB. Worth switching to if memory gets tight during Phase 2.

**Prototyping:** `networkx` in-memory, for fast iteration before any database exists.

---

# PART F — DATA CONTRACTS

Every pipeline stage reads and writes plain JSON on disk. This makes stages independently testable, restartable and debuggable — and lets five people work in parallel without stepping on each other.

## F.1 Detection output → `data/interim/{sheet_id}/detections.json`
```json
{
  "sheet_id": "0",
  "image_size": [7168, 4562],
  "tiling": {"tile": 1024, "stride": 768, "n_tiles": 54},
  "detections": [
    {"id": "d001", "yolo_class": 16, "pidqa_class": 17,
     "bbox": [1685, 798, 1718, 860], "confidence": 0.94}
  ]
}
```

## F.2 OCR output → `data/interim/{sheet_id}/ocr.json`
```json
{
  "sheet_id": "0",
  "lines": [
    {"id": "t001", "text": "ZLO 433", "bbox": [1690, 870, 1790, 900],
     "confidence": 0.91, "angle": 0}
  ]
}
```

## F.3 Entity resolution → `data/interim/{sheet_id}/entities.json`
```json
{
  "entities": [
    {"entity_id": "e001", "pidqa_class": 17, "name": "centrifugal pump",
     "tag": "ZLO 433", "bbox": [1685, 798, 1718, 860],
     "detection_conf": 0.94, "tag_conf": 0.91,
     "match_distance_px": 24, "needs_review": false}
  ]
}
```

## F.4 Relations → `data/interim/{sheet_id}/relations.json`
```json
{
  "relations": [
    {"source": "e001", "target": "e014", "type": "CONNECTED_TO",
     "evidence": "line_trace", "polyline": [[1718,829],[1980,829],[1980,1120]],
     "confidence": 0.82}
  ]
}
```

## F.5 Knowledge-graph schema (must match PIDQA exactly)

```cypher
(:Symbol {
   entity_id: string,
   class: int,          // 1..32  — PIDQA convention, NOT the YOLO 0..31 id
   name: string,        // from config/symbol_ontology.yaml
   tag: string,
   sheet_id: string,
   bbox: [int,int,int,int],
   confidence: float
})

(:Symbol)-[:CONNECTED_TO {confidence: float, evidence: string}]-(:Symbol)
```

Project-specific relations layered on top, per the synopsis: `CONTROLS`, `CONTAINS`, `FEEDS_TO`, `CONTROLLED_BY`. Keep `CONNECTED_TO` semantically identical to PIDQA's so the benchmark stays valid.

Indexes to create on load:
```cypher
CREATE INDEX symbol_class  FOR (s:Symbol) ON (s.class);
CREATE INDEX symbol_tag    FOR (s:Symbol) ON (s.tag);
CREATE INDEX symbol_sheet  FOR (s:Symbol) ON (s.sheet_id);
```

---

# PART G — CONFIGURATION

`config/pipeline.yaml`:

```yaml
input:
  render_dpi: 300
  min_long_edge_px: 2500

tiling:
  tile_size: 1024
  stride: 768
  merge_iou: 0.5

detection:
  weights: models/yolov8s_pid.pt
  imgsz: 1024
  conf_threshold: 0.25
  iou_threshold: 0.45
  device: mps
  review_below_conf: 0.50

ocr:
  engine: paddleocr
  use_angle_cls: true          # vertical pipe-run tags
  lang: en
  roi_only: true               # never OCR blank tiles - CPU-bound stage
  min_confidence: 0.60

entity_resolution:
  max_tag_distance_px: 150     # tag must be near its symbol
  fuzzy_threshold: 85          # rapidfuzz - fixes "V-2O3" -> "V-203"
  review_below: 0.70

relations:
  line_detector: lsd           # OpenCV LineSegmentDetector
  min_line_length_px: 30
  junction_tolerance_px: 8
  max_gap_px: 12

llm:
  cypher_model: qwen3:8b
  cypher_temperature: 0.0
  answer_model: qwen3:8b
  vlm_model: qwen3-vl:8b
  keep_alive_between_phases: 0

retrieval:
  text_embed: bge-m3
  image_embed: google/siglip-base-patch16-224
  top_k: 8
  fewshot_k: 5
  rerank: false

graph:
  backend: neo4j               # or 'kuzu'
  uri: bolt://localhost:7687
```

---

# PART H — CURRENT STATE

> **Note — decisions that changed once measured.** Neo4j was replaced by **Kùzu**
> (no Docker on this machine; embedded, MIT, and verified to run all four PIDQA
> query shapes). bge-m3 was replaced by **TF-IDF** for few-shot selection (0.2s vs
> 93s, and better on templated questions); it remains the choice for document
> retrieval. FAISS and ChromaDB were dropped entirely — at a few thousand vectors
> numpy is instant and it removed the largest Python 3.13 wheel risk. A Python
> **3.12/3.13 venv with `--system-site-packages`** was used instead of a fresh
> 3.11 env, inheriting the already-working torch/ultralytics/paddleocr.
> See [RESULTS.md](RESULTS.md).

## H.1 Done ✅
- Project scaffold at `~/nlp/{data,docs,models,src,scripts,notebooks,scratch}`
- PIDQA cloned (64,000 QA pairs + ground-truth Cypher)
- Reference demo cloned and read
- Dataset-P&ID download running (400 train + 100 val, with YOLO labels)
- `qwen3-vl:8b`, `qwen3:8b`, `bge-m3` pulling
- Hardware envelope, MPS availability and Paddle CPU-only status all verified on-machine
- Resumable fetch script for the two large Zenodo datasets

## H.2 Pending
| Item | Owner | Notes |
|---|---|---|
| `config/symbol_ontology.yaml` — 32 class names | 1 member | Half a day; blocks human-readable answers (§C.2.4) |
| Python 3.11 venv if 3.13 wheels fail | 1 member | ~1 hour (§E.2) |
| Neo4j container up | 1 member | 30 min |
| PID2Graph / PID_dataset | background | `bash scripts/fetch_optional_datasets.sh` overnight |

---

## Appendix — Sources

- Dataset-P&ID / Digitize-PID — https://arxiv.org/abs/2109.03794 · https://huggingface.co/datasets/hamzas/digitize-pid-yolo
- PIDQA — https://github.com/mgupta70/PIDQA · https://www.mdpi.com/2504-4990/7/2/39
- Reference demo — https://github.com/mgupta70/PID-KnowledgeGraph-demo
- PID2Graph — https://zenodo.org/records/14803338
- PID_dataset — https://zenodo.org/records/8028570
