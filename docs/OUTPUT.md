# OUTPUT / SYSTEM SPECIFICATION
### Engineering Drawing Intelligence using Knowledge Graphs and Multimodal RAG
**Companion to `INPUT.md`.** That file says what goes in. This file says what gets built, what comes out, and in what order to build it.

---

## 1. What this system is, in one paragraph

A user drops a P&ID (or mechanical/structural drawing) into the app. The system tiles it, detects all 32 classes of engineering symbol, reads every tag and dimension with OCR, matches each tag to its symbol, traces the pipe/connection lines between symbols, and writes the whole thing into a Neo4j knowledge graph. The user then asks a question in plain English. The system translates that question into a Cypher query, runs it against the graph, retrieves supporting evidence from three sources at once — the graph path, the cropped drawing region, and any attached specification documents — and a local LLM phrases an answer **with the evidence attached**. Everything runs offline on an 18 GB MacBook Pro with no paid API.

The answer is never just text. It always ships with the reasoning path (`P-101 → connected_to → L-302 → controlled_by → V-203`), the drawing crop it came from, and a confidence figure.

---

## 2. System architecture

Nine stages. Stages 1–7 are **offline ingestion** (run once per drawing). Stages 8–9 are **online query** (run per question). Per `INPUT.md` §A.3 these two groups never hold their models in memory simultaneously.

```
┌─────────────────────── OFFLINE INGESTION (per drawing) ───────────────────────┐
│                                                                               │
│  1. ACQUIRE      PDF/PNG/TIFF ──► 300 DPI raster ──► sheets                    │
│         │                                                                     │
│  2. PREPROCESS   deskew · denoise · greyscale · TILE 1024/stride 768           │
│         │        (54 tiles per 7168×4562 sheet)                               │
│  3. DETECT       YOLOv8s per tile ──► map to sheet coords ──► global NMS       │
│         │        → detections.json                                            │
│  4. OCR          PaddleOCR, angle-aware, ROI-restricted                        │
│         │        → ocr.json                                                    │
│  5. RESOLVE      nearest-neighbour tag↔symbol match + fuzzy repair             │
│         │        → entities.json      (low-confidence → review queue)          │
│  6. RELATE       LSD line detection · junction tracing · graph assembly         │
│         │        → relations.json                                              │
│  7. BUILD        Neo4j load  +  FAISS text index  +  SigLIP region index        │
│                  → knowledge graph + multimodal indexes                        │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────── ONLINE QUERY (per question) ───────────────────────────┐
│                                                                               │
│  8. RETRIEVE     question ──► bge-m3 ──► few-shot selector ──► qwen3:8b        │
│         │        ──► Cypher ──► Neo4j ──► subgraph                             │
│         │        ‖ parallel: region retrieval (SigLIP) + doc retrieval (FAISS) │
│  9. ANSWER       qwen3:8b fuses graph + regions + docs                         │
│                  → answer + reasoning path + evidence crops + confidence       │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Stage detail

| # | Stage | Input | Output | Key technique | Hard part |
|---|---|---|---|---|---|
| 1 | Acquire | PDF/image | Sheet rasters | PyMuPDF @ 300 DPI | Multi-page splitting |
| 2 | Preprocess | Sheet raster | 54 tiles + offset map | OpenCV, deskew, adaptive threshold | Getting stride/overlap right |
| 3 | Detect | Tiles | `detections.json` | YOLOv8s fine-tuned, 32 classes | Seam duplicates → global NMS |
| 4 | OCR | ROI crops | `ocr.json` | PaddleOCR `use_angle_cls=True` | **Rotated tags on vertical pipe runs** |
| 5 | Resolve | detections + ocr | `entities.json` | Spatial NN + `rapidfuzz` | `V-2O3` (letter O) vs `V-203` (zero) |
| 6 | Relate | Tiles + entities | `relations.json` | LSD + junction tracing | **The hardest stage — see §7.1** |
| 7 | Build | entities + relations | Graph + indexes | Neo4j, FAISS, SigLIP | Class off-by-one (`INPUT.md` §C.2.3) |
| 8 | Retrieve | Question | Subgraph + regions + docs | NL→Cypher + tri-modal retrieval | Invalid Cypher → repair loop |
| 9 | Answer | Evidence bundle | Answer + evidence | `qwen3:8b` fusion | Grounding, not hallucinating |

---

## 3. Artifacts the system produces

### 3.1 On disk, per sheet
```
data/interim/{sheet_id}/
├── sheet.png              # 300 DPI normalised raster
├── tiles/                 # 54 × 1024px tiles + offsets.json
├── detections.json        # ~89 symbols, class + bbox + confidence
├── ocr.json               # every text line, bbox, angle, confidence
├── entities.json          # symbols fused with their tags
├── relations.json         # CONNECTED_TO edges with traced polylines
├── review_queue.json      # low-confidence items needing a human
└── crops/                 # evidence images, one per entity
```

### 3.2 Global
```
data/graph/          Neo4j database  (~44,500 Symbol nodes across 500 sheets)
data/index/text/     FAISS index over reference documents + OCR text
data/index/region/   SigLIP embeddings of every drawing crop
data/index/fewshot/  bge-m3 index over solved question→Cypher pairs
models/yolov8s_pid.pt
reports/eval_*.json  Metrics per run
```

### 3.3 Per answer (the API response)
```json
{
  "answer": "Two valves act on line L-302, which is connected to pump P-101: V-203 (control) and V-207 (isolation).",
  "reasoning_path": ["P-101", "connected_to", "L-302", "controlled_by", "V-203"],
  "cypher": "MATCH (s1:Symbol)-[:CONNECTED_TO]-(s2:Symbol) WHERE s1.tag = 'P-101' RETURN s2",
  "evidence": {
    "regions":  [{"crop": "crops/e014.png", "bbox": [1685,798,1718,860], "confidence": 0.94}],
    "graph":    {"nodes": ["e001","e014","e022"], "edges": [["e001","e014"]]},
    "documents":[{"source": "specs/valve_schedule.pdf", "section": "4.2", "score": 0.81}]
  },
  "confidence": 0.87,
  "latency_ms": 3400
}
```

Those three evidence types map exactly to the three chips in wireframe **1a**: `◧ region 3`, `◇ graph path`, `¶ spec 4.2`.

---

## 4. The knowledge graph

```cypher
(:Symbol {entity_id, class:int(1..32), name, tag, sheet_id, bbox, confidence})
(:Symbol)-[:CONNECTED_TO {confidence, evidence}]-(:Symbol)
```
Plus the project's own semantic edges: `CONTROLS`, `CONTAINS`, `FEEDS_TO`, `CONTROLLED_BY`.

**Scale:** ~89 symbols/sheet × 500 sheets ≈ **44,500 nodes**, with an estimated 50–70k edges. Trivial for Neo4j; it will be instant.

> **Non-negotiable:** `class` is stored **1-based** to match PIDQA. YOLO emits 0-based. One conversion function, unit-tested. Get this wrong and all 64,000 benchmark queries silently return wrong answers (`INPUT.md` §C.2.3).

---

## 5. The user interface

The wireframe (Claude Design project *"Wireframe screens request"*) contains three competing shells and four supporting flows. **The three shells are alternatives; the four flows are all required regardless of which shell wins.**

### 5.1 Recommendation: build 1b as the primary shell

| Shell | What it is | Verdict |
|---|---|---|
| **1a** ask-first | Conversation is the app; drawing list left, evidence panel right | Easiest to build, but demotes the drawing to a sidebar |
| **1b** drawing-first ✅ | Canvas with live detection overlays, inspector right, ask-bar docked at the bottom | **Recommended primary** |
| **1c** graph-first | Three linked panes, path trace, Cypher/NL toggle | Best as a drill-down, not the front door |

**Why 1b.** The drawing is the artifact of record — an engineer trusts an answer when they can see the box drawn around the thing it is talking about. 1b puts detection overlays, the `unmatched tag?` dashed box and the `37 entities · 21 relations` counter directly in front of the user, which makes the system's *confidence* visible rather than hidden. It also happens to be the shell that best showcases the novelty: region-grounded multimodal evidence.

Adopt the wireframe's own suggestion — **1b as the main screen, with 1a's evidence panel docked to the right, and "see in graph →" opening 1c as a drill-down.** That composition uses all three designs rather than discarding two.

### 5.2 Screens to build

| Screen | From wireframe | Purpose | Priority |
|---|---|---|---|
| Upload & ingest | **1d** | Drop zone, drawing-type picker (P&ID / mechanical / structural), attach reference docs, "Run extraction" | P0 |
| Pipeline progress | **1e** | Live stage tracker: preprocess → detection → OCR → resolution → relations → graph. Per-stage counts, progress bar | P0 |
| Drawing workbench | **1b** | Canvas + overlays, symbol/text/link layer toggles, inspector showing type, tag confidence and relations, docked ask-bar | P0 |
| Answer + evidence | **1a** panel | Answer text, evidence chips, reasoning path, crop viewer with bbox + confidence | P0 |
| Review queue | **1f** | Low-confidence entities. Shows the crop, the OCR read (`V-2O3`), nearest symbol match, ranked suggestions, accept/skip | P1 |
| Graph workbench | **1c** | Node-link view, hop filter, path trace, relation-type filter, Cypher/NL toggle | P1 |
| Evaluation dashboard | **1g** | mAP, OCR accuracy, relation F1, Recall@K tiles + QA-accuracy-by-type chart + failure-case list + report export | P1 |

### 5.3 The review queue is a feature, not a fallback

Screen **1f** deserves emphasis. OCR on engineering drawings will never be perfect — the wireframe's own example, `V-2O3` (capital letter O) versus `V-203` (digit zero), is the canonical failure and it is genuinely ambiguous at 33 px. A system that silently guesses is untrustworthy; a system that surfaces four uncertain reads and asks is trustworthy. Route anything below `review_below: 0.70` here, and **log every human correction** — that log becomes fine-tuning data and a headline result for the report ("human-in-the-loop correction improved effective tag accuracy from X to Y").

### 5.4 Stack
Streamlit for the working app (fastest path, and the reference demo proves it works for this). React only if time permits after evaluation is complete — do not spend Phase-1 time on it.

---

## 6. Evaluation

> **Measured results are in [RESULTS.md](RESULTS.md).** The targets below were set
> before building. As of the first working pipeline: overall **0.8141** on 1,920
> PIDQA questions — simple counting **1.000**, spatial connection **0.902**,
> spatial counting **0.781**, value-based **0.573**, with ground-truth detection.

### 6.1 Metrics and targets

| Stage | Metric | Target | Basis |
|---|---|---|---|
| Detection | mAP@50 | **≥ 0.90** | Bayer et al. (2026) reached 97.4 % on bridge drawings; synthetic P&IDs are cleaner |
| Detection | mAP@50-95 | ≥ 0.65 | Small objects drag this down; 0.65 is honest |
| OCR | Exact-match tag accuracy | **≥ 0.85** | Bayer et al. report 93 %; our tags are smaller |
| Entity resolution | Tag↔symbol match F1 | ≥ 0.88 | |
| Relations | Edge F1 vs ground truth | **≥ 0.75** | The hardest stage — see §7.1. Be honest here |
| Retrieval | Recall@5 | ≥ 0.80 | |
| NL→Cypher | Executable-query rate | ≥ 0.95 | Syntactic validity, cheap to measure |
| NL→Cypher | Exact-answer match | **≥ 0.75** | Against PIDQA `GT` column |
| QA end-to-end | Accuracy by question type | Report all 4 separately | Simple counting will beat spatial connection — show that |

### 6.2 The evaluation harness is nearly free

PIDQA ships 64,000 questions **each with a ground-truth Cypher query and a ground-truth answer**. That gives a three-level diagnostic that most student projects cannot produce:

1. **Graph correctness** — run PIDQA's *reference* Cypher against *our* graph. If the answer is wrong, the fault is in stages 3–7 (vision/KG), not the LLM.
2. **Cypher correctness** — run *our generated* Cypher against a *ground-truth* graph. If wrong, the fault is the LLM.
3. **End-to-end** — our Cypher, our graph.

Being able to say *"our graph is 91 % correct and our Cypher generation is 84 % correct, and here is the ablation"* is a substantially stronger result than a single end-to-end number. **Build this harness in Week 6, before optimising anything.**

### 6.3 Ablations worth running (these become the report's results section)
- Tiled detection vs whole-sheet detection → will show a dramatic gap, and quantifies `INPUT.md` §C.2.1
- With vs without the graph (pure VLM QA baseline) → justifies the entire KG
- With vs without few-shot Cypher selection
- `qwen3:8b` vs `qwen2.5:14b` → is the extra 4 GB worth it?
- With vs without human review-queue corrections

---

## 7. Risks

### 7.1 ⚠️ Relation extraction is the project's real risk
Stages 3–5 (detect, OCR, match) are well-trodden and will work. **Stage 6 — deciding that pump P-101 connects to line L-302 — is where engineering-drawing projects fail.** Pipe lines are broken by symbols, crossed by other lines, interrupted by text, and drawn with jumps. There is no pretrained model for it.

**Mitigation — build it in three escalating tiers, and ship tier 1 early:**
1. **Proximity baseline (Week 6).** Two symbols within N px are connected. Crude, ~50 % F1, but it produces a working end-to-end graph immediately so every other stage can be tested.
2. **Line tracing (Weeks 7–8).** OpenCV LSD → merge collinear segments → snap endpoints to symbol bboxes → walk junctions. Target ~75 %.
3. **Learned refinement (Weeks 9–10, only if time).** A small GNN or a transformer over detected primitives, as in *From Engineering Diagrams to Graphs* (arXiv:2411.13929).

Never let the schedule reach Week 10 with no graph at all. Tier 1 in Week 6 guarantees that.

### 7.2 Other risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Python 3.13 wheel failures (faiss, chromadb) | **High** | Python 3.11 venv (`INPUT.md` §E.2). Do this on day 1 |
| OCR too slow (Paddle is CPU-only ✅) | **High** | ROI-only OCR; batch; cache. Never OCR blank tiles |
| YOLO training too slow on MPS | High | Train on Colab T4, infer locally |
| 18 GB exhausted → swap | Medium | Two-phase model staging (`INPUT.md` §A.3); Kùzu instead of Neo4j if needed |
| Symbols are numbers, not names | **Certain** | `config/symbol_ontology.yaml`, 32 rows, Week 2 (`INPUT.md` §C.2.4) |
| Zenodo datasets never finish | Medium | Not on the critical path; overnight script; Phase 1 needs neither |
| Generated Cypher is invalid | Medium | Validate → repair loop (feed the Neo4j error back to the LLM, retry twice) → fall back to a template query |
| Synthetic-only training won't generalise | **High** | Be honest in the report. State it as a limitation and use PID2Graph for a real-world spot-check if it downloads |

### 7.3 Scope discipline
The synopsis promises P&IDs *and* mechanical *and* structural drawings. **Only P&IDs have training data and a benchmark.** Deliver P&ID fully; demonstrate mechanical/structural qualitatively via the VLM path and label them as such. A complete P&ID system plus an honest limitations section beats three half-built ones.

---

## 8. Repository layout

```
nlp/
├── config/
│   ├── pipeline.yaml
│   └── symbol_ontology.yaml        # the 32 class→name map
├── data/
│   ├── raw/{DigitizePID,PIDQA,demo}/
│   ├── interim/{sheet_id}/
│   ├── graph/
│   └── index/{text,region,fewshot}/
├── src/
│   ├── ingest/      acquire.py  preprocess.py  tiling.py
│   ├── vision/      detect.py   ocr.py         resolve.py
│   ├── graph/       relations.py  build_kg.py  schema.py
│   ├── retrieval/   text_index.py  region_index.py  fewshot.py
│   ├── qa/          nl2cypher.py  answer.py    evidence.py
│   ├── eval/        metrics.py  run_pidqa.py   ablations.py
│   └── app/         streamlit_app.py  screens/
├── scripts/
│   ├── fetch_optional_datasets.sh  ✅ written
│   ├── build_tiles.py
│   └── train_yolo_colab.ipynb
├── tests/
├── docs/  INPUT.md  OUTPUT.md
└── reports/
```

---

## 9. Execution plan — 14 weeks

| Phase | Weeks | Deliverable | Exit criteria |
|---|---|---|---|
| **0 · Setup** | 1 | Python 3.11 venv, Neo4j running, datasets local, models pulled | `pytest` green on a smoke test; Neo4j reachable |
| **1 · Vision** | 2–5 | Tiling + YOLO trained + OCR + entity resolution | mAP@50 ≥ 0.90; `entities.json` for all 500 sheets |
| **2 · Graph** | 6–8 | Relations tier 1 → tier 2; Neo4j loaded; **eval harness** | PIDQA reference Cypher runs on our graph; relation F1 ≥ 0.70 |
| **3 · RAG** | 9–11 | NL→Cypher, tri-modal retrieval, answer + evidence | Exact-answer ≥ 0.70 on a 2,000-question sample |
| **4 · UI** | 11–12 | Screens 1d, 1e, 1b, evidence panel, 1f | End-to-end demo from upload to cited answer |
| **5 · Eval** | 13 | Full benchmark + all ablations + screen 1g | Results table populated; failure cases catalogued |
| **6 · Report** | 14 | Paper, slides, demo video | Submitted |

### 9.1 Week-by-week for the first six (the part that usually slips)

| Wk | Focus | Concrete output |
|---|---|---|
| 1 | Environment | venv, Neo4j container, dataset verified, `symbol_ontology.yaml` started |
| 2 | Tiling | `build_tiles.py` producing 54 tiles/sheet with correct offsets; visual QA on 5 sheets; ontology finished |
| 3 | YOLO train | Colab notebook, 27k tiles, first mAP number |
| 4 | Detection merge | Global NMS across tile seams; `detections.json` for all 500 |
| 5 | OCR + resolve | PaddleOCR angle-aware, ROI-restricted; fuzzy tag matching; `entities.json` |
| 6 | **Graph tier 1 + harness** | Proximity edges → Neo4j loaded → PIDQA reference Cypher executes. **First end-to-end number.** |

Week 6 is the milestone that matters. If a question can be answered end-to-end by Week 6 — even badly — the remaining eight weeks are optimisation, and the project cannot fail. If Week 6 slips, cut relation tiers 2 and 3 and the React UI immediately.

---

## 10. Work split across five members

| Member | Owns | Stages | Primary deliverable |
|---|---|---|---|
| A | **Vision — detection** | 2, 3 | Tiling pipeline, YOLO training, NMS merge, mAP report |
| B | **Vision — text** | 4, 5 | OCR, angle handling, entity resolution, fuzzy matching, review-queue logic |
| C | **Graph** | 6, 7 | Line tracing, relation extraction (all 3 tiers), Neo4j schema + loader |
| D | **RAG / LLM** | 8, 9 | NL→Cypher, few-shot selector, tri-modal retrieval, answer fusion, evidence assembly |
| E | **App + evaluation** | UI, eval | Streamlit screens 1b/1d/1e/1f/1g, PIDQA harness, ablations, report |

The JSON contracts in `INPUT.md` §F are what make this parallel: B can develop against a fake `detections.json` before A finishes, D can develop against a hand-written graph before C finishes. **Freeze those contracts in Week 1 and write a fixture file for each.**

---

## 11. What is genuinely novel here

Positioned honestly against the fifteen surveyed papers:

| Prior work | What it did | What it did not do |
|---|---|---|
| PIDQA (2025) | P&ID → labelled property graph → LLM Cypher QA | No multimodal retrieval; **no visual evidence returned**; graph built offline, not from pixels in-loop |
| Talking like P&IDs (2025) | GraphRAG over P&IDs | Starts from **DEXPI structured data, not from drawing images** |
| BLUEPRINT (2026) | Layout-aware multimodal retrieval | **Retrieval only — no knowledge graph, no multi-hop reasoning** |
| Bayer et al. (2026) | YOLO + OCR + ontology KG on bridges | **Validation via SPARQL, not natural-language QA** |
| MMGraphRAG (2025) | Multimodal KG + structure-aware retrieval | General-domain, **not engineering drawings** |

**The gap this project fills:** every prior system has *either* the pixels-to-graph pipeline *or* the multimodal RAG layer. None has both with **evidence-grounded answers**. Our contribution is the end-to-end path — raw drawing image → knowledge graph → tri-modal retrieval (graph + region + document) → natural-language answer **that cites the drawing region it came from**.

Two secondary contributions worth claiming:
- **A fully free, offline, 18 GB-laptop deployment.** Every comparable system uses GPT-4-class paid APIs. Showing this works on consumer hardware has real practical value, and DesignQA (2024) found even GPT-4-class models struggle here — so the comparison is fair game.
- **A human-in-the-loop review queue** with measured before/after accuracy — a dimension none of the surveyed papers report.

---

## 12. Definition of done

- [ ] Upload a P&ID PDF → graph built end-to-end without manual steps
- [ ] Answer all four PIDQA question types with cited evidence
- [ ] mAP@50 ≥ 0.90, relation F1 ≥ 0.75, QA exact-match ≥ 0.75
- [ ] Every answer carries a reasoning path + drawing crop + confidence
- [ ] Evaluation dashboard populated with real numbers and ≥ 1 ablation
- [ ] Runs fully offline on the 18 GB MacBook with no paid API
- [ ] Review queue demonstrably improves accuracy, with the delta measured
- [ ] Report + slides + demo video
