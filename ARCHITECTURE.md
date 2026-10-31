# ARCHITECTURE

How the code is laid out, how data flows through it, and **why** each decision
was made. Every "why" below was settled by measurement — the numbers are in
`docs/RESULTS.md`.

---

## 1 · Data flow

```
 UPLOAD ──► acquire.py ──► sheet PNG (normalised, ~7000px long edge)
                              │
 ┌────────────── OFFLINE INGESTION (once per drawing) ──────────────┐
 │  1 preprocess   tiling.py      54 tiles @1024, stride 768        │
 │  2 detect       detect.py      YOLO per tile → sheet coords → NMS │
 │                                → detections.json                  │
 │  3 ocr          ocr.py         2 orientations, global assignment  │
 │  4 resolve      ocr.py         tag ↔ symbol, review flags         │
 │                                → ocr.json                         │
 │  5 relate       relations.py   erase symbols → trace runs → MST   │
 │                                → relations.json                   │
 │                 semantics.py   CONTROLS, FEEDS_TO                 │
 │                                → derived.json                     │
 │  6 build        kg.py          Kùzu load (+ edits.json overlay)   │
 └───────────────────────────────────────────────────────────────────┘
                              │
 ┌──────────────── ONLINE QUERY (per question) ─────────────────────┐
 │  fewshot.py   TF-IDF picks similar solved question→Cypher pairs  │
 │  nl2cypher.py qwen3:8b generates Cypher, validates, repairs      │
 │  kg.py        executes against the graph                          │
 │  documents.py retrieves specification excerpts in parallel        │
 │  answer.py    fuses graph rows + crops + documents → answer       │
 └───────────────────────────────────────────────────────────────────┘
```

Each stage writes JSON to `data/interim/{sheet}/`. That is deliberate: stages are
independently testable and restartable, a failure is traceable to one file, and
several people can work in parallel against fixed contracts.

---

## 2 · Code map

```
src/edi/
├── config.py            config loader + THE class-id conversion (0-based ↔ 1-based)
├── pipeline.py          orchestrates the 6 stages with progress callbacks
├── edits.py             manual edit overlay (add/remove/modify, replay, undo)
├── export.py            CSV / JSON / Cypher / zip bundle
├── ingest/
│   ├── acquire.py       PDF & image → normalised sheets
│   ├── tiling.py        tile grid, label remap, coordinate mapping, NMS
│   ├── blueprint.py     detector-free ingestion of real drawings
│   ├── lines.py         pipe line numbers → size · service · spec
│   └── connectors.py    off-page connectors + cross-sheet joining
├── vision/
│   ├── detect.py        YoloDetector + GroundTruthDetector (same interface)
│   ├── ocr.py           PaddleOCR, rotation, tag scoring, global assignment
│   ├── ocr_pool.py      process-level parallelism (Paddle is single-threaded)
│   └── vlm.py           qwen3-vl fallback for tags OCR cannot read
├── graph/
│   ├── relations.py     ink mask → runs → junctions → nets → MST edges
│   ├── semantics.py     CONTROLS / FEEDS_TO derivation
│   ├── isolation.py     which valves actually isolate an item, and what it cannot close
│   ├── flow.py          flow direction from the drawing's own labelling, with confidence
│   └── kg.py            Kùzu store, schema, cross-sheet search
├── retrieval/
│   ├── fewshot.py       TF-IDF few-shot selector for NL→Cypher
│   └── documents.py     reference-document chunking, indexing, retrieval
├── knowledge/           what things ARE (as opposed to what connects to what)
│   ├── isa.py           ISA-5.1 tag decoding + the industry conventions that override it
│   ├── glossary.py      plant terminology for a newcomer + hazard per pipe service
│   └── __init__.py      explain(): glossary -> line number -> ISA -> tag convention
├── qa/
│   ├── nl2cypher.py     prompt, generation, validate-and-repair loop
│   ├── field.py         field-question router, item cards, honest refusals
│   └── answer.py        evidence assembly + answer phrasing
├── rig/                 the rig builder — English → spec → drawing → graph
│   ├── catalog.py       105 oil-rig item types: prefix, category, symbol, design data
│   ├── templates.py     11 standard packages; also the no-LLM fallback
│   ├── spec.py          tagging · auto-complete · HAZOP checks · the registers
│   ├── layout.py        layered graph layout, in sheet pixels
│   ├── render.py        ISA symbols, pipe routing, flare header, title block
│   ├── build.py         writes the ingest contracts, loads the graph
│   └── chat.py          local-LLM conversation, JSON repair, fallback
├── eval/run_pidqa.py    L1 / L3 harness, Cypher repair, answer normalisation
└── app/
    ├── streamlit_app.py 7 screens
    ├── editor.py        bidirectional canvas component (click/drag → Python)
    ├── components/editor/index.html   the component itself, plain JS, no npm
    ├── rigtab.py        the rig builder screen
    ├── fieldtab.py      field mode — the technician's screen
    ├── canvas.py        read-only zoom/pan canvas for the workbench
    ├── graphview.py     node-link graph on the drawing's real geometry
    └── theme.py         the entire visual system
```

Scripts: `dataset_report` · `healthcheck` · `build_tiles` · `train_yolo` ·
`ingest` · `redo_relations` · `build_graph` · `evaluate` · `eval_detection` ·
`sweep_relations` · `diag_ocr` · `symbol_atlas` · `promote_model` · `rebuild_all` ·
`rig_build` · `eval_flow`

---

## 3 · Decisions, and what forced them

### Tiling is mandatory
Symbols are ~33×62 px on a 7168×4562 sheet. At 1280 px they become 6×11 px.
Tiles at 1024/stride 768 keep them at native size; global NMS removes the
duplicates that 25% overlap creates. **Without this nothing else matters.**

### Kùzu, not Neo4j
No Docker on the target machine, and Neo4j's JVM wants ~2 GB of an 18 GB budget.
Kùzu is embedded, MIT, ~50 MB, and was verified to execute **all four** PIDQA
query shapes including undirected 3-hop traversal and `STARTS WITH`.
*Trade-off:* single-writer. Hence ingest→JSON, separate short-lived graph build.

### Edges stored in ONE direction
Kùzu's `-[:R]-` is already undirected. Storing both directions made every
undirected match fire twice and **silently doubled every count**. Sheet 0
reported 308 edges instead of 154.

### TF-IDF, not embeddings, for few-shot
bge-m3 took **93 s** to load under memory contention and costs 1.2 GB of the
query-phase budget. PIDQA questions are templated, so word+char TF-IDF picks the
right pattern in **0.2 s**. Embeddings remain the choice for *documents*, where
wording actually varies.

### Two OCR orientations, not three
15 of 19 inspected failures were vertically-printed tags read as `-44323`
instead of `OP-44323`. Rotating fixes it. Measured: `0/±90` = 187 s at recall
0.938; **`0/-90` = 127 s at the same 0.938** — the orientation classifier
already resolves the 180° flip.

### Global tag assignment, not nearest-text
Choosing independently let symbol `IJ-07591` claim its neighbour's tag
`ERV-7-11`. Assignment is now global: one tag per symbol, one symbol per tag,
cost discounted when the text sits on the symbol's own row or column.

### A pipe run is a chain, not a clique
Emitting all pairs within a net over-counted on **10 of 10** failing questions,
zero under-counts. Each net is reduced to its minimum spanning tree. Geometry
was then swept: `erase_pad 20 / snap 24 / max_edge 1600`.

### Processes, not threads, for OCR
PaddleOCR here is CPU-only and single-threaded: **~1000 ms/crop at
OMP_NUM_THREADS of 1, 6 and 10 alike**. Three worker processes give **1.93×**
with byte-identical output.

### Edits are an overlay, never a rewrite
`edits.json` is replayed on top of pipeline output. Writing corrections back
into `ocr.json` would mean the next re-ingestion silently destroys hours of human
work. The overlay survives re-extraction, model upgrades and parameter changes,
and every edit is logged with a timestamp.

### CONTINUES_ON is the one relation that must NOT be sheet-scoped
Every query is constrained to the current sheet, because the graph holds many
drawings. `CONTINUES_ON` is the exception: it spans sheets by definition, so
scoping both ends asks for a cross-sheet link whose ends are on the same sheet —
which can never match. `scoped()` pins only the near side.

### Cross-sheet matching is on distinctive tokens, not whole strings
One drawing says `TO FLARE`, another says `HP FLARE HEADER`. Exact matching
finds nothing. Generic nouns (HEADER, SYSTEM, LINE, STAGE, LP, HP) are stripped
and the remaining words compared by subset, so `{FLARE} ⊆ {FLARE}` links while
`SUPPLY HEADER` and `RETURN HEADER` correctly do not.

### The editor is a declared component, not `components.html`
`components.html` is one-way, so the first editing UI made the user type pixel
coordinates for a bounding box — which is not something anyone looking at the
symbol should have to do. The editor is a **declared custom component** speaking
Streamlit's postMessage protocol in plain JavaScript (no npm, no build step). It
returns `select` / `draw` / `connect` / `delete` actions, with coordinates
converted back to original sheet pixels so an edit made at any zoom lands
correctly.

*Gotcha:* announcing `componentReady` once on script load races Streamlit's
mount and the host logs "Received component message for unregistered
ComponentInstance!". Keep announcing until the first render arrives.

### GroundTruthDetector is not a stub
It replays the dataset's own labels as detections. That isolates failures — a
wrong answer under ground-truth detection is a relations or LLM bug, never a
detector bug — and gives the "perfect detection" ceiling for the ablation table.

---

## 4 · Contracts between stages

```jsonc
// detections.json
{"sheet_id":"0","image_size":[7168,4562],
 "detections":[{"id":"d0000","yolo_class":20,"pidqa_class":21,
                "bbox":[1669,767,1702,829],"confidence":1.0}]}

// ocr.json  — detections plus resolved tags
{"records":[{"id":"d0000","pidqa_class":21,"bbox":[...],
             "tag":"SDL 101","tag_confidence":1.0,
             "tag_location":"internal|external|none"}]}

// relations.json
{"relations":[{"source":"d0001","target":"d0014",
               "type":"CONNECTED_TO","evidence":"net_trace_mst","confidence":0.8}]}

// derived.json
{"CONTROLS":[{"source":"d0003","target":"d0009","evidence":"instrument_adjacent_to_valve"}],
 "FEEDS_TO":[{"source":"d0011","target":"d0002","evidence":"flow_arrow"}]}

// edits.json  — the human overlay
{"entities":{"added":[...],"removed":["d0007"],"modified":{"d0003":{"tag":"V-203"}}},
 "relations":{"added":[...],"removed":[["d0001","d0014","CONNECTED_TO"]]},
 "log":[{"ts":"...","action":"modify_entity","detail":"..."}]}
```

## 5 · Graph schema

```cypher
(:Symbol {entity_id, class:INT64 /* 1..32, PIDQA convention */, name, tag,
          sheet_id, x0,y0,x1,y1, confidence, tag_confidence, needs_review})

(:Symbol)-[:CONNECTED_TO {confidence, evidence}]-(:Symbol)   // undirected
(:Symbol)-[:CONTROLS     {confidence, evidence}]->(:Symbol)  // instrument → valve
(:Symbol)-[:FEEDS_TO     {confidence, evidence}]->(:Symbol)  // upstream → downstream
(:Symbol)-[:CONTINUES_ON {confidence, evidence}]->(:Symbol)  // spans sheets
```

`class` is stored **1-based** so PIDQA's 64,000 reference queries run unmodified
against our graph — that is what makes the L1 diagnostic possible.

---

## 6 · Testing

`tests/` — 224 tests, all fast (< 2 s), no network, no model loads.

| File | Pins |
|---|---|
| `test_class_ids.py` | the 0↔1 off-by-one, incl. the empirical ZLO anchor |
| `test_tiling.py` | tile grid, edge coverage, seam-clipped boxes, NMS dedup |
| `test_cypher.py` | PIDQA's invalid Cypher repair, fence/think stripping |
| `test_ocr_geometry.py` | rotation mapping vs cv2, tag completeness scoring |
| `test_semantics.py` | CONTROLS direction, arrow direction from ink mass |
| `test_documents.py` | chunking bounds, empty-store safety |
| `test_export.py` | every export format, quote escaping |
| `test_edits.py` | overlay purity, cascade deletes, undo, no duplicates |

Plus two runtime audits: `scripts/dataset_report.py` (data present & complete)
and `scripts/healthcheck.py` (19 subsystems live).
