# USER_GUIDE — Engineering Drawing Intelligence (EDI)

> **Complete Operational Manual & Feature-by-Feature Walkthrough**  
> *A self-contained guide for team members, researchers, and evaluators to navigate, use, and verify every single capability built into the system.*

---

## 1 · Quick Start: Launching the System

Open Terminal on the target machine and run:

```bash
cd /Users/nikhilsridhara/nlp
source .venv/bin/activate

# 1. Verify that all 19 subsystems are active
python scripts/healthcheck.py

# 2. Launch the Web Application
python -m streamlit run src/edi/app/streamlit_app.py --server.port 8520
```

Open your browser to: **`http://localhost:8520`**

---

## 2 · System Overview & Navigation Map

The interface follows a **cockpit telemetry wireframe**:
* **Sidebar (Left)**: Active drawing selector, cross-drawing tag search, upload & extraction engine, reference document indexing, and data bundle export.
* **Top Header & Telemetry Strip**: Displays the active sheet ID, live node/link counts, control loops, flow edges, active local model (`qwen3:8b`), and benchmark accuracy.
* **Main Area**: 7 dedicated functional tabs:
  1. **Workbench** (Visual Q&A with tri-modal grounded evidence)
  2. **Field Mode** (Plain-English technician cards, hazard warnings, isolation planner)
  3. **Knowledge Graph** (Interactive topology layout & raw Cypher query runner)
  4. **Edit Graph** (Interactive canvas for adding/editing symbols and lines)
  5. **Rig Builder** (Conversational inverse P&ID design & procedural generation)
  6. **Review Queue** (Human-in-the-loop OCR verification and correction)
  7. **Evaluation** (PIDQA benchmark metrics and failure case inspection)

---

## 3 · Feature-by-Feature Walkthrough

### Tab 1: Workbench (Visual Q&A & Provenance)
* **What it does**: The central query workbench. Asks natural language questions and receives answers backed by (1) Cypher graph paths, (2) bounding-box drawing crops, and (3) reference citations.
* **How to navigate**:
  * **Ask Box (Top Right)**: Type any question (e.g., *"What is connected to P-1101A?"*, *"How many valves are on this drawing?"*, *"Explain this whole drawing"*). Press **Enter** or click **Ask**.
  * **Suggestion Chips**: If no conversation is active, click any pre-computed suggestion chip to see an instant response.
  * **Interactive Drawing Canvas (Left)**: Drag to pan across the 7000×4500 px drawing; scroll with your mouse wheel or trackpad to zoom in to millimeter-scale symbols. When a question is answered, the relevant symbols are highlighted with glowing rings.
  * **Evidence Drawer (Right)**: Click the **"evidence & query"** expander beneath any answer to inspect the exact Cypher query executed, execution latency (in ms), tabular SQL/Cypher results, and cropped image snippets.
  * **Export Conversation**: Click **"Export conversation (.md)"** to download the complete Q&A transcript with Cypher logs.
  * **Clear**: Resets the chat history.

---

### Tab 2: Field Mode (The Operator & Technician Screen)
* **What it does**: Tailored for personnel who do not read complex P&IDs. It translates topological graphs into plain-English job briefs, fluid hazard assessments, and isolation plans with **zero model hallucinations** (purely graph- and rule-derived).
* **How to navigate**:
  1. **Tag Lookup / Plain Question**:
     * Type an equipment tag like `V-1101A` or ask *"What is V-1101A?"*.
     * **Result**: Displays an **Item Card** containing:
       * Plain-English equipment identification (ISA-5.1 decoded).
       * *On the Job*: Operational significance.
       * *Hazard Warnings*: Flammability, toxicity, or pressure risks derived from connected line service codes.
       * *Lines on it*: Nominal diameter, service name, and hazard classification.
       * *Which way it flows*: Upstream and downstream connections with confidence and rule explanations (e.g., `0.95: connector says FROM`).
       * *Relief Devices*: Safety valves connected to this equipment (explicitly flagged: *"never block these in"*).
       * *Drawing Crop*: Exact visual location on the sheet.
       * *What this does NOT tell you*: Explicit list of unstated variables (e.g., design pressure, wall thickness).
  2. **Valve Isolation Planning**:
     * Type *"Which valves isolate P-1101A?"* or *"How do I lock out V-1101A?"*.
     * **Result**: Generates an **Isolation Checklist**:
       * *Close these*: First genuine block valves found along every branch.
       * *Relief & Blowdown paths*: Highlighted so operators do not inadvertently trap pressure.
       * *Branches with no block valve*: Honest reporting of lines that leave the sheet without a valve.
  3. **Industry Definition Lookup**:
     * Type *"What does SDV mean?"* or *"What is a PSV?"*.
     * **Result**: Decodes standard industrial instrument and valve acronyms instantly from the internal knowledge layer without querying the LLM.

---

### Tab 3: Knowledge Graph (Topology Explorer & Cypher Runner)
* **What it does**: Direct exploration of the underlying graph database (Kùzu).
* **How to navigate**:
  * **Graph Visualization (Left)**: Interactive Node-Link graph laid out on the drawing's real geometry. Click nodes to trace connections; click legend chips to filter equipment categories.
  * **Query Engine (Right)**:
    * Switch between **Natural language** mode (enter a question → model generates Cypher → executes) or **Cypher** mode (enter raw Cypher queries like `MATCH (s:Symbol)-[:CONNECTED_TO]-(n) RETURN s.tag, count(n)`).
    * Displays results in an interactive data table.
  * **Topology Metrics**: Displays node count, connected edges, control loops (`CONTROLS`), flow vectors (`FEEDS_TO`), mean node degree, and isolated component counts.

---

### Tab 4: Edit Graph (Human-in-the-Loop Visual Annotation)
* **What it does**: Fix extraction errors visually on the drawing. All edits are saved as a non-destructive overlay (`edits.json`) that survives re-extractions.
* **How to navigate**:
  * **Canvas Modes (Right Radio Selector)**:
    1. **Inspect**: Click any symbol on the drawing to inspect its tag, category, and connected neighbors.
    2. **Add symbol (draw)**: Click and drag a bounding box on the drawing canvas. A sidebar form opens: select the symbol class (1..32), type the tag name (e.g., `PV-1042`), and click **Add**.
    3. **Connect**: Click Symbol A, then click Symbol B. Select the relationship type (`CONNECTED_TO`, `CONTROLS`, or `FEEDS_TO`) to draw a topological link.
    4. **Delete (erase)**: Click any symbol or edge to remove it from the graph.
  * **History & Undo**: Displays a timestamped log of all manual changes. Click **Undo last** or **Discard all**.

---

### Tab 5: Rig Builder (Conversational Inverse P&ID Design)
* **What it does**: Generates standard 7168×4562 px engineering drawings from natural language specs. Solves the ground-truth problem by generating drawings and graphs from the exact same mathematical spec.
* **How to navigate**:
  * **LLM Selector**: Choose any local Ollama model (`qwen3:8b`, `deepseek-coder`, etc.).
  * **Auto-complete**: Keep checked to automatically generate ASME relief valves, ESD loops, and HAZOP-compliant bypass lines.
  * **Starter Package Chips**: Click any pre-configured package:
    * *Two-stage Separation Train*
    * *Gas Compression & Aftercooling*
    * *Glycol Gas Dehydration*
    * *Flare & Relief System*
    * *Produced Water Treatment*
  * **Natural Language Input**: Type a custom instruction (e.g., *"build a gas compression train with suction scrubber and flare relief"*).
  * **Registers & Deliverables**: Automatically compiles 5 engineering registers available for CSV download:
    * `equipment_list.csv`
    * `line_list.csv`
    * `valve_register.csv`
    * `instrument_index.csv`
    * `cause_and_effect.csv`
  * **Commit to Graph**: Click **"Draw & ingest this sheet"** to render the drawing and register it into the live knowledge graph.

---

### Tab 6: Review Queue (Quality Assurance & Human Correction)
* **What it does**: Automatically isolates low-confidence OCR reads and ambiguous symbols for one-click human verification.
* **How to navigate**:
  * **Confidence Slider**: Adjust the confidence threshold (e.g., `0.70`).
  * **Card Review**: For each flagged symbol, the UI displays:
    * The cropped image snippet of the symbol on the drawing.
    * The OCR candidate read vs. Expected prefix convention.
    * An editable text box.
  * **Apply**: Type the corrected tag and click **Apply corrections**. Updates the Knowledge Graph immediately and sets provenance to `human`.

---

### Tab 7: Evaluation (Quantitative Benchmark Dashboard)
* **What it does**: Real-time inspection of the academic PIDQA benchmark (2,176 questions, 17 sheets).
* **How to navigate**:
  * **Score Cards**: Top metric grid displays accuracy across:
    * *Simple Counting*: 0.991
    * *Spatial Counting*: 0.792
    * *Spatial Connection*: 0.888
    * *Value / Tags*: 0.544
    * *Overall Benchmark*: **0.8038** (0 query errors).
  * **Failure Cases Table**: Scrollable table showing the exact failures (Sheet ID, Question, Expected Ground Truth, System Prediction) with colored diff formatting.

---

### Sidebar Features (Global Controls)

1. **Drawing Selector**: Choose between real uploaded offshore drawings (`u_pid_real...`), synthetic benchmark drawings (`0`, `2`, `5`), or procedurally generated rig drawings (`u_rig...`).
2. **Find Across All Drawings**: Global tag search across all drawings currently loaded in the database.
3. **Add a Drawing (Ingestion)**:
   * **Engine**: Select **Blueprint** (for any real, unseen customer drawing) or **Detector** (for standard benchmark symbols).
   * **File Upload**: Accepts PDFs, TIFFs, PNGs, and JPEGs.
   * **Fast Mode**: Skips multi-scale OCR retries for rapid extraction (~55s vs 130s).
   * **VLM Fallback**: Uses `qwen3-vl` to re-read difficult tags.
   * Click **Run extraction**: Displays a live 6-stage extraction progress monitor.
4. **Reference Documents**: Upload equipment manuals, vendor cut-sheets, or operating procedures (PDF/TXT/MD). Click **Index** to populate the RAG document store.
5. **Data Export**: Click **Data bundle (.zip)** to download a complete export package containing `entities.csv`, `relations.csv`, `graph.json`, and `graph.cypher`.

---

## 4 · Complete Feature Status Matrix

| Feature / Subsystem | Status | Real-World Operational Notes |
| :--- | :---: | :--- |
| **High-Res Canvas (Pan/Zoom)** | **WORKING** | Smooth, sub-pixel coordinate alignment on 7000px sheets. |
| **Workbench Q&A (Suggestions)** | **WORKING** | Pre-computed suggestion queries execute instantly. |
| **Custom Natural Language Q&A** | **WORKING** | Takes 5–9 seconds via local Qwen-8B; grounded in graph. |
| **Tri-Modal Evidence (Crops & Graph)** | **WORKING** | Bounding box crops and Cypher paths display reliably. |
| **Field Mode (Item Cards)** | **WORKING** | Decodes ISA-5.1 tags, fluid hazards, and line specs. |
| **Field Mode (Isolation Planner)** | **WORKING** | Generates block valve lists; refuses to use check valves. |
| **Field Mode (Definition Lookup)** | **WORKING** | Decodes tags like `SDV`, `BDV`, `PSV`, `LSLL` instantly. |
| **Topology Graph Viewer (Tab 3)** | **WORKING** | Interactive Node-Link graph laid out on drawing pixels. |
| **Raw Cypher Runner (Tab 3)** | **WORKING** | Executes custom Cypher queries directly on Kùzu engine. |
| **Edit Graph (Draw/Connect/Erase)**| **WORKING** | Visual postMessage canvas; saves to non-destructive overlay. |
| **Rig Builder (Generative Design)** | **WORKING** | Compiles natural language specs to 7000px drawings + registers. |
| **Review Queue (Human Fixes)** | **WORKING** | Flags low-confidence tags; applies corrections directly to KG. |
| **Benchmark Dashboard (Tab 7)** | **WORKING** | Renders 2,176 question results and failure inspection table. |
| **Blueprint Mode (No-Model Ingest)**| **WORKING** | Tested on real North Sea offshore P&IDs (81 items, 93 links). |
| **Data Export (.zip bundle)** | **WORKING** | Generates CSV, JSON, and Cypher bundles in one click. |
| **Document Citations (RAG)** | *NEEDS PDF* | The store is currently empty. Upload a PDF in the sidebar to activate. |
| **Enter Key in Custom Ask Box** | *WIDGET QUIRK*| When typing custom questions, clicking the **"Ask" button** is 100% reliable. |

---

## 5 · 5-Minute Script for Demoing the System

Follow this exact sequence to demonstrate the platform to a professor or client:

1. **Start on the Workbench (`tab 1`)**:
   * Show the high-res drawing canvas on the left. Zoom in to show how small the symbols and line tags are.
   * Click the suggestion button: **"What is connected to P-1101A?"**.
   * Show the answer: point out the glowing highlighted boxes on the drawing canvas, the graph path, and open the **"evidence & query"** drawer to show the real Cypher query.
2. **Switch to Field Mode (`tab 2`)**:
   * Click **"Which valves isolate P-1101A?"**.
   * Highlight the isolation list, the valve hops, and the honest safety warning (*"Eleven branches leave without a block valve on this sheet"*).
   * Type *"What does SDV mean?"* to show instantaneous ISA-5.1 acronym decoding.
3. **Switch to Knowledge Graph (`tab 3`)**:
   * Show the interactive Node-Link graph laid out over the plant geometry.
   * Point out the topology metrics (nodes, links, control loops, mean degree).
4. **Switch to Rig Builder (`tab 5`)**:
   * Click **"Flare & Relief System"** or **"Two-stage Separation Train"**.
   * Show the 5 generated engineering registers (Equipment list, Line list, Valve list, Instrument index, Cause & Effect matrix).
5. **Switch to Evaluation (`tab 7`)**:
   * Show the academic score: **0.8038 overall accuracy** across 2,176 questions with **0 query execution errors**.
