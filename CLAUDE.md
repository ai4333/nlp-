# CLAUDE.md — read this first, every session

Instructions for working on this project. **[PROJECT.md](PROJECT.md)** is the
full picture; this file is how to behave here and what has already been decided,
so nothing has to be re-explained.

---

## What this is, in one line

Ask questions in plain English about oil-and-gas P&IDs and get answers **with
the drawing region, the graph path and the query attached**. Runs entirely
offline on Nikhil's 18 GB MacBook Pro. Final-year project, five people, but it
is being pitched to a real oil rig manager.

---

## What Nikhil has told me — treat these as settled

**Training is DONE.** A teammate trained it on Colab (12 epochs, 1.9 h) and it
is installed at `models/yolo_pid.pt`. **Full-sheet detection F1 0.992**
(precision 0.991, recall 0.994). Do not suggest more training — the curve was
flat from epoch 1 and the last three epochs gained +0.003. `TRAINING_BUNDLE.zip`
and `HANDOFF.md` remain for retraining on a customer's own symbol legend, which
is the only reason to train again.

**He is short on time.** Do not gold-plate. When he says finish something,
finish that thing and stop — do not start a fourth improvement he did not ask
for. Ask before opening a new front.

**This is being sold, not just submitted.** Two audiences: an oil rig manager,
and a teacher he rates highly. He wants industry-level quality and to be
"100% sure" before showing anyone. So: verify before claiming, and hand him
things he can cross-check himself.

**He cannot get the customer's drawings until he demos something that works on
drawings like theirs.** That is why blueprint mode (detector-free) exists and
why it matters more than accuracy on the synthetic benchmark.

**UI/UX is his priority when he raises it, and he is blunt about it.** He called
an earlier design "total shit". The current direction is **cockpit wireframe**,
rebuilt from the shared reference (Spacecraft Cockpit Wireframe UI on
variant.com): `#030303` stage, Space Mono on everything, square corners,
bright hairline borders, corner brackets and scanlines on every panel, a
blueprint grid, uppercase snake_case system labels, and a fixed telemetry strip.
Colour only as 5px indicator lamps. The rules and the token set are in
**docs/DESIGN.md** — read it before touching anything visual. He rejected the
glassmorphism pass before this one as looking generic; the failure was mostly
that Streamlit was silently rendering the 9.5px labels at 16px.

**He wants to type any question, conversationally.** "Like ChatGPT." The chat
must always work; it is the product. If it breaks, that is a P0.

**Documentation exists so context loss costs nothing.** Keep PROJECT.md,
ARCHITECTURE.md and this file current when something real changes.

## How he communicates

Voice-to-text, so expect typos and run-on sentences — read for intent, do not
ask him to clarify wording. Casual tone back. He wants depth and real reasoning,
not generic advice. Do not pad replies.

---

## How to work on this

**Measure before you build.** Every real win here came from looking at the data
first — rendering failing OCR crops as a contact sheet, dumping the actual
strings on a drawing, sweeping parameters against the benchmark. Every wasted
hour came from assuming a pattern and coding to it. Look first, always.

**Verify before you report.** Run `pytest`, `healthcheck.py`, `dataset_report.py`.
Say what actually happened, including when a number went down.

**Report negative results.** Several good-looking ideas failed (see
docs/RESULTS.md §"Tried and rejected"). Recording them is worth as much as the
wins — it stops them being retried.

**Quote 0.8038, not 0.8612.** `0.8038` is the real end-to-end score with the
trained detector — that is the honest number and the one to use. `0.8612` was
measured with ground-truth detection replayed from the dataset labels; it is a
ceiling, useful only as an ablation ("given perfect detection…"). Never present
the ceiling as the result.

---

## Traps — each of these cost real time

| Trap | What happens |
|---|---|
| **Class ids are off by one** | YOLO is `0..31`, PIDQA is `1..32`. Conversions live only in `src/edi/config.py`. Get it wrong and all 64,000 benchmark queries return confidently wrong answers. |
| **`snap` must exceed `symbol_erase_pad_px`** | Relation extraction erases symbols before tracing; a smaller snap attaches nothing and the graph comes out **completely empty**. |
| **Kùzu is single-writer** | Many readers are fine, one writer locks everyone out. Ingest writes JSON only; `build_graph.py` holds the lock for seconds; the app opens read-only. |
| **Streamlit wraps every vertical block in the same testid** | `stVerticalBlockBorderWrapper` is on bordered AND unbordered containers, so styling it directly drew a second slab around every tab panel. Panels opt in: `st.container(border=True)` + `theme.PANEL` as the first child, matched with `:has()`. |
| **"Explain the whole drawing" had no handler** | The question names no tag, so it fell through to Cypher, which wrote a query returning ONE row — and the model then faithfully described a single pump as though it were the sheet. The most natural question about a drawing was the one it could not answer. `field.sheet_overview()` now gathers equipment, instruments, valves, lines-by-service, flow paths and control loops in one pass, and `answer.narrate_stream()` has the model rewrite that briefing as prose. Grounding beats model size: the model was never the problem, the context was. |
| **A long briefing makes the model run out of budget** | Listing all 47 line numbers pushed the briefing long enough that the model wrote an essay and truncated mid-sentence at 109 s. Summarising lines by service and asking for 350–450 words gives a complete answer in 29 s. |
| **Never force `position` on `.stApp > *`** | Streamlit keeps `[data-testid=stAppViewContainer]` at `position:absolute; inset:0` so the inner `stMain` is viewport-height and scrolls. A blanket `.stApp > *{position:relative}` (added to lift content above the fixed grid layer) made the container grow to full content height; `.stApp`'s `overflow:hidden` then clipped the bottom and **the page could not be scrolled at all**. Lift content with `z-index` only — it is already positioned. |
| **A bracketed panel box must not wrap dynamic content** | The whole sidebar was wrapped in a corner-bracket `PANEL`-style box. Its border is drawn at one height, but the expanders inside grow/shrink, so the frame ended up slicing through the content. Panels frame fixed regions; a scrolling column of controls gets a clean edge instead. |
| **Mono + uppercase + tracking is unreadable for sentences** | Every string in the app was simultaneously Space Mono, UPPERCASE, tracked .12–.22em and 9–11.5px. Each slows reading; stacked they made plain English hard to parse, and the people using this are technicians on a tablet. The rule now: **mono carries data (tags, counts, code, tables), sans carries language.** Body text is Inter ≥13px, sentence case, no tracking. |
| **Do not answer a usability problem with instructions** | A "Start here" tab explaining the app was built and immediately rejected as "uglier and childish". Users can understand the domain — the fix is fewer steps and readable controls, not explanatory copy. Empty states get one short line, not a paragraph. |
| **Streamlit's markdown out-specifies a bare class** | `.eyebrow{font-size:9.5px}` rendered at **16px** — every micro-label was four times its intended size and the whole interface read as generic. Pin `font-size` and `letter-spacing` with `!important` on any custom class. |
| **Uppercase + wide tracking grows a button** | At 10px/0.18em uppercase, labels that fit on one line wrapped to four and buttons came out 91px tall. Controls are `nowrap` + ellipsis; only sentence-like suggestions wrap. |
| **Streamlit paints its own widgets red** | Checkboxes, sliders, spinners and focus rings use `primaryColor`, which defaults to `#FF4B4B`. It is set near-white in `.streamlit/config.toml`; do not chase them one at a time in CSS. |
| **Never `session_state.pop()` a Streamlit widget key** | It unregisters the widget and Streamlit silently stops delivering its value to Python. This is exactly what "I can't type anything" was. Reset by bumping a counter in the key instead. |
| **`st.chat_input` pins to the bottom of the app** | Not to its container. It landed below the canvas, off-screen. The ask box sits ABOVE the conversation so it never moves. |
| **A second resident model destroys answer latency** | Benchmarking four models left them all loaded on the 18 GB machine. The same question then took **1058 s** instead of 5 s — first token stayed fast (3.7 s) but generation collapsed from 72 ch/s to under 1. Warm and alone, qwen3:8b answers in **~5 s (0.7 s to first token)**. `keep_alive="30m"` keeps it resident; run `ollama stop <other>` if anything else was loaded. Never benchmark models while the app is being demoed. |
| **Vision and language models cannot be co-resident in 18 GB** | With YOLO training holding 7.4 GB an Ollama call timed out at 120 s; suspending training gave 5.4 s. Never run training and the app together. |
| **Per-epoch YOLO validation crashes on Apple MPS** | Process vanishes at the end of epoch 1, leaked-semaphore warning, no traceback. Use `--no-val` on a Mac. NVIDIA is fine. |
| **`CONTINUES_ON` must not be sheet-scoped** | It spans sheets by definition; scoping both ends can never match. |
| **PaddleOCR is single-threaded here** | `OMP_NUM_THREADS` does nothing. Only separate processes help (1.93× with 3 workers). |
| **Browser automation loses focus** | Clicks land on `BODY`. Focus the element via JS first, then use real keyboard input. A failed UI test is usually the harness, not the app — confirm before "fixing" working code. |
| **Ultralytics version must match the checkpoint** | The model was trained with 8.4.128. Loading it under 8.3.155 segfaulted (exit 139) with no traceback — the checkpoint's model object would not even deserialize. The venv is pinned to 8.4.128; do not downgrade. |
| **Detector thresholds are tuned, not default** | `conf 0.88 / merge_iou 0.20`. Ultralytics defaults (0.25/0.55) score F1 **0.83** on this model; the tuned pair scores **0.99**. The NMS value matters most — 512px tiles at stride 384 overlap heavily and a lenient IoU leaves duplicates in, inflating every count. |
| **A flare KO drum is not a sink** | Direction is not "sink or not sink": a KO drum is downstream of every relief valve and upstream of the stack. `flow.py` ranks how far along the disposal system a node sits (2 terminal · 1 on the way · 0 the plant) and flows low→high. Treating "has flare in the name" as terminal reversed every relief and drain line touching it. |
| **Rank must look THROUGH inline valves** | An ESD valve spliced into a flare header is rank 0 alone, which made the header run backwards into it. `flow.effective_rank` walks two hops through valves and lines — but must exclude the node being reasoned about, or the rank leaks back around a relief valve and both sides look downstream. |
| **A clipped line number looks exactly like an equipment tag** | OCR returns `DC266` out of `1"-DC-11-266-P1A-HE`. The prefix walk resolves the FIRST letter, so it became a **drum** — 22 of 37 "equipment" on the real offshore sheet were pipe labels. `lines.parse_fragment` now catches multi-letter service codes; single letters are excluded because `P` is both "process line" and "pump". Never widen it to single letters. |
| **An empty briefing is worse than no answer** | `ask()` nested the field evidence under `evidence["field"]`, but the workbench streamed from `evidence["briefing"]` — so it got `""`, asked the model to "explain this drawing" with NO facts, and got a fluent, confident description of a distillation column with HX-101, T-101, CV-101, "1.2 m diameter, 18 m high". None of it exists on any drawing in the corpus. The briefing is now surfaced at the top level, and `narrate_stream`/`answer_stream` REFUSE to generate from a briefing under 40 chars. Never add a streaming caller without checking it is actually grounded. |
| **A thin result makes a thin answer — every time** | The overview bug was not special. ANY question whose generated query returned one narrow row got a one-row answer: "design pressure of V-1101A?" came back describing a separator. `qa/context.py` now assembles the same grounded briefing for EVERY question — query rows, item cards for tags named in the question or returned by it, ISA/glossary definitions, and a keyword search of the sheet when nothing else matches. The model only phrases it. |
| **A question that names a tag is not a request for its card** | `field.route` returned `item` for anything containing a tag, so "what should I check before I open BDV-11350?" got a card. Bare identification ("what is X") is the card; anything else routes to `item_question`, which uses the card as grounding and answers what was asked. The `_BARE_ID` regex must NOT allow spaces in the tag slot — it once matched "the design pressure of V-100" (24 chars) and the behaviour depended on how long the tag happened to be. |
| **A deterministic result must not go to the model** | "How many valves?" returns a single number. Phrasing that with an LLM cost seconds AND invented content — asked for a count of 16 instruments it volunteered "valves, sensors and controllers", none of it in the briefing. `answer.direct_answer()` answers scalar counts in code. |
| **Never let the model fill silence** | `phrase()` is told "if rows are present the answer is NOT empty", so one row containing only the tag you asked about produced an invented paragraph of safety guidance. `answer._rows_are_informative` now blocks that. On a rig a fluent guess is a hazard, not a bad answer. |
| **The prefix walk fires on any word** | It resolved on the first letter, so `FROBNICATOR` classified as a **filter**. `classify()` now requires tag-shaped text first. |
| **The rig builder's LLM must not draw** | The model emits a spec (items + streams) only. Tags, relief valves, instrument loops, line numbers and geometry are generated deterministically in `src/edi/rig/spec.py`. Letting the model produce tags or coordinates puts duplicate tags and invalid geometry on the sheet, and the drawing stops matching the graph. |
| **A long chained background job dies silently** | Running ingest→relations→graph→benchmark as one `nohup bash -c` chain stopped after the first sheet with exit 0 and no error. Run the steps separately. |

---

## Commands

```bash
source .venv/bin/activate

python -m pytest tests/ -q            # 236 tests
python scripts/healthcheck.py         # 19 subsystems
python scripts/dataset_report.py      # datasets present & complete
python scripts/evaluate.py --per-type 32          # the benchmark

python -m streamlit run src/edi/app/streamlit_app.py --server.port 8520

python scripts/ingest_blueprint.py drawing.pdf --graph   # real drawing, no detector
python scripts/rig_build.py "gas compression with flare"  # generate a rig P&ID
python scripts/rig_build.py --list                        # packages + item catalogue
python scripts/eval_flow.py --rules                       # score flow direction
python scripts/eval_flow.py --no-layout                   # ...on stated evidence only
python scripts/ingest_blueprint.py --sheet <id> --graph-only   # reload the graph
                                                          # from interim JSON, no OCR
bash scripts/install_model.sh ~/Downloads/best.pt        # when training comes back
```

**Never run `git add -A` here.** The git root is his HOME directory, not this
project — it tries to stage all of `~`. Do not commit unless asked.

---

## State (2026-08-28)

**End-to-end benchmark 0.8038** (2,176 questions, 0 query errors) ·
**detection F1 0.992** · 236 tests · 19/19 subsystems · graph 2,084 nodes /
2,194 links / 503 control loops / 417 flow edges / 6 cross-sheet · 20 sheets
(17 benchmark + 2 real offshore + generated rig sheets).

**The document store is empty** — the reference PDF was removed, so the
document evidence leg cannot be demonstrated until one is re-indexed from the
sidebar. Everything else runs.

| Question type | end-to-end | perfect-detection ceiling |
|---|---|---|
| simple counting | 0.9908 | 1.0000 |
| spatial connection | 0.8879 | 0.9044 |
| spatial counting | 0.7923 | 0.7941 |
| value / tags | **0.5441** | 0.7463 |
| **overall** | **0.8038** | 0.8612 |

**Value/tags is the weak point and it is where the work is.** It fell hardest
when the real detector went in: the detector's boxes differ slightly from
ground-truth boxes, and OCR crops its search region from those boxes, so tag
reads degrade. Tuning the OCR ROI padding against real detections is the single
highest-value fix left.

Working: trained detector · blueprint mode (detector-free, for a customer's
first drawing) · conversational chat · pipe line sizes · off-page connectors +
cross-sheet links · canvas graph editing · tri-modal evidence · cross-drawing
search · export · review queue · **rig builder** (describe a rig in English →
local LLM → a real 7000px P&ID + its graph + its registers; see
docs/RIG_BUILDER.md) · **field mode** (plain-English item cards, ISA-5.1 tag
decoding, plant glossary, isolation planning — the technician's screen; see
docs/REALITY_CHECK.md).

Not built: dimensions beyond line size · mechanical and structural drawings ·
region classification (stream tables and title blocks are still read as text) ·
`contains` relation (deliberately not faked — nothing in the corpus expresses
containment).

## Next, in order

Reordered after the reality check (docs/REALITY_CHECK.md): the benchmark
measures the questions the system can answer, not the questions people ask.

1. **Process data into the graph** — design P/T, service, fail position, duty.
   The rig builder already has them for a generated sheet and drops them at the
   drawing; for a customer's sheet they come from the line list, which is a
   document import.
2. **`CONTINUES_ON` across a drawing set** — `build_graph.py` makes them, but
   only 3 links exist because only one destination is shared between the two
   real sheets. Direction does not yet propagate across them either.
3. **OCR ROI padding tuned against real detector boxes** — the value/tags
   benchmark drop.
4. Retrain on a customer's symbol legend once drawings are obtained.

Flow direction was #1 and is done — docs/FLOW_DIRECTION.md.

## Where to look

| File | For |
|---|---|
| `PROJECT.md` | full context, status, how to resume cold |
| `ARCHITECTURE.md` | code map, data flow, why each decision was made |
| `docs/RESULTS.md` | every measurement, and what was tried and rejected |
| `docs/BLUEPRINT_MODE.md` | how real drawings work without a detector |
| `docs/DEMO.md` | the 6-minute buyer walkthrough |
| `docs/RIG_BUILDER.md` | generating drawings by conversation — catalogue, auto-complete, HAZOP |
| `docs/REALITY_CHECK.md` | **read before adding features** — what real users ask, what failed, what was fixed, what is still wrong |
| `docs/FLOW_DIRECTION.md` | how direction is read off a drawing with no model, and how it is scored |
| `docs/DESIGN.md` | the visual system — tokens, rules, and what the reference actually does |
| `HANDOFF.md` | training: what the teammate runs, what he runs after |
