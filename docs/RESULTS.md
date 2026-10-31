# RESULTS — measured, not projected

> **Headline (2026-08-28): end-to-end 0.8038 with the trained detector.**
> Detection full-sheet **F1 0.992**. Everything below that reports 0.8612 is the
> perfect-detection ablation. See §0.

## 0 · End to end, with the real detector

A teammate trained YOLOv8s on Colab — 12 epochs, 1.9 hours, 9.6 min/epoch.

**Detection, 20 unseen sheets, 2,311 symbols:**

| | |
|---|---|
| precision | 0.9909 |
| recall | 0.9935 |
| **F1** | **0.9922** |
| false positives / misses | 21 / 15 |

**Twelve epochs was enough — the curve was flat from the first one:**

| epoch | mAP50 | mAP50-95 | recall |
|---|---|---|---|
| 1 | 0.8733 | 0.8179 | 0.966 |
| 6 | 0.8702 | 0.8489 | 0.985 |
| 12 | 0.8730 | 0.8628 | 0.987 |

mAP50 moved 0.8733 → 0.8730 across the entire run; the last three epochs gained
+0.0032. The remaining 48 epochs would have cost ~7.7 h for nothing.

**Thresholds had to be tuned, and this mattered more than any training:**

| conf | NMS IoU | full-sheet F1 |
|---|---|---|
| 0.25 | 0.55 *(ultralytics defaults)* | 0.820 |
| 0.70 | 0.40 | 0.916 |
| **0.88** | **0.20** | **0.993** |

The NMS value dominates. 512px tiles at stride 384 overlap heavily, and a
lenient IoU leaves the duplicates in — the same symbol counted two or three
times. Straight out of the box this model looked under-trained at F1 0.83; it
was not.

**Why per-tile mAP50 (0.873) is lower than full-sheet F1 (0.992):** mAP is
computed per tile across all confidence levels, so low-confidence noise drags it
down. F1 here is per whole sheet, after thresholding and merging. Both are
honest; they measure different things, and the full-sheet figure is the one that
describes the system, because that is what feeds the graph.

**The end-to-end benchmark:**

| Question type | end-to-end | perfect-detection ceiling |
|---|---|---|
| simple counting | 0.9908 | 1.0000 |
| spatial connection | 0.8879 | 0.9044 |
| spatial counting | 0.7923 | 0.7941 |
| value / tags | **0.5441** | 0.7463 |
| **overall** | **0.8038** | 0.8612 |

**Value/tags fell hardest.** OCR crops its search region from the detector's
boxes; those differ slightly from ground-truth boxes, so tag reads degrade.
Tuning the OCR ROI padding against real detections is the largest remaining win.

**One environment trap:** the checkpoint was trained with ultralytics 8.4.128.
Loading it under 8.3.155 **segfaulted (exit 139) with no traceback** — the
model object would not deserialize. The venv is pinned to 8.4.128.


All numbers below were produced on the target machine (M3 Pro, 18 GB, no CUDA),
fully offline, with free models only. Nothing here is an estimate unless labelled.

---

## 1 · Headline

**PIDQA benchmark, 2,176 questions across 17 sheets, 0 query errors.**

| Question type | Accuracy | n | first run | after OCR retry | now |
|---|---|---|---|---|---|
| Simple counting | **1.0000** | 544 | 1.0000 | 1.0000 | 1.0000 |
| Spatial connection | **0.9044** | 544 | 0.9021 | 0.9099 | 0.9044 |
| Spatial counting | **0.7941** | 544 | 0.7812 | 0.7849 | 0.7941 |
| Value-based (tags) | **0.7463** | 544 | 0.5729 | 0.6710 | 0.7463 |
| **OVERALL** | **0.8612** | 2,176 | **0.8141** | **0.8415** | **0.8612** |

Graph: **1,948 symbols · 2,043 `connected_to` · 439 `controls` · 276 `feeds_to`**
across 17 sheets.

Value-based nearly +18 points across three rounds, each driven by looking at the
actual failures rather than tuning blind (§7b).

**Underlying tag recall is much higher than the value-based score suggests:
0.938 on sheet 0, 0.933 on sheet 2.** Value questions demand the *exact set* of
tags for a class, so a single missed tag out of ten fails the whole question. Of
the 138 remaining failures, **45 are missing exactly one tag**.

This is **L1 — graph quality**: PIDQA's *own reference Cypher* executed against
*our* graph. It isolates the vision pipeline from the language model. A wrong
answer here is a detection / OCR / relation-extraction failure, never an LLM
failure.

Detection ran in ground-truth mode (`--detector gt`), so this is the ceiling the
rest of the pipeline can reach — the honest reading is *"given perfect symbol
detection, our OCR and relation extraction answer 81% of PIDQA correctly."*
The trained YOLO detector replaces that input without changing anything else.

---

## 2 · What each fix was worth

Measured on sheet 0 (128 questions), each change isolated:

| Change | Overall | Δ |
|---|---|---|
| First working pipeline | 0.8125 | — |
| Ontology fix: class 23 is *insulation marking*, not an untagged spool | | |
| OCR multi-scale retry ladder for internal tags | | |
| Relations: net **clique → minimum spanning tree** | **0.8672** | **+5.5 pts** |

Per-type on sheet 0, before → after:

| Type | Before | After |
|---|---|---|
| simple_counting | 1.000 | 1.000 |
| spatial_connection | 1.000 | 1.000 |
| spatial_counting | 0.688 | **0.750** |
| value_based | 0.562 | **0.719** |

---

## 3 · Bugs found and fixed (each would have been silent)

**Edges inserted in both directions.** Kùzu's `-[:R]-` is already undirected, so
storing both directions made every undirected match fire twice — silently
doubling every count-based answer. Sheet 0 reported 308 edges instead of 154.

**A pipe run treated as a clique.** Emitting all pairs within a net over-counted
on **10 of 10** failing spatial-count questions, with **zero** under-counts.
Reducing each net to its MST cut sheet 0 from 154 edges to 97 and lifted spatial
counting 0.688 → 0.750 *without* costing anything on connection questions (still
1.000).

**`fresh=True` silently no-op.** Kùzu can persist as a file rather than a
directory, so `shutil.rmtree` did nothing and stale rows survived a "fresh"
rebuild — surfacing as a duplicate-primary-key crash rather than bad data, which
was lucky.

**Unquoted Cypher literals in the benchmark itself.** PIDQA's value-based queries
ship as `s.tag STARTS WITH ZLO` — not valid Cypher in any engine. Repaired in
`repair_cypher()`.

**The LLM splitting tags.** Ungrounded, the model emitted `tag = '101'` for
"SDL 101". Feeding real tags from the graph into the prompt fixed it.

**The LLM ignoring its own results.** Given bare nested lists it replied "no
connections found" while holding two rows. Rendering a labelled table fixed it.

**Reasoning paths revisiting nodes.** `A → B → C → B`. Fixed with `SHORTEST` plus
de-duplication.

**Kùzu's exclusive lock.** A running ingest blocked both the app and the
evaluator — *even read-only*. Fixed architecturally: ingest writes JSON only,
`build_graph.py` holds the lock for seconds.

---

## 4 · Empirical confirmation of the class off-by-one

Not assumed — proven. PIDQA asks for *"class 31 symbols whose tags begin with
ZLO"* with ground truth `['ZLO 433']`. Cropping YOLO class **30** from sheet 0
shows a bubble reading exactly **`ZLO 433`**. Therefore `pidqa = yolo + 1`,
pinned in `tests/test_class_ids.py`.

---

## 5 · Performance on this machine

| Stage | Time | Notes |
|---|---|---|
| Tiling 500 sheets → 20,283 tiles | **30 s** | 8 processes |
| Detection (ground truth) | ~0.1 s/sheet | |
| Detection (YOLO, est.) | ~2–4 s/sheet | 54 tiles, batched, MPS |
| OCR | **55–98 s/sheet** | CPU-only Paddle; the bottleneck |
| Relation extraction | **0.2–0.5 s/sheet** | |
| Graph load, 15 sheets | ~4 s | |
| NL→Cypher | **3.4–11.3 s** | qwen3:8b, warm |
| Full answer + evidence | ~9 s | |

**OCR optimisation:** disabling document-orientation classification and
unwarping — both meaningless on a small upright crop — took a crop from
**2,459 ms to 398 ms (6.2×)**. Per-crop OCR also beat per-tile OCR
(36 s vs 123 s per sheet) *and* preserves which symbol each tag belongs to.

**Few-shot selection:** TF-IDF builds in **0.2 s** versus **93 s** to load
bge-m3 under memory contention, and matches PIDQA's templated questions better.
bge-m3 stays behind the same interface for document retrieval.

---

## 6 · The memory constraint, confirmed the hard way

`INPUT.md` §A.3 predicted that vision and language models cannot be co-resident
in 18 GB. Confirmed in practice: with YOLO training holding ~7.4 GB, an Ollama
call **timed out at 120 s**. Suspending training (`kill -STOP`) and retrying gave
a **5.4 s** response. The two-phase staging is a hard requirement, not an
optimisation.

---

## 7 · Where the remaining error is

Of 205 value-based failures across 15 sheets:

| Cause | Count |
|---|---|
| Missing some tags | 115 |
| Mixed / misread | 48 |
| Nothing extracted | 42 |

**77% is recall, not misreading** — the tag exists on the drawing but OCR
returned nothing or a clipped string (`IJ 43075` → `IJ 4307`, `ST 47622` →
`ST 476`). Addressed with a completeness-scored retry ladder for external tags.

Spatial counting's residual error is the remaining over-connection at genuine
branch points, where the MST still links symbols a strict reading would not.

---

---

## 7b · Accuracy work — what actually moved the number

Every change below was chosen by measurement, not intuition. The diagnostic that
made it possible is `scripts/diag_ocr.py`, which renders the failures as a
contact sheet: looking at 19 failing crops answered in one minute what days of
parameter guessing would not have.

### Vertical tags were the whole OCR problem

Of 19 inspected failures, **15 were text printed vertically along a pipe run**.
The recogniser was not weak — it was reading them partially and dropping the
letter prefix, which is precisely the identifying part:

| ground truth | what OCR returned |
|---|---|
| `OP-44323` | `-44323` |
| `EF-81061` | `-81061` |
| `GH-20054` | `-20054` |
| `DV-76442` | `-76442` |
| `CV-91132` | `71116` |

Rotating the crop so the text is horizontal fixes it outright. Measured on sheet 0
(48 ground-truth tags):

| orientations | time | tag recall |
|---|---|---|
| 0° only (before) | ~60 s | ~0.70 |
| 0° / +90° / −90° | 187.6 s | **0.938** |
| **0° / −90°** | **127.1 s** | **0.938** |
| 0° / +90° | 126.9 s | 0.938 |
| 0° / ±90° at 2× upscale | 110.5 s | 0.896 |

One rotation is enough — PaddleOCR's textline-orientation classifier already
resolves the 180° flip, so the second rotation costs 60 s and returns nothing.

An adaptive variant (upright first, rotate only weak reads) was 17% faster but
dropped sheet 2 from 0.933 to 0.867: a *wrong* horizontal read scores as
complete, so the symbol is never re-examined. Rejected.

### Nearest-text was stealing neighbours' tags

Symbol `IJ-07591` was labelled `ERV-7-11`; `CS-93` was labelled `627` from an
adjacent instrument bubble. Choosing per symbol independently lets two symbols
claim the same text. Replaced with a **global assignment**: one tag per symbol,
one symbol per tag, cost = centre distance discounted when the text sits on the
symbol's own row or column, because engineering tags are set square to the item
they label.

**Tag recall vs ground truth: 0.938 (sheet 0), 0.933 (sheet 2)** — up from ~0.70.

### Relation geometry, swept against the benchmark

`scripts/sweep_relations.py` scores candidate settings on the two question types
that depend on topology. Four edge-construction strategies were compared:

| mode | edges | spatial counting | spatial connection |
|---|---|---|---|
| **mst** (nearest-neighbour tree per net) | 557 | **0.787** | **0.900** |
| walk (blocking BFS from each symbol) | 707 | 0.706 | 0.906 |
| segment (must share one pipe run) | 284 | 0.744 | 0.706 |
| clique (all pairs in a net) | — | 0.688 | 1.000 |

The blocking walk was expected to win and did not — it over-connects more than
the tree, not less. Geometry mattered more than strategy:

| erase pad | snap | spatial counting | spatial connection |
|---|---|---|---|
| 3 | 7 | 0.762 | 0.900 |
| 14 | 18 | 0.775 | 0.900 |
| **20** | **24** | **0.787** | **0.900** |
| 26 | 30 | 0.787 | 0.881 |
| 50 | 54 | 0.750 | 0.869 |

Erasing symbols more aggressively before tracing keeps separate pipe runs from
fusing. Past ~20 px it starts cutting real connections instead.

> ⚠️ **`snap` must exceed `erase_pad`.** Erasing cuts pipes N px away from each
> symbol, so a smaller snap attaches nothing and the graph comes out completely
> empty. An early sweep produced a wall of zeros before this was spotted.

Finally, capping MST edge length (long links usually mean two runs merged):

| max edge px | spatial counting | spatial connection | overall |
|---|---|---|---|
| off | 0.787 | 0.900 | 0.8641 |
| **1600** | **0.794** | **0.900** | **0.8656** |
| 1100 | 0.806 | 0.863 | 0.8594 |
| 800 | 0.825 | 0.838 | 0.8578 |

Tighter caps buy counting accuracy by giving away connection accuracy; 1600 is
the point where both hold.

### Tried and rejected

Recorded because negative results are results, and each of these looked
obviously right beforehand:

| idea | expectation | measured |
|---|---|---|
| Blocking netlist walk instead of a spanning tree | fewer spurious edges | **worse** — 707 edges vs 557, spatial counting 0.706 vs 0.787 |
| Connect only symbols sharing one pipe run | strictest, most precise | **worse** — connection accuracy collapsed to 0.706 |
| Swapping the answer model for llama3.1:8b, mistral or qwen2.5:14b | a non-reasoning model would be faster | **worse on both axes** — llama3.1 answered "closing SDV-11014 does nothing, it is already closed by design" (invented and dangerous); mistral and qwen2.5 both added consequences the briefing never stated; and all three were SLOWER to first token than qwen3:8b (31.2s / 28.0s / 53.4s vs 12.5s). qwen3:8b stays. |
| Split solid (process) from dashed (signal) lines | signal wires create false process edges | **no effect** — after a long directional opening the dashes are already joined; 28 vs 552 edges, overall 0.8641 vs 0.8656 |
| Adaptive OCR: upright first, rotate only weak reads | 17% faster, same accuracy | 17% faster, but sheet 2 fell 0.933 → 0.867 |
| More OpenMP threads for PaddleOCR | use the other 10 cores | **no effect** — ~1000 ms/crop at 1, 6 and 10 threads |

### Parallelism: threads do nothing, processes give 2×

PaddleOCR here is CPU-only and effectively single-threaded — measured **~1000
ms/crop at OMP_NUM_THREADS of 1, 6 and 10 alike**. Raising the thread count is
not a lever. Separate processes are:

| workers | 40 crops | speedup | output identical |
|---|---|---|---|
| serial | 44.0 s | 1.00× | — |
| 3 | 22.9 s | **1.93×** | yes |
| 5 | 22.1 s | 1.99× | yes |

Three workers captures nearly all of it; each holds ~1 GB, so more is not free
on an 18 GB machine.

---

## 8 · L3 — end to end, our Cypher on our graph

Sampled, 48 questions on sheets 0 and 2 (one LLM call per question):

| Question type | L3 (our Cypher) | L1 (reference Cypher, same 2 sheets) |
|---|---|---|
| Simple counting | **1.0000** | 1.0000 |
| Spatial connection | **1.0000** | 0.9688 |
| Spatial counting | 0.7500 | 0.7031 |
| Value-based | 0.6667 | 0.7031 |
| **OVERALL** | **0.8542** | 0.8438 |

**Executable query rate: 1.0000** — every query `qwen3:8b` generated parsed and
ran. No repair attempts were needed.

**The finding that matters: L3 ≈ L1.** The local 8B model contributes
essentially none of the error. Whatever the graph can answer, the model asks for
correctly. All remaining error lives in the vision pipeline — OCR recall and
relation over-connection — not in the language layer. (L3 edging above L1 is
sampling noise at n=48, not the LLM improving on ground-truth queries.)

This is the argument for the "free and local" claim: a model you can run offline
on a laptop is not the bottleneck for this task.

---

## 9 · Also built

**VLM fallback** (`ingest.py --vlm`, `src/edi/vision/vlm.py`) re-reads tags that
PaddleOCR could not resolve, using `qwen3-vl:8b` on a single symbol crop. Scoped
deliberately: never a detector, never a whole page — a sheet downsamples to
~1000px internally, where a symbol is 6px wide. Opt-in because it costs ~2-6s
per crop and 7 GB resident, which cannot coexist with the query phase.

---

## 10 · Honest limitations

- **Detection is not yet trained.** YOLO is mid-training (~25 min/epoch on MPS).
  All results above use ground-truth detection. Real end-to-end numbers will be
  lower and require re-running `--detector yolo`.
- **Synthetic data only.** Dataset-P&ID is generated. Generalisation to real
  scanned drawings is unproven; PID2Graph is queued for that.
- **11 of 32 class names are low-confidence.** Named by eye from ISA-5.1
  convention (`scratch/atlas/*.png`), not an authoritative legend. Naming does
  not affect benchmark correctness — PIDQA references classes by number — only
  answer readability.
- **Mechanical and structural drawings are unimplemented.** Only P&IDs have
  training data and a benchmark.
- **OCR retry cost can spike.** One 162-symbol sheet took **1,004 s** versus
  ~150 s typical, because retry crops grew to ~1250 px. Now capped at 700 px per
  side (`_RETRY_MAX_SIDE`), but that cap landed after this benchmark run — the
  measured accuracy is unaffected, the timing figures are not.
- **`_nets()` is O(n²)** over segments per sheet. Fine at current sizes
  (0.2–0.5 s) but would need spatial bucketing for much denser drawings.
