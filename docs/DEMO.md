# Demo walkthrough — 6 minutes

For showing the system to someone evaluating whether to buy it. The order matters:
lead with the problem, prove the extraction, then show the answer, then show the
receipts.

---

## Before they arrive

```bash
source .venv/bin/activate
python -m streamlit run src/edi/app/streamlit_app.py --server.port 8520
```

Check, in order:

- [ ] `models/yolo_pid.pt` exists (it does — F1 0.992). Uploads work.
- [ ] `ollama ps` shows nothing huge resident; the first question takes ~15 s if
      `qwen3:8b` has to load. **Ask one throwaway question before they walk in**
      so the model is warm and the demo answers in ~5 s.
- [ ] YOLO training is **not** running. It eats 7 GB and will make everything crawl:
      `pgrep -f train_yolo.py` should be empty.
- [ ] Pick a sheet with a good tag density — 41, 31 or 35 are dense and look impressive.

---

## 1 · The problem (30 s — no screen)

> "This is a P&ID. Seven thousand pixels wide, about 120 symbols, every valve and
> instrument tagged. Today, if an engineer needs to know which valves isolate the
> line off a given pump, they zoom in and trace the pipe by eye, then cross-check
> the tags against a separate spec document. Every time. On a rig with hundreds of
> sheets that doesn't scale, and being wrong about which valve isolates which line
> is a safety problem."

## 2 · Show the drawing (45 s)

Open the **Workbench**. Scroll to zoom in on a dense area.

> "Every box is a symbol the system found and classified. Colour is the category —
> blue valves, orange instruments. The text above each box is the tag it read."

Click a symbol. The card shows type, class, both confidences, the pixel bbox, and
`connected_to`.

> "It didn't just find the symbol. It read the tag and worked out what it's
> physically connected to."

## 3 · Ask it something (90 s — the core)

Type a real question, e.g. **"What is connected to SDL 101?"**

Point at three things in the answer, in this order:

1. **The answer** — plain English, naming the actual tags.
2. **The evidence chips** — "these are the exact regions of the drawing the answer
   came from."
3. **The reasoning path** — "and this is the chain it followed through the graph."

Open **Query detail**.

> "That's the Cypher it wrote. It's not guessing from the picture — it translated
> the question into a graph query, ran it, and phrased the result. If you doubt an
> answer you can read the query."

## 4 · The graph (45 s)

Open **Knowledge graph**.

> "Same drawing as a graph. Nodes are positioned where they actually sit on the
> sheet, so the topology still reads like the drawing. This is the thing that makes
> multi-hop questions possible — 'what feeds the tank two steps upstream' is a
> traversal, not an image search."

Click a node to trace its neighbours. Click a legend chip to filter a category out.

## 5 · Honesty (60 s — this wins the room)

Open **Review queue**.

> "OCR on a drawing is never perfect. Rather than silently guessing, anything below
> the confidence threshold comes here with the crop, what it read, and the tag
> prefix it expected. The engineer corrects it, and the correction goes straight
> into the graph."

Then **Evaluation**.

> "And we measure it. This is a public benchmark — 64,000 questions over these
> drawings, with published ground-truth answers. These are our scores, including
> the ones we're bad at, and here are the actual failures."

**Do not skip the failure table.** A vendor showing their own failure cases is more
credible than one showing only wins.

## 6 · Getting data out (30 s)

Sidebar → **Download data bundle**.

> "Entities, relations, node-link JSON, and replayable Cypher. Nothing is locked in
> here — it drops straight into whatever asset system you already run."

## 7 · Their drawing (60 s — only if you're confident)

Upload one of *their* sheets.

> "Watch the pipeline: preprocess, detect, OCR, resolve, relations, graph."

⚠️ **Risk.** The detector is trained on synthetic P&IDs. On a real scanned drawing
with a different symbol legend, recall will drop and it may look bad. **Use
Blueprint mode for their drawing instead** — it reads tags and pipe routing with
no training at all, and it is what works on a sheet nobody has trained on. Only
use the Detector engine on a file you have tested beforehand. If unsure, say:

> "Training on your symbol library is the first step of any deployment — that's a
> few days of work, not a research project, because the pipeline around it already
> works."

---

## Questions you will be asked

**"Does this send our drawings anywhere?"**
No. Every model runs locally — detector, OCR, and the language model. The machine
can be fully offline. For a contractor handling client P&IDs under NDA, that is
usually the deciding factor.

**"What does it run on?"**
This laptop. 18 GB MacBook Pro, no GPU server, no cloud bill.

**"How accurate is it, really?"**
Open the Evaluation tab rather than answering from memory. The honest answer:
**detection finds 99.4% of symbols; end to end it answers 80% of a public
2,176-question benchmark correctly.** Then name the weak category yourself —
tag reading, at 54% — and show the failure table. Volunteering the weakness is
what makes the rest believable.

**"How long per drawing?"**
Detection is seconds. OCR is the bottleneck — roughly a minute to two and a half
per sheet on CPU, and it's a one-off cost per drawing, not per question. Fast mode
roughly halves it.

**"What happens with our symbol library?"**
The detector is retrained on your symbols; everything downstream is unchanged.
The ontology file that maps class numbers to names is 32 lines of YAML.

**"Can it do mechanical / structural drawings?"**
Honestly: not yet. P&ID is what's trained and benchmarked. The pipeline is not
P&ID-specific, but claiming the others today would be overselling.

---

## Do not say

- "It's AI, it understands the drawing." It runs a graph query. Say that — it's
  more impressive *and* it's true.
- Any accuracy figure from memory. Open the tab.
- That mechanical and structural work today.
