# Reality check — what happens when real people ask real questions

This document is the honest audit. It was written by running the questions five
real users would ask against the system exactly as it stood, on a real offshore
P&ID, and writing down what came back. No question was chosen to make a point;
they are the ones each of those people asks on a normal day.

The headline is uncomfortable and worth stating plainly:

> **The benchmark measures the questions the system can answer, not the
> questions people ask.** 0.8038 on PIDQA is real. But of sixteen questions a
> technician, an operator and a design engineer would actually ask, **two were
> answered correctly, and two were answered confidently and wrongly.**

Everything below is measured, on sheet `u_pid_real_p1_52af52` (a dense offshore
P&ID ingested in blueprint mode), with the graph as it stood on 2026-08-28.

---

## 1 · Who actually uses this

| | Who they are | What they need | What breaks for them |
|---|---|---|---|
| **A** | **Field technician** — just joined, standing at the equipment, about to connect something | "What is this? What is it connected to? What is in the pipe? What do I close before I touch it?" | Everything. See §2. |
| **B** | **Control room operator** | "What trips this? What relieves where? What is upstream?" | Direction and cause-and-effect are missing. |
| **C** | **Process / design engineer** | design conditions, line sizes, revisions, the registers | Counts and filters work; there is no process data at all. |
| **D** | **Safety / permit officer** | isolation lists, what is live, what must not be blocked | Was not addressable at all. |
| **E** | **Rig manager / buyer** | "Does it work on *my* drawings, and can I trust it?" | Trust is the whole product, and confident wrong answers destroy it. |

---

## 2 · What actually happened

Sixteen questions, run through the real pipeline. Verbatim.

### Field technician

| Question | What came back | Verdict |
|---|---|---|
| *What is V-1101A?* | "a vessel/separator equipment item **with a size of 0.0**" | Leaks the database schema. Tells a newcomer nothing. |
| *What is connected to V-1101A?* | "a **drum tagged DC266** and a **heat exchanger EAN-11012**" | **Confidently wrong.** DC266 is a closed-drain *line number*. EAN-11012 is an OCR misread. |
| *What does SDV mean?* | ran `MATCH (s:Symbol) WHERE s.tag='SDV'` → "no information available" | The answer was sitting in `config/tag_conventions.yaml` the whole time. The system had a graph and no knowledge. |
| *Which valves do I close to isolate P-1101A?* | 0 rows | No procedural reasoning existed. |
| *What happens if I close SDV-11014?* | *"Closing SDV-11014 would stop the flow of fluid through the valve, isolating the system section it controls."* | **Invented.** The query returned one row containing the string `SDV-11014` and nothing else. Not one word of that answer was supported. |
| *What should I check before I open BDV-11350?* | *"…check if it is properly secured and ensure that all safety protocols are followed."* | **Invented**, and worse than useless — it sounds like guidance. |
| *Where does the flow from V-1101A go next?* | "cannot be determined" | `FEEDS_TO` is empty on every blueprint-mode sheet. |
| *What is DC266?* | "not present in the graph" | It *was* in the graph — as a drum. The generated query looked in `name` instead of `tag`. |

### Operator

| Question | What came back |
|---|---|
| *Which instruments shut down C-1101A?* | one adjacency, hedged. `CONTROLS` means "an instrument bubble is next to it", not "it trips it". |
| *What relieves to the flare?* | 0 rows |
| *What is upstream of P-1101A?* | 0 rows — no flow direction |

### Engineer

| Question | What came back | Verdict |
|---|---|---|
| *What is the design pressure of V-1101A?* | "no information… size_in is 0.0" | There is **no process data in the system at all** — no pressures, temperatures, flows or duties. |
| *Which lines are 6 inch or larger?* | correct | ✓ |
| *How many valves are on this drawing?* | 7 | ✓ (of the ones it found) |
| *Which drawings does this continue on?* | nothing — `CONTINUES_ON` is 0 on this sheet despite 13 off-page connectors | |

**Score: 2 of 16 useful. 2 fabricated.**

---

## 3 · Root causes

### 3.1 · The classifier could not tell a pipe from a vessel — the worst one

The tag-prefix classifier resolves the *first letter* of a tag. On a real sheet
OCR routinely returns a fragment of a line number: the drawing prints
`1"-DC-11-266-P1A-HE` along a pipe and the reader gets back `DC266`. That
fragment is tag-shaped, so:

| What the drawing says | What the system said |
|---|---|
| `DC266`, `DC267`, `DC268`, `DC269`, `DC021`, `DC022`, `DC023`, `DC270`, `DC271`, `DC-6970` — closed drain lines | **drums** |
| `PL101`–`PL105` — process liquid lines | **pumps** |
| `VH082`, `VH083`, `VL014`, `VL015`, `VL016`, `VA004`, `VA005102` — vent lines | **vessels / separators** |

Measured: **22 of the 37 items it called "equipment" on that sheet were pipe
labels.** A technician asking "what is DC266?" was told, with a drawing crop to
back it up, that it is a drum.

Second-order version of the same bug: the prefix walk fired on *any* word.
`explain("frobnicator")` returned **"a filter"** — because the walk fell back to
`F`.

Third: the walk never reached multi-letter meanings. `TA-11003` (temperature
alarm) became a **tank**, `TW-18025` (a thermowell) became a **tank**.

### 3.2 · There was no knowledge layer

The graph knows the plant; it does not know the trade. Every "what is / what
does X mean" question was translated into a Cypher lookup for a node that does
not exist, and answered "no information". Meanwhile ISA-5.1 — which decodes
*every instrument tag ever printed* from three letter tables — was not
represented anywhere in the codebase.

### 3.3 · The system filled silence with fluent text

`phrase()` sends whatever rows came back to the language model with the
instruction "if rows are present, the answer is NOT empty". A query that returns
one row containing only the tag you asked about satisfies that, and the model
writes a paragraph. That is how "what happens if I close SDV-11014" produced
safety-shaped guidance from nothing.

### 3.4 · Flow direction does not exist on real drawings

`semantics.enrich()` derives `FEEDS_TO` from flow-arrow symbols, which only the
trained detector finds. Blueprint mode returns `FEEDS_TO: []`. So "where does
this go", "what is upstream", "what relieves to the flare" — the majority of an
operator's questions — cannot be answered on a customer's drawing.

### 3.5 · There is no process data

The graph carries geometry, tags, categories and pipe sizes. It has no design
pressure, no temperature, no flow, no duty, no material, no fail position.
The rig builder generates those numbers but they stop at the drawing — they are
not in the graph.

### 3.6 · Extraction is not reproducible

Re-running blueprint ingest on the same sheet produced a different tag set
(`C-1101A` vs `C-110A`, `TA-11003` vs `TA-113`). OCR variance moves entity
counts run to run. Nothing downstream is aware of that.

---

## 4 · What has been fixed

### Line fragments are lines (§3.1)

`ingest/lines.parse_fragment` recognises a **multi-letter pipe service code
followed by digits** as a clipped line number. Single letters are excluded on
purpose — `P` is both "process line" and "pump", so `P-1101A` stays a pump.
There is no overlap between the multi-letter service codes and the
equipment/instrument prefixes, except the ones now declared as ambiguous in
`config/tag_conventions.yaml` (`ST` — steam, or speed transmitter).

Fragments become **line** nodes flagged for review, not equipment, and they keep
`size_in = 0` because the size was on the part of the string OCR lost.

Measured on the real sheet, re-ingested:

| | before | after |
|---|---|---|
| items called "equipment" | **37** | **15** |
| line nodes | 21 | 47 |
| phantom drums / pumps / vessels | 22 | **0** |

### The prefix walk only fires on tag-shaped text (§3.1)

`FROBNICATOR`, `FLARE`, `NOTE` now classify as unknown instead of as filters.

### ISA-5.1 decoding, and a plant glossary (§3.2)

`src/edi/knowledge/` is new:

* `isa.py` — the ISA-5.1 letter tables, plus the industry conventions that
  override them (`SDV`, `BDV`, `PSV`, `ZLO`, `NRV`, `RO`, `TW`, `HIPPS`, …).
  It decodes limit letters from the *end* of the code, which is the only thing
  that distinguishes `LSLL` = "level switch low-low" from "level light low".
* `glossary.py` — 64 plant terms written for somebody who has never seen a P&ID,
  each with *what it is* and *what it means for the person standing next to it*;
  plus a hazard note for every pipe service code (what is actually in the line).
* `__init__.explain()` — glossary → complete line number → industry convention →
  clipped line number → ISA → the customer's tag conventions. **If none of them
  knows, it returns nothing.** It never guesses.

Classification now uses ISA as a fallback, so `TA-11003` is a temperature alarm,
`TW-18025` a thermowell, `PDI-11003` a differential pressure indicator — on a
drawing nobody trained a model on.

### Field mode — the technician's screen (§3.2, §3.3)

`qa/field.py` routes a question *before* the text-to-Cypher path:

| the question | who answers it |
|---|---|
| "what does SDV mean" | the knowledge layer |
| "what is V-1101A", "where is PSV-11004", "is that line dangerous" | an **item card** built from the graph |
| "which valves isolate P-1101A", "how do I lock this out" | the **isolation planner** |
| "how many valves", "which lines are 6 inch or larger" | the Cypher path, unchanged |

The item card gives the tag, what it is in plain English, what it means on the
job, the drawing zone and a crop, the lines on it *with what they carry and the
hazard that implies*, what it is connected to, the instruments on it, its relief
devices — and a **"what this does not tell you"** section that is always
populated.

### Isolation planning (§3.3)

`graph/isolation.py` walks outward over the pipe connections and finds, on every
branch, the first valve that can actually hold. It:

* refuses to offer a **check valve, control valve or relief valve** as an
  isolation point — a check valve will not hold for a permit,
* lists **relief and blowdown paths separately** and says not to block them in,
* reports the **branches it could not close** rather than quietly dropping them,
* and states, every time, that it is what the drawing shows and not a permit.

On the real sheet it finds `SDV-11014` and `SDV-18165` around `V-1101A`, and
says explicitly that eleven branches leave the item without passing a block
valve on this sheet. That is the honest answer: the extraction is not yet good
enough for a complete isolation, and it says so instead of inventing three
valves.

### Flow direction, with its reasoning attached (§3.4)

`FEEDS_TO` is populated on every real drawing now, and the field-mode card
answers *"where does this go"* leg by leg with the rule and the confidence for
each one:

```
Flow through V-0103 — vessel / separator
  Comes in from:  FROM LP CONDENSATE   (0.95, connector says FROM)
  Goes out to:    TO 3RD STAGE         (0.95, connector says TO)
                  PL104                (0.50, a P&ID reads left to right)
  > The legs at 0.50 rest on a convention, not on anything the drawing states.
```

That last line is the point. Two of those legs are things the drawing says out
loud; one is an inference from how drawings are drawn. Presenting them as
equally certain would be the same mistake as inventing them.

### It stops filling silence (§3.3)

`answer.ask()` now checks whether the returned rows contain anything that was
not already in the question. If not, the language model is never called, and the
answer is: *"The drawing, as it has been read, does not contain an answer to
that… I am not going to guess."*

---

## 5 · What is still wrong

In priority order, with why.

1. ~~**No flow direction on real drawings**~~ — **fixed**, see
   [FLOW_DIRECTION.md](FLOW_DIRECTION.md). `graph/flow.py` reads direction out of
   the drawing's own labelling: off-page connectors that say TO or FROM, relief
   paths, service codes, check valves on machine discharges, then conservation
   along the chain. On the real offshore sheet: **0 → 96 directed edges, 0.77 of
   its connections**, each carrying the rule that decided it and a confidence.
   Scored against the rig builder's own specs — the only flow ground truth that
   exists here — precision is **0.884 on stated evidence alone** and 0.934 with
   the left-to-right layout convention included.
2. **No process data** (§3.5). Design pressure, temperature, service, fail
   position, duty. For a generated sheet these exist in the rig builder spec and
   simply are not carried into the graph. For a customer's sheet they come from
   the line list and equipment list, which is a document import, not a vision
   problem.
3. **`CONTINUES_ON` is not built on ingest.** 13 off-page connectors on the real
   sheet, 0 cross-sheet links. Most real questions cross sheets.
4. **`CONTROLS` overstates itself.** It means "an instrument bubble is adjacent",
   which is not the same as "it trips this". The cause & effect matrix, which the
   rig builder already generates for a designed sheet, does not exist for an
   ingested one.
5. **OCR non-determinism** (§3.6). The same sheet ingested twice gives different
   tags. Nothing reports the variance, and no answer carries a "this was read
   with low confidence" marker into the graph beyond a single flag.
6. **Second-wave OCR noise.** With the line fragments gone, what is left in the
   "equipment" list on the real sheet still includes misreads (`C-110A`,
   `CAY-11008`, `HHE-11441`). These are tag-shaped garbage; they belong in the
   review queue with their crops, and the review queue should be the first screen
   a new drawing opens on, not the fifth tab.

---

## 6 · The principle that came out of this

Everything above reduces to one rule, and it is now enforced in code rather than
hoped for:

> **On a rig, "I don't know" is a good answer and a fluent guess is a hazard.**

Three places enforce it: `knowledge.explain()` returns `None` rather than
guessing; `answer.ask()` refuses to phrase an answer the rows do not support;
and every item card and isolation plan carries an explicit list of what it does
not know.
