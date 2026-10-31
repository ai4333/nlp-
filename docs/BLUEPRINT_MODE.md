# Blueprint mode — any real drawing, no training

## The problem it solves

The symbol detector only knows the 32 shapes it was trained on. A customer's
drawing is unreadable to it until a model is trained on their legend — and they
will not hand over drawings until they have seen it work on drawings like theirs.

Blueprint mode breaks that deadlock.

## The idea

**On a real P&ID the tag is the identity.** An engineer says "V-1103A", never
"that bowtie shape". And tag prefixes follow long-standing industry convention:
`V-` is a vessel, `C-` a compressor, `PSV-` a pressure safety valve. So the
equipment type can be read from the tag itself.

```
OCR the whole sheet
   → keep text matching an engineering tag pattern
   → read equipment type from the prefix        (config/tag_conventions.yaml)
   → pair stacked instrument bubbles  "PC" + "1103"  →  "PC-1103"
   → drop truncations   ("V-120" when "V-1201" exists)
   → merge duplicates   (one tag = one physical item)
   → trace pipe runs geometrically              (no model)
   → connect tags that share a pipe network
```

No detector. No training. Works on the first drawing you are handed.

It writes the **same** `ocr.json` / `relations.json` / `derived.json` contracts as
the trained path, so the knowledge graph, question answering, evidence, export,
editing and the app all work on the result unchanged.

## Pipe line numbers

Line size is the most common question asked of a P&ID, and the answer is printed
on the drawing:

```
12"-DC-11-185-P1A-HE
 │   │   │   │   │  └ insulation / tracing
 │   │   │   │   └── piping material spec
 │   │   │   └────── line sequence number
 │   │   └────────── area or unit
 │   └────────────── service code
 └────────────────── nominal size, inches
```

Lines become their own graph nodes with a numeric `size_in`, so *"which lines are
4 inch or larger"* is an arithmetic query, not a string search. Service codes live
in `config/tag_conventions.yaml` — 26 of them, editable per operator.

**OCR guard:** piping specs are letter-digit-letter (`P1A`, `A1C`) and OCR
routinely confuses the digit for a letter. The same drawing produced both
`3"-VH-16-060-P1A` and `3"-VH-16-060-PIA`; de-duplication is on the pipe's
identity (size · service · area · number), so that is one pipe, not two.

## Measured on a real offshore drawing

POL Oil Limited · Alpha Field Development · *Gas Treatment and Compression*
(OPITO offshore training material). 4964 × 3510 scanned raster, never seen
before, entirely different symbol legend.

| | |
|---|---|
| Text lines read | 1,892 |
| **Equipment identified** | **15, all correct** |
| Connections traced | 10 |
| Time | 272 s |

```
C-1102A/B  compressor            H-1106     heater / exchanger
C-1103A/B  compressor            JY-1102    metering package
C-1105A/B  compressor            MQ-1101    pig launcher / receiver
H-1103A/B/C/D heater/exchanger   MY-1301    process package
H-1104     heater / exchanger    V-1103A/B  vessel / separator
H-1105A/B  heater / exchanger    V-1104     vessel / separator
V-1105A/B  vessel / separator    V-1201     vessel / separator
V-1301     vessel / separator
```

Connections it traced, each checkable against the printed drawing:

```
MQ-1101   —— C-1105A/B      pig launcher off the export compressor
C-1105A/B —— MY-1301        export compressor to gas metering
V-1104    —— H-1104         dehydration KO drum to exchanger
H-1106    —— V-1301         glycol regeneration
H-1105A/B —— JY-1102        discharge cooler to metering package
```

Questions answered on it, all verified:

| Question | Answer |
|---|---|
| How many vessels are there? | **5** ✓ |
| How many compressors? | **3** ✓ |
| What is connected to C-1105A/B? | **MQ-1101, MY-1301** ✓ |
| What does the pig launcher connect to? | **C-1105A/B** ✓ |
| List all the compressors | **C-1102A/B, C-1103A/B, C-1105A/B** ✓ |

## Use it

**In the app:** sidebar → *Add a drawing* → engine **Blueprint** → upload → *Run extraction*.

**Command line:**
```bash
python scripts/ingest_blueprint.py path/to/drawing.pdf --graph
```

## Tuning it for a customer

`config/tag_conventions.yaml` holds the tag patterns and the prefix → equipment
map. Most operators have a house tagging standard, and it is usually a one-page
document their engineers can hand over in five minutes. Editing that file is the
whole of "training" for blueprint mode.

## Measured on a dense real P&ID

ALPHA-AA-0000-PR-PD-0030-01 · *LP Compressor Suction Cooler and Drum, Train A* —
an instrument-heavy production P&ID.

| | |
|---|---|
| Text lines read | 1,818 |
| Instrument bubbles paired | 57 |
| **Items identified** | **81** — 37 equipment · 21 pipe lines · 10 instruments · 7 valves |
| Connections traced | 93 |
| Control loops derived | 14 |
| Time | 199 s |

Instruments it typed correctly from their tags alone: `TC-1101A` temperature
controller · `RO-11018` restriction orifice · `PI-18003` pressure indicator ·
`LG-11009` level gauge · `FT-11389` flow transmitter.

## Off-page connectors

No P&ID stands alone. A line leaves the sheet edge labelled `TO 3RD STAGE` or
joins `HP FLARE HEADER`, and the rest of the story is on another drawing. On a
rig most real questions cross sheets.

Three shapes, all read off the real drawings:

| Kind | Example |
|---|---|
| destination | `TO 3RD STAGE` · `FROM LP CONDENSATE` · `TO LP COMPRESSOR` |
| shared header | `HP FLARE HEADER` · `CLOSED DRAINS HEADER` · `ATMOS. VENT HEADER` |
| drawing reference | `FD-0002-01` · `FD-0007-01` |

**The hard part is not finding them — it is telling them apart from note prose,**
which uses the same words. `ALL DRAIN LINES TO SLOPE TO DRAIN HEADER.` is a note;
`FLARE HEADER` is a connector. Connectors are short, upper-case and carry no
terminal punctuation; notes are sentences. A junk filter also rejects OCR debris
like `TO WE ME M U`.

Sheets are then joined on **distinctive tokens**, not whole strings — one drawing
says `TO FLARE`, another `HP FLARE HEADER`. Measured across the two real sheets:

```
LP FLARE HEADER  <──CONTINUES_ON──>  TO FLARE     (via FLARE)
HP FLARE HEADER  <──CONTINUES_ON──>  TO FLARE     (via FLARE)
```

*"Where does this drawing continue?"* → **"on sheet …_60e11e, via LP and HP flare
headers"** — a plant connection neither drawing states on its own.

## What it cannot do

- **Untagged symbols are invisible.** A valve with no tag printed beside it does
  not exist to this path. That is precisely what the trained detector adds.
- **Instrument bubbles are under-read.** Bubbles carrying only a two-letter code
  and no loop number are skipped, because a bare "PC" is ambiguous. On the test
  drawing this cost most of the instrumentation.
- **Stream tables and title blocks are read as text.** The tag patterns filter
  most of it out, but region classification would do better.
- **Connections are geometric, not semantic.** Two tags on the same traced pipe
  network are connected; direction and what sits between them come from the
  detector path.
