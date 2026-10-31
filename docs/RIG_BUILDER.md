# Rig builder — design a drawing by conversation

**What it is.** A tab in the app where you describe a piece of an oil rig in
plain English, a local model turns that into a spec, and the system draws a real
P&ID from it — then loads that drawing into the knowledge graph so you can ask
it questions immediately.

```
"three stage separation with gas compression and flare"
        │
        ▼  qwen3:8b (local, offline)
   spec: 12 items, 10 streams
        │
        ▼  deterministic
   tags · relief valves · ESD valves · instrument loops · line numbers
        │
        ├──▶ 7000 × 4600 px P&ID  (data/uploads/<sheet>.png)
        ├──▶ entities / relations / derived  (data/interim/<sheet>/)
        ├──▶ knowledge graph  ──▶ the Workbench can be asked about it
        └──▶ equipment list · line list · valve list · instrument index ·
             cause & effect matrix   (CSV)
```

---

## Why it exists

Two problems it solves at once.

**1 · There was no way to make a drawing.** The system could read a P&ID and
answer questions about it, but every sheet had to come from somewhere else — the
synthetic benchmark corpus or one of two real offshore drawings. You could not
show a rig manager what the system does *on the kind of sheet he cares about*
without already having his sheets.

**2 · Ground truth is free here.** Because the drawing and the graph are
generated from the same spec, they cannot disagree. "How many instruments are on
this sheet" has an exact answer by construction — no OCR error, no detector
miss. That makes a generated sheet the cleanest possible demo, and a useful test
fixture for the rest of the pipeline.

---

## The split that makes it reliable

**The model does not draw.** It emits a spec — a list of items and streams — and
nothing else. Everything downstream is deterministic code:

| The model decides | The code decides |
|---|---|
| which items are on the sheet | their tags (`V-1101`, `K-1201`, `PSV-1101`) |
| what flows into what | line numbers (`12"-PG-12-104-P1B-HE`) |
| the sheet title and notes | which vessels need a relief valve, and where it discharges |
| | which instrument loops each item carries |
| | the layout, the symbols, the title block |

An 8B model is good at the left column and bad at the right one. Splitting them
is why the output is a valid drawing rather than a plausible-looking one.

The vocabulary is closed: the prompt carries the 105 catalogue keys, and
anything the model invents is dropped by `catalog.find` instead of becoming a
mystery box on the sheet.

**It never dead-ends.** If Ollama is down, or the reply has fewer than two
usable items, the request is keyword-matched against the 11 standard packages in
`templates.py` and the sheet is built from those. The chat always produces a
drawing — which matters when someone is watching.

---

## What it can build

**105 item types**, covering a production platform topsides and below:

| System | Items |
|---|---|
| wells | wellhead · christmas tree · SSV · production choke · production and test manifolds · test separator |
| separation | HP / MP / LP separators · free water knockout · electrostatic coalescer · desalter · surge drum · crude heater · oil cooler |
| compression | suction and discharge scrubbers · centrifugal and reciprocating compressors · after and interstage coolers · anti-surge valve · gas turbine driver |
| dehydration | TEG contactor · TEG regeneration package · glycol pump · molecular sieve · gas filter · dew point unit |
| export | export and booster pumps · fiscal metering skid · pig launcher / receiver · export riser · storage tank |
| flare & relief | HP / LP flare KO drums · flare stack · vent stack · PSV · BDV · rupture disc · KO drum pump |
| produced water | hydrocyclone · degasser · compact flotation unit · injection pumps and manifold |
| water injection | seawater lift pump · fine filter · sulphate removal · vacuum deaerator |
| utilities | fuel gas scrubber and superheater · instrument air compressor / dryer / receiver · nitrogen package · diesel tank and pump · heat and cooling medium · chemical and methanol injection |
| safety | firewater pump · ring main · deluge skid · ESD valve · HIPPS |
| power | gas turbine generator · emergency diesel generator |
| drains | closed drain drum · open drain caisson · slop tank |
| drilling | BOP stack · diverter · mud pump · mud tank · shale shaker · desander · desilter · mud gas separator · choke manifold · cementing unit · riser |
| inline | gate / ball / globe / butterfly / check / control valves · restriction orifice · orifice plate · strainer |

**11 standard packages**, one click each: production separation train · gas
compression and dehydration · wellhead and manifold · flare and relief ·
produced water treatment · seawater lift and injection · utilities · firewater
and deluge · drilling mud circulation and well control · crude export and
pigging · offshore platform overview.

---

## What the auto-complete adds

Ask for five items and you get a sheet with sixty, because a bare item list is
not a P&ID. In order:

1. **Relief protection** — a PSV on every pressure envelope ≥ 3 barg, discharging
   to the flare KO drum or stack; if the sheet has neither, an off-page
   `HP FLARE HEADER` connector is created and the relief is routed to it.
2. **Blowdown** — a BDV on gas-holding vessels ≥ 20 barg.
3. **Check valves** — spliced into every pump and compressor discharge.
4. **Anti-surge recycle** — around every centrifugal compressor, fail-open.
5. **ESD valves** — at the sheet's inlet boundaries, fail-close.
6. **Instrument loops** — by equipment type (a horizontal vessel gets PT, LT, LG,
   TT, PSH; a compressor gets PT, TT, VT, ST; a pump gets PT, FT).
7. **Position switches** — ZLO / ZLC on every automatic valve.
8. **Fire & gas** — gas and flame detection wherever there is Zone 1 equipment.

Everything is tagged in ISA / EPC convention, with instruments carrying their
parent's loop number (`PT-1101` on `V-1101`) and pipe runs numbered
`12"-PG-12-104-P1B-HE` — a format the project's own `ingest/lines.py` parser
reads back, which is asserted in the tests.

---

## The design review

Every sheet is checked the way a process safety engineer would check it, and the
findings are reported rather than silently repaired:

| Code | Severity | What it catches |
|---|---|---|
| `PSV-MISSING` | high | a pressure envelope with no relief device |
| `RELIEF-ROUTE` / `NO-FLARE` | high | relief that does not reach a flare or vent |
| `SCRUBBER-MISSING` | high | a compressor with no suction scrubber (liquid carry-over) |
| `F&G` | high | Zone 1 equipment with no fire & gas detection |
| `TAG-DUP` | high | the same tag on two items |
| `LEVEL-MISSING` | medium | a separator with no level instrumentation |
| `SURGE` | medium | a centrifugal compressor with no anti-surge recycle |
| `NRV-MISSING` | medium | a pump with no discharge check valve |
| `ORPHAN` | medium | an item connected to nothing |
| `LINE-DATA` | low | a line with no size or service code |

The upstream walk in `SCRUBBER-MISSING` passes through inline valves, so a
scrubber feeding a compressor through an ESD valve is not falsely reported.

---

## The registers

The drawing is not the whole deliverable on a real project; the registers that
go with it are. All five are generated from the same spec and export to CSV:

* **equipment list** — tag, description, system, design P/T, hazardous area, duty
* **line list** — line number, from, to, size, service, spec, insulation, fluid
* **valve list** — tag, type, what it is on, fail position, size, function
* **instrument index** — tag, service, measured variable, loop, signal type
* **cause & effect matrix** — ESD initiators against shutdown actions

Plus the spec itself as JSON, and the sheet as PNG.

---

## Using it

**In the app** — the *Rig builder* tab.

```bash
python -m streamlit run src/edi/app/streamlit_app.py --server.port 8520
```

Type a request, or click a package. The model dropdown lists whatever Ollama has
pulled; `qwen3:8b` is the default because the rest of the system is tuned
against it, and a coder model is a good alternative since the task is emitting
JSON. Follow-ups edit the sheet you already have ("add a second compressor
train", "route the LP gas to the flare") — the current spec is passed back to
the model each turn. The *Spec (editable)* register is a table: change the items
or the streams by hand and rebuild.

**From the CLI**

```bash
python scripts/rig_build.py "three stage separation with gas compression and flare"
python scripts/rig_build.py --template drilling
python scripts/rig_build.py --list
python scripts/rig_build.py "gas compression" --model deepseek-coder-v2:16b
python scripts/rig_build.py --template utilities --no-graph   # while an ingest holds the lock
```

---

## Notes and limits

* **Kùzu is single-writer.** If a second app window or an ingest holds a reader,
  the graph write fails. That is handled: the drawing and its data are written
  anyway, the tab says so, and a button retries the graph load when the lock
  clears. `--no-graph` does the same on the CLI.
* **A generated sheet is not a scanned sheet.** It is clean, orthogonal and
  perfectly legible — real drawings are none of those. It demonstrates the
  system and it makes a good fixture; it does not replace testing on real
  drawings, which is what blueprint mode is for.
* **The model's process knowledge is ordinary.** qwen3:8b will occasionally
  order a separation train LP → HP. The prompt corrects for the common cases and
  the design review catches the structural mistakes, but the spec is worth a
  glance before you show the sheet to an engineer.
* **One sheet at a time.** Multi-sheet sets with `CONTINUES_ON` links between
  generated drawings are not built yet; the off-page connectors are drawn and
  are in the graph, but nothing continues on the other side.

## Where the code is

| File | Does |
|---|---|
| `src/edi/rig/catalog.py` | the 105 item types — tag prefix, category, symbol, design data |
| `src/edi/rig/templates.py` | the 11 standard packages, and the keyword fallback |
| `src/edi/rig/spec.py` | normalisation, tagging, auto-complete, HAZOP, the registers |
| `src/edi/rig/layout.py` | layered graph layout in sheet pixels |
| `src/edi/rig/render.py` | ISA symbols, pipe routing, flare header, title block, legend |
| `src/edi/rig/build.py` | writes the ingest contracts and loads the graph |
| `src/edi/rig/chat.py` | the local-LLM conversation, repair loop and fallback |
| `src/edi/app/rigtab.py` | the app screen |
| `scripts/rig_build.py` | the CLI |
| `tests/test_rig.py` | 15 tests |
