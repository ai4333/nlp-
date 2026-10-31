# Flow direction — reading which way the fluid goes, with no model

**The gap.** `FEEDS_TO` was empty on every real drawing. The trained path
derives direction from detected flow-arrow symbols; a customer's drawing has no
trained detector, so that route is closed. The consequence was that every
operator question failed:

| Question | Before |
|---|---|
| *Where does the flow from V-1101A go next?* | "cannot be determined" |
| *What is upstream of P-1101A?* | 0 rows |
| *What relieves to the flare?* | 0 rows |

**Now:** 96 directed edges on that same drawing, **0.77 of its connections**,
each one carrying the rule that decided it and a confidence that reflects how
much the drawing actually said.

---

## The idea

A P&ID states direction in several ways that need no model at all. They are what
an engineer reads, and they are all in the drawing already.

| Rule | What it reads | Confidence |
|---|---|---|
| **R1 · the connector says so** | `TO 3RD STAGE` is an outlet; `FROM LP CONDENSATE` is an inlet | 0.95 |
| **R2 · relief flows one way** | protected system → relief valve → flare | 0.90 |
| **R3 · the service code carries intent** | a flare or relief line runs toward the flare, a drain line toward the drain header, a utility line away from its header | 0.80 |
| **R4 · a machine passes flow through** | if something enters a pump, something leaves it | 0.70 |
| **R6 · a check valve is on the discharge** | a non-return valve exists to stop reverse flow through a machine, so it is always on the outlet | 0.70 |
| **R5 · conservation** | at a point where every connection but one is decided, the last one is forced | 0.60, decaying with distance |
| **R7 · a P&ID reads left to right** | last resort, only where nothing else decided and the run is predominantly horizontal | **0.50** |

R1–R3 and R6 are seeds. R4–R5 propagate them along the chain, which is how four
labelled connectors at the sheet edge end up orienting a whole train.

Every edge records the rule that produced it, so a wrong direction can be traced
to the rule that made it rather than argued about. An edge with no evidence stays
undirected, and an edge that would contradict one already placed is dropped.

---

## Measuring it — the generator is the ground truth

Direction on a real drawing has no labels. Nobody has annotated which way the
fluid goes on an offshore P&ID, and scoring it with a language model would only
measure the model.

The rig builder solves that. It generates a sheet **from a spec**, so the true
direction of every connection is known by construction. `scripts/eval_flow.py`
feeds the generated sheet's entities and *undirected* connections into the
inference — exactly what it gets on a customer's drawing, with the answers
withheld — and scores what comes back.

```bash
python scripts/eval_flow.py --rules          # all 11 packages
python scripts/eval_flow.py --no-layout      # stated evidence only
python scripts/eval_flow.py --package flare_relief --detail
```

| | coverage | precision |
|---|---|---|
| stated evidence only (`--no-layout`) | 0.43 | **0.884** |
| with the left-to-right convention | 0.91 | **0.934** |

> **Read the second row carefully.** Generated sheets are laid out left to right
> *by construction*, so R7 cannot fail on them. That number is an upper bound,
> not a claim. It is reported because R7 is genuinely how drawings are read, and
> because its confidence is 0.50 everywhere it fires — the answer says out loud
> that it is a convention and not something the drawing stated.
>
> On the real offshore P&ID the split is 0.57 coverage from stated evidence and
> 0.77 with the convention. The stated-evidence edges there are verifiable by
> eye: `C-1101A → TO LP COMPRESSOR`, `V-0103 → TO 3RD STAGE`,
> `FROM LP CONDENSATE → V-0103`.

---

## Three real bugs the harness found

Building the measurement first paid for itself immediately.

**1 · A flare KO drum is not a sink.** The first rules treated anything with
"flare" in the name as a terminal, so every relief and drain line touching the
KO drum was reversed. Direction is not "sink or not sink" — it is *how far along
the disposal system* each end sits: rank 2 for a stack, header or caisson, rank
1 for equipment on the way there, rank 0 for the plant being protected. Flow
runs from the lower rank to the higher. Precision 0.752 → 0.894.

**2 · The rank has to look through inline valves.** An ESD valve spliced into a
flare header is rank 0 on its own, which made the header appear to run backwards
into it. What matters is what the valve leads to.

**3 · The rig builder was generating a wrong drawing.** Two errors, both found
by scoring against its own spec and both fixed:

* a PSV inlet line was labelled with the *flare* service. In practice the inlet
  is part of the protected system and only the tailpipe is a relief line. The
  mislabelling made inlet and outlet indistinguishable.
* on the flare-and-relief package the flare KO drum was both the protected
  vessel and the relief destination, so a relief valve discharged into the
  vessel it was protecting. The destination now prefers a real terminal and
  never routes a relief valve back into its own host.

---

## What it does not do

* **It does not know about pressure.** Direction here is topological and
  conventional. It has no design pressures to reason from, so it cannot tell you
  a line is flowing backwards against a differential.
* **Recycle loops read poorly.** A glycol circulation or an anti-surge recycle
  is a cycle; the seeds orient the forward path and conservation does the rest,
  but a loop with no seed at all stays undirected.
* **R7 is a convention, not evidence.** Where it is the only thing that fired,
  the confidence is 0.50 and the field-mode card says so in the "what this does
  not tell you" section.
* **It is one sheet at a time.** Direction does not yet propagate across
  `CONTINUES_ON` links.

## Where it is

| File | Does |
|---|---|
| `src/edi/graph/flow.py` | the rules, the propagation, the rank model |
| `scripts/eval_flow.py` | scores inference against the generated ground truth |
| `src/edi/ingest/blueprint.py` | `_derive()` calls it, so every real ingest gets direction |
| `src/edi/qa/field.py` | the item card shows "flows in from / flows out to" with confidence |
| `tests/test_flow.py` | 16 tests: one per rule, plus a scored regression floor |
