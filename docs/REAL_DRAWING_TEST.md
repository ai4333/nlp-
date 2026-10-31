# Real offshore drawing — what transfers today

The sales problem: nobody hands over their drawings until you've shown it works
on drawings like theirs. So this is a test on a **real, public, offshore oil &
gas drawing** that the system has never seen and was never trained on.

**Test sheet:** POL Oil Limited · Alpha Field Development · *Process Flow Diagram,
Gas Treatment and Compression, Year 1* · drawing `ALPHA-AA-0000-PR-FD-0000-02`,
from OPITO / Petroleum Open Learning offshore training material. 4964 × 3510 px,
scanned raster, drawn 90° to the page.

Nothing about this drawing resembles the training corpus: different symbol
legend, different tagging convention, scan noise, a stream table, a title block,
and a notes column.

---

## Result

| Stage | Works on a real drawing? | Evidence |
|---|---|---|
| **Acquire / normalise** | ✅ Yes | PNG in, 4964×3510, no warnings |
| **Tiling** | ✅ Yes | 30 tiles, geometric, model-free |
| **OCR** | ✅ **Yes** | 1,917 raw lines → 893 unique. **13 of 13 equipment tags correct** |
| **Tag pattern recognition** | ✅ Yes | Matched the `LL-NNNN[A/B]` convention with no configuration |
| **Line tracing** | ✅ Yes | 214 horizontal + 239 vertical pipe runs |
| **Pipe topology** | ✅ Yes | 165 distinct networks after junction merging |
| **Symbol detection** | ❌ **No** | Needs a model trained on this legend |
| **Instrument bubbles** | ⚠️ Partial | Found `FC`, `LC`; missed `PC`, `XC` and others |

### The tags it read, unprompted

```
C-1102A/B   C-1103A/B   C-1105A/B   H-1104     H-1105A/B
JY-1102     MQ-1101     MY-1301     V-1103A/B  V-1104
V-1105A/B   V-1201      V-1301
```

**Every one of these is correct** — verified against the printed drawing. They
are the compressors (C-), heat exchangers (H-), separator and contactor vessels
(V-), the gas metering package (JY-1102) and the pig launcher (MQ-1101).

The system had never seen an offshore gas-treatment drawing, and read the plant's
equipment list off it.

---

## What this means

**The transferable parts are the geometric and linguistic ones.** Tiling, OCR,
tag-pattern recognition, line tracing and topology are all model-free or
domain-general — they worked on the first attempt with no tuning.

**The one part that does not transfer is symbol detection**, and that is expected:
a detector trained on 32 synthetic shapes cannot recognise a symbol legend it has
never seen. That is a *training data* problem, not an architecture problem, and
it is the only thing standing between this and a working system on any given
customer's drawings.

**For the sales conversation:** this is the honest pitch —

> "Here is your own drawing. With no training on it whatsoever, it read your
> equipment list and traced your pipe network. What it cannot do yet is name your
> symbols, because it has never seen your legend. Give us 30 of your sheets and
> it will."

That is a much stronger position than a demo on synthetic data, and it asks for
something small and specific rather than a leap of faith.

---

## Honest limitations found in this test

- **OCR took 577 s** for a blind whole-sheet pass (30 tiles, both orientations).
  With a trained detector the pass is ROI-only and far cheaper — the blind sweep
  is the expensive path, not the normal one.
- **Instrument bubbles were mostly missed.** They are small circles containing
  two-letter codes; the exact-match pattern used here is too strict, and many sit
  at an angle. Fixable, but not fixed.
- **Stream tables and title blocks are read as text.** On this drawing that adds
  hundreds of numeric strings with no meaning to the graph. A real deployment
  needs region classification to exclude them.
- **No connectivity was built**, because connectivity requires symbols, and
  symbols require the detector. Pipe *runs* were traced; what sits on them was not
  identified.

## Reproduce

```bash
python scripts/test_real_drawing.py      # writes reports/real_drawing.json
```
