# The visual system — cockpit wireframe

Rebuilt against the shared reference (*Spacecraft Cockpit Wireframe UI*,
variant.com). Read off its stylesheet, not guessed at.

## What the reference actually is

| | value |
|---|---|
| type | **`Space Mono`, monospace — on everything.** No sans anywhere |
| stage | `rgb(3,3,3)` |
| foreground | `rgb(226,232,240)` — slate, not pure white |
| panel | `rgba(0,0,0,.6)` · `1px solid rgba(226,232,240,.4)` · `blur(8px)` · `inset 0 0 20px rgba(226,232,240,.05)` |
| corners | four 10px L-brackets, 2px, on **every** panel |
| scanlines | a 4px repeating horizontal wash |
| grid | 50px blueprint grid at `rgba(226,232,240,.05)` |
| panel header | 10px, `tracking-widest`, bottom hairline, name left / state right at 50% |
| rows | `LABEL:` dim, value bold, one line |
| strokes | 1.5px solid · 0.5px at 50% · 1px dashed 4,4 |
| radius | **none.** Every corner is square |
| accents | `border-l-4 border-l-white/50`, brightening on hover |
| motion | 60s/90s slow rotations, a 1s blink on live indicators |

Naming matters as much as the pixels: `NAV_SYS`, `INPUT_MATRIX`,
`MANUAL OVERRIDE`, `SEQ: 0x99A4F // PROTOCOL: OMEGA`. Uppercase snake_case
system names are half the character.

## What was here before

Soft glassmorphism — 16–24px radii, pill buttons, Inter, 8%-white borders,
blurred colour washes. Fine as a product page; nothing like an instrument. Every
one of those is now the opposite.

## The translation

| Reference | Here |
|---|---|
| four corner-bracket `<div>`s per panel | eight `linear-gradient` background layers in `--bk`. Streamlit owns the DOM; it does not own the CSS |
| `.panel-glass .scanlines` | `.hudp` and the `:has()` panel primitive, scanlines at **.11** not .22 — at the reference's strength they read as damage across a data table |
| scanlines over everything | **never over the drawing.** Over a white P&ID they are stripes across the thing the user came to read |
| `SEQ: 0x99A4F // PROTOCOL: OMEGA` | `theme.telemetry()` — a fixed bottom strip carrying the loaded sheet, node/link/control/flow counts, review backlog, the model in use and the benchmark |
| `NAV_SYS / ONLINE` header | `theme.hud(label, status)` — `ASK_SHEET / READY`, `FIELD_QUERY / PLAIN ENGLISH`, `RIG_BUILDER / OFFLINE` |
| segment level meters | `theme.seg(filled, total)` |
| strictly monochrome | four indicator lamps survive at 5–6px. The reference is a fictional cockpit; this screen has to tell a technician a line is flammable, and hue is the fastest channel there is. Nothing else is coloured |

## Traps hit while building it

**Streamlit's markdown rules out-specify a bare class.** `.eyebrow{font-size:9.5px}`
was rendering at **16px** — the section labels were four times their intended
size, which was most of why it looked generic. Every custom class now pins
`font-size` and `letter-spacing` with `!important`.

**Uppercase + wide tracking makes buttons grow.** A label that fits on one line
at 13px sans wraps to four at 10px/0.18em uppercase — buttons were **91px tall**.
Controls are `white-space:nowrap` with ellipsis at `.12em`; suggestion chips,
which are sentences, wrap on purpose.

**`primaryColor` paints Streamlit's own widgets.** Checkboxes, sliders, spinners
and focus rings default to red. Set to `#E2E8F0` in `.streamlit/config.toml`.

**Every vertical block shares one testid.** Styling
`stVerticalBlockBorderWrapper` directly framed each tab panel a second time.
Panels opt in with `theme.PANEL` as the container's first child, matched by
`:has(> div > div:first-child .pnl-mark)`.

## Rules for anything added later

1. **Mono, uppercase, tracked out.** Labels 9–10px at `.2em`. Values bold.
2. **Square.** No border-radius on anything, ever.
3. **Borders carry the structure**, not fills: `--l1` .14 hairline · `--l2` .26
   controls · `--l3` .42 panel edge · `--l4` .70 emphasis.
4. **Colour is a lamp**, 5–6px, and only for real state:
   `#4ade80` ok · `#7dd3fc` info · `#fbbf24` caution · `#f87171` alarm.
5. **Panels get brackets and a header.** Use `theme.hud()`; give it a status.
6. **Pin your font-size with `!important`** or Streamlit will hand you 16px.
7. **Nothing decorative over data.** Scanlines, grids and washes stop at the
   edge of a table, an image or a drawing.

## Where it is

| File | Holds |
|---|---|
| `src/edi/app/theme.py` | tokens, components, and the `PANEL` / `hud` / `seg` / `telemetry` helpers |
| `.streamlit/config.toml` | widget accent, base colours, telemetry off |
| `src/edi/app/canvas.py` | the drawing viewport — bracketed frame, square controls, no wash over the sheet |
