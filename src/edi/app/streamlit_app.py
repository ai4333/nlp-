"""
Engineering Drawing Intelligence — local application.

Implements the wireframe: 1b drawing-first workbench with 1a's evidence panel
docked, 1c graph workbench, 1d upload, 1e live extraction tracker, 1f review
queue, 1g evaluation dashboard.

Run:  python -m streamlit run src/edi/app/streamlit_app.py --server.port 8520
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml

from src.edi.config import load
from src.edi.ingest.acquire import acquire, save_upload, load_any
from src.edi.pipeline import run_sheet, sheet_manifest, STAGES
from src.edi.ingest import blueprint as BP
from src.edi.graph.kg import KG, class_name
from src.edi.app.theme import CSS, PANEL, hud, telemetry
from src.edi.app import canvas as CV
from src.edi.app import graphview as GV
from src.edi.app import editor as ED_CANVAS
from src.edi.app import rigtab as RIG
from src.edi.app import fieldtab as FLD
from src.edi import export as EX
from src.edi.retrieval.documents import DocumentStore
from src.edi import edits as ED

CFG = load()
_ONT = yaml.safe_load(open(ROOT / "config" / "symbol_ontology.yaml"))
CLASSES = _ONT["classes"]
PREFIX = _ONT.get("tag_prefixes", {}) or {}

st.set_page_config(page_title="Drawing Intelligence", layout="wide",
                   initial_sidebar_state="expanded", page_icon="◧")
st.markdown(CSS, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────── data helpers
@st.cache_data(show_spinner=False, ttl=5)
def manifest():
    return sheet_manifest()


@st.cache_data(show_spinner=False)
def sheet_bundle(sheet_id: str, _v: int = 0):
    """Entities, relations and derived edges with the edit overlay applied."""
    return ED.load_sheet_edited(sheet_id)


def bump():
    """Invalidate cached sheet data after an edit."""
    st.session_state["_rev"] = st.session_state.get("_rev", 0) + 1


def sheet_data(sheet_id: str):
    e, r, _ = sheet_bundle(sheet_id, st.session_state.get("_rev", 0))
    return e, r


@st.cache_data(show_spinner=False)
def sheet_image(sheet_id: str, _v: float = 0):
    return load_any(sheet_id)


@st.cache_resource(show_spinner=False)
def get_kg():
    """Read-only handle.

    Kuzu allows many concurrent READERS but only one writer, so holding a write
    connection open in the app blocked `evaluate.py` and `build_graph.py` for as
    long as the browser tab was open. The app reads read-only and takes a
    short-lived write connection only when it actually writes (see `writable`).
    """
    try:
        return KG(CFG.get_path("graph.path"), read_only=True)
    except RuntimeError as e:
        if "lock" in str(e).lower():
            st.error("The knowledge graph is locked by a writer "
                     "(an ingest or rebuild is running). Retry in a moment.")
            st.stop()
        raise


def writable():
    """Release the cached reader, hand back a writer. Caller must close it."""
    try:
        get_kg().close()
    except Exception:
        pass
    get_kg.clear()
    return KG(CFG.get_path("graph.path"))


@st.cache_resource(show_spinner=False)
def get_docstore():
    return DocumentStore()


@st.cache_resource(show_spinner=False)
def get_generator():
    from src.edi.retrieval.fewshot import build_index
    from src.edi.qa.nl2cypher import CypherGenerator
    return CypherGenerator(build_index(per_type=250))


def stat(k, v, sub="", cls=""):
    return (f'<div class="stat"><div class="k">{k}</div>'
            f'<div class="v {cls}">{v}</div>'
            f'{f"<div class=\'s\'>{sub}</div>" if sub else ""}</div>')


# ───────────────────────────────────────────────────────────── sidebar
with st.sidebar:
    sheets = manifest()
    st.markdown('<p class="eyebrow">Drawings</p>', unsafe_allow_html=True)
    if sheets:
        def _fmt(x):
            m = next(y for y in sheets if y["sheet_id"] == x)
            return f"{'↑ ' if m['uploaded'] else ''}{x}   ·   {m['n_symbols']} symbols"
        ids = [x["sheet_id"] for x in sheets]
        # The list is newest-first so a fresh upload is easy to find, but the
        # SELECTION is chosen separately. Two things were wrong: a drawing you
        # had just ingested was never opened (`goto_sheet` was written and
        # never read), and on a cold start the app landed on whichever file was
        # touched last — which after generating a test rig was an 11-symbol
        # sheet rather than the real 98-symbol drawing.
        # `goto_sheet` is a plain flag, not a widget key, so popping it is safe;
        # `sheet_pick` IS a widget key and is only ever assigned BEFORE the
        # widget is created.
        goto = st.session_state.pop("goto_sheet", None)
        if goto and goto in ids:
            st.session_state["sheet_pick"] = goto
        elif st.session_state.get("sheet_pick") not in ids:
            # Rank what to land on: a REAL uploaded drawing beats a synthetic
            # benchmark sheet, which beats a generated rig sheet — and within a
            # class, the one with the most on it. Plain "richest" landed on a
            # 171-symbol synthetic dataset sheet instead of the real offshore
            # P&ID, which is the one worth showing anybody.
            def _rank(r):
                rig = r["sheet_id"].startswith("u_rig_")
                return (2 if (r["uploaded"] and not rig) else 0 if rig else 1,
                        r["n_symbols"])
            st.session_state["sheet_pick"] = max(sheets, key=_rank)["sheet_id"]
        sheet = st.selectbox("sheet", ids, format_func=_fmt, key="sheet_pick",
                             label_visibility="collapsed")
        meta = next(x for x in sheets if x["sheet_id"] == sheet)
        st.markdown(
            f'<div class="kv" style="margin:9px 0 2px">'
            f'<b>{meta["n_symbols"]}</b> entities &nbsp;·&nbsp; '
            f'<b>{meta["n_relations"]}</b> relations<br>'
            f'<span style="color:var(--tx3)">{meta["n_tagged"]} tagged'
            f'{" · " + str(meta["n_review"]) + " need review" if meta["n_review"] else ""}'
            f'</span></div>', unsafe_allow_html=True)
    else:
        sheet = None
        st.caption("Nothing indexed yet.")

    st.markdown('<p class="eyebrow sp">Find across all drawings</p>', unsafe_allow_html=True)
    term = st.text_input("find", label_visibility="collapsed",
                         placeholder="tag — e.g. ZLO or RO-10")
    if term.strip():
        try:
            hits = get_kg().search_tag(term, limit=30)
        except Exception:
            hits = []
        if not hits:
            st.caption("no match in any indexed drawing")
        else:
            bysheet = {}
            for h in hits:
                bysheet.setdefault(h["sheet"], []).append(h)
            st.caption(f"{len(hits)} hit(s) in {len(bysheet)} drawing(s)")
            for sh, hs in list(bysheet.items())[:6]:
                tags = ", ".join(sorted({h["tag"] for h in hs})[:3])
                st.markdown(f'<div class="kv" style="font-size:11.5px">'
                            f'<b>{sh}</b> <span style="color:var(--tx3)">{tags}</span></div>',
                            unsafe_allow_html=True)

    _w = ROOT / "models" / "yolo_pid.pt"
    with st.expander("Add a drawing", expanded=False):
        engine = st.radio(
            "engine", ["blueprint", "detector"],
            format_func=lambda m: {"blueprint": "Blueprint  ·  any real drawing",
                                   "detector": "Detector  ·  trained symbols"}[m],
            label_visibility="collapsed", key="ing_engine",
            help="Blueprint mode reads the drawing's own tags and pipe routing — "
                 "no training needed, works on any P&ID. Detector mode recognises "
                 "symbols and needs a model trained on that symbol legend.")
        if engine == "detector" and not _w.exists():
            st.caption("⚠︎ No trained detector yet (`models/yolo_pid.pt`). "
                       "Use Blueprint mode.")
        up = st.file_uploader("upload", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                              label_visibility="collapsed")
        dtype = st.selectbox("drawing type", ["P&ID", "mechanical", "structural"],
                             label_visibility="collapsed")
        fast = st.checkbox("Fast mode", value=False,
                           help="Skip multi-scale OCR retries — roughly 2x faster, "
                                "measurably less accurate on clipped tags.")
        use_vlm = st.checkbox("VLM fallback", value=False,
                              help="qwen3-vl re-reads tags OCR could not resolve. "
                                   "Slower, needs ~7 GB.")
        if up is not None and st.button("Run extraction", type="primary",
                                        use_container_width=True, key="btn_run"):
            st.session_state["ingest"] = {"bytes": up.getvalue(), "name": up.name,
                                          "vlm": use_vlm, "type": dtype, "fast": fast,
                                          "engine": engine}
            st.rerun()

    _ds = get_docstore()
    _stats = _ds.stats()
    with st.expander(f"Reference documents ({len(_stats)})", expanded=False):
        if _stats:
            for d, v in _stats.items():
                st.markdown(f'<div class="kv" style="font-size:11.5px;color:var(--tx3)">'
                            f'{v["source"][:30]} · <span class="m">{v["chunks"]}</span> chunks</div>',
                            unsafe_allow_html=True)
        else:
            st.caption("None indexed — answers cite the graph and drawing only.")
        doc = st.file_uploader("docs", type=["pdf", "txt", "md", "csv"],
                               label_visibility="collapsed", key="docup")
        d1, d2 = st.columns([2, 1])
        if doc is not None and d1.button("Index", use_container_width=True, key="btn_docidx"):
            tmp = Path(CFG.get_path("paths.index")) / "documents" / "_in"
            tmp.mkdir(parents=True, exist_ok=True)
            f = tmp / doc.name; f.write_bytes(doc.getvalue())
            with st.spinner("indexing…"):
                n = _ds.add(f)
            st.success(f"{n} chunks"); get_docstore.clear(); st.rerun()
        if _stats and d2.button("Clear", use_container_width=True, key="btn_docclear"):
            _ds.clear(); get_docstore.clear(); st.rerun()

    if sheet:
        with st.expander("Export", expanded=False):
            _e, _r = sheet_data(sheet)
            st.download_button("Data bundle (.zip)", EX.bundle(sheet, _e, _r),
                               f"{sheet}_export.zip", "application/zip",
                               use_container_width=True, key="dl_bundle",
                               help="entities.csv · relations.csv · graph.json · graph.cypher")


# ───────────────────────────────────────────────────────────── extraction flow (1d → 1e)
if "ingest" in st.session_state:
    job = st.session_state["ingest"]
    st.markdown(f"### Extracting · {job['name']}")
    st.caption(f"{job['type']} · full pipeline: preprocess → detect → OCR → resolve → relations → graph")

    try:
        src = save_upload(job["bytes"], job["name"])
        pages = acquire(src)
    except Exception as e:
        st.error(f"Could not read that file — {type(e).__name__}: {e}")
        del st.session_state["ingest"]
        st.stop()

    st.caption(f"{len(pages)} sheet(s) · " +
               " · ".join(f"{p.width}×{p.height}" for p in pages[:3]))
    for p in pages:
        for w in p.warnings:
            st.warning(w) if "Low resolution" in w else st.caption(f"· {w}")

    # run_sheet() opens the graph for writing; the cached reader must go first
    try:
        get_kg().close()
    except Exception:
        pass
    get_kg.clear()

    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        track = st.empty()
    with col_b:
        prev = st.empty()

    done_ids = []
    for p in pages:
        state = {k: {"status": "pending", "detail": "", "elapsed": 0.0} for k, _ in STAGES}

        def paint():
            rows = []
            for key, label in STAGES:
                s = state[key]
                cls = "done" if s["status"] == "done" else ("run" if s["status"] == "running" else "")
                mark = "✓" if s["status"] == "done" else ""
                det = s["detail"] or ("…" if s["status"] == "running" else "")
                tm = f"{s['elapsed']:.1f}s" if s["elapsed"] else ""
                rows.append(f'<div class="stage {cls}"><div class="bul">{mark}</div>'
                            f'<div class="nm">{label}</div>'
                            f'<div class="dt">{det}{"  " + tm if tm else ""}</div></div>')
            track.markdown(f'<div class="card">{"".join(rows)}</div>', unsafe_allow_html=True)

        def cb(pr):
            state[pr.stage] = {"status": pr.status, "detail": pr.detail, "elapsed": pr.elapsed}
            paint()

        paint()
        prev.image(cv2.resize(cv2.imread(str(p.path), cv2.IMREAD_GRAYSCALE),
                              None, fx=0.13, fy=0.13), use_container_width=True,
                   caption=f"{p.source_name} · page {p.page}")
        try:
            if job.get("engine", "blueprint") == "blueprint":
                import cv2 as _cv2
                bimg = _cv2.imread(str(p.path), _cv2.IMREAD_GRAYSCALE)
                for k in ("preprocess",):
                    state[k] = {"status": "done", "detail": f"{p.width}x{p.height}",
                                "elapsed": 0.0}
                state["detect"] = {"status": "done", "detail": "skipped (blueprint)",
                                   "elapsed": 0.0}
                state["ocr"] = {"status": "running", "detail": "reading the sheet",
                                "elapsed": 0.0}
                paint()
                out = BP.extract(bimg, p.sheet_id,
                                 on_progress=lambda m: (
                                     state.__setitem__("ocr", {"status": "running",
                                                               "detail": m, "elapsed": 0.0}),
                                     paint()))
                BP.save(out, p.sheet_id)
                st_ = out["stats"]
                state["ocr"] = {"status": "done", "detail": f"{st_['tags']} tags",
                                "elapsed": st_["t_ocr"]}
                state["resolve"] = {"status": "done",
                                    "detail": ", ".join(f"{k} {v}" for k, v in
                                                        st_["by_category"].items())[:40],
                                    "elapsed": 0.0}
                state["relations"] = {"status": "done",
                                      "detail": f"{st_['relations']} links", "elapsed": 0.0}
                paint()
                kg = KG(CFG.get_path("graph.path"))
                try:
                    kg.conn.execute("MATCH (s:Symbol) WHERE s.sheet_id=$s DETACH DELETE s",
                                    {"s": p.sheet_id})
                except Exception:
                    pass
                kg.add_sheet(p.sheet_id, out["entities"], out["relations"], out["derived"])
                state["graph"] = {"status": "done",
                                  "detail": f"{kg.counts()['symbols']} nodes total",
                                  "elapsed": 0.0}
                kg.close(); paint()
                done_ids.append(p.sheet_id)
                st.success(f"**{p.sheet_id}** — {st_['tags']} tagged items · "
                           f"{st_['relations']} connections · {st_['t_total']}s")
            else:
                res = run_sheet(p.sheet_id, on_progress=cb, use_vlm=job["vlm"],
                                fast=job.get("fast", False))
                done_ids.append(p.sheet_id)
                st.success(f"**{p.sheet_id}** — {res.n_symbols} symbols · {res.n_tagged} tagged · "
                           f"{res.n_relations} relations · {res.timings['total']}s")
        except Exception as e:
            st.error(f"{p.sheet_id} failed — {type(e).__name__}: {e}")

    del st.session_state["ingest"]
    manifest.clear()
    if done_ids:
        st.session_state["goto_sheet"] = done_ids[0]
    if st.button("Open in workbench", type="primary", key="btn_open"):
        st.rerun()
    st.stop()

def close_reader():
    """Hand the Kuzu write lock back. Kuzu is single-writer, so the cached
    read-only connection must go before anything writes."""
    try:
        get_kg().close()
    except Exception:
        pass
    get_kg.clear()


if sheet is None:
    # Nothing ingested yet is not a dead end any more: you can draw a sheet
    # here and then ask questions of it.
    st.markdown("""
<div class="mast">
  <div class="logo"><div class="mark">◧</div>
    <div><h1>Drawing Intelligence</h1>
    <div class="sub">P&amp;ID · knowledge graph · multimodal RAG</div></div></div>
  <div class="pills"><span class="pill"><span class="d"></span>local · offline</span>
  <span class="pill opt">no drawing indexed yet</span></div>
</div>""", unsafe_allow_html=True)
    st.info("Upload a drawing from the sidebar — or build one here.")
    RIG.render(close_reader, manifest.clear, get_kg)
    st.stop()

ents, rels, derived = sheet_bundle(sheet, st.session_state.get("_rev", 0))
sheet_edits = ED.load_edits(sheet)
img = sheet_image(sheet)

try:
    _kc = get_kg().counts()
except Exception:
    _kc = {"symbols": 0, "connected_to": 0, "controls": 0, "feeds_to": 0}
_rep_p = CFG.get_path("paths.reports") / "pidqa_L1.json"
_acc = json.loads(_rep_p.read_text())["overall_accuracy"] if _rep_p.exists() else None

st.markdown(f"""
<div class="mast">
  <div class="logo"><div class="mark">◧</div>
    <div><h1>Drawing Intelligence</h1>
    <div class="sub">P&amp;ID · knowledge graph · multimodal RAG</div></div></div>
  <div class="pills">
    <span class="pill"><span class="d"></span>local · offline</span>
    <span class="pill">sheet <b>{sheet}</b></span>
    <span class="pill opt">graph <b>{_kc['symbols']}</b> nodes ·
      <b>{_kc['connected_to']}</b> links · <b>{_kc['controls']}</b> controls</span>
    {f'<span class="pill">benchmark <b>{_acc:.3f}</b></span>' if _acc else ''}
  </div>
</div>""", unsafe_allow_html=True)

st.markdown(telemetry(
    [("sheet", sheet),
     ("nodes", f"{_kc['symbols']}"),
     ("links", f"{_kc['connected_to']}"),
     ("ctrl", f"{_kc['controls']}"),
     ("flow", f"{_kc.get('feeds_to', 0)}"),
     ("review", f"{meta['n_review']}")],
    seq=f"local · offline // model {CFG.dotted('llm.answer_model','qwen3:8b')} // "
        f"bench {_acc:.4f}" if _acc else "local · offline"),
    unsafe_allow_html=True)

tab_w, tab_f, tab_g, tab_x, tab_b, tab_r, tab_e = st.tabs(
    ["Workbench", "Field mode", "Knowledge graph", "Edit graph", "Rig builder",
     "Review queue", "Evaluation"])


# ───────────────────────────────────────────────────────────── 1b + 1a workbench
with tab_w:
    left, right = st.columns([1.95, 1], gap="large")

    # Each region is a floating glass panel, the way the design reference
    # composes a workspace — a bordered container is that panel primitive and
    # the theme styles it.
    with right, st.container(border=True):
        st.markdown(PANEL + hud("Ask about this drawing", "ready"), unsafe_allow_html=True)

        hist_key = f"chat_{sheet}"
        history = st.session_state.setdefault(hist_key, [])

        err = st.session_state.get("ask_error")
        if err:
            st.error(err[0])
            with st.expander("details"):
                st.code(err[1])

        # The ask box sits ABOVE the conversation, and the newest answer is
        # rendered directly beneath it. Two earlier placements both failed:
        # st.chat_input pins to the bottom of the whole app rather than its
        # container (it landed below the canvas, off-screen), and putting a
        # text input below the conversation pushed it off-screen again as soon
        # as anything was asked. Above the conversation, it never moves.
        # Enter must submit. A bare text_input reruns on Enter WITHOUT setting
        # the button, so the question was silently dropped and the box cleared —
        # which is exactly what "I can't type anything" looked like. The
        # on_change callback captures the value at the moment Enter is pressed.
        # NEVER pop a widget's session_state key to clear it: deleting the key
        # unregisters the widget and Streamlit stops delivering its value to
        # Python entirely — the box accepts typing and nothing ever happens.
        # Bumping a counter in the key makes a brand-new (empty) widget instead.
        # A form is the one construct where pressing Enter in the box reliably
        # submits: Streamlit fires the form on Enter AND on the submit button,
        # so the two paths can never disagree. clear_on_submit empties the box
        # for the next question without the widget-key juggling that used to be
        # needed (and used to swallow input). Clear lives outside the form
        # because a form may only contain submit buttons.
        with st.form(key=f"ask_form_{sheet}", clear_on_submit=True,
                     border=False):
            typed = st.text_input("ask", label_visibility="collapsed",
                                  placeholder="Ask anything about this drawing…")
            send = st.form_submit_button("Ask", type="primary",
                                         use_container_width=True)
        if st.button("Clear", use_container_width=True, key="btn_clearchat"):
            st.session_state[hist_key] = []
            st.session_state.pop("last", None)
            st.session_state.pop("ask_error", None)
            st.rerun()
        if history:
            st.download_button("Export conversation (.md)",
                               "\n\n".join(f"**Q.** {t['q']}\n\n{t['answer']}\n\n"
                                            f"`{t['cypher']}`" for t in history),
                               f"{sheet}_conversation.md", "text/markdown",
                               use_container_width=True, key="dl_chat")
        st.markdown("---")


        if not history:
            tagged = [e["tag"] for e in ents if e.get("tag")]
            names = sorted({e.get("name", "") for e in ents if e.get("name")})
            sugg = []
            if tagged:
                sugg.append(f"What is connected to {tagged[0]}?")
            if names:
                w = names[0].split(" /")[0]
                sugg.append(f"How many {w}s are there?")
            # The whole-drawing question goes first: it is the most natural
            # thing to ask of a new sheet and it used to have no handler.
            sugg = ["Explain this whole drawing"] + sugg + [
                "What equipment is on this drawing?",
                "Which item has the most connections?"]
            st.markdown('<div class="empty" style="padding:16px">'
                        '<b>Ask anything about this drawing</b></div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="sugg">', unsafe_allow_html=True)
            for i, sq in enumerate(sugg[:4]):
                if st.button(sq, key=f"sg{i}", use_container_width=True):
                    st.session_state["pending_q"] = sq; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- conversation ------------------------------------------------
        for turn in reversed(history):
            with st.chat_message("user"):
                st.markdown(turn["q"])
            with st.chat_message("assistant"):
                st.markdown(turn["answer"])
                chips = "".join(
                    f'<span class="chip">◧ {r["tag"] or r.get("name","region")}</span>'
                    for r in turn.get("regions", [])[:4])
                if turn.get("path"):
                    chips += '<span class="chip g">◇ graph path</span>'
                for d in turn.get("docs", [])[:2]:
                    chips += f'<span class="chip d">¶ {d["source"][:14]} p{d["page"]}</span>'
                if chips:
                    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
                if turn.get("path"):
                    st.markdown(f'<div class="path">{" → ".join(turn["path"])}</div>',
                                unsafe_allow_html=True)
                with st.expander("evidence & query"):
                    st.code(turn["cypher"], language="cypher")
                    st.caption(f"{turn['latency']} ms · confidence {turn['confidence']} · "
                               f"{turn['attempts']} attempt(s)")
                    if turn.get("rows"):
                        st.dataframe(pd.DataFrame(turn["rows"],
                                                  columns=turn.get("cols") or None).astype(str),
                                     use_container_width=True, hide_index=True)
                    for r in turn.get("regions", [])[:3]:
                        if r.get("crop") and Path(r["crop"]).exists():
                            st.image(r["crop"], use_container_width=True,
                                     caption=r["tag"] or r.get("name", ""))


    entered = st.session_state.pop("_ask_now", None)          # Enter key
    clicked = typed.strip() if (send and typed and typed.strip()) else None
    pend = st.session_state.pop("pending_q", None)             # suggestion chip
    ask_q = entered or clicked or pend

    if ask_q:
        from src.edi.qa.answer import ask, narrate_stream, answer_stream
        with right:
            with st.spinner("reading the drawing…"):
                try:
                    res = ask(ask_q, get_kg(), get_generator(), sheet, image=img,
                              docstore=get_docstore(),
                              history=st.session_state.get(hist_key, []),
                              stream_overview=True, stream_answer=True)
                # Streaming means the reader watches the answer being written
                # instead of waiting on a spinner, which is most of why a local
                # model can feel as responsive as a hosted one. Every answer
                # that came back empty is one the caller was asked to stream.
                    if not res["answer"]:
                        brief = res["evidence"].get("briefing", "")
                        gen = (narrate_stream if res.get("handler") == "overview"
                               else answer_stream)
                        try:
                            res["answer"] = st.write_stream(gen(ask_q, brief))
                        except Exception:
                            res["answer"] = brief
                    st.session_state[hist_key].append({
                        "q": ask_q, "answer": res["answer"], "cypher": res["cypher"],
                        "rows": res["rows"], "cols": res["columns"],
                        "path": res["reasoning_path"],
                        "regions": res["evidence"]["regions"],
                        "docs": res["evidence"].get("documents", []),
                        "confidence": res["confidence"], "latency": res["latency_ms"],
                        "attempts": res["n_attempts"],
                    })
                    st.session_state["last"] = res
                    st.session_state.pop("ask_error", None)
                    st.session_state["ask_n"] = st.session_state.get("ask_n", 0) + 1
                    st.rerun()
                except Exception as e:
                    # Store it rather than st.error() here: the rerun below used
                    # to wipe the message before anyone could read it, so every
                    # failure looked like nothing happening at all.
                    import traceback
                    st.session_state["ask_error"] = (
                        f"{type(e).__name__}: {e}", traceback.format_exc()[-1400:])

    res = st.session_state.get("last")
    highlight = ([r["entity_id"].split(":", 1)[-1] for r in res["evidence"]["regions"]]
                 if res else [])

    with left, st.container(border=True):
        st.markdown(PANEL, unsafe_allow_html=True)
        components.html(CV.render(img, ents, rels, highlight=highlight,
                                  show_links=False, show_labels=True, height=660,
                                  derived=derived),
                        height=672, scrolling=False)
        st.caption(f"{img.shape[1]}×{img.shape[0]} px · scroll to zoom · drag to pan · "
                   f"click a symbol for detail")


# ───────────────────────────────────────────────────────── field mode
with tab_f:
    FLD.render(get_kg(), sheet, img, ents,
               generator=get_generator(), docstore=get_docstore())


# ───────────────────────────────────────────────────────────── 1c graph workbench
with tab_g:
    gl, gr = st.columns([2.15, 1], gap="medium")
    with gr:
        st.markdown('<p class="eyebrow">Query the graph</p>', unsafe_allow_html=True)
        mode = st.radio("mode", ["Natural language", "Cypher"], horizontal=True,
                        label_visibility="collapsed")
        if mode == "Cypher":
            cy = st.text_area("cypher", height=110, label_visibility="collapsed",
                              value="MATCH (s:Symbol)-[:CONNECTED_TO]-(n:Symbol)\n"
                                    "RETURN s.tag, count(n) AS c ORDER BY c DESC LIMIT 10")
            if st.button("Run", type="primary", use_container_width=True, key="btn_cypher"):
                try:
                    df = get_kg().run(get_kg().scoped(" ".join(cy.split()), sheet))
                    st.dataframe(df.astype(str), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"{type(e).__name__}: {str(e)[:220]}")
        else:
            gq = st.text_input("gq", label_visibility="collapsed",
                               placeholder="what feeds the tank?")
            if st.button("Generate & run", type="primary", use_container_width=True, key="btn_gen") and gq.strip():
                with st.spinner("translating…"):
                    try:
                        g = get_generator().generate(gq, kg=get_kg(), sheet_id=sheet)
                        st.code(g["cypher"], language="cypher")
                        df = get_kg().run(get_kg().scoped(g["cypher"], sheet))
                        st.dataframe(df.astype(str), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"{type(e).__name__}: {str(e)[:220]}")

        st.markdown('<p class="eyebrow" style="margin-top:14px">Topology</p>',
                    unsafe_allow_html=True)
        deg = {}
        for r in rels:
            deg[r["source"]] = deg.get(r["source"], 0) + 1
            deg[r["target"]] = deg.get(r["target"], 0) + 1
        iso = len(ents) - len(deg)
        st.markdown(
            f'<div class="kv"><b>{len(ents)}</b> nodes · <b>{len(rels)}</b> connected_to<br>'
            f'<span style="color:var(--ok)">{len(derived.get("CONTROLS",[]))}</span> controls · '
            f'<span style="color:var(--warn)">{len(derived.get("FEEDS_TO",[]))}</span> feeds_to<br>'
            f'mean degree <span class="m">{(2*len(rels)/max(len(ents),1)):.2f}</span> · '
            f'isolated <span class="m">{iso}</span></div>', unsafe_allow_html=True)

    with gl:
        focus = None
        if st.session_state.get("last"):
            regs = st.session_state["last"]["evidence"]["regions"]
            if regs:
                focus = regs[0]["entity_id"].split(":", 1)[-1]
        path_nodes = [n.split(":", 1)[-1]
                      for n in (st.session_state.get("last", {}).get("evidence", {})
                                .get("graph", {}).get("nodes", []))]
        components.html(GV.render(ents, rels, img.shape[1], img.shape[0],
                                  focus=focus, path_nodes=path_nodes, height_px=660,
                                  derived=derived),
                        height=672, scrolling=False)
        st.caption("laid out on the drawing's real geometry · click a node to trace · "
                   "click a legend chip to filter")


# ───────────────────────────────────────────────────────────── edit graph
with tab_x:
    c = ED.count(sheet_edits)
    ec_left, ec_right = st.columns([2.15, 1], gap="medium")

    with ec_right:
        st.markdown('<p class="eyebrow">Editing</p>', unsafe_allow_html=True)
        mode = st.radio("mode", ["inspect", "draw", "connect", "erase"],
                        format_func=lambda m: {"inspect": "Inspect",
                                               "draw": "Add symbol",
                                               "connect": "Connect",
                                               "erase": "Delete"}[m],
                        horizontal=True, label_visibility="collapsed", key="ed_mode")
        rtype = st.selectbox("relation type", ["CONNECTED_TO", "CONTROLS", "FEEDS_TO"],
                             key="ed_rtype",
                             help="Applied when you connect two symbols.")
        st.markdown(
            f'<div class="kv" style="margin:8px 0 4px;font-size:11.5px">'
            f'<b>+{c["added"]}</b> added · <b>−{c["removed"]}</b> removed · '
            f'<b>~{c["modified"]}</b> changed · '
            f'<b>+{c["rel_added"]}</b>/<b>−{c["rel_removed"]}</b> links<br>'
            f'<span style="color:var(--tx3)">Stored as an overlay — re-extraction '
            f'never destroys them. Run <code>build_graph.py --fresh</code> to push '
            f'into the graph.</span></div>', unsafe_allow_html=True)

    with ec_left:
        spec = ED_CANVAS.build_spec(sheet, ents, rels, derived, img,
                                    st.session_state.get("_rev", 0))
        act = ED_CANVAS.canvas(spec, mode=mode, height=620,
                               selected=st.session_state.get("ed_sel"),
                               key=f"canvas_{sheet}")

    # ---- act on whatever the canvas sent back --------------------------------
    if act and act.get("nonce") != st.session_state.get("_ed_nonce"):
        st.session_state["_ed_nonce"] = act.get("nonce")
        a = act.get("action")
        if a == "select":
            st.session_state["ed_sel"] = act["id"]
            st.rerun()
        elif a == "delete":
            ED.save_edits(sheet, ED.remove_entity(sheet_edits, act["id"], "canvas"))
            sheet_bundle.clear(); bump(); st.rerun()
        elif a == "connect":
            try:
                ED.save_edits(sheet, ED.add_relation(sheet_edits, act["a"], act["b"],
                                                     rtype, "canvas"))
                sheet_bundle.clear(); bump(); st.rerun()
            except ValueError as err:
                st.error(str(err))
        elif a == "draw":
            st.session_state["ed_pending"] = act["bbox"]
            st.rerun()

    with ec_right:
        pending = st.session_state.get("ed_pending")
        class_opts = sorted(CLASSES)
        def _cn(k): return f'{k} — {CLASSES[k]["name"]}'

        if pending:
            st.markdown('<p class="eyebrow sp">New symbol</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="kv" style="font-size:11.5px;color:var(--tx3)">'
                        f'box <span class="m">{pending}</span></div>', unsafe_allow_html=True)
            ncls = st.selectbox("class", class_opts, format_func=_cn, key="ed_ncls")
            ntag = st.text_input("tag", key="ed_ntag", placeholder="e.g. PV-1042")
            k1, k2 = st.columns(2)
            if k1.button("Add", type="primary", use_container_width=True, key="ed_addok"):
                nid = ED.next_id(ents, sheet_edits)
                ED.save_edits(sheet, ED.add_entity(sheet_edits, nid, ncls, pending,
                                                   ntag, "canvas"))
                st.session_state.pop("ed_pending", None)
                sheet_bundle.clear(); bump(); st.rerun()
            if k2.button("Cancel", use_container_width=True, key="ed_addno"):
                st.session_state.pop("ed_pending", None); st.rerun()

        sid = st.session_state.get("ed_sel")
        cur = next((e for e in ents if e["id"] == sid), None)
        if cur is not None and not pending:
            st.markdown('<p class="eyebrow sp">Selected symbol</p>', unsafe_allow_html=True)
            new_tag = st.text_input("tag", value=cur.get("tag") or "", key=f"et_{sid}")
            new_cls = st.selectbox("class", class_opts, format_func=_cn,
                                   index=class_opts.index(cur["pidqa_class"])
                                   if cur["pidqa_class"] in class_opts else 0,
                                   key=f"ec_{sid}")
            s1, s2 = st.columns(2)
            if s1.button("Save", type="primary", use_container_width=True, key="ed_save"):
                ED.save_edits(sheet, ED.modify_entity(
                    sheet_edits, sid,
                    tag=new_tag if new_tag != (cur.get("tag") or "") else None,
                    pidqa_class=new_cls if new_cls != cur["pidqa_class"] else None,
                    reason="canvas"))
                sheet_bundle.clear(); bump(); st.rerun()
            if s2.button("Delete", use_container_width=True, key="ed_del"):
                ED.save_edits(sheet, ED.remove_entity(sheet_edits, sid, "canvas"))
                st.session_state.pop("ed_sel", None)
                sheet_bundle.clear(); bump(); st.rerun()

            nbrs = [(r["target"] if r["source"] == sid else r["source"])
                    for r in rels if sid in (r["source"], r["target"])]
            if nbrs:
                st.markdown('<p class="eyebrow sp">Disconnect</p>', unsafe_allow_html=True)
                by = {e["id"]: e for e in ents}
                opts = {f'{by[n].get("tag") or "(untagged)"} · {n}': n
                        for n in nbrs if n in by}
                if opts:
                    dn = st.selectbox("neighbour", list(opts), key=f"dn_{sid}",
                                      label_visibility="collapsed")
                    if st.button("Remove connection", use_container_width=True, key="ed_rdel"):
                        ED.save_edits(sheet, ED.remove_relation(
                            sheet_edits, sid, opts[dn], "CONNECTED_TO", "canvas"))
                        sheet_bundle.clear(); bump(); st.rerun()
        elif not pending:
            st.markdown('<div class="empty" style="padding:22px 16px"><b>Nothing selected</b>'
                        'Click a symbol on the drawing to edit it.</div>',
                        unsafe_allow_html=True)

        st.markdown('<p class="eyebrow sp">History</p>', unsafe_allow_html=True)
        log = sheet_edits.get("log", [])
        if not log:
            st.caption("no manual edits yet")
        else:
            for entry in reversed(log[-8:]):
                st.markdown(f'<div class="kv" style="font-size:11px">'
                            f'<span style="color:var(--tx3)">{entry["ts"][11:19]}</span> '
                            f'<b>{entry["action"].replace("_"," ")}</b> '
                            f'<span style="color:var(--tx3)">{entry["detail"][:40]}</span></div>',
                            unsafe_allow_html=True)
            u1, u2 = st.columns(2)
            if u1.button("Undo last", use_container_width=True, key="ed_undo"):
                ED.save_edits(sheet, ED.undo(sheet_edits))
                sheet_bundle.clear(); bump(); st.rerun()
            if u2.button("Discard all", use_container_width=True, key="ed_clear"):
                ED.clear(sheet); st.session_state.pop("ed_sel", None)
                sheet_bundle.clear(); bump(); st.rerun()


# ───────────────────────────────────────────────────────────── rig builder
with tab_b:
    RIG.render(close_reader, manifest.clear, get_kg)


# ───────────────────────────────────────────────────────────── 1f review queue
with tab_r:
    st.markdown('<p class="eyebrow">Low-confidence entities</p>', unsafe_allow_html=True)
    tc1, tc2 = st.columns([2, 1])
    thr = tc1.slider("flag tag confidence below", 0.0, 1.0,
                     float(CFG.dotted("entity_resolution.review_below", 0.70)), 0.05)
    queue = [e for e in ents if e.get("tag_location") != "none"
             and (not e.get("tag") or e.get("tag_confidence", 0) < thr)]
    tc2.markdown(f'<div class="stat" style="margin-top:6px"><div class="k">flagged</div>'
                 f'<div class="v warn">{len(queue)}</div>'
                 f'<div class="s">of {len(ents)} entities</div></div>', unsafe_allow_html=True)

    if not queue:
        st.success("Nothing below this threshold — every tag was read confidently.")
    else:
        fixes = {}
        for e in queue[:14]:
            c1, c2 = st.columns([1, 2.6])
            x0, y0, x1, y1 = e["bbox"]; p = 95
            crop = img[max(0, y0-p):y1+p, max(0, x0-p):x1+p]
            with c1:
                if crop.size:
                    st.image(crop, use_container_width=True)
            with c2:
                pref = ", ".join(PREFIX.get(e["pidqa_class"], [])) or "—"
                read = e.get("tag") or "nothing read"
                cls = "b" if e.get("tag") else "bad"
                st.markdown(
                    f'<div class="rd">OCR read &nbsp;<{cls}>{read}</{cls}><br>'
                    f'symbol &nbsp;<b>{class_name(e["pidqa_class"])}</b> '
                    f'· conf <b>{e.get("tag_confidence",0):.2f}</b><br>'
                    f'expected prefix &nbsp;<b>{pref}</b></div>', unsafe_allow_html=True)
                fixes[e["id"]] = st.text_input("corrected tag", value=e.get("tag") or "",
                                               key=f"fx_{sheet}_{e['id']}",
                                               label_visibility="collapsed")
            st.markdown('<hr style="margin:6px 0 12px">', unsafe_allow_html=True)

        if st.button("Apply corrections", type="primary", key="btn_apply"):
            base = CFG.get_path("paths.interim") / sheet
            data = json.load(open(base / "ocr.json"))
            n = 0
            for r in data["records"]:
                nv = fixes.get(r["id"])
                if nv is not None and nv.strip() and nv.strip() != (r.get("tag") or ""):
                    r["tag"] = nv.strip(); r["tag_confidence"] = 1.0
                    r["tag_source"] = "human"; n += 1
            if n:
                json.dump(data, open(base / "ocr.json", "w"), indent=1)
                kg = writable()
                try:
                    for r in data["records"]:
                        if r.get("tag_source") == "human":
                            kg.conn.execute(
                                "MATCH (s:Symbol) WHERE s.entity_id=$i "
                                "SET s.tag=$t, s.tag_confidence=1.0",
                                {"i": f"{sheet}:{r['id']}", "t": r["tag"]})
                finally:
                    kg.close()
                sheet_bundle.clear(); manifest.clear(); bump()
                st.success(f"Applied {n} correction(s) to the graph.")
                st.rerun()
            else:
                st.info("No changes to apply.")


# ───────────────────────────────────────────────────────────── 1g evaluation
with tab_e:
    r1 = CFG.get_path("paths.reports") / "pidqa_L1.json"
    r3 = CFG.get_path("paths.reports") / "pidqa_L3.json"
    if not r1.exists():
        st.info("No evaluation yet. Run:  `python scripts/evaluate.py --per-type 32`")
    else:
        rep = json.loads(r1.read_text())
        order = [("simple_counting", "simple counting"), ("spatial_counting", "spatial counting"),
                 ("spatial_connection", "spatial connection"), ("value_based", "value / tags")]
        cells = "".join(
            stat(lbl, f"{rep['by_type'][k]['accuracy']:.3f}", f"n={rep['by_type'][k]['n']}",
                 "ok" if rep["by_type"][k]["accuracy"] >= .9 else
                 ("acc" if rep["by_type"][k]["accuracy"] >= .75 else "warn"))
            for k, lbl in order if k in rep.get("by_type", {}))
        st.markdown(f'<div class="stat-grid">{cells}</div>', unsafe_allow_html=True)

        summary = stat("overall · L1 graph quality", f"{rep['overall_accuracy']:.4f}",
                       f"{rep['n_questions']} questions · {rep['sheets']} sheets", "acc")
        summary += stat("query errors", rep["n_errors"], "reference Cypher on our graph",
                        "ok" if rep["n_errors"] == 0 else "warn")
        if r3.exists():
            rr = json.loads(r3.read_text())
            summary += stat("overall · L3 end-to-end", f"{rr['overall_accuracy']:.4f}",
                            f"executable {rr['executable_query_rate']:.3f} · "
                            f"{rr['n_questions']} sampled", "ok")
        st.markdown(f'<div class="stat-grid" style="margin-top:12px">{summary}</div>',
                    unsafe_allow_html=True)

        st.markdown('<p class="eyebrow" style="margin-top:18px">Failure cases</p>',
                    unsafe_allow_html=True)
        det = CFG.get_path("paths.reports") / "pidqa_L1_detail.csv"
        if det.exists():
            df = pd.read_csv(det)
            f = df[~df.ok][["sheet", "qtype", "question", "gt", "pred"]].head(40)
            if len(f) == 0:
                st.success("No failures in this run.")
            else:
                # Streamlit's dataframe collapses on these columns (gt/pred hold
                # Python sets for value-based questions, which Arrow cannot
                # serialise). A plain styled table is both reliable and on-theme.
                head = ("<tr><th>sheet</th><th>type</th><th>question</th>"
                        "<th>expected</th><th>got</th></tr>")
                body = "".join(
                    "<tr><td class='m'>{s}</td><td>{t}</td><td>{q}</td>"
                    "<td class='m ok'>{g}</td><td class='m bad'>{p}</td></tr>".format(
                        s=r["sheet"], t=str(r["qtype"]).replace("_", " "),
                        q=str(r["question"])[:110],
                        g=str(r["gt"])[:60], p=str(r["pred"])[:60])
                    for _, r in f.iterrows())
                st.markdown(
                    "<div class='tbl-wrap'><table class='tbl'>"
                    f"<thead>{head}</thead><tbody>{body}</tbody></table></div>",
                    unsafe_allow_html=True)
                st.caption(f"showing {len(f)} of {int((~df.ok).sum())} failures")
