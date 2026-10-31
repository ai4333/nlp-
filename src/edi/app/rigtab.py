"""
Rig builder screen — design a drawing by conversation, then query it.

The loop is: say what you want in plain English → a local model turns it into a
spec → the spec is drawn, checked and loaded into the knowledge graph → the
sheet appears in the sidebar like any other drawing and the workbench can be
asked questions about it.

Everything on this screen is a table you can read and export, because that is
what the deliverable actually is on a project: the drawing plus its registers —
equipment list, line list, valve list, instrument index, and the ESD cause &
effect matrix.
"""
from __future__ import annotations
import json
import time

import cv2
import pandas as pd
import streamlit as st

from ..rig import catalog as CAT
from ..rig import templates as TPL
from ..rig import spec as SP
from ..rig import build as BLD
from ..rig.chat import RigChat, available_models
from .theme import PANEL, hud


def _tbl(rows: list[dict], height: int = 320):
    if not rows:
        st.caption("nothing to list")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 height=height)


def _dl(rows: list[dict], name: str, key: str):
    if rows:
        st.download_button(f"{name}.csv", pd.DataFrame(rows).to_csv(index=False),
                           f"{name}.csv", "text/csv", key=key,
                           use_container_width=True)


def _build_and_ingest(raw: dict, close_reader, clear_manifest) -> dict:
    """Draw the sheet, write the interim contracts, load the graph."""
    sp = SP.build(raw)
    close_reader()                       # Kuzu is single-writer
    res = BLD.build_sheet(sp, write_graph=True)
    clear_manifest()
    st.session_state["rig_raw"] = raw
    st.session_state["rig_result"] = res
    return res


def render(close_reader, clear_manifest, get_kg):
    ask_panel = st.container(border=True)
    ask_panel.__enter__()
    st.markdown(PANEL + hud("Build a rig drawing", "runs offline"), unsafe_allow_html=True)

    models = available_models()
    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        if models:
            # qwen3:8b is the model the rest of this system is tuned against, so
            # it is the default; a coder model is a fine alternative and the
            # dropdown lists whatever Ollama has.
            order = ("qwen3:", "qwen2.5-coder", "qwen3-coder", "deepseek-coder",
                     "qwen2.5:")
            pref = [m for p in order for m in models if m.startswith(p)]
            default = pref[0] if pref else models[0]
            model = st.selectbox("model", models, index=models.index(default),
                                 label_visibility="collapsed",
                                 help="Any local Ollama model. The task is emitting a "
                                      "JSON spec, so a coder model does it well.")
        else:
            model = None
            st.error("Ollama is not answering — the builder will fall back to the "
                     "standard packages, which still produce a complete sheet.")
    with c2:
        auto = st.checkbox("Auto-complete the design", value=True,
                           help="Add relief valves, ESD valves, check valves, "
                                "instrument loops and fire & gas detection the way "
                                "a real sheet has them.")
    with c3:
        st.markdown('<span class="pill"><span class="d"></span>offline</span>',
                    unsafe_allow_html=True)

    # ---- the ask box, ABOVE the conversation so it never moves ----------
    n = st.session_state.get("rig_n", 0)
    key = f"rig_in_{n}"

    def _enter(k=key):
        v = (st.session_state.get(k) or "").strip()
        if v:
            st.session_state["_rig_now"] = v

    st.text_input("build", key=key, label_visibility="collapsed",
                  placeholder="e.g. build a three stage separation train with gas "
                              "compression, dehydration and flare…",
                  on_change=_enter)
    b1, b2, b3, _sp = st.columns([1.5, 1, 1.4, 2.6])
    go = b1.button("Build the sheet", type="primary", use_container_width=True,
                   key="rig_go")
    if b2.button("Clear", use_container_width=True, key="rig_clear"):
        for k in ("rig_raw", "rig_result", "rig_hist", "_rig_now"):
            st.session_state.pop(k, None)
        st.session_state["rig_n"] = n + 1
        st.rerun()
    has_spec = bool(st.session_state.get("rig_raw"))
    if b3.button("Start from scratch" if has_spec else "Surprise me",
                 use_container_width=True, key="rig_seed"):
        st.session_state.pop("rig_raw", None)
        st.session_state["_rig_now"] = "an offshore platform overview sheet"

    # ---- the standard packages -----------------------------------------
    with st.expander("Standard packages — one click", expanded=not has_spec):
        names = list(TPL.TEMPLATES)
        cols = st.columns(4)
        for i, nm in enumerate(names):
            if cols[i % 4].button(TPL.TEMPLATES[nm]["title"], key=f"tpl_{nm}",
                                  use_container_width=True):
                st.session_state["_rig_template"] = nm
                st.rerun()

    ask_panel.__exit__(None, None, None)
    st.write("")

    q = st.session_state.pop("_rig_now", None)
    tpl = st.session_state.pop("_rig_template", None)
    if go and not q:
        q = (st.session_state.get(key) or "").strip() or None

    # ---- run -------------------------------------------------------------
    if tpl:
        with st.spinner(f"drawing the {TPL.TEMPLATES[tpl]['title'].lower()}…"):
            raw = TPL.expand(tpl)
            res = _build_and_ingest(raw, close_reader, clear_manifest)
        st.session_state.setdefault("rig_hist", []).append(
            {"q": TPL.TEMPLATES[tpl]["title"], "summary":
             f"Built the standard {TPL.TEMPLATES[tpl]['title'].lower()} package — "
             f"{res['n_equipment']} items, {res['n_lines']} lines.",
             "source": "package"})
        st.session_state["rig_n"] = n + 1
        st.rerun()

    if q:
        hist = st.session_state.setdefault("rig_hist", [])
        t0 = time.time()
        with st.spinner(f"{model or 'the standard packages'} is laying out the sheet…"):
            try:
                chat = RigChat(model=model) if model else None
                if chat is not None:
                    out = chat.propose(q, current=st.session_state.get("rig_spec_dict"),
                                       history=hist)
                else:
                    nm = (TPL.match(q) or ["production_train"])[0]
                    out = {"spec": TPL.expand(nm), "source": "template",
                           "note": "Ollama unavailable."}
                raw = out["spec"]
                sp = SP.build(raw, auto=auto)
                close_reader()
                res = BLD.build_sheet(sp, write_graph=True)
                clear_manifest()
                st.session_state["rig_raw"] = raw
                st.session_state["rig_result"] = res
                st.session_state["rig_spec_dict"] = sp.dict()
                summary = (chat.describe(sp, res) if chat is not None else
                           f"Built {sp.title} from the standard package.")
                hist.append({"q": q, "summary": summary, "source": out["source"],
                             "note": out.get("note", ""),
                             "secs": round(time.time() - t0, 1)})
            except Exception as e:
                hist.append({"q": q, "summary": f"Could not build that — "
                                                f"{type(e).__name__}: {e}",
                             "source": "error", "note": "", "secs": 0})
        st.session_state["rig_n"] = n + 1
        st.rerun()

    # ---- conversation ----------------------------------------------------
    hist = st.session_state.get("rig_hist", [])
    for turn in reversed(hist[-6:]):
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            st.markdown(turn["summary"])
            chips = f'<span class="chip">◇ {turn["source"]}</span>'
            if turn.get("secs"):
                chips += f'<span class="chip g">{turn["secs"]}s</span>'
            st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
            if turn.get("note"):
                st.caption(turn["note"])

    out_panel = st.container(border=True)
    out_panel.__enter__()
    st.markdown(PANEL, unsafe_allow_html=True)

    res = st.session_state.get("rig_result")
    if not res:
        st.markdown('<div class="empty"><b>Nothing built yet</b>'
                    'Describe the sheet you want, or start from a standard package. '
                    'The drawing, its graph and its registers are generated together, '
                    'so they cannot disagree.</div>', unsafe_allow_html=True)
        with st.expander(f"Equipment catalogue — {len(CAT.CATALOG)} item types"):
            _tbl(CAT.as_rows(), height=420)
        out_panel.__exit__(None, None, None)
        return

    sp = SP.build(st.session_state["rig_raw"])

    # ---- what came out ---------------------------------------------------
    cells = [("equipment", res["n_equipment"]), ("valves", res["n_valves"]),
             ("instruments", res["n_instruments"]), ("lines", res["n_lines"]),
             ("graph nodes", res["n_entities"]), ("graph edges", res["n_relations"])]
    st.markdown('<div class="stat-grid">' + "".join(
        f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in cells) + '</div>', unsafe_allow_html=True)

    if res.get("graph_error"):
        st.warning(res["graph_error"])
        if st.button("Load it into the graph now", key="rig_retry_graph"):
            close_reader()
            try:
                BLD.load_graph(res["sheet_id"], res["entities"], res["relations"],
                               res["derived"])
                res["graph_error"] = ""
                clear_manifest()
                st.rerun()
            except RuntimeError as e:
                st.error(f"still locked — {e}")

    left, right = st.columns([2.1, 1], gap="medium")
    with left:
        img = cv2.imread(res["png"], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            st.image(cv2.resize(img, None, fx=0.30, fy=0.30),
                     use_container_width=True,
                     caption=f'{res["sheet_id"]} · {res["width"]}×{res["height"]} px')
            ok, buf = cv2.imencode(".png", img)
            if ok:
                st.download_button("Download the sheet (.png)", buf.tobytes(),
                                   f'{res["sheet_id"]}.png', "image/png",
                                   use_container_width=True, key="rig_dl_png")
    with right:
        st.markdown('<p class="eyebrow">Design review</p>', unsafe_allow_html=True)
        issues = res.get("issues") or []
        if not issues:
            st.success("No findings — relief, shutdown and instrumentation are complete.")
        else:
            head = "<tr><th>sev</th><th>code</th><th>finding</th></tr>"
            body = "".join(
                f'<tr><td class="m {"bad" if i["severity"]=="high" else "ok"}">'
                f'{i["severity"]}</td><td class="m">{i["code"]}</td>'
                f'<td>{i["finding"]}</td></tr>' for i in issues)
            st.markdown(f'<div class="tbl-wrap"><table class="tbl">'
                        f'<thead>{head}</thead><tbody>{body}</tbody></table></div>',
                        unsafe_allow_html=True)
            _dl(issues, "design_findings", "dl_hazop")

        st.markdown('<p class="eyebrow sp">Ask this sheet</p>', unsafe_allow_html=True)
        if res.get("graph_error"):
            st.caption("Not in the graph yet — load it above, then ask it questions "
                       "from the Workbench tab.")
        else:
            st.caption("It is in the graph already — pick it in the sidebar and ask "
                       "on the Workbench tab.")
        try:
            kc = get_kg().run(
                "MATCH (s:Symbol) WHERE s.sheet_id=$s RETURN s.category AS c, "
                "count(s) AS n ORDER BY n DESC", {"s": res["sheet_id"]})
            st.markdown('<div class="kv">' + "<br>".join(
                f'<b>{int(r.n)}</b> {r.c}' for r in kc.itertuples()) + '</div>',
                unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"graph read pending — {type(e).__name__}")

    # ---- the registers ---------------------------------------------------
    st.markdown('<p class="eyebrow sp">Drawing registers</p>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Equipment list", "Line list", "Valve list", "Instrument index",
         "Cause & effect", "Spec (editable)"])
    with t1:
        rows = SP.equipment_list(sp); _tbl(rows); _dl(rows, "equipment_list", "dl_eq")
    with t2:
        rows = SP.line_list(sp); _tbl(rows); _dl(rows, "line_list", "dl_ln")
    with t3:
        rows = SP.valve_list(sp); _tbl(rows); _dl(rows, "valve_list", "dl_vl")
    with t4:
        rows = SP.instrument_index(sp); _tbl(rows); _dl(rows, "instrument_index", "dl_ii")
    with t5:
        causes, effects, grid = SP.cause_effect(sp)
        st.caption("ESD cause & effect — X marks an action the cause initiates.")
        head = "<tr><th>cause</th>" + "".join(
            f"<th>{e}</th>" for e in effects[:14]) + "</tr>"
        body = "".join("<tr><td class='m'>" + c + "</td>" +
                       "".join(f"<td class='m ok'>{v}</td>" for v in row[:14]) + "</tr>"
                       for c, row in zip(causes, grid))
        st.markdown(f'<div class="tbl-wrap"><table class="tbl">'
                    f'<thead>{head}</thead><tbody>{body}</tbody></table></div>',
                    unsafe_allow_html=True)
        ce = [{"cause": c, **{e: v for e, v in zip(effects, row)}}
              for c, row in zip(causes, grid)]
        _dl(ce, "cause_and_effect", "dl_ce")
    with t6:
        st.caption("Edit the item list or the streams and rebuild. Tags, relief "
                   "devices, instruments and line numbers are regenerated.")
        raw = st.session_state["rig_raw"]
        eq_rows = [{"type": (e.get("type") if isinstance(e, dict) else e),
                    "label": (e.get("label", "") if isinstance(e, dict) else "")}
                   for e in raw.get("equipment", [])]
        st_rows = [{"from": s.get("from", ""), "to": s.get("to", ""),
                    "size": s.get("size", 6), "service": s.get("service", "P")}
                   for s in raw.get("streams", [])]
        ed_eq = st.data_editor(pd.DataFrame(eq_rows), num_rows="dynamic",
                               use_container_width=True, key="rig_ed_eq",
                               column_config={"type": st.column_config.SelectboxColumn(
                                   "type", options=sorted(CAT.CATALOG), width="medium")})
        ed_st = st.data_editor(pd.DataFrame(st_rows), num_rows="dynamic",
                               use_container_width=True, key="rig_ed_st")
        if st.button("Rebuild from the table", type="primary", key="rig_rebuild"):
            new = {"title": raw.get("title", "Oil rig process sheet"),
                   "facility": raw.get("facility", ""),
                   "notes": raw.get("notes", []),
                   "equipment": [{"type": r["type"], "label": r.get("label", "")}
                                 for r in ed_eq.to_dict("records") if r.get("type")],
                   "streams": [{"from": r["from"], "to": r["to"],
                                "size": r.get("size", 6),
                                "service": r.get("service", "P")}
                               for r in ed_st.to_dict("records")
                               if r.get("from") and r.get("to")]}
            with st.spinner("rebuilding…"):
                _build_and_ingest(new, close_reader, clear_manifest)
            st.rerun()
        st.download_button("Spec (.json)", json.dumps(sp.dict(), indent=1),
                           f'{res["sheet_id"]}_spec.json', "application/json",
                           key="dl_spec")

    with st.expander(f"Equipment catalogue — {len(CAT.CATALOG)} item types"):
        _tbl(CAT.as_rows(), height=420)
    out_panel.__exit__(None, None, None)
