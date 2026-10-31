#!/usr/bin/env python
"""End-to-end health check: every subsystem, one command."""
import sys, json, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, FAIL = "  ok  ", " FAIL "
results = []

def check(name, fn):
    t0 = time.time()
    try:
        detail = fn() or ""
        results.append((True, name, detail, time.time() - t0))
    except Exception as e:
        results.append((False, name, f"{type(e).__name__}: {e}"[:110], time.time() - t0))


from src.edi.config import load, yolo_to_pidqa
CFG = load()

def c_config():
    return f"tile {CFG.dotted('tiling.tile_size')} · relations {CFG.dotted('relations.mode')}"

def c_dataset():
    d = CFG.get_path("paths.dataset")
    n = len(list((d / "images" / "train").glob("*.jpg"))) + len(list((d / "images" / "val").glob("*.jpg")))
    assert n >= 500, f"only {n} sheets"
    return f"{n} sheets"

def c_pidqa():
    import pandas as pd
    df = pd.read_csv(CFG.get_path("paths.pidqa") / "Value/value_based.csv")
    return f"{len(df)} value rows"

def c_tiling():
    from src.edi.ingest.tiling import plan_tiles
    t = plan_tiles(7168, 4562, 1024, 768)
    assert len(t) == 54
    return f"{len(t)} tiles/sheet"

def c_interim():
    n = len([d for d in CFG.get_path("paths.interim").glob("*")
             if (d / "ocr.json").exists() and (d / "relations.json").exists()])
    assert n > 0
    return f"{n} sheets ingested"

def c_graph():
    from src.edi.graph.kg import KG
    kg = KG(CFG.get_path("graph.path"), read_only=True)
    c = kg.counts(); kg.close()
    assert c["symbols"] > 0
    return (f"{c['symbols']} symbols · {c['connected_to']} connected · "
            f"{c['controls']} controls · {c['feeds_to']} feeds")

def c_graph_concurrent():
    from src.edi.graph.kg import KG
    a = KG(CFG.get_path("graph.path"), read_only=True)
    b = KG(CFG.get_path("graph.path"), read_only=True)
    a.counts(); b.counts(); a.close(); b.close()
    return "two readers coexist"

def c_ocr():
    from src.edi.vision import ocr as O
    assert O.ORIENTATIONS == (0, -90)
    return f"orientations {O.ORIENTATIONS} · workers {CFG.dotted('ocr.workers')}"

def c_detector():
    w = Path(CFG.get_path("detection.weights"))
    if w.exists():
        return f"YOLO weights present ({w.stat().st_size//1024} KB)"
    return "GT replay only — no trained weights (uploads unavailable)"

def c_llm():
    import ollama
    c = ollama.Client(host=CFG.dotted("llm.host"))
    names = [m.model for m in c.list().models]
    need = [CFG.dotted("llm.cypher_model"), CFG.dotted("llm.vlm_model")]
    missing = [n for n in need if not any(x.startswith(n.split(":")[0]) for x in names)]
    assert not missing, f"missing {missing}"
    return f"{len(names)} models available"

def c_nl2cypher():
    from src.edi.retrieval.fewshot import build_index
    from src.edi.qa.nl2cypher import CypherGenerator, schema_prompt
    idx = build_index(per_type=60)
    g = CypherGenerator(idx)
    assert "CONTROLS" in schema_prompt()
    return f"{len(idx.examples)} few-shot examples"

def c_documents():
    """The document store is one of the three evidence legs.

    An empty store used to report "[ ok ] 0 chunks from 0 document(s)", which
    reads as healthy — but it means document evidence cannot be shown at all.
    Say so plainly instead of quietly passing.
    """
    from src.edi.retrieval.documents import DocumentStore
    ds = DocumentStore()
    n_docs, n_chunks = len(ds.stats()), len(ds.chunks)
    if n_docs == 0:
        return ("EMPTY — no reference documents indexed, so document evidence "
                "cannot be shown. Add one in the sidebar.")
    return f"{n_chunks} chunks from {n_docs} document(s)"

def c_export():
    from src.edi import export as EX
    sid = next(d.name for d in CFG.get_path("paths.interim").glob("*")
               if (d / "ocr.json").exists())
    e = json.load(open(CFG.get_path("paths.interim") / sid / "ocr.json"))["records"]
    r = json.load(open(CFG.get_path("paths.interim") / sid / "relations.json"))["relations"]
    b = EX.bundle(sid, e, r)
    assert len(b) > 500
    return f"{len(b)//1024} KB bundle"

def c_components():
    from src.edi.app import canvas as CV, graphview as GV
    from src.edi.ingest.acquire import load_any
    sid = next(d.name for d in CFG.get_path("paths.interim").glob("*")
               if (d / "ocr.json").exists())
    e = json.load(open(CFG.get_path("paths.interim") / sid / "ocr.json"))["records"]
    r = json.load(open(CFG.get_path("paths.interim") / sid / "relations.json"))["relations"]
    dp = CFG.get_path("paths.interim") / sid / "derived.json"
    dv = json.loads(dp.read_text()) if dp.exists() else None
    img = load_any(sid)
    h1 = CV.render(img, e, r, derived=dv)
    h2 = GV.render(e, r, img.shape[1], img.shape[0], derived=dv)
    assert "<svg" in h1 and "<svg" in h2
    return f"canvas {len(h1)//1024} KB · graph {len(h2)//1024} KB"

def c_reports():
    p = CFG.get_path("paths.reports") / "pidqa_L1.json"
    assert p.exists()
    d = json.loads(p.read_text())
    return f"L1 {d['overall_accuracy']} over {d['n_questions']} questions"

def c_flow():
    """Direction inference, scored against a sheet whose true flow is known."""
    from src.edi.graph import flow as FL
    from src.edi.rig import spec as SP, templates as TPL, build as BLD
    from src.edi.rig.layout import place
    sp = SP.build(TPL.expand("production_train"))
    ents = BLD.to_entities(sp, place(sp))
    rels, _ = BLD.to_relations(sp)
    rels = [r for r in rels if r.get("evidence") != "instrument connection"]
    truth = {}
    for st in sp.streams:
        lid = f"L{st.id}"
        for a, b in ((st.src, lid), (lid, st.dst), (st.src, st.dst)):
            truth[tuple(sorted((a, b)))] = (a, b)
    got = FL.infer(ents, rels)
    right = sum(1 for d in got if truth.get(tuple(sorted((d["source"], d["target"]))))
                == (d["source"], d["target"]))
    judged = sum(1 for d in got if tuple(sorted((d["source"], d["target"]))) in truth)
    prec = right / judged if judged else 0.0
    assert prec >= 0.85, f"flow precision fell to {prec:.3f}"
    pairs = {tuple(sorted((r["source"], r["target"]))) for r in rels}
    return f"precision {prec:.3f} · coverage {len(got)/len(pairs):.2f} on a known sheet"


def c_knowledge():
    """The knowledge layer must decode, and must refuse to guess."""
    from src.edi import knowledge as K
    from src.edi.knowledge import isa as ISA, glossary as G
    assert K.explain(K.strip_question("What does SDV mean?"))["name"] == "shutdown valve"
    assert ISA.decode("PSH-1101").name == "pressure switch, high"
    assert K.explain("frobnicator") is None
    assert K.explain("DC266")["kind"] == "line"
    return (f"{len(G.GLOSSARY)} glossary terms · {len(ISA.CONVENTION)} conventions · "
            f"{len(ISA.FIRST)}x{len(ISA.SUCCEEDING)} ISA letters")


def c_rig_builder():
    """Draw a package end to end without touching the graph."""
    from src.edi.rig import catalog as CAT, templates as TPL, spec as SP, layout as LO
    from src.edi.rig import render as RD
    sp = SP.build(TPL.expand("gas_compression"))
    img, boxes = RD.draw(sp)
    assert img.shape == (LO.SHEET_H, LO.SHEET_W)
    assert all(n.id in boxes for n in sp.all_nodes())
    return (f"{len(CAT.CATALOG)} item types · {len(TPL.TEMPLATES)} packages · "
            f"drew {len(sp.all_nodes())} items")


def c_app_import():
    import ast
    src = Path("src/edi/app/streamlit_app.py").read_text()
    ast.parse(src)
    return f"{len(src.splitlines())} lines parse"


for name, fn in [
    ("config", c_config), ("dataset", c_dataset), ("PIDQA benchmark", c_pidqa),
    ("tiling", c_tiling), ("ingested sheets", c_interim),
    ("knowledge graph", c_graph), ("graph concurrency", c_graph_concurrent),
    ("OCR settings", c_ocr), ("detector", c_detector),
    ("local LLM", c_llm), ("NL→Cypher", c_nl2cypher),
    ("document store", c_documents), ("export", c_export),
    ("UI components", c_components), ("eval report", c_reports),
    ("knowledge layer", c_knowledge), ("flow direction", c_flow), ("rig builder", c_rig_builder), ("app module", c_app_import),
]:
    check(name, fn)

print()
for ok, name, detail, dt in results:
    print(f"[{OK if ok else FAIL}] {name:20s} {detail}")
bad = [r for r in results if not r[0]]
print(f"\n{len(results)-len(bad)}/{len(results)} subsystems healthy")
sys.exit(1 if bad else 0)
