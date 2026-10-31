#!/usr/bin/env python
"""Run the PIDQA benchmark against the built graph."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.edi.config import load
from src.edi.graph.kg import KG
from src.edi.eval.run_pidqa import evaluate_L1, evaluate_L3, save_report

CFG = load()


def _sheet_sort(name: str):
    """Dataset sheets are numeric ("0", "41"); uploaded ones are not
    ("u_pid_real_p1_52af52"). Sorting with int() crashes on the second kind."""
    return (0, int(name), "") if name.isdigit() else (1, 0, name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(CFG.get_path("graph.path")))
    ap.add_argument("--sheets", default="", help="comma list; default = all ingested")
    ap.add_argument("--per-type", type=int, default=32)
    ap.add_argument("--name", default="pidqa_L1")
    ap.add_argument("--l3", type=int, default=0,
                    help="also run end-to-end L3 on N questions per type (slow)")
    a = ap.parse_args()

    if a.sheets:
        sheets = [s.strip() for s in a.sheets.split(",") if s.strip()]
    else:
        # benchmark sheets only — uploaded drawings have no ground truth
        sheets = sorted([d.name for d in CFG.get_path("paths.interim").glob("*")
                         if (d / "relations.json").exists() and d.name.isdigit()],
                        key=_sheet_sort)
    kg = KG(a.graph, read_only=True)   # coexists with the running app
    rep = evaluate_L1(kg, sheets, per_type=a.per_type)
    p = save_report(rep, a.name)

    print(f"\n=== L1 · PIDQA reference Cypher on OUR graph ===")
    print(f"sheets={rep['sheets']}  questions={rep['n_questions']}  errors={rep['n_errors']}")
    for k, v in sorted(rep["by_type"].items()):
        print(f"  {k:20s} {v['accuracy']:.4f}   (n={v['n']})")
    print(f"  {'OVERALL':20s} {rep['overall_accuracy']:.4f}")
    print(f"report -> {p}")

    if a.l3:
        from src.edi.retrieval.fewshot import build_index
        from src.edi.qa.nl2cypher import CypherGenerator
        gen = CypherGenerator(build_index(exclude_sheets=set(sheets), per_type=250))
        r3 = evaluate_L3(kg, gen, sheets, per_type=a.l3)
        p3 = save_report(r3, a.name.replace("L1", "L3"))
        print(f"\n=== L3 · OUR generated Cypher on OUR graph (sampled) ===")
        print(f"questions={r3['n_questions']}  executable={r3['executable_query_rate']:.4f}")
        for k, v in sorted(r3["by_type"].items()):
            print(f"  {k:20s} {v['accuracy']:.4f}   (n={v['n']})")
        print(f"  {'OVERALL':20s} {r3['overall_accuracy']:.4f}")
        print(f"report -> {p3}")
    kg.close()

if __name__ == "__main__":
    main()
