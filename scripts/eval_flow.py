#!/usr/bin/env python
"""
Measure flow-direction inference against known-true drawings.

Direction on a real drawing has no ground truth — nobody has labelled which way
the fluid goes on an offshore P&ID, and asking a model to score it would only
measure the model. The rig builder solves that: it generates a sheet FROM a
spec, so the true direction of every connection is known by construction.

So this harness feeds the generated sheet's entities and undirected connections
into `graph.flow.infer` — exactly what it gets on a customer's drawing, with the
answers withheld — and scores what comes back.

    coverage   how many connections got a direction at all
    precision  of those, how many point the right way
    recall     of the true flow paths, how many were recovered

Run:  python scripts/eval_flow.py            (all packages)
      python scripts/eval_flow.py --package gas_compression --detail
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.edi.rig import spec as SP, templates as TPL, build as BLD
from src.edi.rig.layout import place
from src.edi.graph import flow as FL


def truth_for(sp) -> dict[tuple, tuple]:
    """Every connection the drawing makes, with the direction the spec says."""
    out = {}
    for s in sp.streams:
        lid = f"L{s.id}"
        for a, b in ((s.src, lid), (lid, s.dst), (s.src, s.dst)):
            out[tuple(sorted((a, b)))] = (a, b)
    return out


def score_one(name: str, detail: bool = False, layout: bool = True) -> dict:
    sp = SP.build(TPL.expand(name))
    boxes = place(sp)
    ents = BLD.to_entities(sp, boxes)
    rels, _derived = BLD.to_relations(sp)
    # instrument links are signal, not flow — the same filter the real path uses
    rels = [r for r in rels if r.get("evidence") != "instrument connection"]

    truth = truth_for(sp)
    got = FL.infer(ents, rels, layout_hint=layout)
    cov = FL.coverage(ents, rels, got)

    right = wrong = unknown_pair = 0
    misses = []
    by = {e["id"]: e for e in ents}
    for d in got:
        k = tuple(sorted((d["source"], d["target"])))
        t = truth.get(k)
        if t is None:
            unknown_pair += 1
            continue
        if (d["source"], d["target"]) == t:
            right += 1
        else:
            wrong += 1
            if detail:
                a, b = by[d["source"]], by[d["target"]]
                misses.append(f'  WRONG  {a["tag"]:<24} -> {b["tag"]:<24} '
                              f'[{d["evidence"]}]')

    judged = right + wrong
    prec = right / judged if judged else 0.0
    recall = right / len(truth) if truth else 0.0
    if detail:
        for m in misses[:20]:
            print(m)
    return {"package": name, "connections": cov["connections"],
            "directed": cov["directed"], "coverage": cov["fraction"],
            "right": right, "wrong": wrong, "unscored": unknown_pair,
            "precision": round(prec, 3), "recall": round(recall, 3),
            "by_rule": cov["by_rule"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", help="score one package only")
    ap.add_argument("--detail", action="store_true", help="print the wrong ones")
    ap.add_argument("--rules", action="store_true", help="print the rule histogram")
    ap.add_argument("--no-layout", action="store_true",
                    help="turn off the left-to-right convention. Generated sheets "
                         "are laid out left to right by construction, so that rule "
                         "flatters itself here — this is the conservative number.")
    a = ap.parse_args()

    names = [a.package] if a.package else list(TPL.TEMPLATES)
    rows = [score_one(n, a.detail, layout=not a.no_layout) for n in names]

    w = max(len(r["package"]) for r in rows) + 2
    print(f'{"package":<{w}}{"conns":>7}{"directed":>10}{"cover":>8}'
          f'{"right":>7}{"wrong":>7}{"prec":>7}{"recall":>8}')
    for r in rows:
        print(f'{r["package"]:<{w}}{r["connections"]:>7}{r["directed"]:>10}'
              f'{r["coverage"]:>8.2f}{r["right"]:>7}{r["wrong"]:>7}'
              f'{r["precision"]:>7.3f}{r["recall"]:>8.3f}')

    tot_c = sum(r["connections"] for r in rows)
    tot_d = sum(r["directed"] for r in rows)
    tot_r = sum(r["right"] for r in rows)
    tot_w = sum(r["wrong"] for r in rows)
    judged = tot_r + tot_w
    print(f'\n{"OVERALL":<{w}}{tot_c:>7}{tot_d:>10}{tot_d/tot_c:>8.2f}'
          f'{tot_r:>7}{tot_w:>7}{tot_r/judged if judged else 0:>7.3f}')

    if a.rules:
        agg = {}
        for r in rows:
            for k, v in r["by_rule"].items():
                agg[k] = agg.get(k, 0) + v
        print("\nrules that fired:")
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>5}  {k}")


if __name__ == "__main__":
    main()
