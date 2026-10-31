"""
Flow direction — which way the fluid actually goes.

Pipe tracing only ever produces "these two are plumbed together". Direction is
the thing every operator question needs and the one blueprint mode never had:
`FEEDS_TO` was empty on every real drawing, so *"where does this go"*, *"what is
upstream of the pump"* and *"what relieves to the flare"* all returned nothing.

The trained path derives direction from detected flow-arrow symbols. A customer's
drawing has no trained detector, so that route is closed. But a P&ID states
direction in several other ways that need no model at all, and they are what an
engineer actually reads:

    R1  an off-page connector says it outright — "TO 3RD STAGE" is an outlet,
        "FROM LP CONDENSATE" is an inlet
    R2  a relief device only ever flows one way: protected item → PSV → flare
    R3  the service code carries intent — a flare or relief line runs toward the
        flare, a drain line runs toward the drain header, a utility line runs
        away from its header
    R4  a machine has a through-flow: if something enters a pump, something
        leaves it
    R5  conservation: at a node where every connection but one is already
        directed, the last one is forced

R1-R3 are seeds; R4-R5 propagate them along the chain, which is how a few
labelled connectors at the sheet edge end up orienting a whole train.

Every edge records the rule that produced it and a confidence that decays with
propagation distance, so a wrong direction can be traced to the rule that made
it rather than argued about. Nothing here guesses: an edge with no evidence
stays undirected, and an edge that would contradict one already placed is
dropped.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

# service codes whose direction is implied by what they are
TO_FLARE = {"FL", "RL", "VL", "VA", "VH"}      # runs toward a flare or vent
# RL is the tailpipe out of a relief device; FL is the header it joins. Keeping
# them apart is what separates a PSV's inlet from its outlet.
TO_DRAIN = {"DC", "DO", "CD"}                  # runs toward a drain header
FROM_UTILITY = {"IA", "UA", "N", "FG", "CW", "HW", "ST", "CH", "MO", "GL"}

RELIEF_PREFIX = ("PSV", "PRV", "RV", "ERV", "RD", "BDV")
MACHINE_WORDS = ("pump", "compressor", "blower", "fan", "turbine")
# Deciding a direction is not "sink or not sink". A flare knockout drum is
# DOWNSTREAM of every relief valve and UPSTREAM of the stack, so asking whether
# it is a sink gives the wrong answer half the time either way. What matters is
# how far along the disposal system each end of a line sits:
#
#   rank 2  the terminal — stack, caisson, atmosphere, a named header, an
#           off-page connector that says TO
#   rank 1  equipment in that system on the way there — a flare KO drum, a
#           slop tank, a drain drum
#   rank 0  anything else — the plant the relief protects
#
# Flow in a disposal line always runs from the lower rank to the higher one.
FLARE_TERMINAL = ("stack", "atmosphere", "flare header", "flare tip",
                  "vent header", "to flare", "to vent")
FLARE_WORDS = ("flare", "vent")
DRAIN_TERMINAL = ("drain header", "caisson", "to drain", "closed drains",
                  "open drain")
DRAIN_WORDS = ("drain", "slop", "sump")
MACHINE_WORDS = ("pump", "compressor", "blower", "fan", "turbine")

CONF = {"connector": 0.95, "relief": 0.90, "service": 0.80,
        "machine": 0.70, "conservation": 0.60, "layout": 0.50}


@dataclass
class _N:
    id: str
    tag: str
    name: str
    category: str
    service: str = ""
    connector_dir: str = ""      # out | in | both
    cx: float = 0.0
    cy: float = 0.0


def _service_of(e: dict) -> str:
    ln = e.get("line")
    if isinstance(ln, dict) and ln.get("service"):
        return str(ln["service"]).upper()
    if e.get("service"):
        return str(e["service"]).upper()
    m = re.match(r'^\s*\d*\.?\d*"?\s*-\s*([A-Z]{1,3})\s*-', (e.get("tag") or "").upper())
    return m.group(1) if m else ""


def _is_relief(n: _N) -> bool:
    head = n.tag.upper().split("-")[0].rstrip("0123456789")
    return head in RELIEF_PREFIX or any(
        k in n.name.lower() for k in ("relief", "safety valve", "blowdown", "rupture"))


def _is_check(n: _N) -> bool:
    head = n.tag.upper().split("-")[0].rstrip("0123456789")
    return head in ("NRV", "CV") and "control" not in n.name.lower() or \
        any(k in n.name.lower() for k in ("check valve", "non-return"))


def _is_machine(n: _N) -> bool:
    return any(k in n.name.lower() for k in MACHINE_WORDS)


def _rank(n: _N, terminal: tuple, words: tuple) -> int:
    """How far along a disposal system this node sits. 0 means not in it."""
    t = f"{n.tag} {n.name}".lower()
    if any(k in t for k in terminal):
        return 2
    if any(re.search(rf"\b{w}\b", t) for w in words):
        # a bare "flare"/"vent" on a connector is the destination itself
        return 2 if n.category == "connector" else 1
    return 0


def flare_rank(n: _N) -> int:
    return _rank(n, FLARE_TERMINAL, FLARE_WORDS)


def drain_rank(n: _N) -> int:
    return _rank(n, DRAIN_TERMINAL, DRAIN_WORDS)


def effective_rank(nid: str, nodes, adj, rank_fn, depth: int = 2,
                   avoid: set | None = None) -> int:
    """A node's rank, looking THROUGH inline valves and lines.

    An ESD valve spliced into a flare header is rank 0 on its own, which made
    the header appear to run backwards into it. What matters is what the valve
    leads to.
    """
    # `avoid` is the node we are reasoning ABOUT. Without it the walk goes out
    # one side of a relief valve and back in the other, and every neighbour
    # inherits the flare header's rank — which made both sides look downstream.
    best = rank_fn(nodes[nid])
    frontier, seen = [nid], {nid} | set(avoid or ())
    for _ in range(depth):
        nxt = []
        for cur in frontier:
            for m in adj.get(cur, ()):
                if m in seen:
                    continue
                seen.add(m)
                best = max(best, rank_fn(nodes[m]))
                if nodes[m].category in ("valve", "line"):
                    nxt.append(m)
        frontier = nxt
    return best


def _orient_by_rank(nodes, adj, place, line: _N, rank_fn, rule: str, conf: float):
    """A disposal line runs from its low-ranked end to its high-ranked end."""
    nbrs = adj[line.id]
    if len(nbrs) < 2:
        return
    rk = {m: effective_rank(m, nodes, adj, rank_fn, avoid={line.id}) for m in nbrs}
    ranked = sorted(nbrs, key=lambda m: rk[m])
    lo, hi = ranked[0], ranked[-1]
    if rk[hi] <= rk[lo]:
        return                       # no gradient: nothing to say
    place(lo, line.id, rule, conf)
    place(line.id, hi, rule, conf)
    for m in nbrs:
        if m not in (lo, hi):
            place(m, line.id, rule, conf * 0.9)


def infer(entities: list[dict], relations: list[dict],
          max_rounds: int = 12, layout_hint: bool = True) -> list[dict]:
    """Directed FEEDS_TO edges, each carrying the rule that produced it."""
    nodes: dict[str, _N] = {}
    for e in entities:
        cx = (e.get("connector") or {})
        bb = e.get("bbox") or [0, 0, 0, 0]
        nodes[e["id"]] = _N(id=e["id"], tag=str(e.get("tag") or ""),
                            name=str(e.get("name") or ""),
                            category=str(e.get("category") or ""),
                            service=_service_of(e),
                            connector_dir=str(cx.get("direction") or ""),
                            cx=(bb[0] + bb[2]) / 2.0, cy=(bb[1] + bb[3]) / 2.0)

    undirected: list[tuple[str, str]] = []
    seen = set()
    for r in relations:
        a, b = r.get("source"), r.get("target")
        if a not in nodes or b not in nodes or a == b:
            continue
        k = tuple(sorted((a, b)))
        if k in seen:
            continue
        seen.add(k)
        undirected.append((a, b))

    adj: dict[str, list[str]] = {i: [] for i in nodes}
    for a, b in undirected:
        adj[a].append(b)
        adj[b].append(a)

    # decided[(a,b) sorted] = (source, target, rule, confidence)
    decided: dict[tuple, tuple] = {}

    def place(src: str, dst: str, rule: str, conf: float) -> bool:
        k = tuple(sorted((src, dst)))
        if k not in seen:
            return False
        cur = decided.get(k)
        if cur is not None:
            # already oriented; a contradiction is dropped rather than overwritten
            return cur[0] == src
        decided[k] = (src, dst, rule, round(conf, 3))
        return True

    # ── R1 · off-page connectors state the direction outright
    for n in nodes.values():
        if n.category != "connector" or n.connector_dir not in ("out", "in"):
            continue
        for m in adj[n.id]:
            if n.connector_dir == "out":
                place(m, n.id, "connector says TO", CONF["connector"])
            else:
                place(n.id, m, "connector says FROM", CONF["connector"])

    # ── R2 · a relief device flows one way only: protected system in, flare out
    for n in nodes.values():
        if not _is_relief(n):
            continue
        nbrs = adj[n.id]
        if not nbrs:
            continue
        # The protected vessel is often itself part of the flare system — a PSV
        # on a flare KO drum — so "is it flare-ish" cannot separate inlet from
        # outlet. What can: the outlet side leads FURTHER into the disposal
        # system than the inlet side does.
        rk = {m: effective_rank(m, nodes, adj, flare_rank, avoid={n.id}) for m in nbrs}
        rl = {m: (nodes[m].category == "line" and nodes[m].service == "RL")
              for m in nbrs}
        hi = max(rk.values())
        for m in nbrs:
            # every neighbour that is a tailpipe, or that sits as deep in the
            # disposal system as the deepest one, is downstream — a relief valve
            # can have more than one outlet connection recorded
            if rl[m] or (hi > 0 and rk[m] == hi):
                place(n.id, m, "a relief valve discharges to the flare system",
                      CONF["relief"])
            else:
                place(m, n.id, "a relief valve protects the system upstream of it",
                      CONF["relief"])

    # ── R3 · the service code carries intent
    for n in nodes.values():
        if n.category != "line" or not n.service:
            continue
        if n.service in TO_FLARE:
            _orient_by_rank(nodes, adj, place, n, flare_rank,
                            f"{n.service} line runs toward the flare system",
                            CONF["service"])
        elif n.service in TO_DRAIN:
            _orient_by_rank(nodes, adj, place, n, drain_rank,
                            f"{n.service} line runs toward the drain system",
                            CONF["service"])
        elif n.service in FROM_UTILITY:
            for m in adj[n.id]:
                if nodes[m].category == "connector":
                    place(m, n.id, f"{n.service} is a utility supply",
                          CONF["service"])

    # ── R6 · a check valve next to a machine sits on the discharge
    # A non-return valve exists to stop reverse flow through a pump or a
    # compressor, so it is always on the outlet side. That one fact seeds a
    # direction into the middle of a process train, which nothing else does.
    for n in nodes.values():
        if not _is_check(n):
            continue
        machines = [m for m in adj[n.id] if _is_machine(nodes[m])]
        if len(machines) != 1:
            continue
        # Only the machine-to-valve edge is certain. Fanning the direction out
        # to the valve's other neighbours guessed at the inlet line as well and
        # was wrong half the time; conservation propagates it correctly instead.
        place(machines[0], n.id, "a check valve sits on a machine's discharge",
              CONF["machine"])

    def propagate(rounds: int):
        _propagate(nodes, adj, decided, place, rounds)

    propagate(max_rounds)

    # ── R7 · last resort: a P&ID reads left to right
    # Feed enters on the left, product leaves on the right. It is the strongest
    # layout convention there is and the weakest piece of evidence in this file,
    # so it is applied ONLY to connections nothing else could decide, only where
    # the run is predominantly horizontal, and at a confidence that says so.
    if layout_hint:
        for a, b in undirected:
            if tuple(sorted((a, b))) in decided:
                continue
            na, nb = nodes[a], nodes[b]
            if na.category not in DRAWN or nb.category not in DRAWN:
                continue
            dx, dy = nb.cx - na.cx, nb.cy - na.cy
            if abs(dx) < 60 or abs(dx) <= abs(dy):
                continue
            lo, hi = (a, b) if dx > 0 else (b, a)
            place(lo, hi, "a P&ID reads left to right", CONF["layout"])
        propagate(max_rounds)

    return [{"source": s, "target": t, "type": "FEEDS_TO",
             "evidence": rule, "confidence": conf}
            for (s, t, rule, conf) in decided.values()]


DRAWN = ("equipment", "valve", "line", "connector")


def _propagate(nodes, adj, decided, place, max_rounds: int):
    """Push known directions along the chain: machines pass flow through, and a
    node whose every other connection is decided has its last one forced."""
    for rnd in range(max_rounds):
        changed = 0
        decay = 0.95 ** (rnd + 1)
        for n in nodes.values():
            nbrs = adj[n.id]
            if len(nbrs) < 2:
                continue
            ins, outs, free = [], [], []
            for m in nbrs:
                d = decided.get(tuple(sorted((n.id, m))))
                if d is None:
                    free.append(m)
                elif d[1] == n.id:
                    ins.append(m)
                else:
                    outs.append(m)
            if not free:
                continue

            # a machine with flow in and nothing out must discharge
            if _is_machine(n) and len(free) == 1:
                if ins and not outs:
                    changed += place(n.id, free[0], "a machine discharges what it takes in",
                                     CONF["machine"] * decay)
                    continue
                if outs and not ins:
                    changed += place(free[0], n.id, "a machine takes in what it discharges",
                                     CONF["machine"] * decay)
                    continue

            # conservation: one undecided connection, and the node is not
            # balanced yet
            if len(free) == 1:
                if ins and not outs:
                    changed += place(n.id, free[0],
                                     "everything else at this point flows in",
                                     CONF["conservation"] * decay)
                elif outs and not ins:
                    changed += place(free[0], n.id,
                                     "everything else at this point flows out",
                                     CONF["conservation"] * decay)
        if not changed:
            break


def coverage(entities: list[dict], relations: list[dict],
             directed: list[dict]) -> dict:
    """How much of the sheet's plumbing now has a direction."""
    seen = {tuple(sorted((r["source"], r["target"]))) for r in relations
            if r.get("source") != r.get("target")}
    got = {tuple(sorted((d["source"], d["target"]))) for d in directed}
    by_rule: dict[str, int] = {}
    for d in directed:
        by_rule[d["evidence"]] = by_rule.get(d["evidence"], 0) + 1
    return {"connections": len(seen), "directed": len(got),
            "fraction": round(len(got) / len(seen), 3) if seen else 0.0,
            "by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1]))}
