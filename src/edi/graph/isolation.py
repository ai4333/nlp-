"""
Isolation planning — "which valves do I close to work on this?"

This is the question a technician actually asks, and until now the system could
not answer it at all: it generated a Cypher query against a `FEEDS_TO` relation
that is empty on real drawings, returned no rows, and said so.

It is a graph problem, not a language problem. Walk outwards from the item over
the pipe connections; on every branch, the first valve that can hold pressure is
the isolation point for that branch. Stop there — anything beyond it is outside
the isolation.

Three things this deliberately does:

* **It distinguishes valves that isolate from valves that do not.** A check
  valve stops reverse flow but will not hold for a permit. A control valve is
  positioned by the control system and can drift open. Neither is an isolation
  point, and offering one as if it were is the kind of error that gets somebody
  hurt.
* **It reports the branches it could NOT isolate.** A path that leaves the sheet
  or reaches other equipment without passing a block valve is an open route into
  the work, and saying "3 valves" while quietly dropping that branch would be
  worse than saying nothing.
* **It never claims to be a permit.** The output is what the drawing shows. The
  isolation that gets signed is a person's job, done on the plant.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# Valve types that can be closed and relied on to hold.
BLOCK_NAMES = ("gate", "ball", "globe", "plug", "butterfly", "block", "hand",
               "isolation", "shutdown", "on/off", "esd", "emergency shutdown")
# Valve types that must never be offered as an isolation point.
NOT_BLOCK_NAMES = ("check", "non-return", "nrv", "relief", "safety", "psv",
                   "rupture", "control valve", "anti-surge", "restriction")

# Tag prefixes that are block-capable even when the name is unhelpful.
BLOCK_PREFIX = ("HV", "XV", "SDV", "ESDV", "SSV", "MOV", "ROV", "BV", "GV")
NOT_BLOCK_PREFIX = ("PSV", "PRV", "RV", "ERV", "NRV", "RD", "RO", "BDV",
                    "FV", "PV", "LV", "TV", "CV", "FCV")


@dataclass
class Point:
    tag: str
    name: str
    entity_id: str
    hops: int
    via: list = field(default_factory=list)      # the path from the item


@dataclass
class Plan:
    target: str
    target_name: str
    isolation_valves: list = field(default_factory=list)
    open_branches: list = field(default_factory=list)   # no block valve found
    bleeds: list = field(default_factory=list)          # vents/drains inside
    relief: list = field(default_factory=list)          # relief devices inside
    crosses_sheet: bool = False
    notes: list = field(default_factory=list)

    def dict(self) -> dict:
        return {"target": self.target, "target_name": self.target_name,
                "isolation_valves": [v.__dict__ for v in self.isolation_valves],
                "open_branches": [b.__dict__ for b in self.open_branches],
                "bleeds": [b.__dict__ for b in self.bleeds],
                "relief": [r.__dict__ for r in self.relief],
                "crosses_sheet": self.crosses_sheet, "notes": self.notes}


def _is_block(tag: str, name: str, category: str) -> bool:
    if category != "valve":
        return False
    t, n = (tag or "").upper(), (name or "").lower()
    head = t.split("-")[0].rstrip("0123456789")
    if head in NOT_BLOCK_PREFIX or any(k in n for k in NOT_BLOCK_NAMES):
        return False
    if head in BLOCK_PREFIX or any(k in n for k in BLOCK_NAMES):
        return True
    # an unlabelled valve is treated as block-capable, but flagged in the notes
    return True


def _is_bleed(tag: str, name: str, category: str) -> bool:
    n = (name or "").lower()
    return category == "line" and any(k in n for k in ("drain", "vent"))


def _is_relief(tag: str, name: str, category: str) -> bool:
    head = (tag or "").upper().split("-")[0].rstrip("0123456789")
    return head in ("PSV", "PRV", "RV", "ERV", "RD", "BDV") or \
        any(k in (name or "").lower() for k in ("relief", "safety valve", "blowdown",
                                                "rupture"))


def plan(kg, sheet_id: str, tag: str, max_hops: int = 6) -> Plan | None:
    """Isolation points for `tag`, from what this sheet shows."""
    t = (tag or "").strip().upper()
    df = kg.run("MATCH (s:Symbol) WHERE s.sheet_id=$s AND toUpper(s.tag)=$t "
                "RETURN s.entity_id AS eid, s.tag AS tag, s.name AS nm, "
                "s.category AS cat LIMIT 1", {"s": sheet_id, "t": t})
    if not len(df):
        return None
    root = df.iloc[0]
    P = Plan(target=str(root.tag), target_name=str(root.nm))

    edges = kg.run(
        "MATCH (a:Symbol)-[:CONNECTED_TO]-(b:Symbol) "
        "WHERE a.sheet_id=$s AND b.sheet_id=$s "
        "RETURN a.entity_id AS a, b.entity_id AS b", {"s": sheet_id})
    nodes = kg.run(
        "MATCH (n:Symbol) WHERE n.sheet_id=$s RETURN n.entity_id AS eid, "
        "n.tag AS tag, n.name AS nm, n.category AS cat", {"s": sheet_id})
    info = {r.eid: (str(r.tag), str(r.nm), str(r.cat)) for r in nodes.itertuples()}
    adj: dict[str, set] = {e: set() for e in info}
    for r in edges.itertuples():
        if r.a in adj and r.b in adj:
            adj[r.a].add(r.b)
            adj[r.b].add(r.a)

    start = str(root.eid)
    seen = {start}
    frontier = [(n, [start]) for n in adj.get(start, ())]
    unlabelled = 0

    while frontier:
        nxt = []
        for eid, path in frontier:
            if eid in seen:
                continue
            seen.add(eid)
            tg, nm, cat = info.get(eid, ("", "", ""))
            route = [info[p][0] or "(untagged)" for p in path[1:]]

            if _is_relief(tg, nm, cat):
                P.relief.append(Point(tg, nm, eid, len(path), route))
                continue                        # a relief path is not isolated
            if _is_block(tg, nm, cat):
                if not tg or not nm:
                    unlabelled += 1
                P.isolation_valves.append(Point(tg or "(untagged valve)", nm, eid,
                                                len(path), route))
                continue                        # stop: beyond it is outside
            if _is_bleed(tg, nm, cat):
                P.bleeds.append(Point(tg, nm, eid, len(path), route))
            if cat == "connector":
                P.crosses_sheet = True
                P.open_branches.append(Point(tg or "(off-page)", nm or "off-page connector",
                                             eid, len(path), route))
                continue
            if len(path) >= max_hops:
                P.open_branches.append(Point(tg or "(untagged)", nm, eid, len(path), route))
                continue
            for n in adj.get(eid, ()):
                if n not in seen:
                    nxt.append((n, path + [eid]))
        frontier = nxt

    # dead ends that were never closed off by a valve
    if not P.isolation_valves:
        P.notes.append("No block valve was found on any branch of this item on this "
                       "sheet. Either the isolation is on a connected drawing, or the "
                       "drawing has not been read completely. Do not treat this as "
                       "'nothing to close'.")
    if P.open_branches:
        P.notes.append(f"{len(P.open_branches)} branch(es) leave this item without "
                       f"passing a block valve on this sheet — follow them onto the "
                       f"connected drawings before isolating.")
    if P.relief:
        P.notes.append("Relief and blowdown paths are listed separately and must not "
                       "be blocked in. They are normally car-sealed open.")
    if unlabelled:
        P.notes.append(f"{unlabelled} valve(s) on the isolation had no readable tag "
                       f"and are shown by position only — confirm them on the plant.")
    if P.crosses_sheet:
        P.notes.append("This isolation crosses a sheet boundary.")
    P.notes.append("This is what the drawing shows. It is not a permit and not an "
                   "isolation certificate — those are raised and proved on the plant.")
    return P
