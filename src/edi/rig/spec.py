"""
The rig spec — the structured thing in the middle.

The language model does NOT draw. It emits a spec: which items exist and what
flows into what. Everything after that is deterministic — tagging, instrument
loops, relief paths, line numbers, layout, the drawing, the graph. That split is
the whole design:

  * the model does the part it is good at (turning "three-stage separation with
    flare and produced-water treatment" into a list of items and streams),
  * and never the part it is bad at (drawing valid geometry, keeping tags
    unique, remembering that every vessel needs a PSV).

Because the drawing and the graph are both generated from this one object, they
can never disagree — the answer to "how many instruments are on this sheet" is
true by construction, not by OCR.

Normalisation adds what a real drawing has and a bare item list does not:
ISA tags, instrument loops per equipment type, relief valves routed to flare,
check valves downstream of pumps, anti-surge recycle around compressors, line
numbers in EPC format, and hazardous-area classification.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict

from . import catalog as CAT

# unit number per system — "V-1101" reads as unit 11, item 01
UNIT = {"wells": 10, "separation": 11, "compression": 12, "dehydration": 13,
        "export": 14, "flare": 15, "water": 16, "utilities": 17, "power": 18,
        "drains": 19, "drilling": 20, "safety": 21, "process": 11}


@dataclass
class Node:
    id: str
    tag: str
    item: str                 # catalogue key
    name: str
    category: str             # equipment | valve | instrument | connector
    symbol: str
    system: str = "process"
    label: str = ""           # printed description under the tag
    parent: str = ""          # instruments and relief valves hang off equipment
    design_p: float = 0.0
    design_t: float = 0.0
    zone: str = "Zone 2"
    duty: str = ""
    auto: bool = False        # added by normalisation, not asked for
    col: int = 0              # layout grid
    row: int = 0

    def dict(self) -> dict:
        return asdict(self)


@dataclass
class Stream:
    id: str
    src: str                  # node id
    dst: str                  # node id
    size_in: float = 6.0
    service: str = "P"
    area: str = "11"
    number: str = "001"
    spec: str = "P1A"
    insulation: str = ""
    fluid: str = ""
    auto: bool = False

    @property
    def tag(self) -> str:
        s = f'{self.size_in:g}"-{self.service}-{self.area}-{self.number}-{self.spec}'
        return s + (f"-{self.insulation}" if self.insulation else "")

    def dict(self) -> dict:
        d = asdict(self); d["tag"] = self.tag; return d


@dataclass
class Spec:
    title: str = "Oil rig process sheet"
    drawing_no: str = "RIG-AA-0000-PR-PID-0001"
    facility: str = "Offshore production platform"
    revision: str = "A"
    area: str = "11"
    nodes: list[Node] = field(default_factory=list)
    streams: list[Stream] = field(default_factory=list)
    controls: list[tuple] = field(default_factory=list)     # (instrument id, target id)
    connectors: list[Node] = field(default_factory=list)    # off-page, kept apart
    notes: list[str] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    prompt: str = ""

    # ---- lookups
    def by_id(self, nid: str) -> Node | None:
        for n in self.all_nodes():
            if n.id == nid:
                return n
        return None

    def by_tag(self, tag: str) -> Node | None:
        t = (tag or "").strip().upper()
        for n in self.all_nodes():
            if n.tag.upper() == t:
                return n
        return None

    def all_nodes(self) -> list[Node]:
        return self.nodes + self.connectors

    def equipment(self) -> list[Node]:
        return [n for n in self.nodes if n.category == "equipment"]

    def of_category(self, c: str) -> list[Node]:
        return [n for n in self.all_nodes() if n.category == c]

    def dict(self) -> dict:
        return {"title": self.title, "drawing_no": self.drawing_no,
                "facility": self.facility, "revision": self.revision,
                "area": self.area, "prompt": self.prompt,
                "nodes": [n.dict() for n in self.nodes],
                "connectors": [n.dict() for n in self.connectors],
                "streams": [s.dict() for s in self.streams],
                "controls": [list(c) for c in self.controls],
                "notes": self.notes, "issues": self.issues}


# ───────────────────────────────────────────────────────────── tag allocation
class _Tagger:
    """Hands out conventional tags and never repeats one."""

    def __init__(self):
        self.seq: dict[tuple, int] = {}
        self.taken: set[str] = set()

    def take(self, tag: str) -> str:
        t = tag.strip().upper()
        if t and t not in self.taken:
            self.taken.add(t)
            return t
        # a caller-supplied duplicate gets a suffix rather than being dropped
        for suf in "ABCDEFGH":
            if f"{t}{suf}" not in self.taken:
                self.taken.add(f"{t}{suf}")
                return f"{t}{suf}"
        return t

    def next(self, prefix: str, system: str) -> str:
        unit = UNIT.get(system, 11)
        k = (prefix, unit)
        while True:
            self.seq[k] = self.seq.get(k, 0) + 1
            t = f"{prefix}-{unit}{self.seq[k]:02d}"
            if t not in self.taken:
                self.taken.add(t)
                return t

    def loop(self, code: str, parent_tag: str) -> str:
        """An instrument carries its parent's loop number: PT-1101 on V-1101."""
        m = re.search(r"(\d{3,5})", parent_tag or "")
        num = m.group(1) if m else "1101"
        base = f"{code}-{num}"
        if base not in self.taken:
            self.taken.add(base)
            return base
        for suf in "ABCDEFGHJK":
            if f"{base}{suf}" not in self.taken:
                self.taken.add(f"{base}{suf}")
                return f"{base}{suf}"
        return base


# ───────────────────────────────────────────────────────────── building
def _node_from_item(it: CAT.Item, nid: str, tag: str, label: str = "",
                    auto: bool = False, parent: str = "") -> Node:
    return Node(id=nid, tag=tag, item=it.key, name=it.name, category=it.category,
                symbol=it.symbol, system=it.system, label=label or it.name.upper(),
                parent=parent, design_p=it.design_p, design_t=it.design_t,
                zone=it.zone, duty=it.duty, auto=auto)


def build(raw: dict, auto: bool = True) -> Spec:
    """Turn a raw (model- or template-produced) dict into a complete Spec."""
    sp = Spec(
        title=str(raw.get("title") or "Oil rig process sheet")[:70],
        drawing_no=str(raw.get("drawing_no") or "RIG-AA-0000-PR-PID-0001")[:40],
        facility=str(raw.get("facility") or "Offshore production platform")[:60],
        revision=str(raw.get("revision") or "A")[:3],
        notes=[str(n)[:120] for n in (raw.get("notes") or [])][:8],
        prompt=str(raw.get("prompt") or "")[:400],
    )
    tg = _Tagger()

    # pre-claim any tag the caller supplied so auto-numbering never collides
    for e in (raw.get("equipment") or []):
        if isinstance(e, dict) and e.get("tag"):
            tg.taken.add(str(e["tag"]).strip().upper())

    alias: dict[str, str] = {}          # every name the caller might use -> node id
    n_i = 0
    for e in (raw.get("equipment") or []):
        if isinstance(e, str):
            e = {"type": e}
        key = str(e.get("type") or e.get("item") or e.get("name") or "").strip()
        it = CAT.find(key)
        if it is None:
            continue
        nid = f"n{n_i:03d}"; n_i += 1
        tag = str(e.get("tag") or "").strip().upper()
        tag = tg.take(tag) if tag else tg.next(it.tag_prefix, it.system)
        node = _node_from_item(it, nid, tag, str(e.get("label") or "").upper()[:34])
        if e.get("design_p"):
            try: node.design_p = float(e["design_p"])
            except Exception: pass
        sp.nodes.append(node)
        for a in {key.lower(), tag.lower(), it.key, it.name.lower(),
                  str(e.get("label") or "").lower(), str(e.get("id") or "").lower()}:
            if a:
                alias.setdefault(a, nid)

    if not sp.nodes:
        return sp

    sp.area = str(UNIT.get(sp.nodes[0].system, 11))

    def resolve(ref) -> str | None:
        r = str(ref or "").strip().lower()
        if not r:
            return None
        if r in alias:
            return alias[r]
        for n in sp.nodes:
            if n.id == r or n.tag.lower() == r:
                return n.id
        hit = CAT.find(r)
        if hit:
            for n in sp.nodes:
                if n.item == hit.key:
                    return n.id
        return None

    # ---- streams -------------------------------------------------------
    seen_pair = set()
    for i, s in enumerate(raw.get("streams") or raw.get("lines") or []):
        if not isinstance(s, dict):
            continue
        a = resolve(s.get("from") or s.get("source") or s.get("src"))
        b = resolve(s.get("to") or s.get("target") or s.get("dst"))
        if not a or not b or a == b or (a, b) in seen_pair:
            continue
        seen_pair.add((a, b))
        na = sp.by_id(a)
        svc = str(s.get("service") or "").strip().upper()
        if svc not in CAT.PIPE_SPEC:
            svc = CAT.DEFAULT_SERVICE.get(na.system if na else "process", "P")
        try:
            size = float(s.get("size") or s.get("size_in") or CAT.SIZE_BY_SERVICE.get(svc, 6))
        except Exception:
            size = CAT.SIZE_BY_SERVICE.get(svc, 6)
        sp.streams.append(Stream(
            id=f"s{i:03d}", src=a, dst=b, size_in=round(max(0.5, min(48, size)), 1),
            service=svc, area=str(UNIT.get(na.system if na else "process", 11)),
            number=f"{100 + i:03d}", spec=CAT.PIPE_SPEC.get(svc, "P1A"),
            insulation="HE" if svc in ("PG", "ST", "HW") else "",
            fluid=str(s.get("fluid") or "")[:20]))

    # a spec with items but no stated flow is still useful: chain them in order
    if not sp.streams and len(sp.nodes) > 1:
        for i in range(len(sp.nodes) - 1):
            svc = CAT.DEFAULT_SERVICE.get(sp.nodes[i].system, "P")
            sp.streams.append(Stream(id=f"s{i:03d}", src=sp.nodes[i].id,
                                     dst=sp.nodes[i + 1].id,
                                     size_in=CAT.SIZE_BY_SERVICE.get(svc, 6),
                                     service=svc, area=sp.area,
                                     number=f"{100 + i:03d}",
                                     spec=CAT.PIPE_SPEC.get(svc, "P1A"), auto=True))

    # ---- caller-supplied instruments -----------------------------------
    for j, ins in enumerate(raw.get("instruments") or []):
        if isinstance(ins, str):
            ins = {"type": ins}
        code = str(ins.get("type") or ins.get("code") or "").strip().upper()
        if code not in CAT.INSTRUMENTS:
            continue
        host = resolve(ins.get("on") or ins.get("parent") or ins.get("equipment"))
        if host is None:
            continue
        hn = sp.by_id(host)
        nid = f"i{j:03d}"
        nm, _ = CAT.INSTRUMENTS[code]
        sp.nodes.append(Node(id=nid, tag=tg.loop(code, hn.tag), item=code.lower(),
                             name=nm, category="instrument", symbol="bubble",
                             system=hn.system, label=code, parent=host,
                             zone=hn.zone))
        sp.controls.append((nid, host))

    if auto:
        _autofit(sp, tg)
    _renumber_streams(sp)
    sp.issues = hazop(sp)
    return sp


def _renumber_streams(sp: Spec):
    for i, s in enumerate(sp.streams):
        s.number = f"{100 + i:03d}"


# ───────────────────────────────────────────────── automatic completion
def _autofit(sp: Spec, tg: _Tagger):
    """Add what a real P&ID has and an item list does not.

    Order matters: relief devices first (they create the flare demand), then the
    flare destination, then instrument loops, then the inline valves.
    """
    added: list[Node] = []
    new_streams: list[Stream] = []
    equip = [n for n in sp.nodes if n.category == "equipment"]
    n_ai = 0

    # 1 · relief protection for every pressure envelope
    # Prefer a real terminal. A knockout drum is only a fallback, and a relief
    # valve must never be routed into the vessel it is protecting — which is
    # what happened on the flare-and-relief package, where the KO drum was both
    # the protected item and the destination.
    def _flare_dest(exclude: str | None = None):
        for kinds in (("flare_stack", "vent_stack"),
                      ("hp_flare_ko", "lp_flare_ko")):
            for n in equip:
                if n.item in kinds and n.id != exclude:
                    return n
        return None

    flare_dest = _flare_dest()
    psv_hosts = [n for n in equip if n.symbol in CAT.NEEDS_PSV and n.design_p >= 3]
    if psv_hosts and flare_dest is None:
        cx = Node(id="x000", tag="HP FLARE HEADER", item="flare_header",
                  name="off-page connector", category="connector", symbol="offpage",
                  system="flare", label="TO HP FLARE HEADER", auto=True, zone="Zone 1")
        sp.connectors.append(cx)
        flare_dest = cx
    for h in psv_hosts:
        dest = _flare_dest(exclude=h.id) or flare_dest
        if dest is not None and dest.id == h.id:
            dest = None
        it = CAT.CATALOG["psv"]
        nid = f"a{n_ai:03d}"; n_ai += 1
        psv = _node_from_item(it, nid, tg.loop("PSV", h.tag), auto=True, parent=h.id)
        psv.design_p = h.design_p
        psv.zone = h.zone
        added.append(psv)
        # The PSV INLET is part of the protected system and carries its service;
        # only the tailpipe downstream of the valve is a relief/flare line.
        # Labelling the inlet "RL" was wrong, and it is exactly the kind of
        # error that teaches a direction rule the wrong lesson.
        in_svc = CAT.DEFAULT_SERVICE.get(h.system, "P")
        new_streams.append(Stream(id="", src=h.id, dst=nid, size_in=4,
                                  service=in_svc, spec=CAT.PIPE_SPEC.get(in_svc, "P1A"),
                                  auto=True, fluid="to relief"))
        if dest is None:
            # nothing on this sheet can take it: hand it to the flare header
            cx = next((c for c in sp.connectors if c.item == "flare_header"), None)
            if cx is None:
                cx = Node(id=f"x{len(sp.connectors):03d}", tag="HP FLARE HEADER",
                          item="flare_header", name="off-page connector",
                          category="connector", symbol="offpage", system="flare",
                          label="TO HP FLARE HEADER", auto=True, zone="Zone 1")
                sp.connectors.append(cx)
            dest = cx
        # The tailpipe is a RELIEF line (RL); FL is reserved for the flare
        # headers it discharges into. Keeping them distinct is what lets the
        # direction rules tell a PSV inlet from its outlet.
        new_streams.append(Stream(id="", src=nid, dst=dest.id, size_in=6,
                                  service="RL", spec="F1A", auto=True, fluid="relief"))

    # 2 · blowdown on gas-holding vessels
    for h in [n for n in equip if n.symbol in ("vessel_v", "vessel_h", "column")
              and n.system in ("compression", "dehydration", "separation")
              and n.design_p >= 20]:
        it = CAT.CATALOG["bdv"]
        nid = f"a{n_ai:03d}"; n_ai += 1
        bdv = _node_from_item(it, nid, tg.loop("BDV", h.tag), auto=True, parent=h.id)
        bdv.zone = h.zone
        bdest = _flare_dest(exclude=h.id) or flare_dest
        if bdest is not None and bdest.id == h.id:
            bdest = None
        added.append(bdv)
        bd_svc = CAT.DEFAULT_SERVICE.get(h.system, "P")
        new_streams.append(Stream(id="", src=h.id, dst=nid, size_in=3,
                                  service=bd_svc, spec=CAT.PIPE_SPEC.get(bd_svc, "P1A"),
                                  auto=True, fluid="to blowdown"))
        if bdest is None:
            bdest = next((c for c in sp.connectors if c.item == "flare_header"), None)
        if bdest is not None:
            new_streams.append(Stream(id="", src=nid, dst=bdest.id, size_in=4,
                                      service="RL", spec="F1A", auto=True))

    # 3 · check valve downstream of every pump and compressor
    for h in [n for n in equip if n.symbol in ("pump", "pump_recip", "compressor",
                                               "compressor_recip")]:
        it = CAT.CATALOG["check_valve"]
        nid = f"a{n_ai:03d}"; n_ai += 1
        nrv = _node_from_item(it, nid, tg.loop("NRV", h.tag), auto=True, parent=h.id)
        nrv.zone = h.zone
        added.append(nrv)
        # splice it into the discharge: pump -> NRV -> whatever the pump fed
        outs = [s for s in sp.streams if s.src == h.id]
        if outs:
            s0 = outs[0]
            new_streams.append(Stream(id="", src=nid, dst=s0.dst, size_in=s0.size_in,
                                      service=s0.service, spec=s0.spec, auto=True))
            s0.dst = nid
        else:
            new_streams.append(Stream(id="", src=h.id, dst=nid, size_in=6,
                                      service="P", spec="P1A", auto=True))

    # 4 · anti-surge recycle around every centrifugal compressor
    for h in [n for n in equip if n.symbol == "compressor"]:
        it = CAT.CATALOG["antisurge_valve"]
        nid = f"a{n_ai:03d}"; n_ai += 1
        asv = _node_from_item(it, nid, tg.loop("FV", h.tag), auto=True, parent=h.id)
        asv.label = "ANTI-SURGE"
        asv.zone = h.zone
        added.append(asv)
        new_streams.append(Stream(id="", src=h.id, dst=nid, size_in=8, service="PG",
                                  spec="P1B", auto=True, fluid="recycle"))
        new_streams.append(Stream(id="", src=nid, dst=h.id, size_in=8, service="PG",
                                  spec="P1B", auto=True, fluid="recycle"))

    # 5 · ESD valve at the boundary — the item with no upstream source
    fed = {s.dst for s in sp.streams}
    sources = [n for n in equip if n.id not in fed]
    sources = [h for h in sources if any(s.src == h.id for s in sp.streams)]
    for h in sources[:3]:
        it = CAT.CATALOG["esdv"]
        nid = f"a{n_ai:03d}"; n_ai += 1
        sdv = _node_from_item(it, nid, tg.loop("SDV", h.tag), auto=True, parent=h.id)
        sdv.zone = h.zone
        added.append(sdv)
        outs = [s for s in sp.streams if s.src == h.id]
        if outs:
            s0 = outs[0]
            new_streams.append(Stream(id="", src=nid, dst=s0.dst, size_in=s0.size_in,
                                      service=s0.service, spec=s0.spec, auto=True))
            s0.dst = nid

    sp.nodes.extend(added)

    # 6 · instrument loops, by equipment type
    n_ii = 500
    for h in equip:
        for code in CAT.AUTO_INSTRUMENTS.get(h.symbol, [])[:5]:
            if any(n.parent == h.id and n.item == code.lower() for n in sp.nodes):
                continue
            nm, _fam = CAT.INSTRUMENTS[code]
            nid = f"i{n_ii:03d}"; n_ii += 1
            sp.nodes.append(Node(id=nid, tag=tg.loop(code, h.tag), item=code.lower(),
                                 name=nm, category="instrument", symbol="bubble",
                                 system=h.system, label=code, parent=h.id,
                                 zone=h.zone, auto=True))
            sp.controls.append((nid, h.id))

    # 7 · every automatic valve gets its position switches and solenoid
    for v in [n for n in sp.nodes if n.category == "valve"
              and n.symbol in ("esd_valve", "control_valve")]:
        for code in ("ZLO", "ZLC") if v.symbol == "esd_valve" else ("ZLO",):
            nm, _f = CAT.INSTRUMENTS[code]
            nid = f"i{n_ii:03d}"; n_ii += 1
            sp.nodes.append(Node(id=nid, tag=tg.loop(code, v.tag), item=code.lower(),
                                 name=nm, category="instrument", symbol="bubble",
                                 system=v.system, label=code, parent=v.id,
                                 zone=v.zone, auto=True))
            sp.controls.append((nid, v.id))

    # 8 · fire & gas coverage for every hazardous area on the sheet
    zones = sorted({n.zone for n in equip if n.zone == "Zone 1"})
    for z in zones:
        for code in ("GD", "FD"):
            nm, _f = CAT.INSTRUMENTS[code]
            nid = f"i{n_ii:03d}"; n_ii += 1
            host = next((n for n in equip if n.zone == z), None)
            sp.nodes.append(Node(id=nid, tag=tg.loop(code, host.tag if host else "1101"),
                                 name=nm, item=code.lower(), category="instrument",
                                 symbol="bubble", system="safety", label=code,
                                 parent=host.id if host else "", zone=z, auto=True))
            if host:
                sp.controls.append((nid, host.id))

    for k, s in enumerate(new_streams):
        s.id = f"s{900 + k:03d}"
    sp.streams.extend(new_streams)


# ───────────────────────────────────────────────────────────── HAZOP checks
def hazop(sp: Spec) -> list[dict]:
    """Design checks a process safety engineer would run over the sheet.

    Reported, never silently repaired — an issue list you can hand to a
    reviewer is worth more than a drawing that quietly patched itself.
    """
    out: list[dict] = []

    def add(sev, code, msg, tag=""):
        out.append({"severity": sev, "code": code, "item": tag, "finding": msg})

    ids = {n.id: n for n in sp.all_nodes()}
    deg: dict[str, int] = {n.id: 0 for n in sp.all_nodes()}
    for s in sp.streams:
        deg[s.src] = deg.get(s.src, 0) + 1
        deg[s.dst] = deg.get(s.dst, 0) + 1

    tags: dict[str, int] = {}
    for n in sp.all_nodes():
        tags[n.tag] = tags.get(n.tag, 0) + 1
    for t, k in tags.items():
        if k > 1:
            add("high", "TAG-DUP", f"tag {t} is used {k} times", t)

    protected = {n.parent for n in sp.nodes if n.item in ("psv", "rupture_disc")}
    for n in sp.equipment():
        if n.symbol in CAT.NEEDS_PSV and n.design_p >= 3 and n.id not in protected:
            add("high", "PSV-MISSING",
                f"{n.name} {n.tag} is a pressure envelope with no relief device", n.tag)
        if deg.get(n.id, 0) == 0:
            add("medium", "ORPHAN", f"{n.tag} is not connected to anything", n.tag)
        if n.symbol in ("vessel_h", "column") and not any(
                x.parent == n.id and x.item in ("lt", "lg", "lic")
                for x in sp.nodes):
            add("medium", "LEVEL-MISSING",
                f"{n.name} {n.tag} has no level instrumentation", n.tag)

    def upstream_items(nid: str, depth: int = 3) -> list[Node]:
        """Walk back through inline valves — a scrubber feeding through an ESD
        valve is still the scrubber, and reporting it missing is a false alarm."""
        found, frontier = [], [nid]
        for _ in range(depth):
            nxt = []
            for cur in frontier:
                for st in sp.streams:
                    if st.dst != cur:
                        continue
                    u = ids.get(st.src)
                    if u is None:
                        continue
                    if u.category == "valve":
                        nxt.append(u.id)
                    else:
                        found.append(u)
            frontier = nxt
        return found

    comps = [n for n in sp.equipment() if n.symbol in ("compressor", "compressor_recip")]
    for c in comps:
        upstream = upstream_items(c.id)
        if not any(u and u.symbol in ("vessel_v", "vessel_h") for u in upstream):
            add("high", "SCRUBBER-MISSING",
                f"compressor {c.tag} has no suction scrubber — liquid carry-over risk", c.tag)
        if not any(n.parent == c.id and n.item == "antisurge_valve" for n in sp.nodes):
            add("medium", "SURGE", f"compressor {c.tag} has no anti-surge recycle", c.tag)

    for p in [n for n in sp.equipment() if n.symbol in ("pump", "pump_recip")]:
        if not any(n.parent == p.id and n.item == "check_valve" for n in sp.nodes):
            add("medium", "NRV-MISSING", f"pump {p.tag} has no discharge check valve", p.tag)

    relief = [n for n in sp.nodes if n.item in ("psv", "bdv")]
    flare_ids = {n.id for n in sp.all_nodes()
                 if n.item in ("flare_stack", "hp_flare_ko", "lp_flare_ko",
                               "vent_stack", "flare_header")}
    for r in relief:
        if flare_ids and not any(s.src == r.id and s.dst in flare_ids for s in sp.streams):
            add("high", "RELIEF-ROUTE",
                f"{r.tag} discharge does not reach a flare or vent", r.tag)
        elif not flare_ids:
            add("high", "NO-FLARE", f"{r.tag} relieves to nowhere — no flare system on the sheet", r.tag)

    if any(n.zone == "Zone 1" for n in sp.equipment()):
        if not any(n.item in ("gd", "fd") for n in sp.nodes):
            add("high", "F&G", "Zone 1 equipment with no fire & gas detection")

    for s in sp.streams:
        if s.size_in <= 0 or not s.service:
            add("low", "LINE-DATA",
                f"line {s.tag} is missing a size or service code", s.tag)

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(out, key=lambda r: order[r["severity"]])


# ───────────────────────────────────────────────────── deliverable registers
def equipment_list(sp: Spec) -> list[dict]:
    return [{"tag": n.tag, "description": n.label or n.name, "type": n.name,
             "system": n.system, "design P (barg)": n.design_p,
             "design T (°C)": n.design_t, "area": n.zone, "duty / note": n.duty}
            for n in sorted(sp.equipment(), key=lambda n: n.tag)]


def line_list(sp: Spec) -> list[dict]:
    rows = []
    for s in sp.streams:
        a, b = sp.by_id(s.src), sp.by_id(s.dst)
        rows.append({"line number": s.tag, "from": a.tag if a else s.src,
                     "to": b.tag if b else s.dst, "size (in)": s.size_in,
                     "service": s.service,
                     "service description": _svc_name(s.service),
                     "spec": s.spec, "insulation": s.insulation or "—",
                     "fluid": s.fluid or _svc_name(s.service)})
    return sorted(rows, key=lambda r: r["line number"])


def _svc_name(code: str) -> str:
    from ..ingest.lines import SERVICE
    return (SERVICE.get(code) or {}).get("name", code)


def valve_list(sp: Spec) -> list[dict]:
    rows = []
    for n in sp.of_category("valve"):
        p = sp.by_id(n.parent)
        rows.append({"tag": n.tag, "type": n.name, "on": p.tag if p else "—",
                     "fail position": _fail(n.item), "size (in)": _valve_size(sp, n),
                     "area": n.zone, "function": n.duty or n.label})
    return sorted(rows, key=lambda r: r["tag"])


def _fail(item: str) -> str:
    return {"esdv": "fail close", "bdv": "fail open", "antisurge_valve": "fail open",
            "psv": "spring", "check_valve": "n/a", "hipps": "fail close"}.get(item, "fail close")


def _valve_size(sp: Spec, n: Node) -> float:
    for s in sp.streams:
        if s.src == n.id or s.dst == n.id:
            return s.size_in
    return 0.0


def instrument_index(sp: Spec) -> list[dict]:
    rows = []
    for n in sp.of_category("instrument"):
        p = sp.by_id(n.parent)
        code = n.item.upper()
        nm, fam = CAT.INSTRUMENTS.get(code, (n.name, ""))
        rows.append({"tag": n.tag, "service": nm, "measured": fam,
                     "on": p.tag if p else "—", "loop": n.tag.split("-")[-1],
                     "area": n.zone, "signal": "4-20 mA HART" if code.endswith("T")
                     else ("discrete" if code.startswith(("Z", "S")) else "local")})
    return sorted(rows, key=lambda r: r["tag"])


def cause_effect(sp: Spec) -> tuple[list[str], list[str], list[list[str]]]:
    """The ESD cause & effect matrix — the safety deliverable that goes with a P&ID."""
    causes, effects = [], []
    for n in sp.of_category("instrument"):
        if n.item.upper() in ("PSH", "PSL", "LSH", "LSL", "GD", "FD", "SD", "HD"):
            p = sp.by_id(n.parent)
            causes.append(f"{n.tag}  {n.name}" + (f"  ({p.tag})" if p else ""))
    causes.append("ESD-1  manual emergency shutdown")
    for n in sp.of_category("valve"):
        if n.item in ("esdv", "bdv", "hipps"):
            effects.append(f"{n.tag}  {'open' if n.item == 'bdv' else 'close'}")
    for n in sp.equipment():
        if n.symbol in ("pump", "compressor", "compressor_recip", "pump_recip", "turbine"):
            effects.append(f"{n.tag}  trip")
    if not effects:
        effects.append("no shutdown actions defined")

    grid = []
    for c in causes:
        row = []
        fire = c.split("  ")[1].startswith(("gas", "flame", "smoke", "heat")) or "manual" in c
        for e in effects:
            if "manual" in c or fire:
                row.append("X")
            elif e.split("  ")[1] == "close" and c.startswith(("PSH", "LSH")):
                row.append("X")
            elif e.endswith("trip") and c.startswith(("PSL", "LSL")):
                row.append("X")
            else:
                row.append("")
        grid.append(row)
    return causes, effects, grid


def tie_in_list(sp: Spec) -> list[dict]:
    rows = []
    for i, c in enumerate(sp.connectors, 1):
        feed = [s for s in sp.streams if s.dst == c.id or s.src == c.id]
        rows.append({"tie-in": f"TI-{i:03d}", "destination": c.tag,
                     "lines": len(feed),
                     "size (in)": max([s.size_in for s in feed], default=0),
                     "continues on": "adjacent sheet"})
    return rows
