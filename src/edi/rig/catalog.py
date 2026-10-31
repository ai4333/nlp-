"""
The oil-rig equipment catalogue.

Everything the rig builder can draw, tag and register lives here — one entry per
item type, holding the four things every downstream stage needs:

    tag_prefix   ISA / EPC letter, so tags come out conventional (V-1101, K-1201)
    category     equipment · valve · instrument · connector, the graph facet
    symbol       which renderer draws it (see render.py)
    design       plate data that fills the equipment list (design P/T, duty, area zone)

The catalogue is deliberately data, not code: a customer with a different house
convention edits this table, not the drawing engine.

Coverage is the whole of a production platform, topsides and below:
wellheads and manifolds, separation, compression, dehydration, condensate and
oil export, produced-water treatment, water injection, flare and relief, the
utilities (fuel gas, instrument air, nitrogen, diesel, firewater, seawater,
chemical injection), power generation, and the drilling package.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    key: str
    name: str
    category: str            # equipment | valve | instrument | connector
    tag_prefix: str
    symbol: str              # renderer key
    system: str              # which rig system it belongs to
    design_p: float = 0.0    # barg
    design_t: float = 0.0    # degC
    zone: str = "Zone 2"     # hazardous area classification
    duty: str = ""
    ports: int = 2
    aliases: tuple = ()


def _I(*a, **k) -> Item:
    return Item(*a, **k)


# ─────────────────────────────────────────────────────────── wells & manifold
CATALOG: dict[str, Item] = {}


def _add(items: list[Item]):
    for it in items:
        CATALOG[it.key] = it


_add([
 _I("wellhead",        "wellhead",                     "equipment", "WH", "wellhead",   "wells", 345, 120, "Zone 1",
    duty="production wellhead", aliases=("well", "producer")),
 _I("xmas_tree",       "christmas tree",               "equipment", "XT", "xmas_tree",  "wells", 345, 120, "Zone 1",
    duty="surface tree, 5k/10k", aliases=("tree", "wellhead tree")),
 _I("choke",           "production choke",             "valve",     "FCV","choke",      "wells", 345, 120, "Zone 1",
    duty="well flow control", aliases=("choke valve",)),
 _I("smpv",            "surface master valve",         "valve",     "SSV","gate_valve", "wells", 345, 120, "Zone 1"),
 _I("ssv",             "surface safety valve",         "valve",     "SSV","esd_valve",  "wells", 345, 120, "Zone 1",
    duty="fail-close on ESD"),
 _I("production_manifold","production manifold",       "equipment", "MF", "manifold",   "wells", 150, 120, "Zone 1",
    duty="commingled production header", ports=6, aliases=("manifold", "prod manifold")),
 _I("test_manifold",   "test manifold",                "equipment", "MF", "manifold",   "wells", 150, 120, "Zone 1",
    duty="well test routing", ports=6),
 _I("test_separator",  "test separator",               "equipment", "V",  "vessel_h",   "wells", 25, 100, "Zone 1",
    duty="well test, 3-phase"),
])

# ─────────────────────────────────────────────────────────── separation train
_add([
 _I("hp_separator",    "HP separator",                 "equipment", "V",  "vessel_h",   "separation", 100, 120, "Zone 1",
    duty="1st stage 3-phase separation", ports=4,
    aliases=("first stage separator", "1st stage separator", "hp sep")),
 _I("mp_separator",    "MP separator",                 "equipment", "V",  "vessel_h",   "separation", 40, 110, "Zone 1",
    duty="2nd stage 3-phase separation", ports=4,
    aliases=("second stage separator", "2nd stage separator")),
 _I("lp_separator",    "LP separator",                 "equipment", "V",  "vessel_h",   "separation", 12, 100, "Zone 1",
    duty="3rd stage separation", ports=4,
    aliases=("third stage separator", "3rd stage separator", "lp sep")),
 _I("separator",       "separator",                    "equipment", "V",  "vessel_h",   "separation", 40, 110, "Zone 1",
    duty="3-phase separation", ports=4, aliases=("2-phase separator", "3-phase separator")),
 _I("fwko",            "free water knockout",          "equipment", "V",  "vessel_h",   "separation", 40, 110, "Zone 1",
    duty="bulk water removal", ports=4, aliases=("free-water knockout",)),
 _I("coalescer",       "electrostatic coalescer",      "equipment", "V",  "vessel_h",   "separation", 20, 110, "Zone 1",
    duty="oil dehydration to 0.5% BSW"),
 _I("desalter",        "desalter",                     "equipment", "V",  "vessel_h",   "separation", 20, 110, "Zone 1"),
 _I("surge_drum",      "surge drum",                   "equipment", "D",  "vessel_h",   "separation", 12, 100, "Zone 1"),
 _I("crude_heater",    "crude oil heater",             "equipment", "H",  "exchanger",  "separation", 40, 150, "Zone 1",
    duty="heats crude for separation"),
 _I("oil_cooler",      "oil cooler",                   "equipment", "E",  "exchanger",  "separation", 40, 120, "Zone 2"),
])

# ─────────────────────────────────────────────────────────── gas compression
_add([
 _I("suction_scrubber","compressor suction scrubber",  "equipment", "V",  "vessel_v",   "compression", 100, 100, "Zone 1",
    duty="liquid removal upstream of compressor", aliases=("scrubber", "suction ko")),
 _I("discharge_scrubber","compressor discharge scrubber","equipment","V", "vessel_v",   "compression", 150, 120, "Zone 1"),
 _I("compressor",      "gas compressor",               "equipment", "K",  "compressor", "compression", 150, 150, "Zone 1",
    duty="centrifugal, motor driven", aliases=("gas compressor", "centrifugal compressor")),
 _I("lp_compressor",   "LP compressor",                "equipment", "K",  "compressor", "compression", 40, 150, "Zone 1"),
 _I("hp_compressor",   "HP compressor",                "equipment", "K",  "compressor", "compression", 150, 150, "Zone 1"),
 _I("recip_compressor","reciprocating compressor",     "equipment", "K",  "compressor_recip", "compression", 200, 150, "Zone 1"),
 _I("booster_compressor","booster compressor",         "equipment", "K",  "compressor", "compression", 60, 150, "Zone 1"),
 _I("aftercooler",     "compressor aftercooler",       "equipment", "E",  "air_cooler", "compression", 150, 150, "Zone 2",
    duty="air-cooled, forced draft", aliases=("air cooler", "gas cooler", "cooler")),
 _I("interstage_cooler","interstage cooler",           "equipment", "E",  "air_cooler", "compression", 100, 150, "Zone 2"),
 _I("antisurge_valve", "anti-surge valve",             "valve",     "FV", "control_valve","compression", 150, 150, "Zone 1",
    duty="recycle, fail-open"),
 _I("gas_turbine",     "gas turbine driver",           "equipment", "GT", "turbine",    "compression", 0, 550, "Zone 2",
    duty="compressor driver"),
])

# ─────────────────────────────────────────────────────────── dehydration
_add([
 _I("teg_contactor",   "TEG contactor",                "equipment", "C",  "column",     "dehydration", 100, 100, "Zone 1",
    duty="gas dehydration to 4 lb/MMscf", aliases=("glycol contactor", "contactor")),
 _I("teg_regenerator", "TEG regeneration package",     "equipment", "MY", "package",    "dehydration", 5, 205, "Zone 1",
    duty="glycol reboiler + still column", aliases=("glycol regen",)),
 _I("glycol_pump",     "glycol circulation pump",      "equipment", "P",  "pump",       "dehydration", 100, 100, "Zone 1"),
 _I("mol_sieve",       "molecular sieve dehydrator",   "equipment", "V",  "vessel_v",   "dehydration", 100, 250, "Zone 1"),
 _I("gas_filter",      "gas filter / coalescer",       "equipment", "F",  "filter",     "dehydration", 100, 100, "Zone 1"),
 _I("dew_point_unit",  "dew point control unit",       "equipment", "MY", "package",    "dehydration", 100, 60, "Zone 1"),
])

# ─────────────────────────────────────────────────────────── export & metering
_add([
 _I("export_pump",     "crude export pump",            "equipment", "P",  "pump",       "export", 100, 100, "Zone 1",
    duty="centrifugal, motor driven", aliases=("oil export pump", "shipping pump")),
 _I("booster_pump",    "booster pump",                 "equipment", "P",  "pump",       "export", 40, 100, "Zone 1"),
 _I("metering_skid",   "fiscal metering skid",         "equipment", "JY", "package",    "export", 100, 80, "Zone 1",
    duty="fiscal measurement", aliases=("metering package", "fiscal meter")),
 _I("pig_launcher",    "pig launcher",                 "equipment", "MQ", "pig_trap",   "export", 150, 80, "Zone 1",
    aliases=("launcher",)),
 _I("pig_receiver",    "pig receiver",                 "equipment", "MQ", "pig_trap",   "export", 150, 80, "Zone 1",
    aliases=("receiver",)),
 _I("export_riser",    "export riser",                 "equipment", "RS", "riser",      "export", 150, 80, "Zone 1"),
 _I("storage_tank",    "storage tank",                 "equipment", "TK", "tank",       "export", 1, 80, "Zone 1"),
])

# ─────────────────────────────────────────────────────────── flare & relief
_add([
 _I("hp_flare_ko",     "HP flare KO drum",             "equipment", "D",  "vessel_h",   "flare", 20, 120, "Zone 1",
    duty="liquid removal before the stack", aliases=("flare knockout drum", "hp ko drum")),
 _I("lp_flare_ko",     "LP flare KO drum",             "equipment", "D",  "vessel_h",   "flare", 10, 120, "Zone 1"),
 _I("flare_stack",     "flare stack",                  "equipment", "FS", "flare",      "flare", 0, 200, "Zone 1",
    duty="elevated flare with pilots", ports=1, aliases=("flare", "flare tip")),
 _I("flare_ko_pump",   "flare KO drum pump",           "equipment", "P",  "pump",       "flare", 20, 100, "Zone 1"),
 _I("vent_stack",      "atmospheric vent stack",       "equipment", "VS", "flare",      "flare", 0, 100, "Zone 1", ports=1),
 _I("psv",             "pressure safety valve",        "valve",     "PSV","psv",        "flare", 0, 0, "Zone 1",
    duty="relief to flare", aliases=("relief valve", "safety valve")),
 _I("bdv",             "blowdown valve",               "valve",     "BDV","esd_valve",  "flare", 0, 0, "Zone 1",
    duty="depressurisation to flare"),
 _I("rupture_disc",    "rupture disc",                 "valve",     "RD", "rupture_disc","flare", 0, 0, "Zone 1"),
])

# ─────────────────────────────────────────────────────── produced water & injection
_add([
 _I("hydrocyclone",    "produced water hydrocyclone",  "equipment", "V",  "hydrocyclone","water", 20, 100, "Zone 1",
    duty="bulk oil removal to 100 ppm", aliases=("deoiling hydrocyclone",)),
 _I("degasser",        "produced water degasser",      "equipment", "V",  "vessel_v",   "water", 5, 100, "Zone 1"),
 _I("cfu",             "compact flotation unit",       "equipment", "V",  "vessel_v",   "water", 10, 100, "Zone 1",
    duty="polishing to <30 ppm OiW", aliases=("flotation unit",)),
 _I("water_inj_pump",  "water injection pump",         "equipment", "P",  "pump",       "water", 350, 100, "Zone 1",
    duty="high pressure injection", aliases=("injection pump",)),
 _I("seawater_lift_pump","seawater lift pump",         "equipment", "P",  "pump",       "water", 10, 60, "Zone 2"),
 _I("sw_filter",       "seawater fine filter",         "equipment", "F",  "filter",     "water", 10, 60, "Zone 2"),
 _I("deaerator",       "vacuum deaerator",             "equipment", "V",  "vessel_v",   "water", 5, 60, "Zone 2"),
 _I("so4_removal",     "sulphate removal unit",        "equipment", "MY", "package",    "water", 20, 60, "Zone 2"),
 _I("water_inj_manifold","water injection manifold",   "equipment", "MF", "manifold",   "water", 350, 80, "Zone 1", ports=6),
])

# ─────────────────────────────────────────────────────────── utilities
_add([
 _I("fuel_gas_scrubber","fuel gas scrubber",           "equipment", "V",  "vessel_v",   "utilities", 40, 100, "Zone 1"),
 _I("fuel_gas_heater", "fuel gas superheater",         "equipment", "H",  "exchanger",  "utilities", 40, 150, "Zone 1"),
 _I("air_compressor",  "instrument air compressor",    "equipment", "K",  "compressor", "utilities", 10, 100, "Zone 2"),
 _I("air_dryer",       "instrument air dryer",         "equipment", "MY", "package",    "utilities", 10, 60, "Zone 2"),
 _I("air_receiver",    "instrument air receiver",      "equipment", "V",  "vessel_v",   "utilities", 10, 60, "Zone 2"),
 _I("n2_package",      "nitrogen generation package",  "equipment", "MY", "package",    "utilities", 10, 60, "Zone 2"),
 _I("diesel_tank",     "diesel storage tank",          "equipment", "TK", "tank",       "utilities", 1, 60, "Zone 2"),
 _I("diesel_pump",     "diesel transfer pump",         "equipment", "P",  "pump",       "utilities", 10, 60, "Zone 2"),
 _I("firewater_pump",  "firewater pump",               "equipment", "P",  "pump",       "safety", 15, 60, "Zone 2",
    duty="diesel driven, 100% capacity"),
 _I("firewater_ringmain","firewater ring main",        "equipment", "MF", "manifold",   "safety", 15, 60, "Zone 2", ports=6),
 _I("deluge_skid",     "deluge skid",                  "equipment", "MY", "package",    "safety", 15, 60, "Zone 1"),
 _I("chemical_skid",   "chemical injection skid",      "equipment", "MY", "package",    "utilities", 350, 60, "Zone 1",
    duty="corrosion inhibitor / demulsifier", aliases=("chemical injection",)),
 _I("methanol_skid",   "methanol injection skid",      "equipment", "MY", "package",    "utilities", 350, 60, "Zone 1"),
 _I("heat_medium",     "heat medium package",          "equipment", "MY", "package",    "utilities", 10, 180, "Zone 2"),
 _I("cooling_medium",  "cooling medium package",       "equipment", "MY", "package",    "utilities", 10, 90, "Zone 2"),
 _I("power_generator", "gas turbine generator",        "equipment", "GT", "turbine",    "power", 0, 550, "Zone 2",
    duty="power generation", aliases=("generator", "turbine generator")),
 _I("emergency_gen",   "emergency diesel generator",   "equipment", "DG", "turbine",    "power", 0, 400, "Zone 2"),
 _I("closed_drain",    "closed drain drum",            "equipment", "D",  "vessel_h",   "drains", 10, 100, "Zone 1"),
 _I("open_drain",      "open drain caisson",           "equipment", "D",  "vessel_v",   "drains", 1, 60, "Zone 1"),
 _I("slop_tank",       "slop tank",                    "equipment", "TK", "tank",       "drains", 1, 60, "Zone 1"),
])

# ─────────────────────────────────────────────────────────── drilling package
_add([
 _I("bop_stack",       "BOP stack",                    "equipment", "BOP","bop",        "drilling", 690, 120, "Zone 1",
    duty="15k annular + rams", aliases=("bop", "blowout preventer")),
 _I("diverter",        "diverter",                     "equipment", "DIV","bop",        "drilling", 35, 100, "Zone 1"),
 _I("mud_pump",        "mud pump",                     "equipment", "P",  "pump_recip", "drilling", 350, 80, "Zone 1",
    duty="triplex, 7500 psi"),
 _I("mud_tank",        "active mud tank",              "equipment", "TK", "tank",       "drilling", 1, 60, "Zone 1"),
 _I("shale_shaker",    "shale shaker",                 "equipment", "SS", "package",    "drilling", 1, 60, "Zone 1"),
 _I("desander",        "desander",                     "equipment", "V",  "hydrocyclone","drilling", 10, 60, "Zone 1"),
 _I("desilter",        "desilter",                     "equipment", "V",  "hydrocyclone","drilling", 10, 60, "Zone 1"),
 _I("degasser_mud",    "mud gas separator",            "equipment", "V",  "vessel_v",   "drilling", 10, 60, "Zone 1",
    aliases=("poor boy degasser",)),
 _I("choke_manifold",  "drilling choke manifold",      "equipment", "MF", "manifold",   "drilling", 690, 100, "Zone 1", ports=6),
 _I("cement_unit",     "cementing unit",               "equipment", "MY", "package",    "drilling", 700, 60, "Zone 1"),
 _I("riser",           "drilling riser",               "equipment", "RS", "riser",      "drilling", 35, 60, "Zone 1"),
])

# ─────────────────────────────────────────────────────────── valves & inline
_add([
 _I("esdv",            "emergency shutdown valve",     "valve",     "SDV","esd_valve",  "safety", 0, 0, "Zone 1",
    duty="fail-close on ESD", aliases=("esd valve", "shutdown valve", "sdv")),
 _I("control_valve",   "control valve",                "valve",     "CV", "control_valve","process", 0, 0, "Zone 1"),
 _I("lcv",             "level control valve",          "valve",     "LV", "control_valve","process"),
 _I("pcv",             "pressure control valve",       "valve",     "PV", "control_valve","process"),
 _I("fcv",             "flow control valve",           "valve",     "FV", "control_valve","process"),
 _I("tcv",             "temperature control valve",    "valve",     "TV", "control_valve","process"),
 _I("gate_valve",      "gate valve",                   "valve",     "HV", "gate_valve", "process"),
 _I("ball_valve",      "ball valve",                   "valve",     "HV", "ball_valve", "process"),
 _I("check_valve",     "check valve",                  "valve",     "NRV","check_valve","process",
    duty="non-return", aliases=("non-return valve", "nrv")),
 _I("globe_valve",     "globe valve",                  "valve",     "HV", "globe_valve","process"),
 _I("butterfly_valve", "butterfly valve",              "valve",     "HV", "butterfly_valve","process"),
 _I("restriction_orifice","restriction orifice",       "instrument","RO", "orifice",    "process"),
 _I("orifice_plate",   "orifice plate",                "instrument","FE", "orifice",    "process"),
 _I("strainer",        "strainer",                     "valve",     "ST", "strainer",   "process"),
 _I("hipps",           "HIPPS valve",                  "valve",     "ZV", "esd_valve",  "safety", 345, 120, "Zone 1",
    duty="high integrity pressure protection, SIL 3"),
])

# ─────────────────────────────────────────────────── instruments (ISA-5.1)
INSTRUMENTS: dict[str, tuple[str, str]] = {
    "PT": ("pressure transmitter", "pressure"),
    "PI": ("pressure indicator", "pressure"),
    "PIC": ("pressure indicating controller", "pressure"),
    "PSH": ("pressure switch high", "pressure"),
    "PSL": ("pressure switch low", "pressure"),
    "PAH": ("pressure alarm high", "pressure"),
    "LT": ("level transmitter", "level"),
    "LI": ("level indicator", "level"),
    "LIC": ("level indicating controller", "level"),
    "LG": ("level gauge", "level"),
    "LSH": ("level switch high", "level"),
    "LSL": ("level switch low", "level"),
    "FT": ("flow transmitter", "flow"),
    "FI": ("flow indicator", "flow"),
    "FIC": ("flow indicating controller", "flow"),
    "FQI": ("flow totaliser", "flow"),
    "TT": ("temperature transmitter", "temperature"),
    "TI": ("temperature indicator", "temperature"),
    "TIC": ("temperature indicating controller", "temperature"),
    "AT": ("analyser transmitter", "analysis"),
    "MT": ("moisture transmitter", "analysis"),
    "ZLO": ("position switch, open", "position"),
    "ZLC": ("position switch, closed", "position"),
    "SDL": ("shutdown logic", "logic"),
    "XC": ("logic controller", "logic"),
    "VT": ("vibration transmitter", "machine"),
    "ST": ("speed transmitter", "machine"),
    "GD": ("gas detector", "fire & gas"),
    "FD": ("flame detector", "fire & gas"),
    "SD": ("smoke detector", "fire & gas"),
    "HD": ("heat detector", "fire & gas"),
}

# What each kind of equipment is normally instrumented with. Applied
# automatically so a generated sheet is instrumented the way a real one is —
# a bare vessel with no level or pressure control is not a P&ID, it is a cartoon.
AUTO_INSTRUMENTS: dict[str, list[str]] = {
    "vessel_h":   ["PT", "LT", "LG", "TT", "PSH", "LSH", "LSL"],
    "vessel_v":   ["PT", "LT", "LG", "PSH", "LSL"],
    "column":     ["PT", "LT", "TT", "PSH"],
    "compressor": ["PT", "TT", "VT", "ST"],
    "compressor_recip": ["PT", "TT", "VT"],
    "pump":       ["PT", "FT"],
    "pump_recip": ["PT", "FT"],
    "exchanger":  ["TT", "PT"],
    "air_cooler": ["TT"],
    "tank":       ["LT", "LG"],
    "filter":     ["PT"],
    "package":    ["PT"],
    "manifold":   ["PT", "FT"],
    "hydrocyclone": ["PT", "FT", "AT"],
    "turbine":    ["ST", "TT", "VT"],
    "bop":        ["PT", "ZLO"],
    "pig_trap":   ["PT"],
    "flare":      ["FT", "TT"],
    "riser":      ["PT"],
}

# Equipment that must be protected by a relief device routed to flare.
# Drives both the auto-build and the HAZOP check.
NEEDS_PSV = {"vessel_h", "vessel_v", "column", "tank", "filter", "exchanger",
             "pig_trap", "hydrocyclone"}

# Which service code a stream carries out of each kind of item.
DEFAULT_SERVICE: dict[str, str] = {
    "wells": "P", "separation": "PL", "compression": "PG", "dehydration": "PG",
    "export": "PL", "flare": "FL", "water": "WA", "utilities": "IA",
    "power": "FG", "drains": "DC", "drilling": "P", "safety": "WA",
    "process": "P",
}

SIZE_BY_SERVICE: dict[str, float] = {
    "PG": 12, "PL": 8, "P": 10, "FL": 16, "RL": 4, "WA": 8, "IA": 2, "UA": 2,
    "FG": 4, "DC": 4, "DO": 6, "CH": 1, "N": 2, "CW": 10, "HW": 6, "ST": 4,
    "CD": 3, "MO": 1, "GL": 3, "VL": 3, "VA": 6, "VH": 8, "PA": 2,
}

# The piping material spec printed at the end of a line number. Must cover every
# service code the templates use — a missing entry silently falls back to P1A
# and puts the wrong spec on the drawing, which is a real error on a real sheet.
PIPE_SPEC = {"P": "P1A", "PL": "P1A", "PG": "P1B", "PA": "A1A", "FL": "F1A",
             "RL": "F1A", "WA": "W1A", "IA": "A1A", "UA": "A1A", "FG": "P1B",
             "DC": "D1C", "DO": "D1A", "CD": "D1A", "CH": "S1A", "N": "A1A",
             "CW": "W1A", "HW": "W1B", "ST": "S1B", "MO": "S1A", "GL": "P1A",
             "VL": "V1A", "VA": "V1A", "VH": "V1A"}


def find(text: str) -> Item | None:
    """Resolve a free-text item name to a catalogue entry."""
    t = (text or "").strip().lower().replace("_", " ")
    if not t:
        return None
    for it in CATALOG.values():
        if t == it.key.replace("_", " ") or t == it.name.lower():
            return it
    for it in CATALOG.values():
        if t in it.aliases:
            return it
    # longest partial match wins, so "hp separator" beats "separator"
    best, blen = None, 0
    for it in CATALOG.values():
        for cand in (it.key.replace("_", " "), it.name.lower()) + it.aliases:
            if cand and (cand in t or t in cand) and len(cand) > blen:
                best, blen = it, len(cand)
    return best


def systems() -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = {}
    for it in CATALOG.values():
        out.setdefault(it.system, []).append(it)
    return out


def as_rows() -> list[dict]:
    """The catalogue as a table — what the app shows and the LLM is given."""
    return [{"key": i.key, "name": i.name, "system": i.system,
             "category": i.category, "tag prefix": i.tag_prefix,
             "design P (barg)": i.design_p, "design T (°C)": i.design_t,
             "area": i.zone, "note": i.duty}
            for i in sorted(CATALOG.values(), key=lambda x: (x.system, x.key))]
