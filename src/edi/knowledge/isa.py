"""
ISA-5.1 tag decoding — reading an instrument tag the way an engineer reads it.

Asked "what does SDV mean?", the system used to generate
`MATCH (s:Symbol) WHERE s.tag = 'SDV'`, find nothing, and answer "no
information available". It had a graph and no knowledge: it could tell you what
was connected to what, but not what any of it *was*.

An instrument tag is not an arbitrary label. ISA-5.1 makes it a sentence:

    P   S    H        -1101
    │   │    │
    │   │    └─ modifier: high
    │   └────── function: switch
    └────────── measured variable: pressure

So `PSH-1101` is "pressure switch, high, on loop 1101" — and that is derivable,
not learnable. Every instrument tag on every drawing in the world decodes the
same way, which means one table replaces both a lookup list that can never be
complete and a language model that will confidently guess.

Industry practice overrides the standard in a handful of places (`SDV` is a
shutdown valve, not a "speed... valve"), so an explicit convention table is
checked first. Those are the tags everyone on a rig actually says out loud.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict

# ─────────────────────────────────────────────── ISA-5.1 letter tables
FIRST = {
    "A": "analysis",           "B": "burner / combustion", "C": "conductivity",
    "D": "density",            "E": "voltage",             "F": "flow",
    "G": "gauging / position", "H": "hand (manual)",       "I": "current",
    "J": "power",              "K": "time / schedule",     "L": "level",
    "M": "moisture",           "N": "user defined",        "O": "user defined",
    "P": "pressure",           "Q": "quantity",            "R": "radiation",
    "S": "speed / frequency",  "T": "temperature",         "U": "multivariable",
    "V": "vibration",          "W": "weight / force",      "X": "unclassified",
    "Y": "event / state",      "Z": "position / dimension",
}

# second-position modifiers of the measured variable
MOD_FIRST = {"D": "differential", "F": "ratio", "Q": "total",
             "S": "safety", "K": "rate of change", "M": "momentary"}

SUCCEEDING = {
    "A": "alarm",       "B": "user defined",  "C": "controller",
    "E": "element / sensor", "G": "gauge / sight glass", "I": "indicator",
    "K": "control station", "L": "light",     "N": "user defined",
    "O": "orifice / restriction", "P": "test point", "R": "recorder",
    "S": "switch",      "T": "transmitter",   "U": "multifunction",
    "V": "valve / damper", "W": "well / probe", "X": "accessory",
    "Y": "relay / computing device", "Z": "actuator / final element",
}

MOD_SUCC = {"H": "high", "L": "low", "M": "middle", "HH": "high-high",
            "LL": "low-low"}

# what the reading means to somebody standing in front of it
VARIABLE_HELP = {
    "pressure": "how hard the fluid is pushing on the pipe or vessel wall",
    "level": "how full the vessel is",
    "flow": "how much fluid is moving through the line",
    "temperature": "how hot the fluid is",
    "vibration": "how much the machine is shaking — a bearing or balance warning",
    "speed / frequency": "how fast the machine is turning",
    "analysis": "the composition of the fluid (for example oil in water, or gas dew point)",
    "moisture": "how much water is in the gas",
    "position / dimension": "whether something is open, closed or where it is",
    "weight / force": "how heavy the contents are",
    "density": "how heavy the fluid is for its volume",
}

FUNCTION_HELP = {
    "transmitter": "sends the reading to the control room as a 4-20 mA signal",
    "indicator": "shows the reading on a local dial — you read it by eye",
    "controller": "compares the reading to a setpoint and moves a valve to correct it",
    "switch": "makes or breaks a contact when the reading crosses a set limit",
    "alarm": "annunciates in the control room when the reading crosses a limit",
    "element / sensor": "the sensing element itself, in contact with the fluid",
    "well / probe": "a thermowell — the pocket a temperature sensor sits in, so the "
                    "sensor can be pulled without breaking containment",
    "gauge / sight glass": "a local glass or gauge you look at directly",
    "valve / damper": "the valve itself",
    "relay / computing device": "takes signals in and does logic or arithmetic on them",
    "recorder": "logs the reading over time",
    "orifice / restriction": "a fixed plate that restricts flow, used to measure it "
                             "or to limit it",
    "actuator / final element": "the thing that physically moves the valve",
    "control station": "the operator's local control point for the loop",
}

# ─────────────── industry conventions that override the strict standard
CONVENTION: dict[str, dict] = {
    "SDV":  dict(name="shutdown valve", category="valve",
                 plain="An automatic valve that slams shut when the shutdown system "
                       "trips. It is how a section of the plant is isolated in an "
                       "emergency. Fails closed on loss of air or signal.",
                 on_action="closes"),
    "ESDV": dict(name="emergency shutdown valve", category="valve",
                 plain="The main emergency isolation valve for a system. Closes on "
                       "ESD to cut hydrocarbon inventory.", on_action="closes"),
    "BDV":  dict(name="blowdown valve", category="valve",
                 plain="Opens on demand to dump the pressure in a vessel or system "
                       "to the flare. Fails OPEN — the safe state is depressurised.",
                 on_action="opens"),
    "PSV":  dict(name="pressure safety valve", category="valve",
                 plain="A spring-loaded relief valve. It is the last mechanical "
                       "protection against overpressure: when the pressure reaches "
                       "the set point it lifts and relieves to flare, with no "
                       "electricity, signal or operator involved.",
                 on_action="lifts"),
    "PRV":  dict(name="pressure relief valve", category="valve",
                 plain="A relief valve protecting against overpressure.",
                 on_action="lifts"),
    "RV":   dict(name="relief valve", category="valve",
                 plain="A relief valve protecting against overpressure.",
                 on_action="lifts"),
    "ERV":  dict(name="emergency relief valve", category="valve",
                 plain="A relief valve protecting against overpressure.",
                 on_action="lifts"),
    "RD":   dict(name="rupture disc", category="valve",
                 plain="A thin disc that bursts at a set pressure. Once it goes it "
                       "stays open until it is replaced.", on_action="bursts"),
    "XV":   dict(name="on/off valve", category="valve",
                 plain="An automatic valve driven fully open or fully shut by the "
                       "control system — no throttling.", on_action="opens or closes"),
    "MOV":  dict(name="motor operated valve", category="valve",
                 plain="A valve driven by an electric motor actuator.",
                 on_action="strokes"),
    "ROV":  dict(name="remote operated valve", category="valve",
                 plain="A valve operated from the control room rather than by hand.",
                 on_action="strokes"),
    "NRV":  dict(name="non-return (check) valve", category="valve",
                 plain="Lets flow go one way only. It stops a stopped pump being "
                       "driven backwards by the pressure downstream of it.",
                 on_action="shuts on reverse flow"),
    "HV":   dict(name="hand valve", category="valve",
                 plain="A manual valve — you turn it yourself. Usually the block "
                       "valve you would use to isolate a job.", on_action="is turned by hand"),
    "RO":   dict(name="restriction orifice", category="instrument",
                 plain="A fixed plate with a hole in it, welded into the line to "
                       "limit how much can flow. It cannot be adjusted.",
                 on_action="restricts flow permanently"),
    "FE":   dict(name="flow element (orifice plate)", category="instrument",
                 plain="The plate the flow measurement is taken across."),
    "TW":   dict(name="thermowell", category="instrument",
                 plain="The pocket welded into the pipe that a temperature sensor "
                       "sits in, so the sensor can be pulled without opening the line."),
    "SP":   dict(name="specification break", category="annotation",
                 plain="A marker on the drawing where the piping specification "
                       "changes — different material or rating each side of it."),
    "HIPPS": dict(name="high integrity pressure protection system", category="valve",
                  plain="A fast-acting instrumented protection system that closes "
                        "valves before the pressure can reach the design limit. "
                        "Used instead of relieving to flare.", on_action="closes"),
    "SDL":  dict(name="shutdown logic", category="instrument",
                 plain="The logic block that decides when the shutdown fires — the "
                       "cause and effect matrix lives behind it."),
    "ESD":  dict(name="emergency shutdown", category="instrument",
                 plain="The system that shuts the plant down safely — closes the "
                       "shutdown valves, trips the machines, and opens blowdown."),
    "FO":   dict(name="fail open", category="annotation",
                 plain="A note on a valve: on loss of air or signal it goes OPEN."),
    "FC":   dict(name="fail closed", category="annotation",
                 plain="A note on a valve: on loss of air or signal it goes CLOSED."),
    "LO":   dict(name="locked open", category="annotation",
                 plain="This valve is locked in the open position. It must not be "
                       "closed without an authorised permit."),
    "LC":   dict(name="locked closed", category="annotation",
                 plain="This valve is locked shut. It must not be opened without an "
                       "authorised permit."),
    "NC":   dict(name="normally closed", category="annotation",
                 plain="This valve sits closed in normal operation."),
    "NO":   dict(name="normally open", category="annotation",
                 plain="This valve sits open in normal operation."),
    "CSO":  dict(name="car sealed open", category="annotation",
                 plain="Sealed open — a relief path that must never be blocked."),
    "CSC":  dict(name="car sealed closed", category="annotation",
                 plain="Sealed closed — opening it needs authorisation."),
    "ZLO":  dict(name="position switch, open", category="instrument",
                 plain="Tells the control room that the valve has actually reached "
                       "the OPEN position. It confirms the valve moved — not just "
                       "that it was told to.", on_action="signals open"),
    "ZLC":  dict(name="position switch, closed", category="instrument",
                 plain="Tells the control room that the valve has actually reached "
                       "the CLOSED position.", on_action="signals closed"),
    "ZSO":  dict(name="position switch, open", category="instrument",
                 plain="Confirms the valve has reached the open position."),
    "ZSC":  dict(name="position switch, closed", category="instrument",
                 plain="Confirms the valve has reached the closed position."),
    "DCS":  dict(name="distributed control system", category="annotation",
                 plain="The control room system. A tag pointing here means the "
                       "signal goes to the operator's screens."),
}

_TAG = re.compile(r"^([A-Z]{1,6})\s*[-–]?\s*([0-9][0-9A-Z/\-]*)?$")


@dataclass
class Decoded:
    tag: str
    code: str                 # the letters
    loop: str                 # the number
    name: str                 # short engineering name
    category: str             # equipment | valve | instrument | annotation
    variable: str = ""        # what it measures
    function: str = ""        # what it does with the reading
    modifier: str = ""        # high / low / differential …
    plain: str = ""           # plain English, for somebody who has never seen one
    source: str = "isa"       # isa | convention
    on_action: str = ""       # what it does when it operates

    def dict(self) -> dict:
        return asdict(self)


def split(tag: str) -> tuple[str, str]:
    """('PSH-1101') -> ('PSH', '1101'). Returns ('', '') if it is not tag-shaped."""
    t = (tag or "").strip().upper().replace("—", "-").replace("–", "-")
    m = _TAG.match(t)
    if not m:
        return "", ""
    return m.group(1), (m.group(2) or "")


def decode(tag: str) -> Decoded | None:
    """Decode an instrument or valve tag. None if it is not one."""
    code, loop = split(tag)
    if not code:
        return None

    if code in CONVENTION:
        c = CONVENTION[code]
        return Decoded(tag=tag.upper(), code=code, loop=loop, name=c["name"],
                       category=c["category"], plain=c["plain"], source="convention",
                       on_action=c.get("on_action", ""))

    if code[0] not in FIRST:
        return None

    # ISA writes the letters in a fixed order: variable, optional variable
    # modifier, function letters, then trailing limit modifiers (H/L/M). Parsing
    # the limits from the END is what stops `LSLL` being read as
    # "level light low" — L is both "light" and "low", and only its position
    # tells them apart.
    #
    # The variable modifier is ambiguous with a function letter (`S` is both
    # "safety" and "switch"), so the plain reading is tried first and the
    # modifier reading only if that fails. PSH is a pressure switch high; PSV,
    # which really is "pressure safety valve", is in the convention table above
    # and never reaches here.
    for consume_modifier in (False, True):
        parsed = _parse_letters(code, consume_modifier)
        if parsed is not None:
            variable, modifier, funcs, limit = parsed
            break
    else:
        return None

    name = " ".join(funcs)
    if funcs == ["valve / damper"]:
        name = "control valve"          # "flow valve / damper" reads like a typo
    full = f"{variable} {name}" + (f", {limit}" if limit else "")
    category = "valve" if "valve / damper" in funcs else "instrument"
    plain = _plain(variable, funcs, limit, loop)
    return Decoded(tag=tag.upper(), code=code, loop=loop, name=full,
                   category=category, variable=variable,
                   function=", ".join(funcs), modifier=limit, plain=plain,
                   source="isa", on_action=_on_action(funcs, limit))


def _parse_letters(code: str, consume_modifier: bool):
    """(variable, modifier, functions, limit) or None if the letters do not decode."""
    variable = FIRST[code[0]]
    rest = code[1:]
    modifier = ""
    if consume_modifier:
        if not rest or rest[0] not in MOD_FIRST or len(rest) < 2:
            return None
        modifier = MOD_FIRST[rest[0]]
        variable = f"{modifier} {variable}"
        rest = rest[1:]
    if not rest:
        return None

    i = len(rest)
    while i > 0 and rest[i - 1] in ("H", "L", "M"):
        i -= 1
    func_part, limit_part = rest[:i], rest[i:]
    if not func_part:                     # every letter looked like a limit
        func_part, limit_part = rest[:1], rest[1:]

    funcs = []
    for ch in func_part:
        if ch not in SUCCEEDING:
            return None
        funcs.append(SUCCEEDING[ch])
    if not funcs:
        return None

    limit = {"HH": "high-high", "LL": "low-low", "H": "high", "L": "low",
             "M": "middle"}.get(limit_part, "")
    if limit_part and not limit:
        return None
    return variable, modifier, funcs, limit


def _plain(variable: str, funcs: list[str], limit: str, loop: str) -> str:
    base = VARIABLE_HELP.get(variable, f"the {variable}")
    parts = [f"This measures {base}."]
    for f in funcs:
        h = FUNCTION_HELP.get(f)
        if h:
            parts.append(f"It {h}.")
    if limit:
        parts.append(f"The '{limit}' means it acts when the reading goes "
                     f"{'above' if limit.startswith('high') else 'below'} its set point"
                     + (" — and a doubled limit (HH or LL) normally means a trip, "
                        "not just an alarm." if limit in ("high-high", "low-low") else "")
                     + ".")
    if loop:
        parts.append(f"{loop} is the loop number: everything sharing it belongs to "
                     f"the same control loop.")
    return " ".join(parts)


def _on_action(funcs: list[str], limit: str) -> str:
    if "valve / damper" in funcs:
        return "strokes"
    if "switch" in funcs or "alarm" in funcs:
        return "trips" if limit in ("high-high", "low-low") else "alarms"
    return ""


def category_of(tag: str) -> str | None:
    d = decode(tag)
    return d.category if d else None
