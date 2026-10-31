"""
The plant glossary — what the words on a rig actually mean.

Written for the person the drawing was not written for: the technician who has
just joined, is standing in front of a vessel, and needs to know what it is
before he touches anything. Every entry answers three things — what it is, what
it is for, and what it means for the person standing next to it.

This is deliberately a curated table and not a model. A hallucinated definition
of "blowdown" on an oil rig is not a bad answer, it is a hazard.
"""
from __future__ import annotations
import re

# term -> (short definition, why it matters on the job)
GLOSSARY: dict[str, tuple[str, str]] = {

 # ── separation and process
 "separator": ("A pressure vessel that splits the well stream into gas, oil and water "
               "by letting it slow down so the phases settle out.",
               "It is the heart of the process. It runs part full of liquid — the level "
               "you see on the gauge glass is real liquid, and losing level control "
               "sends gas down a liquid line or liquid to the compressor."),
 "three phase separator": ("A separator that takes off gas, oil and water separately.",
               "Three outlets, three level controls. Most upsets on a train start here."),
 "knockout drum": ("A vessel that catches liquid out of a gas stream before the gas goes "
               "somewhere that liquid would damage.",
               "A compressor or a flare tip will be wrecked by liquid carry-over. The KO "
               "drum is what stops that."),
 "free water knockout": ("A vessel that removes the bulk of the water early, before the "
               "oil is treated.", "Getting water out early makes everything downstream smaller."),
 "coalescer": ("A vessel that merges tiny water droplets in oil until they are heavy "
               "enough to fall out, often helped by an electric field.",
               "It is what gets crude down to export spec (BS&W)."),
 "scrubber": ("A small vessel that takes the last liquid out of a gas stream.",
               "Always sits immediately upstream of a compressor."),
 "compressor": ("A machine that raises gas pressure.",
               "High energy, hot discharge, and very intolerant of liquid. Never bypass "
               "a suction scrubber level trip."),
 "centrifugal compressor": ("A compressor that raises pressure with a spinning impeller.",
               "Has a surge limit: pushed too far it flow-reverses violently. That is what "
               "the anti-surge valve exists to prevent."),
 "surge": ("Flow reversal in a centrifugal compressor when the flow drops too low for the "
               "pressure it is making.", "It sounds like a bang and it damages the machine. "
               "The anti-surge valve recycles gas back to keep flow up."),
 "anti-surge valve": ("A fast recycle valve that sends discharge gas back to the compressor "
               "suction.", "It fails OPEN, because recycling is always safer than surging."),
 "pump": ("A machine that moves liquid and raises its pressure.",
               "Running one against a closed discharge, or with no liquid in it, wrecks it "
               "in minutes."),
 "heat exchanger": ("Equipment that moves heat between two streams without mixing them.",
               "Both sides are usually under pressure. Isolate and drain both before opening."),
 "air cooler": ("A bank of finned tubes with fans blowing across them, cooling the process "
               "with ambient air.", "The fans are the only cooling. Losing them raises "
               "discharge temperature fast."),
 "teg contactor": ("A tall column where gas is washed with triethylene glycol to dry it.",
               "Wet gas forms hydrates and corrodes. The contactor is what prevents that."),
 "glycol": ("A liquid that absorbs water out of gas.", "Hot glycol in the regeneration "
               "package will burn you; it runs near 200 degrees C."),
 "hydrate": ("An ice-like solid that forms when water and gas meet at high pressure and low "
               "temperature.", "It blocks lines and can fire out like a bullet when it lets "
               "go. Methanol injection is one defence against it."),
 "hydrocyclone": ("A cone that spins produced water to fling the oil droplets out of it.",
               "It is what gets water clean enough to discharge overboard."),
 "produced water": ("Water that comes up out of the well with the oil and gas.",
               "It is oily, often hot, sometimes radioactive scale. It is not clean water."),
 "manifold": ("A header with several branches, used to route flow from many wells into "
               "common lines.", "How a well is lined up to production or to test."),
 "pig launcher": ("A barrel used to send a cleaning or inspection tool down a pipeline.",
               "Opening a launcher is one of the highest-risk routine jobs on a facility. "
               "Prove it is depressurised, every time."),
 "metering skid": ("A package that measures how much is exported, for money.",
               "Fiscal metering is sealed and legally controlled. Do not adjust anything on it."),
 "wellhead": ("The equipment at the top of the well that contains the pressure.",
               "It is full pressure from the reservoir. Nothing upstream of the choke is low "
               "pressure."),
 "christmas tree": ("The stack of valves on top of a wellhead.",
               "Each valve has a defined role — master, wing, swab. Their order matters."),
 "choke": ("A restriction that controls how much a well flows.",
               "It is the pressure break between reservoir pressure and the plant."),

 # ── flare, relief and safety
 "flare": ("A stack that burns off gas that has to be got rid of safely.",
               "The flare header is always live. Never open a relief path to atmosphere."),
 "flare header": ("The common collection pipe that takes every relief and blowdown to the "
               "flare.", "It must never be blocked in. Isolation valves on relief paths are "
               "car-sealed open for that reason."),
 "blowdown": ("Deliberately dumping the pressure of a system to flare.",
               "It is fast, loud and cold — depressurising gas chills metal enough to "
               "cause brittle fracture and frostbite."),
 "relief valve": ("A spring valve that opens by itself when pressure reaches its set point.",
               "It is the last line of defence and needs no power or signal. Its outlet "
               "must never be isolated."),
 "esd": ("Emergency shutdown: the system that safely stops the plant.",
               "ESD closes the shutdown valves, trips the machines and can start blowdown. "
               "An ESD level (1, 2, 3) says how much it shuts down."),
 "cause and effect": ("The matrix that says which detected condition causes which shutdown "
               "action.", "It is the authority on what happens when something trips. If you "
               "want to know what closing a valve will do, this is the document."),
 "hipps": ("High integrity pressure protection system — instrumented valves that shut fast "
               "enough to protect a low-pressure system from a high-pressure one.",
               "It replaces relieving to flare, so it is safety critical and heavily tested."),
 "fire and gas": ("Detectors that look for gas leaks, flame and smoke, and act on them.",
               "Gas detection voting can trip the plant on its own."),
 "deluge": ("A fixed system that drops a curtain of water over an area on demand.",
               "It goes off without warning during a test. Know where the manual call points are."),
 "hazardous area": ("An area classified by how likely a flammable atmosphere is: Zone 0 "
               "(continuous), Zone 1 (likely), Zone 2 (unlikely).",
               "It sets what equipment is allowed and what work needs a permit. No non-Ex "
               "electrical kit, no hot work without authorisation."),
 "loto": ("Lock out, tag out — physically locking an energy source off and tagging it so "
               "nobody can restore it while you work.",
               "Your lock goes on the isolation yourself. You take it off yourself."),
 "permit to work": ("The written authorisation that says a specific job may be done, by "
               "whom, with what isolations in place.",
               "No permit, no job. The isolation list on the permit comes from the P&ID."),
 "double block and bleed": ("Two closed valves in series with a vent between them, so any "
               "leak past the first valve is bled away rather than reaching you.",
               "The standard isolation for opening a hydrocarbon system."),
 "spading": ("Inserting a solid steel plate into a flange to guarantee positive isolation.",
               "A valve can leak; a spade cannot. Long jobs on live systems get spades."),
 "car seal": ("A physical seal fitted to a valve to show it must stay in its set position.",
               "Car sealed open means a relief path. Breaking that seal without "
               "authorisation is a serious matter."),

 # ── piping and drawing conventions
 "p&id": ("Piping and instrumentation diagram — the drawing that shows every item of "
               "equipment, every line, every valve and every instrument, and how they connect.",
               "It is not to scale and it is not a layout. It tells you what connects to "
               "what, not where it physically is."),
 "pfd": ("Process flow diagram — the simplified version showing only the main flow and the "
               "main equipment.", "Good for understanding the process; useless for isolating "
               "a job."),
 "line number": ("The label printed along a pipe, e.g. 12\"-PG-11-185-P1B-HE.",
               "It carries the size, what is in it, which unit it belongs to, the piping "
               "material specification and whether it is insulated or traced."),
 "piping spec": ("A code such as P1A that fixes the material, pressure rating and wall "
               "thickness of a run of pipe.", "Where the spec changes there is a spec break "
               "on the drawing. Fitting the wrong class of flange across one is dangerous."),
 "spec break": ("The point on a drawing where the piping specification changes.",
               "Each side has a different rating. Never take a gasket or bolt set across it."),
 "off-page connector": ("An arrow at the sheet edge saying the line continues on another "
               "drawing.", "Half of any real question crosses sheets. Follow the reference."),
 "isolation valve": ("A valve whose job is to shut a section off, not to control flow.",
               "The block valves either side of an item are what you close to work on it."),
 "check valve": ("A valve that only lets flow go one way.",
               "It is not an isolation valve. It will not hold pressure for a permit."),
 "control valve": ("A valve positioned continuously by the control system to hold a "
               "measurement at a setpoint.", "It has a fail position — open or closed — that "
               "the drawing states. Know it before you pull air off it."),
 "fail position": ("Where a valve goes when its air or signal is lost.",
               "Fail-closed on hydrocarbon inlets, fail-open on relief paths. It is the "
               "designed safe state."),
 "normally open": ("The valve sits open in normal operation.", "Closing it changes the "
               "process. Check why it is open first."),
 "normally closed": ("The valve sits closed in normal operation.", "Opening it may connect "
               "two systems that are not meant to be connected."),
 "instrument loop": ("A measurement, a controller and a final element sharing one number, "
               "e.g. LT-1101, LIC-1101, LV-1101.", "Same number means same loop. That is how "
               "you find the valve a transmitter drives."),
 "loop number": ("The digits in an instrument tag.", "Everything on the same loop shares it."),
 "setpoint": ("The value a controller is trying to hold.", "Changing one is an operations "
               "decision, not a maintenance one."),
 "trip": ("An automatic shutdown action caused by a measurement crossing a limit.",
               "A trip is not a fault — it is protection doing its job. Find out why before "
               "resetting it."),
 "interlock": ("Logic that prevents an action unless conditions are met.",
               "Overriding one needs authorisation and a record."),

 # ── units and quantities
 "barg": ("Bar gauge — pressure above atmospheric.", "0 barg is atmospheric, not vacuum."),
 "design pressure": ("The highest pressure the equipment is built and certified for.",
               "Protection is set below it. Exceeding it is a loss-of-containment risk."),
 "mmscfd": ("Million standard cubic feet per day — a gas rate.", "How gas volumes are quoted."),
 "bsw": ("Basic sediment and water — the water and solids left in export crude.",
               "Export contracts set a limit, usually well under one percent."),
 "ppm": ("Parts per million — one part of something in a million parts of the whole.",
               "Overboard produced water is typically limited to 30 ppm oil, and gas "
               "dryness and H2S limits are quoted the same way."),
 "dp": ("Differential pressure — the difference across something.",
               "A rising dP across a filter means it is blocking."),
}

# so "what is a KO drum?" and "what's the KO drum" find the same entry
_ALIAS = {
 "knock out drum": "knockout drum", "k.o. drum": "knockout drum",
 "3 phase separator": "three phase separator", "3-phase separator": "three phase separator",
 "fwko": "free water knockout", "lock out tag out": "loto",
 "lockout tagout": "loto", "dbb": "double block and bleed",
 "ko drum": "knockout drum", "k o drum": "knockout drum",
 "psv": "relief valve", "block valve": "isolation valve",
 "pid": "p&id", "p and id": "p&id", "c&e": "cause and effect",
 "cause & effect": "cause and effect", "f&g": "fire and gas",
 "emergency shutdown": "esd", "pressure safety valve": "psv",
 "safety valve": "relief valve", "prv": "relief valve",
 "zone 1": "hazardous area", "zone 2": "hazardous area", "zone 0": "hazardous area",
 "ptw": "permit to work", "spade": "spading", "blind": "spading",
 "nrv": "check valve", "non return valve": "check valve",
 "spec": "piping spec", "line no": "line number",
}

_CLEAN = re.compile(r"[^a-z0-9&\s.]")


def _norm(t: str) -> str:
    t = _CLEAN.sub(" ", (t or "").lower()).strip()
    return re.sub(r"\s+", " ", t)


# Keys are written the way an engineer writes them ("anti-surge valve"); lookups
# arrive however somebody typed them. Normalise both sides once.
_KEYS = {_norm(k): k for k in GLOSSARY}
_ALIAS_N = {_norm(k): _norm(v) for k, v in _ALIAS.items()}


def lookup(term: str) -> dict | None:
    """Exact or aliased glossary hit."""
    t = _norm(term)
    if not t:
        return None
    t = _ALIAS_N.get(t, t)
    hit = GLOSSARY.get(_KEYS.get(t, t))
    if hit is None and t.endswith("s"):
        t2 = _ALIAS_N.get(t[:-1], t[:-1])
        hit = GLOSSARY.get(_KEYS.get(t2, t2))
        if hit:
            t = t2
    if hit is None:
        return None
    return {"term": t, "definition": hit[0], "why_it_matters": hit[1],
            "source": "glossary"}


def search(text: str) -> list[dict]:
    """Every glossary term mentioned in a sentence, longest first.

    Used to explain the jargon inside an answer, not just a bare lookup.
    """
    t = _norm(text)
    hits = []
    for term in sorted(set(list(_KEYS) + list(_ALIAS_N)), key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", t):
            r = lookup(term)
            if r and not any(h["term"] == r["term"] for h in hits):
                hits.append(r)
    return hits


# ---------------------------------------------------------------------------
# What is actually in the pipe, and what that means for the person next to it.
# Keyed on the service code in the line number. This is the difference between
# "a 3 inch line" and "a 3 inch line full of flammable gas at 100 bar".
# ---------------------------------------------------------------------------
SERVICE_HAZARD: dict[str, tuple[str, str]] = {
    "P":  ("flammable", "Hydrocarbon under pressure. Gas or liquid — assume flammable."),
    "PL": ("flammable", "Hydrocarbon liquid under pressure. Flammable, and it will keep "
                        "coming until the source is isolated."),
    "PG": ("flammable", "Hydrocarbon gas under pressure. Flammable, expands violently on "
                        "release, and can chill metal enough to embrittle it."),
    "PA": ("flammable", "Process/air service — check the line list before assuming."),
    "FG": ("flammable", "Fuel gas. Flammable, and usually routed to fired equipment."),
    "FL": ("flammable", "Flare line. Always assume live and never block it in."),
    "RL": ("flammable", "Relief line. Live whenever the equipment it protects is live; "
                        "must not be isolated."),
    "VL": ("flammable", "Vent line — may carry hydrocarbon vapour."),
    "VH": ("flammable", "Vent header — shared, so it is live even if your equipment is not."),
    "VA": ("flammable", "Vent to atmosphere. The discharge point is a hazardous area."),
    "DC": ("flammable", "Closed drain. It has had hydrocarbon in it and it holds pressure."),
    "DO": ("flammable", "Open drain. Oily, and can carry gas back from the caisson."),
    "CD": ("hot",       "Condensate — hot liquid, scalding risk."),
    "GL": ("hot",       "Glycol. Runs hot in the regeneration loop and burns on contact."),
    "MO": ("toxic",     "Methanol. Toxic if swallowed or absorbed, and flammable."),
    "CH": ("toxic",     "Chemical injection. Check the specific chemical's data sheet — "
                        "many are corrosive."),
    "N":  ("asphyxiant","Nitrogen. It is not air. It displaces oxygen silently and has "
                        "killed people who entered spaces purged with it."),
    "IA": ("stored energy", "Instrument air. Not toxic, but stored energy — a released "
                            "line whips, and losing it strokes valves to their fail position."),
    "UA": ("stored energy", "Utility air. Stored energy; never use it to blow yourself down."),
    "ST": ("hot",       "Steam. Severe burn risk, and a leak can be invisible."),
    "HW": ("hot",       "Hot water. Scalding risk."),
    "CW": ("low",       "Cooling water. Low hazard in itself, but it can be under "
                        "significant pressure."),
    "WA": ("low",       "Water service. Check whether it is produced water, which is oily "
                        "and can carry scale."),
}


def hazard(service_code: str) -> dict | None:
    hit = SERVICE_HAZARD.get((service_code or "").upper())
    if not hit:
        return None
    return {"service": service_code.upper(), "class": hit[0], "note": hit[1]}
