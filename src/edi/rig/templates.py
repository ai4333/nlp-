"""
Standard rig packages, pre-wired.

Two jobs. First, they are the fallback: if Ollama is not running or the model
returns nonsense, the builder still produces a correct sheet, because the chat
must never be the thing that breaks. Second, they are the few-shot examples —
showing the model a real spec is far more effective than describing the schema.

Each package is the minimum set of items an engineer would expect to see on that
sheet. Everything else (relief, instrument loops, ESD valves, line numbers) is
added by `spec.build`.
"""
from __future__ import annotations

TEMPLATES: dict[str, dict] = {

 "production_train": {
   "title": "Production separation train",
   "keywords": ["separation", "separator", "production", "train", "oil", "three stage",
                "3 stage", "crude"],
   "equipment": ["xmas_tree", "choke", "production_manifold", "hp_separator",
                 "mp_separator", "lp_separator", "crude_heater", "coalescer",
                 "export_pump", "metering_skid", "hp_flare_ko", "flare_stack"],
   "streams": [
     ["xmas_tree", "choke", 6, "P"], ["choke", "production_manifold", 10, "P"],
     ["production_manifold", "hp_separator", 12, "P"],
     ["hp_separator", "mp_separator", 10, "PL"],
     ["mp_separator", "crude_heater", 8, "PL"],
     ["crude_heater", "lp_separator", 8, "PL"],
     ["lp_separator", "coalescer", 8, "PL"],
     ["coalescer", "export_pump", 8, "PL"],
     ["export_pump", "metering_skid", 8, "PL"],
     ["hp_flare_ko", "flare_stack", 24, "FL"]],
   "notes": ["THREE STAGE SEPARATION, HP / MP / LP.",
             "CRUDE HEATED UPSTREAM OF FINAL STAGE.",
             "EXPORT ON SPEC AT 0.5% BSW."]},

 "gas_compression": {
   "title": "Gas compression and dehydration",
   "keywords": ["compression", "compressor", "gas", "dehydration", "teg", "glycol",
                "export gas"],
   "equipment": ["suction_scrubber", "compressor", "aftercooler", "discharge_scrubber",
                 "teg_contactor", "teg_regenerator", "glycol_pump", "gas_filter",
                 "hp_flare_ko", "flare_stack"],
   "streams": [
     ["suction_scrubber", "compressor", 12, "PG"],
     ["compressor", "aftercooler", 10, "PG"],
     ["aftercooler", "discharge_scrubber", 10, "PG"],
     ["discharge_scrubber", "teg_contactor", 10, "PG"],
     ["teg_contactor", "gas_filter", 10, "PG"],
     ["teg_contactor", "teg_regenerator", 3, "GL"],
     ["teg_regenerator", "glycol_pump", 3, "GL"],
     ["glycol_pump", "teg_contactor", 3, "GL"],
     ["hp_flare_ko", "flare_stack", 24, "FL"]],
   "notes": ["GAS DEHYDRATED TO 4 LB/MMSCF.",
             "ANTI-SURGE RECYCLE FAILS OPEN.",
             "GLYCOL CIRCULATION IS A CLOSED LOOP."]},

 "wellhead": {
   "title": "Wellhead and production manifold",
   "keywords": ["wellhead", "well", "manifold", "xmas", "tree", "choke", "test"],
   "equipment": ["wellhead", "xmas_tree", "ssv", "choke", "production_manifold",
                 "test_manifold", "test_separator", "chemical_skid", "flare_stack"],
   "streams": [
     ["wellhead", "xmas_tree", 6, "P"], ["xmas_tree", "ssv", 6, "P"],
     ["ssv", "choke", 6, "P"], ["choke", "production_manifold", 10, "P"],
     ["choke", "test_manifold", 6, "P"],
     ["test_manifold", "test_separator", 6, "P"],
     ["chemical_skid", "xmas_tree", 1, "CH"]],
   "notes": ["SSV FAILS CLOSED ON ESD.",
             "CONTINUOUS CORROSION INHIBITOR AT THE TREE.",
             "ONE WELL SHOWN, TYPICAL OF SIX."]},

 "flare_relief": {
   "title": "Flare and relief system",
   "keywords": ["flare", "relief", "blowdown", "vent", "psv", "depressur"],
   "equipment": ["hp_flare_ko", "lp_flare_ko", "flare_ko_pump", "flare_stack",
                 "vent_stack", "closed_drain"],
   "streams": [
     ["hp_flare_ko", "flare_stack", 24, "FL"],
     ["lp_flare_ko", "flare_stack", 16, "FL"],
     ["hp_flare_ko", "flare_ko_pump", 4, "DC"],
     ["flare_ko_pump", "closed_drain", 4, "DC"],
     ["lp_flare_ko", "vent_stack", 8, "VA"]],
   "notes": ["FLARE KO DRUMS SIZED FOR THE LARGEST SINGLE RELIEF CASE.",
             "KO DRUM PUMPS ON LEVEL CONTROL.",
             "PILOT AND IGNITION PACKAGE NOT SHOWN."]},

 "produced_water": {
   "title": "Produced water treatment",
   "keywords": ["produced water", "water treatment", "hydrocyclone", "flotation",
                "oily water", "deoil"],
   "equipment": ["hydrocyclone", "degasser", "cfu", "slop_tank", "open_drain",
                 "water_inj_pump", "water_inj_manifold"],
   "streams": [
     ["hydrocyclone", "degasser", 8, "WA"], ["degasser", "cfu", 8, "WA"],
     ["cfu", "water_inj_pump", 8, "WA"],
     ["water_inj_pump", "water_inj_manifold", 6, "WA"],
     ["cfu", "slop_tank", 4, "DO"], ["slop_tank", "open_drain", 4, "DO"]],
   "notes": ["DISCHARGE TO SEA AT LESS THAN 30 PPM OIL IN WATER.",
             "REJECT RETURNS TO THE CLOSED DRAIN SYSTEM."]},

 "water_injection": {
   "title": "Seawater lift and injection",
   "keywords": ["water injection", "seawater", "injection", "sulphate", "deaerat"],
   "equipment": ["seawater_lift_pump", "sw_filter", "so4_removal", "deaerator",
                 "chemical_skid", "water_inj_pump", "water_inj_manifold"],
   "streams": [
     ["seawater_lift_pump", "sw_filter", 16, "WA"],
     ["sw_filter", "so4_removal", 14, "WA"],
     ["so4_removal", "deaerator", 14, "WA"],
     ["deaerator", "water_inj_pump", 12, "WA"],
     ["water_inj_pump", "water_inj_manifold", 10, "WA"],
     ["chemical_skid", "water_inj_pump", 1, "CH"]],
   "notes": ["OXYGEN SCAVENGER INJECTED DOWNSTREAM OF THE DEAERATOR.",
             "INJECTION PRESSURE 350 BARG."]},

 "utilities": {
   "title": "Utilities — fuel gas, air and nitrogen",
   "keywords": ["utility", "utilities", "fuel gas", "instrument air", "nitrogen",
                "diesel", "air compressor"],
   "equipment": ["fuel_gas_scrubber", "fuel_gas_heater", "air_compressor",
                 "air_dryer", "air_receiver", "n2_package", "diesel_tank",
                 "diesel_pump", "power_generator"],
   "streams": [
     ["fuel_gas_scrubber", "fuel_gas_heater", 4, "FG"],
     ["fuel_gas_heater", "power_generator", 4, "FG"],
     ["air_compressor", "air_dryer", 3, "IA"],
     ["air_dryer", "air_receiver", 3, "IA"],
     ["air_receiver", "n2_package", 2, "N"],
     ["diesel_tank", "diesel_pump", 3, "DO"]],
   "notes": ["FUEL GAS SUPERHEATED 20 DEG C ABOVE DEW POINT.",
             "INSTRUMENT AIR DRIED TO -40 DEG C DEW POINT."]},

 "safety": {
   "title": "Firewater and deluge",
   "keywords": ["firewater", "fire water", "deluge", "fire", "safety", "f&g"],
   "equipment": ["seawater_lift_pump", "firewater_pump", "firewater_ringmain",
                 "deluge_skid"],
   "streams": [
     ["seawater_lift_pump", "firewater_pump", 12, "WA"],
     ["firewater_pump", "firewater_ringmain", 10, "WA"],
     ["firewater_ringmain", "deluge_skid", 8, "WA"]],
   "notes": ["FIREWATER PUMPS DIESEL DRIVEN, EACH 100% CAPACITY.",
             "RING MAIN KEPT PRESSURISED BY THE JOCKEY PUMP."]},

 "drilling": {
   "title": "Drilling mud circulation and well control",
   "keywords": ["drilling", "mud", "bop", "shaker", "riser", "well control",
                "choke manifold", "cement"],
   "equipment": ["mud_tank", "mud_pump", "riser", "bop_stack", "choke_manifold",
                 "degasser_mud", "shale_shaker", "desander", "desilter", "flare_stack"],
   "streams": [
     ["mud_tank", "mud_pump", 6, "P"], ["mud_pump", "riser", 5, "P"],
     ["riser", "bop_stack", 5, "P"], ["bop_stack", "choke_manifold", 4, "P"],
     ["choke_manifold", "degasser_mud", 4, "P"],
     ["degasser_mud", "shale_shaker", 6, "P"],
     ["shale_shaker", "desander", 6, "P"], ["desander", "desilter", 6, "P"],
     ["desilter", "mud_tank", 6, "P"],
     ["degasser_mud", "flare_stack", 8, "FL"]],
   "notes": ["BOP RATED 15,000 PSI WORKING PRESSURE.",
             "MUD RETURNS THROUGH THE SOLIDS CONTROL TRAIN.",
             "GAS FROM THE MUD GAS SEPARATOR GOES TO THE VENT."]},

 "export": {
   "title": "Crude export and pigging",
   "keywords": ["export", "pig", "pipeline", "metering", "offloading", "riser"],
   "equipment": ["storage_tank", "booster_pump", "export_pump", "metering_skid",
                 "pig_launcher", "export_riser"],
   "streams": [
     ["storage_tank", "booster_pump", 12, "PL"],
     ["booster_pump", "export_pump", 10, "PL"],
     ["export_pump", "metering_skid", 10, "PL"],
     ["metering_skid", "pig_launcher", 10, "PL"],
     ["pig_launcher", "export_riser", 10, "PL"]],
   "notes": ["FISCAL METERING TO THE OPERATOR'S MEASUREMENT STANDARD.",
             "PIG LAUNCHER ISOLATED BY DOUBLE BLOCK AND BLEED."]},

 "full_platform": {
   "title": "Offshore platform overview",
   "keywords": ["platform", "whole rig", "full", "overview", "everything",
                "complete", "entire", "topsides"],
   "equipment": ["xmas_tree", "production_manifold", "hp_separator", "lp_separator",
                 "suction_scrubber", "compressor", "aftercooler", "teg_contactor",
                 "export_pump", "metering_skid", "hydrocyclone", "water_inj_pump",
                 "power_generator", "hp_flare_ko", "flare_stack"],
   "streams": [
     ["xmas_tree", "production_manifold", 10, "P"],
     ["production_manifold", "hp_separator", 12, "P"],
     ["hp_separator", "lp_separator", 10, "PL"],
     ["hp_separator", "suction_scrubber", 12, "PG"],
     ["suction_scrubber", "compressor", 12, "PG"],
     ["compressor", "aftercooler", 10, "PG"],
     ["aftercooler", "teg_contactor", 10, "PG"],
     ["teg_contactor", "power_generator", 4, "FG"],
     ["lp_separator", "export_pump", 8, "PL"],
     ["export_pump", "metering_skid", 8, "PL"],
     ["lp_separator", "hydrocyclone", 6, "WA"],
     ["hydrocyclone", "water_inj_pump", 6, "WA"],
     ["hp_flare_ko", "flare_stack", 24, "FL"]],
   "notes": ["OVERVIEW SHEET — SEE THE UNIT P&IDS FOR DETAIL.",
             "ALL RELIEF ROUTED TO THE HP FLARE HEADER."]},
}


def expand(name: str) -> dict:
    """Template → the raw spec dict that `spec.build` consumes."""
    t = TEMPLATES[name]
    return {"title": t["title"],
            "equipment": [{"type": e} for e in t["equipment"]],
            "streams": [{"from": a, "to": b, "size": s, "service": v}
                        for a, b, s, v in t["streams"]],
            "notes": t.get("notes", [])}


def match(text: str) -> list[str]:
    """Templates whose subject matter the request mentions, best first."""
    t = (text or "").lower()
    scored = []
    for name, tpl in TEMPLATES.items():
        hits = sum(1 for k in tpl["keywords"] if k in t)
        if hits:
            scored.append((hits, name))
    scored.sort(reverse=True)
    return [n for _, n in scored]


def merged(names: list[str], title: str = "") -> dict:
    """Combine several packages into one sheet, de-duplicating shared items."""
    eq, st, notes = [], [], []
    for n in names:
        t = TEMPLATES[n]
        for e in t["equipment"]:
            if e not in eq:
                eq.append(e)
        for s in t["streams"]:
            if s not in st:
                st.append(s)
        notes += t.get("notes", [])
    return {"title": title or " + ".join(TEMPLATES[n]["title"] for n in names)[:70],
            "equipment": [{"type": e} for e in eq],
            "streams": [{"from": a, "to": b, "size": s, "service": v}
                        for a, b, s, v in st],
            "notes": notes[:6]}
