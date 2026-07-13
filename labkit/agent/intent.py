"""The INTENT CONTRACT — deterministic. No model.

The physics validator answers "is this plan legal?". It cannot answer "is this the
simulation the user asked for?". A plan with salt_conc_M = 0.0 is perfectly legal
physics and completely wrong if the user said "physiological salt". That gap is
exactly where an LLM gets to vibe.

So: everything semantically load-bearing is extracted from the request BY CODE, turned
into hard assertions, and ENFORCED on the plan. The language model never decides these.
It is left with only the parts that cannot be wrong (stage naming, layout), and even the
protocol shape is a deterministic template when the user asks for one.

    request --> extract()   -> Intent (assertions)      [pure code, no model]
    request --> translate() -> Plan   (structure)       [grammar-constrained LLM]
    Plan + Intent --> enforce()  -> Plan                [pure code: OVERWRITES the LLM]
    Plan --> validate()                                 [pure code: physics]

An intent term this module does not recognise is reported as UNCOVERED rather than
silently trusted to the model — you always know which parts were decided by code and
which were left to the model.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

# molecule name -> PDB id. Deterministic lookup; NEVER let the model guess an id
# (it guessed "1LYS" for lysozyme; the canonical structure is 1AKI).
KNOWN_PDB = {
    "lysozyme": "1AKI", "hen egg white lysozyme": "1AKI",
    "ubiquitin": "1UBQ",
    "trp-cage": "1L2Y", "trp cage": "1L2Y",
    "chignolin": "1UAO",
    "villin": "1VII", "villin headpiece": "1VII",
    "haemoglobin": "4HHB", "hemoglobin": "4HHB",
    "myoglobin": "1MBN",
    "insulin": "4INS",
    "bpti": "5PTI",
    "sars-cov-2 main protease": "6LU7", "main protease": "6LU7", "mpro": "6LU7",
    "b-dna": "1BNA", "dna": "1BNA",
}


@dataclass
class Intent:
    temperature_K: float = None
    salt_M: float = None
    pdb_id: str = None
    kind: str = None
    ensemble: str = None
    production_ns: float = None
    equilibrate: bool = None
    forcefield: str = None
    box_padding_nm: float = None
    dt_ps: float = None
    ambiguous: list = field(default_factory=list)   # things we REFUSE to guess at
    box_size_nm: float = None
    water_model: str = None
    uncovered: list = field(default_factory=list)   # phrases we did NOT understand

    def assertions(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if k != "uncovered" and v is not None}


def _pdb_lookup(name: str) -> str | None:
    """Resolve a molecule name to a PDB id. Curated map first; RCSB search as fallback.
    Deterministic and verifiable — not a model guess."""
    n = name.lower().strip()
    if n in KNOWN_PDB:
        return KNOWN_PDB[n]
    try:
        q = {"query": {"type": "terminal", "service": "full_text",
                       "parameters": {"value": name}},
             "return_type": "entry",
             "request_options": {"paginate": {"start": 0, "rows": 1}}}
        url = ("https://search.rcsb.org/rcsbsearch/v2/query?json="
               + urllib.parse.quote(json.dumps(q)))
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.loads(r.read())
        ids = [x["identifier"] for x in d.get("result_set", [])]
        return ids[0] if ids else None
    except Exception:  # noqa: BLE001
        return None


def extract(nl: str) -> Intent:
    """Request -> hard assertions. Pure code."""
    t = nl.lower()
    it = Intent()

    # ---- temperature ----------------------------------------------------- #
    # \b37\b, NOT 37: unanchored, "137 C" contains "37 c" and was silently pinned to
    # 310 K instead of 410.15 K -- a 100 K thermostat error that verify() called clean.
    if re.search(r"body temperature|physiological temperature|\b37\s*°?\s*c\b", t):
        it.temperature_K = 310.0
    elif re.search(r"room temperature|ambient", t):
        it.temperature_K = 300.0
    m = re.search(r"(\d{2,4}(?:\.\d+)?)\s*k\b", t)
    if m:
        it.temperature_K = float(m.group(1))
    m = re.search(r"(\d{1,3})\s*°?\s*c\b", t)
    if m and not re.search(r"\b37\s*°?\s*c\b", t):
        it.temperature_K = float(m.group(1)) + 273.15

    # ---- salt ------------------------------------------------------------- #
    if re.search(r"physiolog\w*\s+salt|physiolog\w*\s+(?:nacl|ionic)|saline|"
                 r"physiological conditions", t):
        it.salt_M = 0.15
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|molar)\s*(?:nacl|salt)", t)
    if m:
        it.salt_M = float(m.group(1))
    if re.search(r"\b(no|without|zero)\s+salt\b|salt[- ]free", t):
        it.salt_M = 0.0

    # ---- ensemble --------------------------------------------------------- #
    if re.search(r"\bnpt\b|constant pressure", t):
        it.ensemble = "NPT"
    elif re.search(r"\bnvt\b|constant volume", t):
        it.ensemble = "NVT"

    # ---- duration (production) -------------------------------------------- #
    # This took the FIRST duration in the string. "equilibrate for 1 ns, then run 20 ns
    # of production" pinned production=1 ns -- a 20x error, reported as no violation.
    # Now: prefer a duration that is ADJACENT to the word production; fall back to the
    # sole duration if there is only one; otherwise pin NOTHING. A wrong pin is worse
    # than no pin, because verify() cannot see through it.
    durs = [(mm.start(), mm.end(),
             float(mm.group(1)) * (1.0 if mm.group(2).startswith(("ns", "nano")) else 0.001))
            for mm in re.finditer(r"(\d+(?:\.\d+)?)\s*(ns\b|nanosecond|ps\b|picosecond)", t)]
    if len(durs) == 1:
        it.production_ns = durs[0][2]
    elif len(durs) > 1:
        # Drop any duration that is the EQUILIBRATION time. The word can sit on EITHER
        # side of the number: "equilibrate for 1 ns" and "100 ps of equilibration".
        def _is_equil(a, b):
            # The connective must be a PREPOSITION that introduces the duration
            # ("equilibrate FOR 1 ns"). Allowing any two words let "equilibration THEN
            # 5 ns" swallow the production time as if it were the equilibration time.
            return bool(re.search(r"equilibrat\w*\s*(?:for|of|with|:)?\s*$", t[:a])
                        or re.match(r"\s*(?:of\s+|for\s+)?equilibrat", t[b:]))
        cand = [(pos, v) for pos, end, v in durs if not _is_equil(pos, end)]
        prod_at = [mm.start() for mm in re.finditer(r"produc|\brun\b|\bsimulat", t)]
        if len(cand) == 1:
            it.production_ns = cand[0][1]
        elif cand and prod_at:
            # nearest to the word "production" wins, and only if it wins CLEARLY
            scored = sorted((min(abs(pos - q) for q in prod_at), v) for pos, v in cand)
            if len(scored) == 1 or (scored[0][0] < 30 and scored[0][0] * 2 < scored[1][0]):
                it.production_ns = scored[0][1]
        if it.production_ns is None:
            it.ambiguous.append(
                f"{len(durs)} durations, none unambiguously the production time")

    # ---- equilibration protocol ------------------------------------------- #
    if re.search(r"\b(no|without|skip|skipping|omit|don'?t)\s+(the\s+)?equilibrat\w*|"
                 r"\bunequilibrated\b", t):
        it.equilibrate = False        # an explicit REFUSAL. It used to switch it on.
    elif re.search(r"proper equilibration|equilibrat\w*|full protocol|production run", t):
        it.equilibrate = True

    # ---- force field ------------------------------------------------------- #
    from ..plan.jsonschema import legal_values
    for ff in legal_values()["forcefield"]:
        if ff.lower() in t:
            it.forcefield = ff
            break
    else:
        if "charmm" in t:
            it.forcefield = "charmm27"
        elif "opls" in t:
            it.forcefield = "oplsaa"
        elif "gromos" in t:
            it.forcefield = "gromos54a7"
        elif "amber" in t:
            it.forcefield = "amber99sb-ildn"

    # ---- timestep ----------------------------------------------------------- #
    m = re.search(r"(\d+(?:\.\d+)?)\s*fs\s*(?:time\s*step|timestep|step)", t)
    if m:
        it.dt_ps = float(m.group(1)) / 1000.0
    else:
        m = re.search(r"(?:time\s*step|timestep)\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*fs", t)
        if m:
            it.dt_ps = float(m.group(1)) / 1000.0

    # ---- box geometry ------------------------------------------------------ #
    # These were UNCOVERED, so an explicit request ("use 2.0 nm of padding") was
    # silently overwritten by the deterministic default. Coverage IS the fix.
    m = re.search(r"(\d+(?:\.\d+)?)\s*nm\s*(?:of\s+)?(?:padding|pad|buffer)", t)
    if not m:
        m = re.search(r"(?:padding|pad|buffer)\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*nm", t)
    if m:
        it.box_padding_nm = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*nm\s*(?:cubic\s*)?box", t)
    if m:
        it.box_size_nm = float(m.group(1))

    # ---- water model -------------------------------------------------------- #
    # MOST SPECIFIC FIRST. r"tip4p" matched inside "tip4p-ew" and pinned plain TIP4P --
    # a different water model, with different charges, pinned so nothing downstream fixed it.
    for wm, pat in (("tip4pew", r"tip4p\s*[-/ ]?\s*ew"), ("spce", r"spc\s*/?\s*e\b"),
                    ("tip3p", r"tip3p"), ("tip4p", r"tip4p"),
                    ("tip5p", r"tip5p"), ("spc", r"\bspc\b")):
        if re.search(pat, t):
            it.water_model = wm
            break

    # ---- what molecule ---------------------------------------------------- #
    # A PDB id is [0-9][A-Za-z0-9]{3} -- but so is "10ns", "20ps", "310k". This regex
    # fabricated pdb_id="10NS" from "simulate lysozyme for 10ns", which 404s at RCSB,
    # AND it ran before the name lookup, so it destroyed the correct id (1AKI) too.
    UNIT = re.compile(r"^\d+(?:ns|ps|fs|nm|k|m|c|s|g|l)$")
    pdb = None
    for mm in re.finditer(r"\b([0-9][a-z0-9]{3})\b", t):
        tok = mm.group(1)
        if not UNIT.match(tok):
            pdb = tok.upper()
            break
    if pdb:
        it.pdb_id = pdb
        it.kind = "protein"
    else:
        for name in sorted(KNOWN_PDB, key=len, reverse=True):
            if name in t:
                it.pdb_id = KNOWN_PDB[name]
                it.kind = "protein"
                break
    if it.kind is None:
        if re.search(r"\bwater\b|solvent box|spc/?e|tip3p|water box", t):
            it.kind = "solvent"
        elif re.search(r"argon|lennard[- ]jones|\blj\b", t):
            it.kind = "fluid"
        elif re.search(r"protein|peptide|enzyme", t):
            it.kind = "protein"
            guess = _pdb_lookup(nl)
            if guess:
                it.pdb_id = guess
            else:
                it.uncovered.append("a protein was requested but no structure could be resolved")

    return it


def verify(plan: dict, it: Intent) -> list:
    """Does the plan satisfy the contract? Returns violations."""
    v = []
    s = plan.get("system", {})
    stages = plan.get("stages", [])
    dyn = [x for x in stages if x.get("type") == "dynamics"]

    if it.kind and s.get("kind") != it.kind:
        v.append(f"kind: asked for {it.kind}, plan says {s.get('kind')}")
    if it.pdb_id and str(s.get("pdb_id", "")).upper() != it.pdb_id:
        v.append(f"structure: asked for {it.pdb_id}, plan says '{s.get('pdb_id')}'")
    if it.salt_M is not None and abs(float(s.get("salt_conc_M", 0)) - it.salt_M) > 0.02:
        v.append(f"salt: asked for {it.salt_M} M, plan says {s.get('salt_conc_M')} M")
    if it.forcefield and s.get("forcefield") != it.forcefield:
        v.append(f"forcefield: asked for {it.forcefield}, plan says {s.get('forcefield')}")
    if it.temperature_K is not None:
        temps = [float((x.get("params") or {}).get("temperature", 0)) for x in dyn]
        if not temps or any(abs(t - it.temperature_K) > 1 for t in temps):
            v.append(f"temperature: asked for {it.temperature_K} K, plan says {temps}")
    if it.production_ns is not None and dyn:
        prod = dyn[-1].get("sim_time_ns", 0)
        if abs(float(prod) - it.production_ns) > 1e-6:
            v.append(f"production time: asked for {it.production_ns} ns, plan says {prod}")
    if it.equilibrate and len(stages) < 3:
        v.append(f"equilibration: asked for a protocol, plan has {len(stages)} stage(s)")
    if it.box_padding_nm is not None and abs(
            float(s.get("box_padding_nm", 0)) - it.box_padding_nm) > 0.01:
        v.append(f"padding: asked for {it.box_padding_nm} nm, plan says {s.get('box_padding_nm')}")
    if it.box_size_nm is not None and abs(
            float(s.get("box_size_nm", 0)) - it.box_size_nm) > 0.01:
        v.append(f"box: asked for {it.box_size_nm} nm, plan says {s.get('box_size_nm')}")
    if it.water_model and s.get("water_model") != it.water_model:
        v.append(f"water: asked for {it.water_model}, plan says {s.get('water_model')}")
    if it.dt_ps is not None:
        for st in stages:
            if st.get("type") == "dynamics":
                got = (st.get("params") or {}).get("dt")
                if got is None or abs(float(got) - it.dt_ps) > 1e-6:
                    v.append(f"dt: asked for {it.dt_ps} ps, plan says {got}")
                break
    if it.ensemble:                        # there was NO ensemble check at all
        dyn = [st for st in stages if st.get("type") == "dynamics"]
        if dyn:
            got = (dyn[-1].get("params") or {}).get("ensemble")
            if got != it.ensemble:
                v.append(f"ensemble: asked for {it.ensemble}, production stage says {got}")
    return v


def enforce(plan: dict, it: Intent) -> dict:
    """OVERWRITE the plan so it satisfies the contract. The model does not get a vote."""
    from ..plan.defaults import complete_system
    p = json.loads(json.dumps(plan))     # deep copy
    s = p.setdefault("system", {})

    if it.kind:
        s["kind"] = it.kind
    if it.pdb_id:
        s["pdb_id"] = it.pdb_id
        s["structure_source"] = "rcsb"
    if it.salt_M is not None:
        s["salt_conc_M"] = it.salt_M
        s["neutralize"] = True
    if it.forcefield:
        s["forcefield"] = it.forcefield
    if it.box_padding_nm is not None:
        s["box_padding_nm"] = it.box_padding_nm
    if it.box_size_nm is not None:
        s["box_size_nm"] = it.box_size_nm
    if it.water_model:
        s["water_model"] = it.water_model
    # protocol: if the user asked for equilibration, the SHAPE is a deterministic
    # template, not something the model improvises.
    if it.equilibrate and it.kind == "protein":
        T = it.temperature_K or 300.0
        prod = it.production_ns if it.production_ns is not None else 0.1
        # The user's ensemble applies to PRODUCTION. The equilibration stages are NVT
        # then NPT by construction (you relax the box before you sample from it), but an
        # explicit "NVT production" used to be silently overwritten with NPT here, and
        # verify() had no ensemble check to catch it.
        prod_ens = it.ensemble or "NPT"
        p["stages"] = [
            {"name": "minimize", "type": "minimize", "max_steps": 5000, "sim_time_ns": 0.0,
             "posres_fc_kj": 0.0, "params": {"ensemble": "NVT", "temperature": T}},
            {"name": "nvt", "type": "dynamics", "sim_time_ns": 0.05, "posres_fc_kj": 1000.0,
             "params": {"ensemble": "NVT", "temperature": T}},
            {"name": "npt", "type": "dynamics", "sim_time_ns": 0.05, "posres_fc_kj": 1000.0,
             "params": {"ensemble": "NPT", "temperature": T}},
            {"name": "production", "type": "dynamics", "sim_time_ns": prod,
             "posres_fc_kj": 0.0, "params": {"ensemble": prod_ens, "temperature": T}},
        ]
        p["analyses"] = ["rmsd", "gyrate", "rmsf"]
    else:
        for st in p.get("stages", []):
            pr = st.setdefault("params", {})
            if it.temperature_K is not None:
                pr["temperature"] = it.temperature_K
            if it.ensemble and st.get("type") == "dynamics":
                pr["ensemble"] = it.ensemble
        dyn = [x for x in p.get("stages", []) if x.get("type") == "dynamics"]
        if it.production_ns is not None and dyn:
            dyn[-1]["sim_time_ns"] = it.production_ns
        if not p.get("analyses"):
            p["analyses"] = (["rdf_ow", "msd_ow"] if s.get("kind") in ("solvent", "fluid")
                             else ["rmsd", "gyrate", "rmsf"])

    # the last four physically-consequential fields the model still touched
    # (water_model, box_shape, box_size_nm, box_padding_nm) are deterministic
    # functions of (force field, kind, cutoff). Take them off the model.
    pinned = {k for k in ("forcefield", "water_model", "box_shape",
                          "box_size_nm", "box_padding_nm")
              if getattr(it, k, None) is not None}
    if it.forcefield:
        pinned.add("forcefield")
    s2, prov = complete_system(p["system"], rcoulomb_nm=1.0, pinned=pinned)
    p["system"] = s2
    p["_provenance"] = prov
    # dt is pinned LAST and on `p` -- the dict we actually return.
    # It used to be written into `plan` (the caller's input) BEFORE the template block,
    # so it mutated a dict nobody reads and was then wiped by the template anyway. The
    # returned plan kept the MODEL's timestep: an unpinned physical field, silently.
    if it.dt_ps is not None:
        for st in p.get("stages") or []:
            if st.get("type") == "dynamics":
                st.setdefault("params", {})["dt"] = it.dt_ps

    return p
