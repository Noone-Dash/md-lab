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
    described: bool = False        # the protein was DESCRIBED, not named -> semantic
    raw_request: str = ""          # so enforce() can gate the model on what was ASKED
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


_WORDNUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}


def _digitise(t: str) -> str:
    """Turn spelled-out quantities into digits BEFORE any numeric regex runs.

    "half a nanosecond" and "two hundred picoseconds" were parsed by NOTHING -- not the
    contract, and not the LLM either, which returned sim_time_ns = 0.0 (a zero-length
    simulation, submitted without complaint). A number written as a word is still a
    number; refusing to read it is not determinism, it is a gap.
    """
    t = re.sub(r"\bhalf\s+an?\b", "0.5", t)
    t = re.sub(r"\ba\s+quarter\s+of\s+an?\b", "0.25", t)
    t = re.sub(r"\bthree\s+quarters\s+of\s+an?\b", "0.75", t)

    def _mul(m):
        a = _WORDNUM.get(m.group(1), 0)
        b = _WORDNUM.get(m.group(2), 1)
        return str(a * b)
    t = re.sub(r"\b(" + "|".join(_WORDNUM) + r")\s+(hundred|thousand)\b", _mul, t)
    for w, v in sorted(_WORDNUM.items(), key=lambda kv: -len(kv[0])):
        if w in ("a", "an"):
            continue
        t = re.sub(r"\b" + w + r"\b(?=\s*(ns|ps|fs|nanosecond|picosecond|femtosecond|"
                   r"nm|nanometre|nanometer|angstrom|kelvin|k\b))", str(v), t)
    return t


def extract(nl: str) -> Intent:
    """Request -> hard assertions. Pure code."""
    t = _digitise(nl.lower())
    it = Intent()
    it.raw_request = nl

    # ---- temperature ----------------------------------------------------- #
    # \b37\b, NOT 37: unanchored, "137 C" contains "37 c" and was silently pinned to
    # 310 K instead of 410.15 K -- a 100 K thermostat error that verify() called clean.
    if re.search(r"body temperature|physiological temperature|\b37\s*°?\s*c\b", t):
        it.temperature_K = 310.0
    elif re.search(r"room temperature|ambient", t):
        it.temperature_K = 300.0
    m = re.search(r"(\d{2,4}(?:\.\d+)?)\s*(?:k\b|kelvin\b)", t)
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
    if re.search(r"\b(no|without|zero)\s+(salt|ions?|counter[- ]?ions?)\b|salt[- ]free|"
                 r"\bion[- ]free\b", t):
        it.salt_M = 0.0

    # ---- ensemble --------------------------------------------------------- #
    if re.search(r"\bnpt\b|constant pressure|fixed pressure|"
                 r"box (?:can )?breathe|barostat|isobaric", t):
        it.ensemble = "NPT"
    elif re.search(r"\bnvt\b|constant volume|fixed volume|volume fixed|"
                   r"volume constant|isochoric", t):
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
    NM = r"(?:nm|nanometre?s?|nanometer?s?)"
    ANG = r"(?:a|ang|angstroms?|\u00c5)"
    for pat, scale in ((rf"(\d+(?:\.\d+)?)\s*{NM}\s*(?:of\s+)?(?:padding|pad|buffer)", 1.0),
                       (rf"(?:padding|pad|buffer)\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*{NM}", 1.0),
                       (rf"(\d+(?:\.\d+)?)\s*{ANG}\s*(?:of\s+)?(?:padding|pad|buffer)", 0.1),
                       (rf"(?:padding|pad|buffer)\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*{ANG}", 0.1),
                       # "leave 15 angstroms around the solute"
                       (rf"leave\s+(\d+(?:\.\d+)?)\s*{ANG}\b", 0.1),
                       (rf"leave\s+(\d+(?:\.\d+)?)\s*{NM}\b", 1.0)):
        m = re.search(pat, t)
        if m:
            it.box_padding_nm = round(float(m.group(1)) * scale, 4)
            break
    for pat in (rf"(\d+(?:\.\d+)?)\s*{NM}\s*(?:cubic\s*)?box",
                rf"(?:cube|box)\s+(?:of\s+\w+\s+)?(\d+(?:\.\d+)?)\s*{NM}",
                rf"(\d+(?:\.\d+)?)\s*{NM}\s+on\s+(?:a|each)\s+side"):
        m = re.search(pat, t)
        if m:
            it.box_size_nm = float(m.group(1))
            break

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
    it.described = bool(re.search(
        r"\b(enzyme|protein|molecule|peptide|receptor|channel)\s+"
        r"(that|which|responsible|involved|used)\b", t))

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
    elif not it.described:          # a description names its subject only semantically
        for name in sorted(KNOWN_PDB, key=len, reverse=True):
            if name in t:
                it.pdb_id = KNOWN_PDB[name]
                it.kind = "protein"
                break
    if it.described:
        it.kind = "protein"
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


def enforce(plan: dict, it: Intent, request: str = "") -> dict:
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
    # EVERY stage parameter is now (intent -> default), never the model's. Measured: on
    # a bare request the model used to control 12 of 36 mdp keys, dt=10 fs among them.
    from ..plan.defaults import complete_stages, system_provenance
    sysprov = system_provenance(p.get("system") or {}, it)
    complete_stages(p, it, request=request or it.raw_request or "")
    p.setdefault("_provenance", {}).update(sysprov)

    # dt is pinned LAST and on `p` -- the dict we actually return.
    # It used to be written into `plan` (the caller's input) BEFORE the template block,
    # so it mutated a dict nobody reads and was then wiped by the template anyway. The
    # returned plan kept the MODEL's timestep: an unpinned physical field, silently.
    if it.dt_ps is not None:
        for st in p.get("stages") or []:
            if st.get("type") == "dynamics":
                st.setdefault("params", {})["dt"] = it.dt_ps

    return p


# Words that name no protein. If a DESCRIPTION reduces to nothing but these, it carries no
# information and there is nothing to resolve.
_GENERIC = {
    "enzyme", "protein", "molecule", "peptide", "receptor", "channel", "thing", "stuff",
    "something", "one", "it", "does", "do", "make", "makes", "work", "works", "some",
    "any", "this", "that", "which", "what", "kind", "sort", "type", "structure",
}


def _has_content(request: str) -> bool:
    """Does the request actually SAY anything about which protein?

    "Simulate the protein that does the thing with the stuff" produced 1AKI -- lysozyme --
    a real, confident, completely invented answer to a request that named nothing. The
    model will always propose SOME protein, and grounding that name in the PDB only proves
    the NAME is real, never that it ANSWERS the question. So the request itself must carry
    content before we are willing to resolve anything from it.
    """
    from ..structures import _clean
    words = [w for w in _clean(request).split() if len(w) > 2 and w not in _GENERIC]
    return bool(words)


def resolve_structure(it, request: str, model_name: str = None) -> dict:
    """Fill pdb_id from the PDB itself, not from the model's memory.

    The benchmark localised every remaining translator error to this one field. All three
    local models score 100% on covered intents after enforce(); the ONLY thing they still
    source is a PDB id for a protein outside KNOWN_PDB, and they are bad at it -- they
    hallucinate well-formed-but-wrong ids (1GFP, 1TND, 1A4M contain none of the requested
    proteins) or emit no id at all ("GFP", "tendamistat").

    Since d P(correct) / d q = P(not covered), the fix is to SHRINK the uncovered set, not
    to buy a bigger model. A PDB id is a lookup, and the PDB has a search API.

    -> {'pdb_id', 'source', 'title'} and mutates `it`; source=None if it could not be
       resolved (offline, or no verified hit), in which case the caller must fall back to
       the model AND say so.
    """
    if it.pdb_id or it.kind not in ("protein", None):
        return {"pdb_id": it.pdb_id, "source": "intent" if it.pdb_id else None}
    from ..structures import resolve as _resolve
    if not _has_content(request):
        return {"pdb_id": None, "source": None, "reason": "the request names no protein"}
    r = _resolve(request, KNOWN_PDB)
    if not r and model_name:
        # The request named the protein SEMANTICALLY ("the enzyme that digests starch").
        # Code cannot resolve that; the model can ("alpha-amylase"). So the model supplies
        # the NAME and the PDB supplies the ID -- and the hit is still title-verified, so
        # a hallucinated name yields NO id rather than a wrong one.
        r = _resolve(model_name, KNOWN_PDB)
        if r:
            r = dict(r, source="rcsb-via-model-name", via=model_name)
    if r:
        it.pdb_id = r["pdb_id"]
        it.kind = "protein"
        return r
    return {"pdb_id": None, "source": None}


def verify_structure(plan: dict) -> dict:
    """A structure the model supplied and we did NOT resolve must never ship unchecked.

    If the resolver comes up empty the model's own pdb_id survives into the plan. It may
    be right (it gave 1A3N for "the protein that carries oxygen in blood") or it may be a
    hallucination (1B2Q for "the enzyme that digests starch" — not an amylase). Either way
    it is UNVERIFIED, and an unverified structure that runs silently is the failure we keep
    finding. So: check the id actually exists in the PDB, and hand back its real title so
    the user can see what they are about to simulate.
    """
    sysd = plan.get("system") or {}
    pid = sysd.get("pdb_id")
    if not pid or sysd.get("kind") != "protein":
        return {"ok": True}
    from ..structures import _title
    title = _title(pid)
    if title is None:
        return {"ok": True, "note": "offline: could not verify the structure"}
    if title == "":
        return {"ok": False, "error": f"pdb_id {pid!r} does not exist in the PDB "
                                      f"(the model invented it)"}
    return {"ok": True, "title": title[:120]}


SKELETON = {"name": "run", "system": {}, "stages": [{"name": "production",
                                                    "type": "dynamics"}]}


def plan_from_request(request: str, model=None, use_llm: bool = True) -> dict:
    """THE pipeline. One entrypoint so every caller gets the same guarantees.

        extract   (pure)   -> the assertions the request makes
        resolve   (PDB)    -> the structure, deterministically
        translate (LLM)    -> shape only; grammar-constrained     [OPTIONAL]
        enforce   (pure)   -> overwrites every physical value the model touched

    use_llm=False runs the whole thing with NO MODEL AT ALL.

    That is not a degraded mode, and the measurement says so: across all 10 benchmark
    requests, a null skeleton produces a .mdp IDENTICAL to gpt-oss:20b's, key for key.
    Once every physical field is intent-or-default and the structure comes from the PDB,
    the model has nothing left to contribute on a request the contract covers. It earns
    its place only on phrasing the contract does NOT parse -- which is exactly what
    `uncovered` reports, so you can see when that is happening instead of assuming.
    """
    from .translate import translate as _translate
    import copy
    it = extract(request)

    if not use_llm:
        raw = copy.deepcopy(SKELETON)
        raw["system"]["kind"] = it.kind or "solvent"
    else:
        r = _translate(request, model=model)
        raw = r.get("plan")
        if not raw:      # the model failed -> fall back to determinism, do not fail the user
            raw = copy.deepcopy(SKELETON)
            raw["system"]["kind"] = it.kind or "solvent"

    # protein_name is the model's SEMANTIC contribution. It never reaches the Plan (which
    # is strict); it is consumed here, as a search query, and the ID comes from the PDB.
    mname = (raw.get("system") or {}).pop("protein_name", None)
    struct = resolve_structure(it, request, model_name=mname)

    # The model does not get to choose a structure. If the user did not type an explicit
    # PDB id and we could not ground one, DROP the model's id rather than simulate a
    # protein nobody chose. (It gave 1B2Q -- not an amylase -- for "the enzyme that
    # digests starch"; it is right often enough to be dangerous.)
    if struct.get("source") is None:
        (raw.get("system") or {}).pop("pdb_id", None)

    final = enforce(raw, it)
    if (final.get("system") or {}).get("kind") == "protein" and not struct.get("source"):
        return {"plan": None,
                "error": ("could not identify the protein. Name it, or give a 4-character "
                          "PDB id." + (f" (the model suggested {mname!r}, which the PDB "
                                       f"could not confirm)" if mname else "")),
                "intent": it.assertions(), "model_named": mname}

    check = verify_structure(final)
    if not check.get("ok"):
        return {"plan": None, "error": check["error"], "intent": it.assertions(),
                "model_named": mname}
    final.setdefault("_provenance", {})["pdb_id"] = struct.get("source") or "default"
    return {
        "plan": final,
        "raw": raw,
        "intent": it.assertions(),
        "violations": verify(final, it),
        "structure_source": struct.get("source"),   # curated | rcsb | intent | None
        "structure_title": struct.get("title"),
        "used_llm": bool(use_llm),
        "uncovered": list(getattr(it, "ambiguous", [])),
        "model_sourced_structure": struct.get("source") is None and bool(
            (final.get("system") or {}).get("pdb_id")),
    }
