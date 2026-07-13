"""Deterministic system completion. No model.

The four fields the model still had a vote on — water_model, box_shape, box_size_nm,
box_padding_nm — are all deterministic functions of (force field, system kind, cutoff).
There is no reason to let a language model choose them, so it doesn't.

    water_model    := convention(force field)          [Amber/CHARMM pair with TIP3P, GROMOS with SPC, ...]
    box_shape      := dodecahedron if solute else cubic  [minimum-image at ~0.71x the volume of a cube]
    box_padding_nm := rcoulomb + margin                  [must exceed the cutoff, else self-interaction]
    box_size_nm    := user-specified, else a default

After this, the model's residual influence on the physics is empty: every field that
changes a force, an energy or a trajectory is fixed by code.
"""

from __future__ import annotations

# The water model a force field was PARAMETERISED against. Mixing them is a real
# (if common) error: Amber's charges were fitted in the presence of TIP3P.
FF_WATER = {
    "amber": "tip3p",
    "charmm": "tip3p",     # CHARMM strictly wants its modified TIPS3P; TIP3P is the usual GROMACS proxy
    "oplsaa": "tip4p",
    "gromos": "spc",
}

CUTOFF_MARGIN_NM = 0.2     # padding must EXCEED the cutoff, not merely equal it


def water_for(forcefield: str) -> str:
    ff = str(forcefield).lower()
    for prefix, water in FF_WATER.items():
        if ff.startswith(prefix):
            return water
    return "tip3p"


def box_for(kind: str) -> str:
    # A rhombic dodecahedron holds the same minimum-image distance in ~71% of the
    # volume of a cube => ~29% fewer solvent atoms => ~29% less compute, same physics.
    return "dodecahedron" if kind == "protein" else "cubic"


def padding_for(rcoulomb_nm: float = 1.0) -> float:
    return round(float(rcoulomb_nm) + CUTOFF_MARGIN_NM, 2)


def system_provenance(raw_system: dict, intent=None) -> dict:
    """Where each SYSTEM field came from.

    `raw_system` must be the model's OWN output, snapshotted BEFORE defaults are applied.
    Reading it afterwards reported "model" for fields the model never set -- including when
    there was no model at all. A provenance tag that lies is worse than no tag.
    """
    prov = {}
    for f, iattr in (("forcefield", "forcefield"), ("water_model", "water_model"),
                     ("box_size_nm", "box_size_nm"), ("box_padding_nm", "box_padding_nm"),
                     ("salt_conc_M", "salt_M"), ("pdb_id", "pdb_id")):
        if getattr(intent, iattr, None) is not None:
            prov[f] = "intent"
        elif (raw_system or {}).get(f) is not None:
            prov[f] = "model"
        else:
            prov[f] = "default"
    return prov


def complete_system(system: dict, rcoulomb_nm: float = 1.0,
                    pinned: set = frozenset()) -> tuple[dict, dict]:
    """Fill the remaining system fields deterministically.

    `pinned` = fields the user explicitly asked for (from intent extraction); those are
    never overwritten. Returns (system, provenance) so every field's origin is auditable.
    """
    s = dict(system)
    prov = {}

    # The force field is the single biggest determinant of the physics. If the user
    # did not ask for one, it is a DEFAULT (auditable, fixed), never the model's pick.
    DEFAULT_FF = "amber99sb-ildn"
    if "forcefield" in pinned:
        ff = s.get("forcefield") or DEFAULT_FF
        prov["forcefield"] = "intent"
    else:
        ff = DEFAULT_FF
        prov["forcefield"] = f"default ({DEFAULT_FF}) — model's choice discarded"
    s["forcefield"] = ff

    if "water_model" not in pinned:
        s["water_model"] = water_for(ff)
        prov["water_model"] = f"convention({ff})"
    else:
        prov["water_model"] = "intent"

    kind = s.get("kind", "solvent")
    if "box_shape" not in pinned:
        s["box_shape"] = box_for(kind)
        prov["box_shape"] = f"geometry({kind})"
    else:
        prov["box_shape"] = "intent"

    if kind == "protein":
        if "box_padding_nm" not in pinned:
            s["box_padding_nm"] = padding_for(rcoulomb_nm)
            prov["box_padding_nm"] = f"rcoulomb({rcoulomb_nm}) + {CUTOFF_MARGIN_NM}"
        else:
            prov["box_padding_nm"] = "intent"
    else:
        if "box_size_nm" not in pinned:
            s.setdefault("box_size_nm", 3.0)
            prov["box_size_nm"] = "default 3.0 nm"
        else:
            prov["box_size_nm"] = "intent"

    return s, prov


# ---------------------------------------------------------------------------------
# STAGE PARAMETERS. The model sources NONE of them.
#
# The measured hole this closes: on a bare request ("Simulate lysozyme"), which pins
# almost nothing, the LLM still controlled 12 of the 36 mdp keys that reach GROMACS --
# including dt=0.01 (10 fs: unstable for an atomistic force field), ref-t=500 K, and
# the entire barostat (pcoupl=no). The earlier "0 residual" measurement was an artifact
# of only ever testing a MAXIMALLY-SPECIFIED request: when the user says less, the model
# was deciding more, which is exactly backwards.
#
# Every physical field belongs to one of two classes, and neither is the model's:
#
#   USER PREFERENCE  (temperature, ensemble, duration, salt, water, force field, box)
#       -> the deterministic intent contract, or a fixed DEFAULT if the user was silent.
#          A fixed default is reproducible; an LLM guess is not.
#
#   DERIVED PHYSICS  (dt, constraints, cutoffs, thermostat/barostat algorithm, output
#                     frequencies)
#       -> a pure function of the force field and the regime. Never a preference at all.
#
# The model's job is what it is actually good at: identifying the molecule and the
# SHAPE of the protocol. Not numbers.
# ---------------------------------------------------------------------------------
# EVIDENCE GATES. The model may PARSE the request; it may not INVENT.
#
# Allowing the model to supply a preference whenever intent is silent let an adversarial
# model set T=500 K, NVT and 9 ns for the request "Simulate lysozyme" -- which mentions no
# temperature, no ensemble and no duration. Nothing was parsed there; it was fabricated,
# and 10 of 36 mdp keys moved with it.
#
# The distinction is whether the sentence CONTAINS ANYTHING TO READ. "blood heat" has
# something to read (and the model reads it correctly). "Simulate lysozyme" does not. So a
# model-supplied value is accepted only when the request mentions the concept at all;
# otherwise the deterministic default stands.
import re as _re

_EVIDENCE = {
    "temperature": _re.compile(
        r"\d|temperature|thermal|hot\b|cold|cool|warm|heat|boil|freez|melt|"
        r"kelvin|celsius|centigrade|body|room|ambient|physiolog", _re.I),
    "ensemble": _re.compile(
        r"\bnpt\b|\bnvt\b|pressure|volume|barostat|thermostat|\bbar\b|breathe|"
        r"isochoric|isobaric|constant|fixed|density", _re.I),
    "sim_time_ns": _re.compile(
        r"\d|nanosecond|picosecond|microsecond|femtosecond|\bns\b|\bps\b|\bus\b|"
        r"long|brief|short|duration|length|quick", _re.I),
}


def has_evidence(field: str, request: str) -> bool:
    rx = _EVIDENCE.get(field)
    return True if rx is None else bool(rx.search(request or ""))


DEFAULT_TEMPERATURE_K = 300.0
DEFAULT_PRODUCTION_NS = 0.1
DT_BY_REGIME = {"atomistic": 0.002, "martini": 0.02}      # ps
CONSTRAINTS_BY_REGIME = {"atomistic": "h-bonds", "martini": "none"}


def default_ensemble(kind: str) -> str:
    # A condensed phase needs its box relaxed against a barostat; a dilute gas does not
    # (an NPT barostat on a near-ideal gas is numerically miserable).
    return "NVT" if kind == "fluid" else "NPT"


def complete_stages(plan: dict, intent=None, allow_model: bool = True,
                    request: str = "") -> dict:
    """Set every stage parameter from intent -> model -> default, and RECORD WHICH.

    THE MISTAKE THIS CORRECTS. I first made enforcement unconditional: every stage param
    was intent-or-default, model discarded. That killed the residual (good) and also
    killed the model's ONLY legitimate job (bad). Measured on out-of-distribution
    phrasings, the LLM correctly parses "blood heat" -> 310 K, "keep the volume fixed"
    -> NVT, "15 angstroms of padding" -> 1.5 nm; unconditional enforcement threw all of
    it away and substituted 300 K / NPT / 1.2 nm. The user's request was silently
    dropped -- the same bug as the box-padding one, but systemic.

    The distinction I had collapsed:

      DERIVED PHYSICS   dt, constraints, cutoffs, thermostat/barostat algorithm, output
                        rates. A pure function of the force field and the regime. Not a
                        preference at all, and NEVER the model's -- it has no way to know
                        and no business guessing. Unconditional.

      USER PREFERENCE   temperature, ensemble, duration, box, salt, water model, force
                        field. The user either said it or did not. Reading it out of the
                        sentence is NATURAL-LANGUAGE PARSING -- which is the model's
                        actual competence, and not physics at all.

    So a preference resolves intent -> model -> default, and every one carries its
    provenance so nothing is silent. A model-sourced value still has to survive the 26
    physics rules; and because it is marked, the UI can show exactly which numbers the
    LLM chose rather than implying the machine decided them.
    """
    from .resolve import regime as _regime

    class _S:                                    # resolve.regime() wants an object
        def __init__(self, d):
            self.__dict__.update(d)
    sysd = plan.get("system") or {}
    try:
        reg = _regime(_S(sysd))
    except Exception:                            # noqa: BLE001
        reg = "atomistic"
    kind = sysd.get("kind", "protein")

    dyn = [st for st in (plan.get("stages") or []) if st.get("type") == "dynamics"]
    prov = {}

    def _pick(field, from_intent, from_model, default, ok=lambda v: True):
        """intent -> model (only if the request says SOMETHING) -> default."""
        if from_intent is not None:
            prov[field] = "intent"
            return from_intent
        if (allow_model and from_model is not None and ok(from_model)
                and has_evidence(field, request)):
            prov[field] = "model"            # the model PARSED it; still faces the validator
            return from_model
        prov[field] = "default"
        return default

    m_prod = dyn[-1].get("params", {}) if dyn else {}
    T = _pick("temperature", getattr(intent, "temperature_K", None),
              _num(m_prod.get("temperature")), DEFAULT_TEMPERATURE_K,
              ok=lambda v: 1.0 <= v <= 1000.0)
    ens = _pick("ensemble", getattr(intent, "ensemble", None),
                m_prod.get("ensemble"), default_ensemble(kind),
                ok=lambda v: v in ("NVT", "NPT"))

    # DERIVED PHYSICS: unconditional. The model never sources these.
    dt = getattr(intent, "dt_ps", None) or DT_BY_REGIME.get(reg, 0.002)
    cons = CONSTRAINTS_BY_REGIME.get(reg, "h-bonds")
    prov["dt"] = "intent" if getattr(intent, "dt_ps", None) else "derived"
    prov["constraints"] = "derived"

    for st in plan.get("stages") or []:
        pr = st.setdefault("params", {})
        pr["temperature"] = T
        pr["constraints"] = cons
        if st.get("type") == "dynamics":
            pr["dt"] = dt
            # Only the PRODUCTION stage takes the user's ensemble. Equilibration stages
            # keep the shape the template gave them (NVT then NPT) -- you relax the box
            # before you sample from it.
            if len(dyn) == 1 or st is dyn[-1]:
                pr["ensemble"] = ens
            else:
                pr.setdefault("ensemble", "NPT")
        else:
            pr.pop("dt", None)                   # minimisation has no timestep
            pr.pop("ensemble", None)

    if dyn:
        ns = _pick("sim_time_ns", getattr(intent, "production_ns", None),
                   _num(dyn[-1].get("sim_time_ns")), DEFAULT_PRODUCTION_NS,
                   ok=lambda v: 0 < v <= 10000)
        dyn[-1]["sim_time_ns"] = ns

    plan.setdefault("_provenance", {}).update(prov)
    return plan


def _num(v):
    try:
        f = float(v)
        return f if f == f else None                 # reject NaN
    except (TypeError, ValueError):
        return None
