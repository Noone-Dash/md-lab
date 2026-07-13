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
DEFAULT_TEMPERATURE_K = 300.0
DEFAULT_PRODUCTION_NS = 0.1
DT_BY_REGIME = {"atomistic": 0.002, "martini": 0.02}      # ps
CONSTRAINTS_BY_REGIME = {"atomistic": "h-bonds", "martini": "none"}


def default_ensemble(kind: str) -> str:
    # A condensed phase needs its box relaxed against a barostat; a dilute gas does not
    # (an NPT barostat on a near-ideal gas is numerically miserable).
    return "NVT" if kind == "fluid" else "NPT"


def complete_stages(plan: dict, intent=None) -> dict:
    """Overwrite every stage parameter with intent-or-default. Discards the model's."""
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

    T = getattr(intent, "temperature_K", None) or DEFAULT_TEMPERATURE_K
    ens = getattr(intent, "ensemble", None) or default_ensemble(kind)
    dt = getattr(intent, "dt_ps", None) or DT_BY_REGIME.get(reg, 0.002)

    dyn = [st for st in (plan.get("stages") or []) if st.get("type") == "dynamics"]
    for st in plan.get("stages") or []:
        pr = st.setdefault("params", {})
        pr["temperature"] = T
        pr["constraints"] = CONSTRAINTS_BY_REGIME.get(reg, "h-bonds")
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
        prod_ns = getattr(intent, "production_ns", None)
        if prod_ns is None and dyn[-1].get("sim_time_ns") is None:
            dyn[-1]["sim_time_ns"] = DEFAULT_PRODUCTION_NS
    return plan
