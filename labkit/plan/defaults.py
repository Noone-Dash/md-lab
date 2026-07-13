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
