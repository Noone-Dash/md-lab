"""The Plan JSON Schema — generated from the dataclasses AND the ontology.

This is the grammar the translator model is constrained to. Every enum is pulled
from what is actually installed on this machine, so an illegal value is not
"caught later" — it is UNREPRESENTABLE. The model cannot emit forcefield="spce"
because that token sequence is not in the grammar.

Determinism note: this schema is the boundary. Everything downstream of it
(resolve -> validate -> mdp_emit -> GROMACS) is pure deterministic code with no
model in the loop.
"""

from __future__ import annotations

from .ontology import get_param
from .schema import STAGE_TYPES, SYSTEM_KINDS


def _opts(key, fallback):
    p = get_param(key)
    o = (p or {}).get("options")
    return [str(x) for x in o] if o else fallback


def plan_schema() -> dict:
    ff = _opts("forcefield", ["amber99sb-ildn"])
    water = _opts("water_model", ["tip3p", "spce"])
    box = _opts("box_shape", ["cubic", "dodecahedron"])
    src = _opts("structure_source", ["rcsb", "alphafold", "file", "none"])

    stage = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": list(STAGE_TYPES)},
            "sim_time_ns": {"type": "number", "minimum": 0, "maximum": 100},
            "max_steps": {"type": "integer", "minimum": 100, "maximum": 500000},
            "posres_fc_kj": {"type": "number", "minimum": 0, "maximum": 10000},
            "params": {
                "type": "object",
                "properties": {
                    "ensemble": {"type": "string", "enum": ["NVE", "NVT", "NPT"]},
                    "temperature": {"type": "number", "minimum": 1, "maximum": 1000},
                    "dt": {"type": "number", "minimum": 0.0001, "maximum": 0.04},
                    "constraints": {"type": "string",
                                    "enum": ["none", "h-bonds", "all-bonds"]},
                },
                "additionalProperties": False,
            },
        },
        # temperature/ensemble are REQUIRED: a silently-omitted field inherits a
        # default, which is how "body temperature" quietly became 300 K.
        "required": ["name", "type", "params"],
        "additionalProperties": False,
    }
    stage["properties"]["params"]["required"] = ["ensemble", "temperature"]

    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "system": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(SYSTEM_KINDS)},
                    "forcefield": {"type": "string", "enum": ff},
                    "water_model": {"type": "string", "enum": water},
                    "box_shape": {"type": "string", "enum": box},
                    "box_size_nm": {"type": "number", "minimum": 1.8, "maximum": 20},
                    "box_padding_nm": {"type": "number", "minimum": 0.8, "maximum": 3},
                    "salt_conc_M": {"type": "number", "minimum": 0, "maximum": 3},
                    "neutralize": {"type": "boolean"},
                    "structure_source": {"type": "string", "enum": src},
                    "pdb_id": {"type": "string"},
                },
                # same reasoning: make the model COMMIT to salt and structure,
                # rather than omitting them and inheriting 0.0 / "".
                "required": ["kind", "forcefield", "water_model",
                             "salt_conc_M", "structure_source", "pdb_id"],
                "additionalProperties": False,
            },
            "stages": {"type": "array", "items": stage, "minItems": 1, "maxItems": 6},
            "analyses": {
                "type": "array",
                "items": {"type": "string",
                          "enum": ["rmsd", "gyrate", "rmsf", "rdf_ow", "msd_ow"]},
            },
        },
        "required": ["name", "system", "stages"],
        "additionalProperties": False,
    }


def legal_values() -> dict:
    """What the grammar allows — for prompts, docs and the UI."""
    s = plan_schema()
    sysp = s["properties"]["system"]["properties"]
    return {
        "system.kind": sysp["kind"]["enum"],
        "forcefield": sysp["forcefield"]["enum"],
        "water_model": sysp["water_model"]["enum"],
        "box_shape": sysp["box_shape"]["enum"],
        "structure_source": sysp["structure_source"]["enum"],
        "stage.type": s["properties"]["stages"]["items"]["properties"]["type"]["enum"],
        "analyses": s["properties"]["analyses"]["items"]["enum"],
    }
