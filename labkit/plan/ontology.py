"""Loader/query over labkit/data/ontology.json — the parameter comprehension surface."""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
_raw = json.loads((DATA / "ontology.json").read_text())

ONTOLOGY: list[dict] = _raw["parameters"]
_BY_KEY = {p["key"]: p for p in ONTOLOGY}


def get_param(key: str) -> dict | None:
    return _BY_KEY.get(key)


def known(key: str) -> bool:
    return key in _BY_KEY


def params_for(area: str = None, applies_to: str = None) -> list[dict]:
    out = ONTOLOGY
    if area:
        out = [p for p in out if p["area"] == area]
    if applies_to:
        out = [p for p in out if not p["applies_to"] or applies_to in p["applies_to"]]
    return out


def mdp_keys() -> dict:
    """logical key -> GROMACS mdp key, for every param that maps to one."""
    return {p["key"]: p["mdp_key"] for p in ONTOLOGY if p.get("mdp_key")}


def defaults(applies_to: str = None) -> dict:
    return {p["key"]: p["default"] for p in params_for(applies_to=applies_to)
            if p.get("default") is not None}


def areas() -> list[str]:
    return sorted({p["area"] for p in ONTOLOGY})


def summary() -> dict:
    return {
        "n_parameters": len(ONTOLOGY),
        "areas": {a: len(params_for(area=a)) for a in areas()},
        "n_mdp_mapped": len(mdp_keys()),
    }
