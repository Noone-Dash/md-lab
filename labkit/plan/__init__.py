"""The agent-comprehensible layer: a simulation is DATA (a Plan), not a hardcoded recipe.

    Plan(json) -> validate() -> estimate() -> resolve() -> mdp_emit() -> build() -> run

Every MD setting an agent may touch is declared in labkit/data/ontology.json with
its unit, meaning, dependencies and guidance. mdp_emit is the ONLY module in the
repo allowed to write .mdp text, so a knob that isn't in the ontology cannot exist.
"""

from .schema import Plan, Stage, SystemSpec
from .ontology import ONTOLOGY, get_param, params_for, mdp_keys
from .resolve import resolve
from .mdp_emit import emit_mdp
from .validate import validate
from .cost import estimate

__all__ = ["Plan", "Stage", "SystemSpec", "ONTOLOGY", "get_param", "params_for",
           "mdp_keys", "resolve", "emit_mdp", "validate", "estimate"]
