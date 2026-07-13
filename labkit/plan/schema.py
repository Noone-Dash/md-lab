"""The Plan contract — a simulation as data an LLM can emit and a human can read."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

SCHEMA_VERSION = "plan/1"
STAGE_TYPES = ("minimize", "dynamics")
SYSTEM_KINDS = ("fluid", "solvent", "protein", "membrane", "qm")


class PlanError(ValueError):
    pass


@dataclass
class SystemSpec:
    kind: str = "solvent"                 # fluid | solvent | protein | membrane | qm
    forcefield: str = "amber99sb-ildn"
    water_model: str = "tip3p"
    box_shape: str = "cubic"
    box_size_nm: float = 3.0              # solvent/fluid: explicit edge
    box_padding_nm: float = 1.2           # protein: solute-to-wall distance
    salt_conc_M: float = 0.0
    neutralize: bool = True
    structure_source: str = "none"        # rcsb | alphafold | file | none
    pdb_id: str = ""
    ignore_hydrogens: bool = True
    lipid: str = "POPC"                   # membrane only
    extras: dict = field(default_factory=dict)


@dataclass
class Stage:
    name: str
    type: str = "dynamics"                # minimize | dynamics
    sim_time_ns: float = 0.0              # dynamics: physical time (nsteps is DERIVED)
    max_steps: int = 50000                # minimize: step cap
    posres_fc_kj: float = 0.0             # 0 = unrestrained
    params: dict = field(default_factory=dict)   # any ontology key -> value


@dataclass
class Plan:
    name: str = "unnamed"
    system: SystemSpec = field(default_factory=SystemSpec)
    stages: list = field(default_factory=list)
    analyses: list = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    # ---- (de)serialisation ------------------------------------------------ #
    def to_dict(self):
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        if not isinstance(d, dict):
            raise PlanError("plan must be a JSON object")
        unknown = set(d) - {"name", "system", "stages", "analyses", "schema_version", "_provenance"}
        if unknown:
            raise PlanError(f"unknown top-level key(s): {sorted(unknown)}")

        sysd = d.get("system", {}) or {}
        sys_fields = {f for f in SystemSpec.__dataclass_fields__}
        bad = set(sysd) - sys_fields
        if bad:
            # loud, not silent: a fabricated knob must not slip through
            raise PlanError(f"unknown system key(s): {sorted(bad)}. "
                            f"Valid: {sorted(sys_fields)}")
        system = SystemSpec(**sysd)
        if system.kind not in SYSTEM_KINDS:
            raise PlanError(f"system.kind must be one of {SYSTEM_KINDS}, got {system.kind!r}")

        stages = []
        for i, s in enumerate(d.get("stages", []) or []):
            sf = {f for f in Stage.__dataclass_fields__}
            badk = set(s) - sf
            if badk:
                raise PlanError(f"stage[{i}] unknown key(s): {sorted(badk)}. Valid: {sorted(sf)}")
            st = Stage(**s)
            if st.type not in STAGE_TYPES:
                raise PlanError(f"stage[{i}].type must be one of {STAGE_TYPES}, got {st.type!r}")
            stages.append(st)
        if not stages:
            raise PlanError("a plan needs at least one stage")

        return Plan(name=d.get("name", "unnamed"), system=system, stages=stages,
                    analyses=list(d.get("analyses", []) or []))
