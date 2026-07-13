"""The agent tool surface: pure functions an LLM can call to drive the lab.

Each returns a JSON-able dict. This is the single source of truth for what an
agent can do; the REST endpoints and any MCP server are thin wrappers over it.
"""

from __future__ import annotations

from ..plan import Plan, validate as _validate, estimate as _estimate, resolve, emit_mdp
from ..plan import ontology as onto
from ..plan.schema import PlanError


def list_capabilities() -> dict:
    """What this lab can simulate, and how big the knob surface is."""
    from ..recipes import REGISTRY
    from ..tracks import TRACKS
    return {
        "engines": ["GROMACS 2026.2 (CUDA)", "OpenMM 8.5", "PySCF 2.13", "NumPy cell model"],
        "system_kinds": ["solvent", "fluid", "protein", "membrane", "qm"],
        "plan_supported_kinds": ["solvent", "fluid", "protein"],
        "presets": sorted(list(REGISTRY) + list(TRACKS)),
        "ontology": onto.summary(),
        "note": "Presets are canned. A Plan gives you the full parameter surface.",
    }


def describe_parameters(area: str = None, applies_to: str = None,
                        search: str = None, limit: int = 8) -> dict:
    """The comprehension surface: every knob with unit, meaning and guidance.

    Search is token-based across key/label/meaning/guidance — an LLM asks for
    "thermostat coupling time", not for the literal key 'tau-t'.
    """
    # an LLM will happily pass area="thermostat" or applies_to="water". Those aren't
    # real filter values — ignore a filter we don't recognise rather than silently
    # returning zero results (which made the model think the ontology was empty).
    if area and area not in onto.areas():
        area = None
    if applies_to and applies_to not in ("fluid", "solvent", "protein", "membrane", "qm"):
        applies_to = None

    ps = onto.params_for(area=area, applies_to=applies_to)

    if search:
        toks = [t for t in
                str(search).lower().replace("-", " ").replace("_", " ").split()
                if len(t) > 2]
        scored = []
        for p in ps:
            hay = " ".join([
                p["key"], p.get("label", ""), p.get("meaning", ""),
                p.get("agent_guidance", ""), p.get("area", ""), p.get("unit", ""),
            ]).lower().replace("-", " ").replace("_", " ")
            key = p["key"].lower().replace("-", " ").replace("_", " ")
            score = 0
            for t in toks:
                if t in key:
                    score += 5
                if t in hay:
                    score += 1
            if score:
                scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        ps = [p for _, p in scored]

    total = len(ps)
    trimmed = [{
        "key": p["key"], "unit": p.get("unit"), "type": p.get("type"),
        "default": p.get("default"), "options": p.get("options"),
        "meaning": p.get("meaning", "")[:320],
        "guidance": p.get("agent_guidance", "")[:280],
        "applies_to": p.get("applies_to"),
    } for p in ps[:limit]]

    return {"count": total, "showing": len(trimmed),
            "areas": onto.areas(), "parameters": trimmed}


def validate_plan(plan: dict, autofix: bool = False) -> dict:
    try:
        p = Plan.from_dict(plan)
    except PlanError as e:
        return {"ok": False, "errors": 1, "warnings": 0,
                "findings": [{"rule": "schema", "severity": "error",
                              "message": str(e), "fix": "Correct the plan JSON."}]}
    res = _validate(p, autofix=autofix)
    if autofix:
        res["fixed_plan"] = p.to_dict()
    return res


def estimate_cost(plan: dict, n_atoms: int = None) -> dict:
    try:
        p = Plan.from_dict(plan)
    except PlanError as e:
        return {"error": str(e)}
    return _estimate(p, n_atoms=n_atoms)


def preview_mdp(plan: dict) -> dict:
    """Show the exact GROMACS input a plan will produce — no hidden knobs."""
    try:
        p = Plan.from_dict(plan)
    except PlanError as e:
        return {"error": str(e)}
    rp = resolve(p)
    return {"stages": [{"name": s["name"], "type": s["type"], "mdp": emit_mdp(s)}
                       for s in rp["stages"]]}


def submit_plan(plan: dict) -> dict:
    """Validate then queue a plan on the scheduler (respects the GPU budget)."""
    v = validate_plan(plan)
    if not v["ok"]:
        return {"submitted": False, "validation": v}
    from ..scheduler import SCHED
    run_id = SCHED.submit_plan(plan)
    return {"submitted": True, "run_id": run_id, "validation": v}


def get_run(run_id: str) -> dict:
    from ..engine import load_run
    m = load_run(run_id)
    if not m:
        return {"error": f"no such run {run_id}"}
    return {
        "id": m["id"], "status": m["status"], "error": m.get("error"),
        "steps": m.get("steps"), "outputs": m.get("outputs"),
        "energy_terms": (m.get("energy") or {}).get("legends"),
        "analyses": [a["name"] for a in m.get("analyses", [])],
        "validation": m.get("validation"),
    }


def wait_for_run(run_id: str, timeout_s: int = 900) -> dict:
    """Block until a run finishes (or times out). Use this instead of polling get_run
    in a loop — polling wastes your tool-call budget."""
    import time
    from ..engine import load_run
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        m = load_run(run_id)
        if not m:
            return {"error": f"no such run {run_id}"}
        if m["status"] in ("done", "error", "killed"):
            return {"run_id": run_id, "status": m["status"], "error": m.get("error"),
                    "waited_s": round(time.time() - t0, 1),
                    "next": "call get_results(run_id) to judge the physics"}
        time.sleep(3)
    return {"run_id": run_id, "status": "timeout", "waited_s": timeout_s}


def get_results(run_id: str) -> dict:
    """Results of a finished run, with the physics ALREADY MEASURED.

    Returns a 'measured' block (equilibrium averages, not raw arrays) plus reference
    values, so the judgement is about physics rather than about reading a plot.
    """
    from ..engine import load_run
    from ..evals.metrics import extract
    m = load_run(run_id)
    if not m:
        return {"error": f"no such run {run_id}"}
    if m["status"] != "done":
        return {"id": run_id, "status": m["status"], "error": m.get("error"),
                "note": "not finished — nothing to judge yet"}

    def num(spec):
        v = extract(m, spec)
        return None if v is None else round(float(v), 4)

    measured = {
        "temperature_mean_K": num({"type": "energy_mean", "term": "Temperature"}),
        "density_mean_kg_m3": num({"type": "energy_mean", "term": "Density"}),
        "pressure_mean_bar": num({"type": "energy_mean", "term": "Pressure"}),
        "potential_energy_mean_kJ_mol": num({"type": "energy_mean", "term": "Potential"}),
        "potential_drift_kJ_mol": num({"type": "drift", "term": "Potential"}),
        "rdf_first_peak_nm": num({"type": "rdf_peak", "which": "pos"}),
        "rdf_first_peak_height": num({"type": "rdf_peak", "which": "height"}),
        "rmsd_plateau_nm": num({"type": "analysis_final", "name": "rmsd"}),
        "radius_of_gyration_nm": num({"type": "analysis_final", "name": "gyrate"}),
    }
    measured = {k: v for k, v in measured.items() if v is not None}

    reference = {
        "density_mean_kg_m3": "liquid water ~997 (experiment). 985-1010 is healthy.",
        "temperature_mean_K": "must equal the thermostat setpoint within a few K.",
        "rdf_first_peak_nm": "water O-O first shell = 0.28 nm (neutron scattering).",
        "rmsd_plateau_nm": "a stable folded protein plateaus below ~0.3 nm; >0.5 nm means it moved a lot or unfolded.",
        "pressure_mean_bar": "instantaneous pressure fluctuates by hundreds of bar; only the MEAN (~1 bar in NPT) is meaningful.",
    }

    o = m.get("outputs") or {}
    return {
        "id": m["id"], "status": m["status"],
        "system": {"n_atoms": o.get("n_atoms"), "n_frames": o.get("n_frames")},
        "measured": measured,
        "reference_values": {k: v for k, v in reference.items() if k in measured},
        "available_series": list((m.get("energy") or {}).get("legends") or [])
                            + [a["name"] for a in m.get("analyses", [])],
    }


TOOLS = {
    "list_capabilities": list_capabilities,
    "describe_parameters": describe_parameters,
    "validate_plan": validate_plan,
    "estimate_cost": estimate_cost,
    "preview_mdp": preview_mdp,
    "submit_plan": submit_plan,
    "get_run": get_run,
    "wait_for_run": wait_for_run,
    "get_results": get_results,
}
