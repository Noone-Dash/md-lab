"""The rules engine. Turns a Plan into structured, fixable findings an agent can act on."""

from __future__ import annotations

import json
from pathlib import Path

from .expr import compile_expr, eval_expr
from .resolve import resolve, regime
from .schema import Plan

DATA = Path(__file__).resolve().parent.parent / "data"
_raw = json.loads((DATA / "rules.json").read_text())

# compile at import: a rules.json that doesn't compile fails loudly here, not at submit
RULES = []
for r in _raw["rules"]:
    RULES.append({
        **r,
        "_when": compile_expr(r["when"]),
        "_assert": compile_expr(r["assert"]),
    })


def _ctx(plan: Plan, rstage: dict) -> dict:
    """Flat context: resolved mdp (hyphens->underscores) + system facts."""
    s = plan.system
    mdp = rstage["mdp"]
    ctx = {k.replace("-", "_"): v for k, v in mdp.items()}

    # list-valued couplings -> scalars the rules can compare
    reft = mdp.get("ref-t", [0])
    ctx["ref_t_max"] = max(reft) if isinstance(reft, (list, tuple)) else float(reft or 0)

    box_min = float(s.box_size_nm or 0)
    if s.kind == "protein":
        # a protein box isn't known until editconf runs; validate what we CAN:
        # padding-vs-cutoff is checked by its own rule, so disable the box rule here
        box_min = 0.0

    ctx.update({
        "kind": s.kind, "forcefield": s.forcefield, "water_model": s.water_model,
        "box_min_nm": box_min, "box_size_nm": float(s.box_size_nm or 0),
        "box_padding_nm": float(s.box_padding_nm or 0),
        "salt_conc_M": float(s.salt_conc_M or 0), "neutralize": bool(s.neutralize),
        "regime": regime(s),
        "stage_type": rstage["type"], "stage_name": rstage["name"],
        "posres_fc": float(rstage.get("posres_fc_kj") or 0),
    })
    ctx.setdefault("gen_vel", False)
    ctx.setdefault("continuation", False)
    ctx.setdefault("pcoupl", "no")
    ctx.setdefault("pcoupltype", "")
    ctx.setdefault("tcoupl", "no")
    ctx.setdefault("nsteps", 0)
    ctx.setdefault("nstlist", 20)
    ctx.setdefault("epsilon_r", 1)
    ctx.setdefault("dt", 0.0)
    ctx.setdefault("constraints", "none")
    return ctx


def _apply_autofix(plan: Plan, name: str, rstage: dict) -> bool:
    s = plan.system
    if name == "clamp_cutoff":
        box = float(s.box_size_nm or 0)
        if box <= 0:
            return False
        rc = round(0.49 * box, 3)
        for st in plan.stages:
            st.params["rvdw"] = rc
            st.params["rcoulomb"] = rc
        return True
    if name == "set_neutralize":
        s.neutralize = True
        return True
    if name == "set_hbonds":
        for st in plan.stages:
            if st.type == "dynamics":
                st.params["constraints"] = "h-bonds"
        return True
    if name == "no_pcoupl_in_min":
        for st in plan.stages:
            if st.type == "minimize":
                st.params["pcoupl"] = "no"
        return True
    return False


def _check_enums(plan: Plan) -> list:
    """Catch a value that simply does not exist — e.g. an LLM setting
    forcefield='spce' (that's a WATER MODEL). Options come from the ontology,
    which reads the force fields actually installed on this machine."""
    from .ontology import get_param
    out = []
    checks = [
        ("forcefield", plan.system.forcefield),
        ("water_model", plan.system.water_model),
        ("box_shape", plan.system.box_shape),
        ("structure_source", plan.system.structure_source),
    ]
    for key, val in checks:
        p = get_param(key)
        if not p or not p.get("options") or val in (None, ""):
            continue
        opts = [str(o) for o in p["options"]]
        if str(val) not in opts:
            out.append({
                "rule": f"enum.{key}", "severity": "error", "stage": "system",
                "message": f"system.{key} = '{val}' is not a valid value. "
                           f"It is not one of the {len(opts)} available options"
                           + (" installed on this machine." if key == "forcefield" else "."),
                "fix": f"Use one of: {', '.join(opts[:8])}"
                       + (" …" if len(opts) > 8 else ""),
                "autofixable": False,
            })
    # a very common LLM confusion, worth its own message
    if str(plan.system.forcefield).lower() in ("spce", "spc", "tip3p", "tip4p", "tip5p"):
        out[-1]["message"] += (" — you have given a WATER MODEL as the force field. "
                               "The force field parameterises the solute; water_model "
                               "parameterises the solvent. They are different fields.")
    return out


def validate(plan: Plan, autofix: bool = False) -> dict:
    """-> {'ok': bool, 'errors': n, 'warnings': n, 'findings': [...]}"""
    findings = []
    for _pass in range(3 if autofix else 1):
        findings = []
        rp = resolve(plan)
        for rstage in rp["stages"]:
            ctx = _ctx(plan, rstage)
            for r in RULES:
                gate = eval_expr(r["_when"], ctx)
                if gate is not True:
                    continue
                ok = eval_expr(r["_assert"], ctx)
                if ok is None:          # unevaluable -> skip, never a false pass
                    continue
                if not ok:
                    findings.append({
                        "rule": r["id"], "severity": r["severity"],
                        "stage": rstage["name"], "message": r["message"],
                        "fix": r["fix"], "autofixable": bool(r.get("autofix")),
                    })
        if not autofix:
            break
        applied = False
        for f in findings:
            rid = f["rule"]
            rule = next(x for x in RULES if x["id"] == rid)
            if rule.get("autofix") and _apply_autofix(plan, rule["autofix"], None):
                f["applied"] = True
                applied = True
        if not applied:
            break

    findings = _check_enums(plan) + findings

    errors = sum(1 for f in findings if f["severity"] == "error")
    warnings = sum(1 for f in findings if f["severity"] == "warning")
    return {"ok": errors == 0, "errors": errors, "warnings": warnings,
            "findings": findings, "n_rules": len(RULES) + 4}
