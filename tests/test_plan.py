"""Regression tests for the plan/agent layer. Each pins a bug that actually happened.

    ./.venv/bin/python tests/test_plan.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labkit.plan import Plan, validate, estimate, resolve, emit_mdp          # noqa: E402
from labkit.plan.mdp_emit import MdpError                                    # noqa: E402
from labkit.plan.schema import PlanError                                     # noqa: E402
from labkit.plan.ontology import summary, known                              # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    (print(f"  ok   {name}") if cond
     else (print(f"  FAIL {name}  {detail}"), FAILED.append(name)))


def water(**stage_params):
    return Plan.from_dict({
        "system": {"kind": "solvent", "box_size_nm": 3.0},
        "stages": [{"name": "prod", "type": "dynamics", "sim_time_ns": 0.01,
                    "params": {"temperature": 300, **stage_params}}]})


# --- BUG: ref-t was clobbered to a scalar while tau-t stayed a per-group list,
#     so grompp aborted: "2 groups, 1 ref-t values and 2 tau-t values".
def test_per_group_coupling():
    p = Plan.from_dict({
        "system": {"kind": "protein", "structure_source": "rcsb", "pdb_id": "1AKI"},
        "stages": [{"name": "nvt", "type": "dynamics", "sim_time_ns": 0.01,
                    "params": {"ensemble": "NVT", "temperature": 310}}]})
    mdp = resolve(p)["stages"][0]["mdp"]
    ngrp = len(str(mdp["tc-grps"]).split())
    check("per-group ref-t/tau-t counts match tc-grps",
          len(mdp["ref-t"]) == ngrp and len(mdp["tau-t"]) == ngrp,
          f"grps={ngrp} ref-t={mdp['ref-t']} tau-t={mdp['tau-t']}")


# --- INVARIANT: no undocumented mdp key may ever be written.
def test_ontology_guard():
    st = {"name": "x", "type": "dynamics", "mdp": {"integrator": "md", "totally_made_up": 1}}
    try:
        emit_mdp(st)
        check("mdp_emit rejects an undeclared knob", False, "it accepted it")
    except MdpError:
        check("mdp_emit rejects an undeclared knob", True)
    check("'define' is declared (it was the one that got rejected for real)", known("define"))


# --- INVARIANT: a fabricated plan key fails loudly instead of being silently dropped.
def test_schema_strict():
    try:
        Plan.from_dict({"system": {"kind": "protein", "ph": 7.4},
                        "stages": [{"name": "a", "type": "dynamics"}]})
        check("schema rejects fabricated key", False, "accepted 'ph'")
    except PlanError:
        check("schema rejects fabricated key", True)


def test_rules():
    cases = [
        ("minimum image", water(rvdw=1.6, rcoulomb=1.6), "geom.minimum_image"),
        ("dt too big unconstrained", water(constraints="none", dt=0.002),
         "dt.unconstrained_too_large"),
        ("semiisotropic on a liquid", water(ensemble="NPT", pcoupltype="semiisotropic"),
         "barostat.semiisotropic_membrane_only"),
        ("minimiser in a dynamics stage", water(integrator="steep"), "integrator.dynamics"),
    ]
    for label, plan, rule in cases:
        r = validate(plan)
        check(f"validator catches: {label}",
              any(f["rule"] == rule for f in r["findings"]),
              f"got {[f['rule'] for f in r['findings']]}")

    good = Plan.from_dict({
        "system": {"kind": "solvent", "box_size_nm": 3.0, "water_model": "spce"},
        "stages": [{"name": "minimize", "type": "minimize", "max_steps": 5000},
                   {"name": "prod", "type": "dynamics", "sim_time_ns": 0.05,
                    "params": {"ensemble": "NPT", "temperature": 300}}]})
    r = validate(good)
    check("a sane plan validates clean", r["ok"] and r["warnings"] == 0,
          str([f["rule"] for f in r["findings"]]))


def test_derivations():
    p = water(dt=0.002)
    mdp = resolve(p)["stages"][0]["mdp"]
    # 0.01 ns / 0.002 ps = 5000 steps
    check("nsteps derived from physical time", mdp["nsteps"] == 5000, str(mdp["nsteps"]))
    p2 = Plan.from_dict({
        "system": {"kind": "solvent", "box_size_nm": 3.0},
        "stages": [{"name": "a", "type": "dynamics", "sim_time_ns": 0.01,
                    "params": {"temperature": 300}},
                   {"name": "b", "type": "dynamics", "sim_time_ns": 0.01,
                    "params": {"temperature": 300}}]})
    s = resolve(p2)["stages"]
    check("gen-vel only on the first dynamics stage",
          s[0]["mdp"]["gen-vel"] is True and s[1]["mdp"]["gen-vel"] is False)
    check("continuation set on the second stage",
          s[1]["mdp"]["continuation"] is True)


def test_cost():
    e = estimate(water())
    check("cost estimator returns a time", e["total_seconds"] > 0, str(e["total_seconds"]))


if __name__ == "__main__":
    print(f"ontology: {summary()}")
    for fn in (test_per_group_coupling, test_ontology_guard, test_schema_strict,
               test_rules, test_derivations, test_cost):
        fn()
    print("\n" + (f"{len(FAILED)} FAILED: {FAILED}" if FAILED else "ALL PLAN TESTS PASS"))
    sys.exit(1 if FAILED else 0)
