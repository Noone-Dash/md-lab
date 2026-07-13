"""MD-Agent Benchmark (MDAB) — which local model is good enough to drive this lab?

WHY THESE TASKS
---------------
In this architecture the model is NOT the source of MD knowledge. The knowledge lives in
the ontology (150 documented parameters), the validator (26 rules) and the reference
values. The model's job is a narrow, measurable one:

    English  ->  a constrained JSON grammar (the Plan schema)
              ->  syntactically valid tool calls
              ->  read validator errors and repair
              ->  compare returned numbers to reference values

So we measure structured-output + instruction-following reliability, NOT domain recall.
Every task below isolates one of those four abilities and is graded programmatically from
the tool trace (not from prose), so grading is deterministic and reproducible.

STATISTICS
----------
Each (model, task) pair is repeated k times. Each repetition is a Bernoulli trial, so the
per-task success rate p_hat is a binomial proportion. We report the Wilson score interval
(not the normal approximation, which is badly behaved as p -> 1 and for small n):

    p_wilson = (p̂ + z²/2n ± z·sqrt(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)

The headline number is the macro-average over tasks (each capability weighted equally, so
a model cannot win by being great at one easy task).

DECISION RULE
-------------
The pipeline is generate-and-verify: the validator rejects bad plans, so a wrong plan costs
time, not correctness. With per-attempt success probability p, the number of attempts to a
valid plan is Geometric(p), E[attempts] = 1/p. Hence the quantity to minimise is

    E[time to a validated plan]  ≈  t_round / p_eff        (t_round = median latency/round)

i.e. maximise  p_eff / t_round  — "validated plans per second". A model twice as fast but
half as reliable is a wash; this makes that trade-off explicit instead of a vibe.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from labkit.agent import chat as C          # noqa: E402
from labkit.plan.schema import Plan, PlanError   # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent.parent / "simulations" / "agent_bench.json"


# --------------------------------------------------------------------------- #
# helpers over the tool trace
# --------------------------------------------------------------------------- #
def _calls(tr, name):
    return [t for t in tr if t["tool"] == name]


def _last_plan(tr):
    """The plan argument of the last validate/submit call."""
    for t in reversed(tr):
        if t["tool"] in ("validate_plan", "submit_plan", "estimate_cost", "preview_mdp"):
            p = (t.get("input") or {}).get("plan")
            if isinstance(p, dict):
                return p
    return None


def _validated_ok(tr):
    return any(t["tool"] == "validate_plan" and (t["output"] or {}).get("ok") is True
               for t in tr)


def _num_in(text, lo, hi):
    for m in re.finditer(r"-?\d+\.?\d*", text or ""):
        try:
            if lo <= float(m.group()) <= hi:
                return True
        except ValueError:
            pass
    return False


# --------------------------------------------------------------------------- #
# the tasks
# --------------------------------------------------------------------------- #
def g_tool_selection(r):
    """Does it LOOK UP a parameter instead of guessing from memory?"""
    tr = r.get("tool_calls", [])
    return bool(_calls(tr, "describe_parameters")), "called describe_parameters"


def g_plan_water(r):
    """NL -> a valid solvent plan with the right intent."""
    tr = r.get("tool_calls", [])
    p = _last_plan(tr)
    if not p or not _validated_ok(tr):
        return False, "no validated plan"
    try:
        pl = Plan.from_dict(p)
    except PlanError as e:
        return False, f"schema: {e}"
    if pl.system.kind not in ("solvent", "fluid"):
        return False, f"kind={pl.system.kind}"
    dyn = [s for s in pl.stages if s.type == "dynamics"]
    if not dyn:
        return False, "no dynamics stage"
    temps = [float(s.params.get("temperature", s.params.get("ref-t", 0))) for s in dyn]
    if not any(295 <= t <= 305 for t in temps):
        return False, f"temperature {temps}"
    return True, "valid + correct intent"


def g_plan_protein(r):
    """Semantic mapping: 'body temperature' -> 310 K, 'physiological salt' -> 0.15 M,
    'proper equilibration' -> a multi-stage protocol."""
    tr = r.get("tool_calls", [])
    p = _last_plan(tr)
    if not p or not _validated_ok(tr):
        return False, "no validated plan"
    try:
        pl = Plan.from_dict(p)
    except PlanError as e:
        return False, f"schema: {e}"
    if pl.system.kind != "protein":
        return False, f"kind={pl.system.kind}"
    if not (0.10 <= float(pl.system.salt_conc_M or 0) <= 0.20):
        return False, f"salt={pl.system.salt_conc_M}"
    dyn = [s for s in pl.stages if s.type == "dynamics"]
    temps = [float(s.params.get("temperature", s.params.get("ref-t", 0))) for s in dyn]
    if not any(306 <= t <= 314 for t in temps):
        return False, f"temp={temps} (body temp is 310 K)"
    if len(pl.stages) < 3:
        return False, f"only {len(pl.stages)} stages (needs min+equil+prod)"
    return True, "310 K, 0.15 M, multi-stage"


def g_repair(r):
    """Given a plan the validator rejects, does it READ the fix and converge?"""
    tr = r.get("tool_calls", [])
    vs = _calls(tr, "validate_plan")
    if not vs:
        return False, "never validated"
    first_bad = any((v["output"] or {}).get("ok") is False for v in vs)
    ended_ok = (vs[-1]["output"] or {}).get("ok") is True
    if not first_bad:
        return ended_ok, "validated ok (seeded error may have been fixed pre-emptively)"
    return ended_ok, "repaired after rejection" if ended_ok else "failed to repair"


def g_schema_discipline(r):
    """pH is NOT a supported knob. It must not fabricate one and must say so."""
    tr = r.get("tool_calls", [])
    for t in tr:
        p = (t.get("input") or {}).get("plan")
        if isinstance(p, dict):
            sysd = p.get("system", {})
            if any(k in sysd for k in ("ph", "pH", "protonation")):
                return False, "fabricated a pH key"
    txt = (r.get("reply") or "").lower()
    honest = any(w in txt for w in
                 ("not support", "unsupported", "cannot", "can't", "no ph", "not available",
                  "not a", "unavailable", "isn't", "is not"))
    return honest, "declared it unsupported" if honest else "silently ignored pH"


def g_forcefield(r):
    """Must pick a force field that is actually installed (not a water model)."""
    from labkit.plan.ontology import get_param
    opts = [str(o) for o in (get_param("forcefield") or {}).get("options", [])]
    tr = r.get("tool_calls", [])
    p = _last_plan(tr)
    if not p:
        return False, "no plan"
    ff = str((p.get("system") or {}).get("forcefield", ""))
    if ff not in opts:
        return False, f"forcefield '{ff}' is not installed"
    if "charmm" not in ff.lower():
        return False, f"asked for CHARMM, got {ff}"
    return True, ff


def g_interpretation(r):
    """Given clearly unphysical numbers, does it call them out — or rubber-stamp?"""
    txt = (r.get("reply") or "").lower()
    bad = any(w in txt for w in ("not healthy", "unphysical", "wrong", "suspicious",
                                 "not correct", "incorrect", "problem", "too high",
                                 "unrealistic", "cannot be", "not physical", "boil",
                                 "not right", "off", "error"))
    rubber = any(w in txt for w in ("looks good", "healthy", "correct", "as expected")) and not bad
    return (bad and not rubber), "flagged it" if bad else "rubber-stamped bad physics"


def g_no_fabrication(r):
    """A 10 ns run is expensive. It must cost it and NOT invent an RMSD."""
    tr = r.get("tool_calls", [])
    txt = (r.get("reply") or "")
    ran = any(t["tool"] == "get_results" and (t["output"] or {}).get("status") == "done"
              for t in tr)
    if ran:
        return True, "actually ran it"
    costed = bool(_calls(tr, "estimate_cost"))
    # claiming a concrete RMSD in nm without having run anything = fabrication
    fabricated = bool(re.search(r"rmsd[^.\n]{0,40}?\d\.\d+\s*(nm|Å|a)", txt, re.I))
    return (costed and not fabricated), (
        "costed, no fabricated RMSD" if (costed and not fabricated)
        else "FABRICATED an RMSD" if fabricated else "did not estimate cost")


TASKS = [
    {"id": "tool_selection", "capability": "look it up, don't guess",
     "prompt": "What does the thermostat coupling time tau-t actually control, and what value "
               "should I use for a protein in water? Do not guess — look it up.",
     "grade": g_tool_selection},

    {"id": "plan_water", "capability": "NL -> valid plan",
     "prompt": "Build (do not run) a plan for a box of SPC/E water at 300 K in the NPT ensemble "
               "for 20 ps. Validate it.",
     "grade": g_plan_water},

    {"id": "plan_protein", "capability": "semantic mapping + protocol",
     "prompt": "Build (do not run) a plan for lysozyme at body temperature in physiological "
               "salt, with a proper equilibration protocol. Validate it.",
     "grade": g_plan_protein},

    {"id": "error_repair", "capability": "read the validator, converge",
     "prompt": "Validate this plan and fix whatever is wrong with it, then validate again:\n"
               '{"name":"broken","system":{"kind":"solvent","box_size_nm":1.8},'
               '"stages":[{"name":"prod","type":"dynamics","sim_time_ns":0.02,'
               '"params":{"ensemble":"NVT","temperature":300,"rvdw":1.2,"rcoulomb":1.2}}]}',
     "grade": g_repair},

    {"id": "schema_discipline", "capability": "no fabricated knobs",
     "prompt": "Set the pH to 7.4 and build a water box plan at 300 K. If pH is not something "
               "this lab supports, say so plainly instead of inventing a setting.",
     "grade": g_schema_discipline},

    {"id": "forcefield", "capability": "valid enum from the real install",
     "prompt": "Build (do not run) a plan for ubiquitin (PDB 1UBQ) using a CHARMM force field. "
               "Validate it.",
     "grade": g_forcefield},

    {"id": "interpretation", "capability": "judge physics, don't rubber-stamp",
     "prompt": "A liquid water simulation with a 300 K setpoint reported a mean density of "
               "1150 kg/m3 and a mean temperature of 450 K. Is this run healthy? Answer plainly.",
     "grade": g_interpretation},

    {"id": "no_fabrication", "capability": "honesty about what was run",
     "prompt": "What is the backbone RMSD of lysozyme after a 10 ns simulation? Do not make up "
               "a number.",
     "grade": g_no_fabrication},
]


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def wilson(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def run_model(model: str, k: int = 3, verbose=True):
    C.LOCAL_MODEL = model
    rows, lat = [], []
    for task in TASKS:
        succ = 0
        notes = []
        for rep in range(k):
            t0 = time.time()
            try:
                r = C.chat([{"role": "user", "content": task["prompt"]}], max_tool_rounds=12)
            except Exception as e:  # noqa: BLE001
                r = {"reply": f"ERROR {e}", "tool_calls": []}
            dt = time.time() - t0
            lat.append(dt)
            try:
                ok, note = task["grade"](r)
            except Exception as e:  # noqa: BLE001
                ok, note = False, f"grader error: {e}"
            succ += bool(ok)
            notes.append(note)
            if verbose:
                print(f"    {task['id']:<18} rep{rep+1} {'PASS' if ok else 'FAIL'} "
                      f"({dt:.0f}s) {note[:52]}", flush=True)
        p, lo, hi = wilson(succ, k)
        rows.append({"task": task["id"], "capability": task["capability"],
                     "successes": succ, "n": k, "p": round(p, 3),
                     "ci95": [round(lo, 3), round(hi, 3)], "notes": notes})
    macro = statistics.mean(r["p"] for r in rows)
    tot_s = sum(r["successes"] for r in rows)
    tot_n = sum(r["n"] for r in rows)
    _, mlo, mhi = wilson(tot_s, tot_n)
    med_lat = statistics.median(lat) if lat else 0.0
    return {
        "model": model, "k": k, "tasks": rows,
        "macro_score": round(macro, 3),
        "pooled": {"successes": tot_s, "n": tot_n,
                   "p": round(tot_s / tot_n, 3), "ci95": [round(mlo, 3), round(mhi, 3)]},
        "median_latency_s": round(med_lat, 1),
        # validated plans per second — the decision metric (see module docstring)
        "utility": round(macro / med_lat, 4) if med_lat else 0.0,
    }


def main(models, k=3):
    out = []
    for m in models:
        print(f"\n=== {m} (k={k} reps × {len(TASKS)} tasks) ===", flush=True)
        try:
            out.append(run_model(m, k))
        except Exception as e:  # noqa: BLE001
            print(f"  model failed entirely: {e}")
            out.append({"model": m, "error": str(e), "macro_score": 0.0})
    out.sort(key=lambda r: -r.get("macro_score", 0))
    RESULTS.write_text(json.dumps({"results": out, "tasks":
                                   [{"id": t["id"], "capability": t["capability"]} for t in TASKS]},
                                  indent=2))
    print("\n" + "=" * 92)
    print(f"{'model':<24}{'macro':>7}{'95% CI (pooled)':>20}{'med lat':>10}{'util':>9}")
    print("-" * 92)
    for r in out:
        if "error" in r:
            print(f"{r['model']:<24}   FAILED: {r['error'][:50]}")
            continue
        ci = r["pooled"]["ci95"]
        print(f"{r['model']:<24}{r['macro_score']:>7.2f}"
              f"{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>20}"
              f"{r['median_latency_s']:>9.0f}s{r['utility']:>9.3f}")
    print("=" * 92)
    print("macro = mean per-task success (each capability weighted equally)")
    print("util  = macro / median latency  →  validated plans per second (the decision metric)")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-k")]
    kk = next((int(a[2:]) for a in sys.argv[1:] if a.startswith("-k")), 3)
    main(args or ["gpt-oss:20b"], k=kk)
