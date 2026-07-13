"""STAGE 1 — the TRANSLATOR.  English -> Plan JSON.  Nothing else.

This is the ONLY place a language model touches the simulation setup, and it is
boxed in on all sides:

  * grammar-constrained decoding: the model is forced to emit tokens that satisfy
    plan_schema(). An illegal force field is not "rejected later" — it is not in
    the grammar, so it cannot be generated.
  * temperature 0 + fixed seed: the mapping is a function, not a sample.
  * it emits INTENT only (what to simulate). It never supplies a physics constant.
    Cutoffs, PME grids, coupling constants, buffers, integrator settings all come
    from the deterministic resolver + force-field presets, never from the model.
  * everything downstream (resolve -> validate -> mdp_emit -> GROMACS) is pure code.

So the LLM is a NATURAL-LANGUAGE PARSER, not a physicist. If it is wrong, the plan
is wrong about *what you asked for* — it cannot be wrong about the physics, because
it never had the pen.
"""

from __future__ import annotations

import json
import os
import urllib.request

from ..plan.jsonschema import plan_schema, legal_values
from ..plan.schema import Plan, PlanError
from ..plan.validate import validate

from ..config import OLLAMA_HOST as OLLAMA, pick_model, CHAT_MODEL_PREF, TRANSLATE_MODEL_PREF
# a small CODING model: its job is grammar/syntax, which is exactly what it is good at
TRANSLATOR = pick_model(TRANSLATE_MODEL_PREF) or "qwen3:8b"  # whatever is ACTUALLY pulled
SEED = 7

SYSTEM = """You translate a request for a molecular-dynamics simulation into a Plan object.

You are a PARSER, not a physicist. Emit ONLY what the user actually asked for:
what to simulate, at what temperature, how long, in which ensemble.

DO NOT invent physics settings. Cutoffs, PME, coupling constants, timestep buffers and
integrator details are chosen by the lab's deterministic resolver from the force field.
Leave them out.

Conventions you must apply:
  "body temperature"        -> 310 K          "room temperature" -> 300 K
  "physiological salt"      -> salt_conc_M 0.15
  "proper equilibration"    -> minimize, then NVT (posres_fc_kj 1000),
                               then NPT (posres_fc_kj 1000), then unrestrained production
  a plain water/solvent box -> system.kind "solvent"
  a named protein/PDB id    -> system.kind "protein", structure_source "rcsb"
  protein analyses          -> ["rmsd","gyrate","rmsf"]   water -> ["rdf_ow","msd_ow"]
Durations are in sim_time_ns (20 ps = 0.02).
"""


def _prompt(nl: str) -> str:
    return (f"{SYSTEM}\nLegal values (the grammar allows nothing else):\n"
            f"{json.dumps(legal_values(), indent=1)}\n\nRequest: {nl}\n\nEmit the Plan object.")


def translate(nl: str, model: str = None, temperature: float = 0.0) -> dict:
    """NL -> {plan, valid, findings, raw}.  Deterministic at temperature 0."""
    model = model or TRANSLATOR
    body = {
        "model": model, "stream": False,
        "format": plan_schema(),                 # <- grammar-constrained decoding
        "options": {"temperature": temperature, "seed": SEED, "num_ctx": 8192},
        "messages": [{"role": "user", "content": _prompt(nl)}],
    }
    req = urllib.request.Request(f"{OLLAMA}/api/chat",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(req, timeout=150).read())["message"]["content"]

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"plan": None, "valid": False, "error": f"not JSON: {e}", "raw": raw}

    # STAGE 2 — pure deterministic code from here on. No model.
    try:
        p = Plan.from_dict(obj)
    except PlanError as e:
        return {"plan": obj, "valid": False, "error": str(e), "raw": raw}

    v = validate(p, autofix=True)          # deterministic repair
    return {
        "plan": p.to_dict(),
        "valid": v["ok"],
        "findings": v["findings"],
        "model": model,
        "raw": raw,
    }


def determinism_check(nl: str, n: int = 5, model: str = None) -> dict:
    """Same request N times at temperature 0 -> is the plan IDENTICAL every time?

    This is the property that matters: the setup must be a function of the request,
    not a sample from a distribution.
    """
    outs = []
    for _ in range(n):
        r = translate(nl, model=model)
        outs.append(json.dumps(r.get("plan"), sort_keys=True))
    uniq = set(outs)
    return {"n": n, "distinct_plans": len(uniq), "deterministic": len(uniq) == 1,
            "model": model or TRANSLATOR}
