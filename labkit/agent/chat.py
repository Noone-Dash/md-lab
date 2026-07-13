"""The agentic chemist: an LLM driving the lab through the REAL tool surface.

Default brain is LOCAL — Ollama on this machine's own GPU (no API key, no cloud).
If ANTHROPIC_API_KEY happens to be set, it uses that instead.

It never "generates" a result. It calls the same tools you can call yourself
(describe_parameters, validate_plan, estimate_cost, submit_plan, get_results),
so anything it tells you is something the lab actually did. And because every
plan goes through the validator, a small local model cannot make the lab do
bad physics — the guardrails, not the model, are what keep it honest.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from . import tools as T

from ..config import OLLAMA_HOST as OLLAMA, pick_model, CHAT_MODEL_PREF, TRANSLATE_MODEL_PREF
# gpt-oss:20b is the one of the locally-pulled models that emits real tool_calls
LOCAL_MODEL = pick_model(CHAT_MODEL_PREF) or "gpt-oss:20b"   # whatever is ACTUALLY pulled
CLOUD_MODEL = os.environ.get("MDLAB_CLOUD_MODEL", "claude-sonnet-4-5")

SYSTEM = """You are the resident computational chemist for a local molecular-dynamics lab \
running GROMACS 2026.2 on an NVIDIA GB10 GPU, plus OpenMM and PySCF.

You drive the lab ONLY through your tools. Never invent a number you could measure. \
Never claim a simulation ran unless a tool told you it did.

Method:
1. If you are unsure what a setting does or what values are legal, call describe_parameters. \
The ontology has ~150 documented knobs with units and guidance. Look it up; do not guess.
2. Build a Plan (JSON, schema below). A real protein protocol is: minimize -> NVT (restrained) \
-> NPT (restrained) -> production (unrestrained). Give sim_time_ns; the lab derives the steps.
3. ALWAYS validate_plan before submitting. If it returns errors, read each 'fix' and repair the \
plan, then explain in plain words what was wrong.
4. ALWAYS estimate_cost and state the runtime before submitting. If it is more than a couple of \
minutes, tell the user and ask first.
5. submit_plan queues it on the GPU scheduler. Poll get_run until done.
6. After submit_plan, call wait_for_run ONCE (never poll get_run in a loop). Then call get_results ONCE and JUDGE the numbers against physics: water density ~1000 kg/m3; \
O-O g(r) first peak ~0.28 nm; temperature must sit at the setpoint; a stable protein's backbone \
RMSD plateaus below ~0.3 nm. Say plainly whether it looks healthy or suspicious, and why.

Be concise, concrete and honest. If it failed, say so. Do not flatter."""

PLAN_HINT = """
PLAN SCHEMA (unknown keys are REJECTED, so use exactly these):

{
  "name": "Lysozyme 310 K, 0.15 M NaCl",
  "system": {"kind": "protein", "structure_source": "rcsb", "pdb_id": "1AKI",
             "forcefield": "amber99sb-ildn", "water_model": "tip3p",
             "box_shape": "dodecahedron", "box_padding_nm": 1.2,
             "salt_conc_M": 0.15, "neutralize": true},
  "stages": [
    {"name": "minimize", "type": "minimize", "max_steps": 5000},
    {"name": "nvt", "type": "dynamics", "sim_time_ns": 0.05, "posres_fc_kj": 1000,
     "params": {"ensemble": "NVT", "temperature": 310}},
    {"name": "npt", "type": "dynamics", "sim_time_ns": 0.05, "posres_fc_kj": 1000,
     "params": {"ensemble": "NPT", "temperature": 310}},
    {"name": "production", "type": "dynamics", "sim_time_ns": 0.1,
     "params": {"ensemble": "NPT", "temperature": 310}}
  ],
  "analyses": ["rmsd", "gyrate", "rmsf"]
}

system.kind: "solvent" | "fluid" | "protein"
forcefield: any installed FF (amber99sb-ildn, amber14sb, charmm27, oplsaa, gromos54a7, ...)
analyses: protein -> rmsd, gyrate, rmsf | water -> rdf_ow, msd_ow
Keep sim_time_ns small (0.05-0.2) unless asked otherwise: this is a workstation.
"""

# one tool spec, rendered into whichever dialect the backend wants
TOOLS_SPEC = [
    ("list_capabilities", "What this lab can simulate: engines, system kinds, presets, ontology size.",
     {"type": "object", "properties": {}}),
    ("describe_parameters",
     "Search the MD parameter ontology (~150 knobs): unit, physical meaning, dependencies, guidance. "
     "Use this instead of guessing any GROMACS setting.",
     {"type": "object", "properties": {
         "search": {"type": "string", "description": "free text, e.g. 'thermostat', 'cutoff', 'salt'"},
         "applies_to": {"type": "string", "enum": ["fluid", "solvent", "protein", "membrane", "qm"]}}}),
    ("validate_plan",
     "Check a plan against the physics rules BEFORE running. Returns findings, each with a concrete fix.",
     {"type": "object", "properties": {"plan": {"type": "object"}}, "required": ["plan"]}),
    ("estimate_cost", "Estimate wall-clock time and memory for a plan on this machine.",
     {"type": "object", "properties": {"plan": {"type": "object"}}, "required": ["plan"]}),
    ("preview_mdp", "Show the exact GROMACS .mdp input a plan will produce.",
     {"type": "object", "properties": {"plan": {"type": "object"}}, "required": ["plan"]}),
    ("submit_plan", "Validate and queue a plan on the GPU scheduler. Returns run_id.",
     {"type": "object", "properties": {"plan": {"type": "object"}}, "required": ["plan"]}),
    ("get_run", "Status of a run: queued/running/done/error, step progress.",
     {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}),
    ("wait_for_run",
     "Block until a run finishes. USE THIS instead of calling get_run in a loop — polling wastes "
     "your tool budget.",
     {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}),
    ("get_results",
     "Results of a finished run with the physics ALREADY MEASURED (equilibrium averages) plus the "
     "reference values to judge them against. Call this ONCE after wait_for_run.",
     {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}),
]


def _ollama_tools():
    return [{"type": "function",
             "function": {"name": n, "description": d, "parameters": p}}
            for n, d, p in TOOLS_SPEC]


def _anthropic_tools():
    return [{"name": n, "description": d, "input_schema": p} for n, d, p in TOOLS_SPEC]


def backend() -> dict:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {"kind": "anthropic", "model": CLOUD_MODEL, "where": "cloud"}
    return {"kind": "ollama", "model": LOCAL_MODEL, "where": "local GPU"}


def local_up() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3)
        return True
    except Exception:  # noqa: BLE001
        return False


def status() -> dict:
    b = backend()
    ready = True if b["kind"] == "anthropic" else local_up()
    return {**b, "ready": ready}


def _call_tool(name, args):
    fn = T.TOOLS.get(name)
    if not fn:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**(args or {}))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _clip(obj, n=12000):
    s = json.dumps(obj)
    return s if len(s) <= n else s[:n] + " …[truncated]"


# --------------------------------------------------------------------------- #
_LAST_TRACE = {"t": []}


def _chat_ollama(messages, max_rounds):
    convo = [{"role": "system", "content": SYSTEM + PLAN_HINT}] + messages
    trace = []
    _LAST_TRACE["t"] = trace
    for _ in range(max_rounds):
        body = {"model": LOCAL_MODEL, "stream": False,
                "tools": _ollama_tools(), "messages": convo,
                "options": {"temperature": 0.2, "num_ctx": 16384}}
        req = urllib.request.Request(f"{OLLAMA}/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        except urllib.error.HTTPError as e:      # surface WHAT the server said
            raise RuntimeError(f"ollama HTTP {e.code}: {e.read().decode()[:300]}") from e

        msg = r.get("message", {}) or {}
        calls = msg.get("tool_calls") or []
        # append the assistant turn VERBATIM. gpt-oss carries a 'thinking' field and its
        # harmony template rejects the conversation if you strip it — that was a hard 500.
        convo.append(msg)

        if not calls:
            return {"reply": msg.get("content", "") or "(no reply)",
                    "tool_calls": trace, "messages": messages}

        for c in calls:
            f = c.get("function", {})
            name = f.get("name")
            args = f.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            out = _call_tool(name, args)
            trace.append({"tool": name, "input": args, "output": out})
            convo.append({"role": "tool", "content": _clip(out), "tool_name": name})

    return {"reply": "I ran out of tool-call rounds. Try a narrower request.",
            "tool_calls": trace, "messages": messages}


def _chat_anthropic(messages, max_rounds):
    import anthropic
    client = anthropic.Anthropic()
    convo = list(messages)
    trace = []
    for _ in range(max_rounds):
        resp = client.messages.create(
            model=CLOUD_MODEL, max_tokens=2048,
            system=SYSTEM + PLAN_HINT, tools=_anthropic_tools(), messages=convo)
        convo.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
        if resp.stop_reason != "tool_use":
            return {"reply": "".join(b.text for b in resp.content if b.type == "text"),
                    "tool_calls": trace, "messages": messages}
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            out = _call_tool(b.name, b.input)
            trace.append({"tool": b.name, "input": b.input, "output": out})
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": _clip(out)})
        convo.append({"role": "user", "content": results})
    return {"reply": "Ran out of tool-call rounds.", "tool_calls": trace, "messages": messages}


def chat(messages: list, max_tool_rounds: int = 16):
    b = backend()
    try:
        if b["kind"] == "anthropic":
            out = _chat_anthropic(messages, max_tool_rounds)
        else:
            if not local_up():
                return {"error": "no_backend",
                        "reply": "The local model server isn't running. On the lab machine:\n\n"
                                 "    ollama serve &\n    ollama pull gpt-oss:20b\n\n"
                                 "No API key needed — it runs on your own GPU."}
            out = _chat_ollama(messages, max_tool_rounds)
        out["backend"] = b
        return out
    except Exception as e:  # noqa: BLE001
        # keep whatever tools already ran — swallowing the trace made a round-2
        # failure look like "0 tool calls", which sent me hunting the wrong bug.
        return {"error": "backend_error",
                "reply": f"The model backend failed: {e}",
                "tool_calls": _LAST_TRACE.get("t", []), "backend": b}
