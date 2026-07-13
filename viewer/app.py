#!/usr/bin/env python3
"""Flask backend for the GROMACS/MD Lab UI.

Everything — GROMACS recipes and the Python engines (OpenMM, PySCF, cell model) —
emits the same run.json manifest, so a single viewer renders them all. The UI is
organised into *tracks* (pages), each labelled live / model / unavailable.
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, abort, render_template

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labkit import list_recipes, run_recipe, load_run, list_runs, gmx_version  # noqa: E402
from labkit.engine import RUNS_DIR                                            # noqa: E402
from labkit.recipes import REGISTRY                                           # noqa: E402
from labkit.tracks import TRACKS, list_tracks, get_track                      # noqa: E402
from labkit.scheduler import SCHED                                            # noqa: E402

app = Flask(__name__, static_folder="static", template_folder="templates")
SCHED.start()

# ---- the pages (tracks) ---------------------------------------------------- #
TRACK_INFO = [
    {"id": "classical", "name": "Classical MD", "icon": "⚛",
     "engine": "GROMACS 2026.2",
     "blurb": "All-atom molecular dynamics: Lennard-Jones fluids, water, ions, proteins."},
    {"id": "cg", "name": "Coarse-grained", "icon": "⬡",
     "engine": "GROMACS + Martini 3",
     "blurb": "~4 atoms per bead — reach membrane & cell scales for ~1000× less compute."},
    {"id": "openmm", "name": "OpenMM + ML", "icon": "🧬",
     "engine": "OpenMM 8.5",
     "blurb": "Programmable GPU MD with implicit solvent; the on-ramp to ML potentials."},
    {"id": "qmmm", "name": "QM / QM-MM", "icon": "🔬",
     "engine": "PySCF 2.13",
     "blurb": "Real quantum chemistry: HF/DFT optimisation + MM electrostatic embedding."},
    {"id": "cell", "name": "Cell-scale", "icon": "🦠",
     "engine": "NumPy particle model",
     "blurb": "Spatial stochastic reaction-diffusion inside a cell (Smoldyn/ReaDDy-class)."},
    {"id": "plan", "name": "Plan (agent surface)", "icon": "🤖",
     "engine": "GROMACS 2026.2 · plan/1",
     "blurb": "Simulations as DATA: full parameter ontology, multi-stage protocols, "
              "validation and cost estimation. This is what an agent drives."},
]

# a labelled placeholder for the not-yet-installed ML potential
ML_STUB = {
    "key": "ml_potential", "name": "ML potential (MACE-OFF)", "category": "ML",
    "track": "openmm", "engine": "MACE + PyTorch", "mode": "unavailable", "est": "—",
    "description": ("Near-DFT accuracy at force-field cost via a machine-learned "
                    "potential. Needs PyTorch + mace-torch (~2 GB) — not installed "
                    "yet, so this card is a labelled placeholder."),
    "params": [],
}


# default classification (a cross-cutting tag) derived from category
_CLASSIFY = {
    "Fundamental": "fluids & solvents", "Solvent": "fluids & solvents",
    "Coarse-grained": "membranes", "Biomolecular": "structure & dynamics",
    "Quantum": "electronic structure", "Reactions": "reactions",
    "Cell-scale": "cells", "ML": "machine learning",
}


# plain-language "what am I looking at" caption shown over the 3D viewer
WHAT_YOU_SEE = {
    "lj_argon": "Every sphere is one argon atom. Tightly packed and jostling = liquid; "
                "spread out and flying = gas; locked in a lattice = solid. The g(r) plot "
                "below tells you which one you got.",
    "water_box": "Red = oxygen, white = hydrogen — every molecule here is water. Watch them "
                 "tumble and swap neighbours. The box quietly resizes as it settles to water's "
                 "real density (~1000 kg/m³, see the Density plot).",
    "nacl_water": "Blue = Na⁺, green = Cl⁻, red = water oxygens. Watch water crowd tightly "
                  "around each ion — that shell is why salt dissolves. The first g(r) peak "
                  "below IS that shell, measured. Tick 'hide water' to see the ions alone.",
    "protein": "The ribbon traces the protein's backbone, coloured blue→red from its first "
               "residue to its last. Loops flop, helices hold. The RMSD plot says how far it "
               "has drifted from the starting structure.",
    "martini_bilayer": "Each bead is ~4 atoms lumped together. Two facing sheets of lipids = a "
                       "cell membrane. It ripples like a fluid sheet while lipids slide past each "
                       "other — that's exactly how real membranes behave.",
    "openmm_implicit": "A mini-protein tumbling in invisible (implicit) water. Watch it curl up; "
                       "the radius-of-gyration plot is it compacting — the beginning of folding.",
    "qmmm_opt": "One molecule finding its lowest-energy shape, computed with real quantum "
                "mechanics (electrons, not springs). The bonds visibly settle as the energy drops.",
    "reaction_scan": "One chemical bond being pulled apart, step by step. The energy curve below "
                     "IS the reaction coordinate — the height of the climb is the bond's strength.",
    "cell_rd": "Blue = molecule A, red = molecule B, yellow = product C. They wander randomly "
               "inside a cell-shaped volume and react whenever they collide. The kinetics plot "
               "counts them as A + B get consumed and C builds up.",
}


def _all_experiments():
    exps = list_recipes() + list_tracks() + [ML_STUB]
    for e in exps:
        e.setdefault("classification", _CLASSIFY.get(e.get("category", ""), "other"))
        e.setdefault("what_you_see", WHAT_YOU_SEE.get(e["key"], ""))
    return exps


def _catalog():
    exps = _all_experiments()
    out = []
    for info in TRACK_INFO:
        items = [e for e in exps if e.get("track") == info["id"]]
        modes = {e["mode"] for e in items}
        if info["id"] == "plan":
            track_mode = "live"      # driven by the API/agent, not by preset cards
        else:
            track_mode = ("model" if "model" in modes and "live" not in modes
                          else "live" if "live" in modes else "unavailable")
        out.append({**info, "experiments": items, "mode": track_mode,
                    "count": len(items)})
    return out


# ---- pages ----------------------------------------------------------------- #
@app.route("/")
def hub():
    return render_template("hub.html")


@app.route("/track/<tid>")
def track_page(tid):
    info = next((t for t in TRACK_INFO if t["id"] == tid), None)
    if not info:
        abort(404)
    return render_template("track.html", tid=tid, tname=info["name"],
                           ticon=info["icon"])


# ---- api ------------------------------------------------------------------- #
@app.route("/api/meta")
def meta():
    return jsonify({"gmx_version": gmx_version()})


@app.route("/api/catalog")
def catalog():
    return jsonify(_catalog())


@app.route("/api/track/<tid>")
def track_detail(tid):
    c = next((t for t in _catalog() if t["id"] == tid), None)
    if not c:
        abort(404)
    return jsonify(c)


@app.route("/api/runs")
def runs():
    return jsonify(list_runs())


@app.route("/api/run", methods=["POST"])
def start():
    body = request.get_json(force=True)
    key = body.get("key") or body.get("recipe")
    params = body.get("params", {})
    if not key:
        abort(400, "key required")
    try:
        run_id = SCHED.submit(key, params)      # queued/launched via the scheduler
    except (KeyError, ValueError) as e:
        abort(400, str(e))
    return jsonify({"id": run_id})


# ---- agent surface: simulations as data ------------------------------------ #
from labkit.agent import tools as agent_tools                                # noqa: E402


@app.route("/plan")
def plan_page():
    return render_template("plan.html")


@app.route("/api/agent/capabilities")
def agent_caps():
    return jsonify(agent_tools.list_capabilities())


@app.route("/api/ontology")
def api_ontology():
    return jsonify(agent_tools.describe_parameters(
        area=request.args.get("area"),
        applies_to=request.args.get("applies_to"),
        search=request.args.get("search")))


@app.route("/api/plan/validate", methods=["POST"])
def api_plan_validate():
    body = request.get_json(force=True) or {}
    return jsonify(agent_tools.validate_plan(body.get("plan", body),
                                             autofix=bool(body.get("autofix"))))


@app.route("/api/plan/estimate", methods=["POST"])
def api_plan_estimate():
    body = request.get_json(force=True) or {}
    return jsonify(agent_tools.estimate_cost(body.get("plan", body)))


@app.route("/api/plan/mdp", methods=["POST"])
def api_plan_mdp():
    body = request.get_json(force=True) or {}
    return jsonify(agent_tools.preview_mdp(body.get("plan", body)))


@app.route("/api/plan/run", methods=["POST"])
def api_plan_run():
    body = request.get_json(force=True) or {}
    res = agent_tools.submit_plan(body.get("plan", body))
    return jsonify(res), (200 if res.get("submitted") else 400)


@app.route("/api/agent/results/<run_id>")
def api_agent_results(run_id):
    return jsonify(agent_tools.get_results(run_id))



# ---- agentic chatbot + evaluation harness ---------------------------------- #
from labkit.agent import chat as agent_chat                                  # noqa: E402
from labkit.evals import runner as evals                                     # noqa: E402

_eval_lock = threading.Lock()
_eval_state = {"running": False, "progress": None}


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


@app.route("/evals")
def evals_page():
    return render_template("evals.html")


@app.route("/api/chat/status")
def chat_status():
    return jsonify({"ready": agent_chat.have_key(), "model": agent_chat.MODEL})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(force=True) or {}
    msgs = body.get("messages") or []
    return jsonify(agent_chat.chat(msgs))


@app.route("/api/evals")
def api_evals():
    return jsonify({"benchmarks": evals.BENCH, "last": evals.load_last(),
                    "running": _eval_state["running"], "progress": _eval_state["progress"]})


@app.route("/api/evals/run", methods=["POST"])
def api_evals_run():
    body = request.get_json(force=True) or {}
    ids = body.get("ids")
    if _eval_state["running"]:
        return jsonify({"error": "already running"}), 409

    def work():
        _eval_state["running"] = True
        try:
            evals.run(ids, progress=lambda pr: _eval_state.update(progress=pr))
        finally:
            _eval_state["running"] = False
            _eval_state["progress"] = None

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"started": True})


# ---- structure explorer (load real molecules from public databases) --------- #
from labkit.config import DATA_DIR as _DATA
STRUCT_CACHE = _DATA / "structures"


@app.route("/explore")
def explore_page():
    return render_template("explore.html")


@app.route("/api/structure/<db>/<pid>")
def fetch_structure(db, pid):
    """Stream a structure from RCSB PDB or the AlphaFold DB (cached on disk)."""
    import re
    import urllib.request
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", pid):
        abort(400, "bad id")
    pid = pid.upper()
    if db == "pdb":
        url = f"https://files.rcsb.org/download/{pid}.pdb"
    elif db == "alphafold":
        url = f"https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_v4.pdb"
    else:
        abort(400, "unknown db")

    STRUCT_CACHE.mkdir(parents=True, exist_ok=True)
    cached = STRUCT_CACHE / f"{db}_{pid}.pdb"
    if not cached.exists():
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                cached.write_bytes(r.read())
        except Exception as e:  # noqa: BLE001
            abort(404, f"could not fetch {pid}: {e}")
    return send_from_directory(cached.parent, cached.name, mimetype="chemical/x-pdb")


# ---- monitor / scheduler --------------------------------------------------- #
@app.route("/monitor")
def monitor_page():
    return render_template("monitor.html")


@app.route("/api/monitor")
def monitor():
    return jsonify({"telemetry": SCHED.telemetry(), "jobs": SCHED.list_jobs()})


@app.route("/api/budget", methods=["POST"])
def set_budget():
    SCHED.set_budget(**(request.get_json(force=True) or {}))
    return jsonify(SCHED.telemetry()["budget"])


@app.route("/api/job/<jid>/<action>", methods=["POST"])
def job_action(jid, action):
    if action == "pause":
        SCHED.pause(jid)
    elif action == "resume":
        SCHED.resume(jid)
    elif action == "kill":
        SCHED.kill(jid)
    elif action == "clear":
        SCHED.clear_finished()
    else:
        abort(400, "unknown action")
    return jsonify({"ok": True})


@app.route("/api/run/<run_id>")
def run_detail(run_id):
    m = load_run(run_id)
    if m is None:
        abort(404)
    return jsonify(m)


@app.route("/api/run/<run_id>/log")
def run_log(run_id):
    f = RUNS_DIR / run_id / "run.log"
    if not f.exists():
        return ("", 200)
    lines = f.read_text(errors="replace").splitlines()
    return ("\n".join(lines[-400:]), 200, {"Content-Type": "text/plain"})


@app.route("/api/run/<run_id>/traj")
def run_traj(run_id):
    """Serve the trajectory, gzipped — these are 10-15 MB of text and the browser
    was silently staring at nothing while they downloaded."""
    import gzip as _gzip
    from flask import Response
    f = RUNS_DIR / run_id / "traj.pdb"
    if not f.exists():
        abort(404)
    raw = f.read_bytes()
    if "gzip" in (request.headers.get("Accept-Encoding") or ""):
        body = _gzip.compress(raw, 6)
        return Response(body, mimetype="chemical/x-pdb", headers={
            "Content-Encoding": "gzip", "Content-Length": str(len(body)),
            "X-Uncompressed-Length": str(len(raw)),
        })
    return Response(raw, mimetype="chemical/x-pdb",
                    headers={"Content-Length": str(len(raw))})


if __name__ == "__main__":
    from labkit import config as C
    port = int(sys.argv[1]) if len(sys.argv) > 1 else C.UI_PORT
    # Loopback by DEFAULT. Binding 0.0.0.0 on a shared cluster would expose an API
    # that can launch compute. Opt in explicitly: MDLAB_HOST=0.0.0.0
    host = C.UI_HOST
    print(f"MD Lab UI  ->  http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}")
    if host == "0.0.0.0":
        print("  WARNING: bound to 0.0.0.0 — reachable from the network, and /api/plan/run "
              "can start jobs. Do not do this on a shared machine without auth.")
    app.run(host=host, port=port, threaded=True, debug=False)
