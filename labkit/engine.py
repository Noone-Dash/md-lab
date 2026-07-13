"""The run engine: materialise a recipe, execute its GROMACS steps, then
post-process into viewer + plot data.  State is persisted to ``run.json`` after
every step so the web UI can poll live progress.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .gmx import gmx, gmx_version, GmxError
from .recipes import get_recipe
from .xvg import parse_xvg

from .config import REPO_ROOT as ROOT, RUNS_DIR   # honours $MDLAB_DATA (cluster scratch)
FRAME_CAP = 150                     # keep viewer trajectories light


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _save(manifest, run_dir):
    # Stamp WHO is running this and WHEN we last heard from them. Without it, a run whose
    # process dies leaves status="running" in the manifest forever and the UI shows it
    # spinning for eternity (tau_water_probe did exactly this). Nothing could tell a live
    # run from a corpse, because nothing recorded that a live run has an owner.
    if manifest.get("status") == "running":
        manifest["pid"] = os.getpid()
        manifest["heartbeat"] = _now()
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))


STALE_AFTER_S = 300


def _reconcile(m: dict) -> dict:
    """A manifest that CLAIMS to be running, but is not. Say so."""
    if m.get("status") != "running":
        return m
    pid = m.get("pid")
    alive = False
    if pid:
        try:
            os.kill(int(pid), 0)          # signal 0: does the process exist?
            alive = True
        except (OSError, ValueError, TypeError):
            alive = False
    if alive:
        return m
    # No owner. If we never recorded one, fall back to the heartbeat so that runs written
    # by an older version of this code are still reconciled rather than spinning forever.
    hb = m.get("heartbeat") or m.get("created")
    if pid is None and hb:
        try:
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(str(hb)).replace(tzinfo=timezone.utc)).total_seconds()
            if age < STALE_AFTER_S:
                return m                  # young and unowned: probably just started
        except Exception:  # noqa: BLE001
            pass
    m = dict(m)
    m["status"] = "interrupted"
    m["detail"] = ("the process running this died without finishing "
                   "(killed, crashed, or the machine went away)")
    return m


def _count_pdb(path):
    """(n_frames, n_atoms) from a multi-model PDB."""
    frames, atoms0, in_first = 0, 0, True
    with open(path) as fh:
        for line in fh:
            if line.startswith("MODEL"):
                frames += 1
                if frames > 1:
                    in_first = False
            elif line.startswith(("ATOM", "HETATM")) and in_first:
                atoms0 += 1
    if frames == 0:                 # single-frame pdb without MODEL records
        frames = 1
    return frames, atoms0


def run_recipe(recipe_key, params, run_id=None, progress_cb=None):
    recipe = get_recipe(recipe_key)
    p = recipe.coerce(params)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"{ts}_{recipe_key}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    manifest = {
        "id": run_id,
        "recipe": recipe_key,
        "recipe_name": recipe.name,
        "category": recipe.category,
        "track": recipe.track,
        "engine": recipe.engine,
        "mode": recipe.mode,
        "params": p,
        "status": "preparing",
        "created": _now(),
        "gmx_version": None,
        "steps": [],
        "error": None,
        "outputs": {},
        "energy": None,
        "analyses": [],
    }

    def push():
        _save(manifest, run_dir)
        if progress_cb:
            progress_cb(manifest)

    push()

    try:
        manifest["gmx_version"] = gmx_version()
        steps = recipe.prepare(run_dir, p)
        manifest["steps"] = [{"name": s.name, "status": "pending", "seconds": None}
                             for s in steps]
        manifest["status"] = "running"
        push()

        for i, step in enumerate(steps):
            manifest["steps"][i]["status"] = "running"
            push()
            t0 = time.time()
            if step.func is not None:
                step.func(run_dir, log_path)          # python step (e.g. insane)
            else:
                gmx(step.argv, cwd=run_dir, log_path=log_path,
                    stdin_text=step.stdin, timeout=step.timeout)
            manifest["steps"][i]["status"] = "done"
            manifest["steps"][i]["seconds"] = round(time.time() - t0, 1)
            push()

        manifest["status"] = "analysing"
        push()
        _postprocess(recipe, run_dir, p, manifest, log_path)

        manifest["status"] = "done"
        manifest["finished"] = _now()
        push()
    except Exception as e:  # noqa: BLE001
        manifest["status"] = "error"
        manifest["error"] = str(e)
        (run_dir / "error.txt").write_text(traceback.format_exc())
        push()

    return manifest


def run_plan(plan, run_id=None, progress_cb=None):
    """Execute a Plan (labkit.plan.schema.Plan) and write the standard manifest."""
    from .plan.build import build
    from .plan.validate import validate

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = run_id or f"{ts}_plan"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    (run_dir / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2))

    manifest = {
        "id": run_id, "recipe": "plan", "recipe_name": plan.name,
        "category": "Biomolecular" if plan.system.kind == "protein" else "Solvent",
        "track": "plan", "engine": "GROMACS 2026.2", "mode": "live",
        "params": {"kind": plan.system.kind, "forcefield": plan.system.forcefield,
                   "stages": len(plan.stages)},
        "status": "preparing", "created": _now(), "steps": [], "error": None,
        "outputs": {}, "energy": None, "analyses": [], "plan": plan.to_dict(),
    }

    def push():
        _save(manifest, run_dir)
        if progress_cb:
            progress_cb(manifest)

    push()
    try:
        v = validate(plan)
        manifest["validation"] = v
        if not v["ok"]:
            raise RuntimeError("plan is invalid: " +
                               "; ".join(f["message"] for f in v["findings"]
                                         if f["severity"] == "error"))
        steps, precipe = build(plan, run_dir)
        manifest["steps"] = [{"name": s.name, "status": "pending", "seconds": None}
                             for s in steps]
        manifest["status"] = "running"
        push()

        for i, step in enumerate(steps):
            manifest["steps"][i]["status"] = "running"
            push()
            t0 = time.time()
            gmx(step.argv, cwd=run_dir, log_path=log_path, stdin_text=step.stdin)
            manifest["steps"][i]["status"] = "done"
            manifest["steps"][i]["seconds"] = round(time.time() - t0, 1)
            push()

        manifest["status"] = "analysing"
        push()
        _postprocess(precipe, run_dir, {}, manifest, log_path)
        manifest["status"] = "done"
        manifest["finished"] = _now()
        push()
    except Exception as e:  # noqa: BLE001
        manifest["status"] = "error"
        manifest["error"] = str(e)
        (run_dir / "error.txt").write_text(traceback.format_exc())
        push()
    return manifest


def _postprocess(recipe, run_dir, p, manifest, log_path):
    out = recipe.outputs
    tpr, xtc, edr = out["tpr"], out["xtc"], out["edr"]
    v = recipe.viewer_opts(p)

    # 1) viewer trajectory: unwrap/center, then dump to multi-model PDB ----------
    traj_argv = ["trjconv", "-s", tpr, "-f", xtc, "-o", "traj.pdb", "-pbc", v["pbc"]]
    if v.get("center"):
        traj_argv.append("-center")
        stdin = f"{v['center']}\n{v['select']}\n"
    else:
        stdin = f"{v['select']}\n"
    gmx(traj_argv, cwd=run_dir, log_path=log_path, stdin_text=stdin, check=False)

    traj = run_dir / "traj.pdb"
    if traj.exists():
        nframes, natoms = _count_pdb(traj)
        # thin out if there are too many frames for smooth browser playback
        if nframes > FRAME_CAP:
            skip = max(2, nframes // FRAME_CAP)
            gmx(traj_argv + ["-skip", skip], cwd=run_dir, log_path=log_path,
                stdin_text=stdin, check=False)
            nframes, natoms = _count_pdb(traj)
        manifest["outputs"] = {"trajectory_pdb": "traj.pdb",
                               "n_frames": nframes, "n_atoms": natoms}

    # 2) thermodynamics from the .edr -------------------------------------------
    if (run_dir / edr).exists():
        terms = recipe.energy_terms
        stdin = "\n".join(terms) + "\n\n"
        rc, _ = gmx(["energy", "-f", edr, "-o", "energy.xvg"], cwd=run_dir,
                    log_path=log_path, stdin_text=stdin, check=False)
        exvg = run_dir / "energy.xvg"
        if exvg.exists():
            manifest["energy"] = parse_xvg(exvg)

    # 3) recipe-specific analyses -----------------------------------------------
    for a in recipe.analyses(run_dir, p):
        try:
            gmx(a.argv, cwd=run_dir, log_path=log_path, stdin_text=a.stdin, check=False)
            xvg = run_dir / a.xvg
            if xvg.exists():
                manifest["analyses"].append({
                    "name": a.name, "label": a.label, "kind": a.kind,
                    "help": a.help, "data": parse_xvg(xvg),
                })
        except Exception:  # noqa: BLE001
            continue

    # 4) python-derived analyses (e.g. area-per-lipid from the box) ---------------
    try:
        for d in recipe.derived(run_dir, p, manifest):
            manifest["analyses"].append(d)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# read-side helpers for the UI
# --------------------------------------------------------------------------- #
def load_run(run_id):
    f = RUNS_DIR / run_id / "run.json"
    if not f.exists():
        return None
    return _reconcile(json.loads(f.read_text()))


def list_runs():
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        f = d / "run.json"
        if f.exists():
            m = _reconcile(json.loads(f.read_text()))
            runs.append({k: m.get(k) for k in
                         ("id", "recipe", "recipe_name", "category", "track",
                          "engine", "mode", "status", "created", "params", "detail")})
    return runs
