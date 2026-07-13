"""Opportunistic backfill: long MD campaigns that fill idle GPU and yield on demand.

WHY THIS AND NOT A FIXED DUTY CYCLE
-----------------------------------
MD throughput is linear in GPU time: work = duty x K, with no batching gain. So a fixed
"use 20% of the GPU" cap is strictly wasteful — it idles the other 80% whenever you are
not using the machine. The optimal policy is instead:

    run whenever the GPU is idle;  yield instantly when it is wanted.

Expected utilisation -> (1 - your_usage), which is far above 20%.

WHY IT IS SAFE TO PREEMPT
-------------------------
Measured, not assumed: on SIGTERM, `gmx mdrun` stops at the next neighbour-search step and
writes a checkpoint (verified: SIGTERM at step 81,600 -> resumed to 107,200, log confirms
"Restarting from checkpoint"). So preemption costs at most the work since the last
checkpoint, and we checkpoint every minute. Resume is exact: mdrun restores positions,
velocities, thermostat/barostat state and the RNG stream, so the trajectory is continuous
— not a restart with fresh velocities.

A campaign therefore = one built system + many preemptible mdrun chunks, accumulating ns.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .engine import RUNS_DIR, _now, _save
from .gmx import gmx
from .plan.build import build
from .plan.schema import Plan

CHECKPOINT_MIN = 1.0          # -cpt: max work lost to a preemption


def _cpt_progress(run_dir: Path, name: str = "production"):
    """(step, time_ps) from the checkpoint. GROMACS knows exactly where it is."""
    cpt = run_dir / f"{name}.cpt"
    if not cpt.exists():
        return 0, 0.0
    rc, out = gmx(["dump", "-cp", str(cpt)], cwd=run_dir, check=False)
    step = re.search(r"^\s*step\s*=\s*(\d+)", out, re.M)
    t = re.search(r"^\s*t\s*=\s*([\d.eE+-]+)", out, re.M)
    return (int(step.group(1)) if step else 0,
            float(t.group(1)) if t else 0.0)


def run_campaign(plan_dict: dict, run_id: str, target_ns: float, progress_cb=None):
    """Run a long campaign in preemptible chunks. Safe to kill; safe to resume."""
    plan = Plan.from_dict(plan_dict)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "run.log"
    mf = run_dir / "run.json"

    manifest = json.loads(mf.read_text()) if mf.exists() else {}
    manifest.update({
        "id": run_id, "recipe": "backfill", "recipe_name": plan.name,
        "category": "Biomolecular" if plan.system.kind == "protein" else "Solvent",
        "track": "plan", "engine": "GROMACS 2026.2", "mode": "live",
        "klass": "backfill", "target_ns": target_ns,
        "params": {"kind": plan.system.kind, "target_ns": target_ns},
        "status": "running", "created": manifest.get("created", _now()),
        "plan": plan_dict,
    })

    def push():
        _save(manifest, run_dir)
        if progress_cb:
            progress_cb(manifest)

    # ---- build the system ONCE; later launches reuse it ---------------------
    built = run_dir / ".built"
    dt_ps = 0.002
    total_steps = int(round(target_ns * 1000.0 / dt_ps))

    if not built.exists():
        manifest["steps"] = [{"name": "build + equilibrate", "status": "running",
                              "seconds": None}]
        push()
        steps, precipe = build(plan, run_dir)
        # run everything EXCEPT the final production mdrun; that one we chunk
        for st in steps[:-1]:
            gmx(st.argv, cwd=run_dir, log_path=log, stdin_text=st.stdin)
        built.write_text("ok")
        manifest["steps"][0]["status"] = "done"
        manifest["steps"].append({"name": f"production ({target_ns} ns)",
                                  "status": "running", "seconds": None})
        push()

    # ---- the preemptible production chunk ----------------------------------
    name = plan.stages[-1].name
    step0, t0_ps = _cpt_progress(run_dir, name)
    manifest["done_ns"] = round(t0_ps / 1000.0, 4)
    manifest["progress_pct"] = round(100.0 * manifest["done_ns"] / target_ns, 1)
    push()

    argv = ["mdrun", "-deffnm", name,
            "-nsteps", str(total_steps),
            "-cpt", str(CHECKPOINT_MIN)]
    if (run_dir / f"{name}.cpt").exists():
        argv += ["-cpi", f"{name}.cpt"]          # exact continuation

    # Publish progress WHILE mdrun runs. Without this the manifest only updates after
    # the chunk ends, so nothing outside can see how far along the campaign is — and a
    # preemption policy that cannot observe progress cannot act on it.
    import threading
    stop = threading.Event()

    def _poll():
        while not stop.wait(5.0):
            try:
                _, t_ps = _cpt_progress(run_dir, name)
                if t_ps > 0:
                    manifest["done_ns"] = round(t_ps / 1000.0, 4)
                    manifest["progress_pct"] = round(
                        min(100.0, 100.0 * t_ps / 1000.0 / target_ns), 1)
                    push()
            except Exception:  # noqa: BLE001
                pass

    watcher = threading.Thread(target=_poll, daemon=True)
    watcher.start()

    t_start = time.time()
    gmx(argv, cwd=run_dir, log_path=log, check=False)   # SIGTERM here is FINE
    wall = time.time() - t_start
    stop.set()

    step1, t1_ps = _cpt_progress(run_dir, name)
    done_ns = t1_ps / 1000.0
    manifest["done_ns"] = round(done_ns, 4)
    manifest["progress_pct"] = round(min(100.0, 100.0 * done_ns / target_ns), 1)
    manifest["last_chunk_s"] = round(wall, 1)
    if step1 > 0 and wall > 5:
        manifest["measured_ns_per_day"] = round(
            (t1_ps - t0_ps) / 1000.0 / (wall / 86400.0), 1)

    complete = step1 >= total_steps - 1
    if complete:
        manifest["status"] = "done"
        manifest["steps"][-1]["status"] = "done"
        manifest["finished"] = _now()
        from .engine import _postprocess
        _, precipe = build(plan, run_dir)      # rebuild the adapter for post-processing
        try:
            _postprocess(precipe, run_dir, {}, manifest, log)
        except Exception:  # noqa: BLE001
            pass
    else:
        # preempted: keep the checkpoint, go back in the queue
        manifest["status"] = "preempted"
    push()
    return manifest
