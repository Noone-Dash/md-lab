"""Cost estimator — so an agent can plan inside the GPU budget instead of guessing.

Calibrated from THIS machine's own completed runs (simulations/runs/*/run.json),
not from a made-up constant. Falls back to a measured default if no history exists.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent.parent / "simulations" / "runs"

# fallback throughput, atom-steps per second, measured on the GB10 during this build
DEFAULT_ATOM_STEPS_PER_S = {"atomistic": 2.6e7, "martini": 6.0e7}


def _calibrate() -> dict:
    """Learn atom-steps/s from finished GROMACS runs on this box."""
    out = {}
    if not RUNS.exists():
        return out
    for d in RUNS.iterdir():
        f = d / "run.json"
        if not f.exists():
            continue
        try:
            m = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        if m.get("status") != "done" or not str(m.get("engine", "")).startswith("GROMACS"):
            continue
        secs = next((s.get("seconds") for s in m.get("steps", [])
                     if s.get("name", "").startswith("MD production") and s.get("seconds")), None)
        atoms = (m.get("outputs") or {}).get("n_atoms")
        nsteps = m.get("params", {}).get("nsteps")
        if not (secs and atoms and nsteps) or secs < 1.0:
            continue
        reg = "martini" if m.get("recipe") == "martini_bilayer" else "atomistic"
        out.setdefault(reg, []).append(int(atoms) * int(nsteps) / float(secs))
    return {k: sorted(v)[len(v) // 2] for k, v in out.items() if v}   # median


_CAL = None


def throughput(reg: str) -> tuple[float, str]:
    global _CAL
    if _CAL is None:
        _CAL = _calibrate()
    if reg in _CAL:
        return _CAL[reg], "calibrated from this machine's finished runs"
    return DEFAULT_ATOM_STEPS_PER_S.get(reg, 2.6e7), "default (no calibration history yet)"


def estimate(plan, n_atoms: int | None = None) -> dict:
    """Wall-clock + memory estimate for a Plan."""
    from .resolve import resolve, regime
    rp = resolve(plan)
    reg = rp["regime"]
    s = plan.system

    if n_atoms is None:
        # crude but honest: ~100 atoms/nm^3 for atomistic water, ~10 beads/nm^3 for CG
        if s.kind in ("solvent", "fluid"):
            vol = float(s.box_size_nm) ** 3
        elif s.kind == "membrane":
            vol = float(s.box_size_nm) ** 3
        else:
            vol = (4.0 + 2 * float(s.box_padding_nm)) ** 3   # rough protein box
        density = 10.0 if reg == "martini" else 100.0
        n_atoms = int(vol * density)

    rate, source = throughput(reg)
    per_stage, total_s = [], 0.0
    for st in rp["stages"]:
        nsteps = int(st["mdp"].get("nsteps", 0) or 0)
        if st["type"] == "minimize":
            secs = max(2.0, n_atoms * 500 / rate)      # EM rarely runs its full cap
        else:
            secs = n_atoms * nsteps / rate
        total_s += secs
        per_stage.append({"stage": st["name"], "type": st["type"],
                          "nsteps": nsteps, "seconds": round(secs, 1)})

    mem_gb = round(max(0.5, n_atoms * 3e-5), 2)   # ~30 kB/atom incl. GPU buffers
    return {
        "n_atoms_estimated": n_atoms,
        "regime": reg,
        "throughput_atom_steps_per_s": f"{rate:.2e}",
        "throughput_source": source,
        "per_stage": per_stage,
        "total_seconds": round(total_s, 1),
        "total_human": _human(total_s),
        "peak_memory_gb_estimated": mem_gb,
    }


def _human(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s/60:.1f} min"
    return f"{s/3600:.1f} h"
