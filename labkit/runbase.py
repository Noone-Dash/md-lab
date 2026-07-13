"""Shared plumbing for non-GROMACS engines (OpenMM, PySCF QM/MM, cell model).

They all write the *same* ``run.json`` manifest + ``traj.pdb`` that the GROMACS
engine produces, so the web viewer and plots render every track identically.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import RUNS_DIR

# element -> role mapping we use for coloured particles in 3Dmol
#   N blue · O red · S yellow · C grey · P orange
def _now():
    return datetime.now().isoformat(timespec="seconds")


def _pdb_atom(i, name, resn, resi, x, y, z, elem):
    """One ATOM record in strict PDB column layout.

    Columns (1-based): 1-6 record · 7-11 serial · 13-16 name · 17 altLoc ·
    18-20 resName · 22 chainID · 23-26 resSeq · 31-38 x · 39-46 y · 47-54 z ·
    55-60 occupancy · 61-66 tempFactor · 77-78 element.
    Getting this off by even one column makes 3Dmol silently drop atoms.
    """
    nm = name[:4] if len(name) >= 4 else " " + name        # short names start at col 14
    return (
        f"ATOM  "            # 1-6
        f"{i % 100000:5d}"   # 7-11  serial
        f" "                 # 12
        f"{nm:<4s}"          # 13-16 atom name
        f" "                 # 17    altLoc
        f"{resn[:3]:>3s}"    # 18-20 resName
        f" "                 # 21
        f"A"                 # 22    chainID
        f"{resi % 10000:4d}" # 23-26 resSeq
        f" "                 # 27    iCode
        f"   "               # 28-30
        f"{x:8.3f}{y:8.3f}{z:8.3f}"   # 31-54
        f"{1.0:6.2f}{0.0:6.2f}"       # 55-66
        f"          "        # 67-76
        f"{elem[:2]:>2s}"    # 77-78 element
        f"\n")


def write_pdb_traj(path, frames, symbols, resnames=None, resids=None,
                   names=None, box=None):
    """Write a multi-model PDB.

    frames   : list of (N,3) coordinate arrays in Angstrom
    symbols  : list[str] element symbol per atom (length N)
    """
    n = len(symbols)
    resnames = resnames or ["MOL"] * n
    resids = resids or [1] * n
    names = names or symbols
    with open(path, "w") as fh:
        if box:
            fh.write(f"CRYST1{box[0]:9.3f}{box[1]:9.3f}{box[2]:9.3f}"
                     f"  90.00  90.00  90.00 P 1           1\n")
        for fi, coords in enumerate(frames, start=1):
            fh.write(f"MODEL     {fi:4d}\n")
            for i in range(n):
                x, y, z = coords[i]
                fh.write(_pdb_atom(i + 1, names[i], resnames[i], resids[i],
                                   float(x), float(y), float(z), symbols[i]))
            fh.write("ENDMDL\n")


class Run:
    """A lightweight run recorder that mirrors the GROMACS manifest schema."""

    def __init__(self, key, name, params, *, track, engine, mode, category,
                 step_names, progress_cb=None, run_id=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"{ts}_{key}"
        self.dir = RUNS_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "run.log"
        self.progress_cb = progress_cb
        self.m = {
            "id": self.run_id, "recipe": key, "recipe_name": name,
            "category": category, "track": track, "engine": engine, "mode": mode,
            "params": params, "status": "running", "created": _now(),
            "steps": [{"name": s, "status": "pending", "seconds": None} for s in step_names],
            "error": None, "outputs": {}, "energy": None, "analyses": [],
        }
        self.save()

    def save(self):
        (self.dir / "run.json").write_text(json.dumps(self.m, indent=2))
        if self.progress_cb:
            self.progress_cb(self.m)

    def log(self, text):
        with open(self.log_path, "a") as fh:
            fh.write(text + "\n")

    def step(self, i, status, seconds=None):
        self.m["steps"][i]["status"] = status
        if seconds is not None:
            self.m["steps"][i]["seconds"] = round(seconds, 2)
        self.save()

    def set_traj(self, frames, symbols, **kw):
        write_pdb_traj(self.dir / "traj.pdb", frames, symbols, **kw)
        self.m["outputs"] = {"trajectory_pdb": "traj.pdb",
                             "n_frames": len(frames), "n_atoms": len(symbols)}
        self.save()

    def set_energy(self, x, series, legends, xaxis="step", yaxis=""):
        self.m["energy"] = {"title": "", "xaxis": xaxis, "yaxis": yaxis,
                            "legends": legends, "x": x, "series": series}
        self.save()

    def add_analysis(self, name, label, x, series, *, legends=None,
                     xaxis="", yaxis="", kind="timeseries", help=""):
        self.m["analyses"].append({
            "name": name, "label": label, "kind": kind, "help": help,
            "data": {"title": "", "xaxis": xaxis, "yaxis": yaxis,
                     "legends": legends or [], "x": x, "series": series},
        })
        self.save()

    def finish(self, status="done", error=None):
        self.m["status"] = status
        self.m["error"] = error
        self.m["finished"] = _now()
        self.save()
        return self.m
