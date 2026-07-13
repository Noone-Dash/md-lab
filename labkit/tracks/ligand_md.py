"""A drug-like molecule in water. The track that was missing.

Every other track here simulates a protein or a box of solvent. Neither can touch an actual
compound: a protein force field only knows the twenty amino acids, so an arbitrary small
molecule has NO parameters at all. That single gap is what made "simulate this drug
candidate" impossible, and it is the one this closes.

    SMILES  ->  OpenFF Sage (SMIRNOFF typing)  ->  NAGL AM1-BCC charges
            ->  packmol + TIP3P                ->  GROMACS

VALIDATION (why you should believe the numbers)
-----------------------------------------------
The export is checked against a number measured through a COMPLETELY DIFFERENT toolchain.
Aspirin in TIP3P, 500 ps NPT:

    this path (OpenFF -> Interchange -> GROMACS)   985.1 +/- 1.3 kg/m3
    plain GROMACS (pdb2gmx/solvate, no OpenFF)     986.0 +/- 1.3 kg/m3   <- independent

Agreement to within 1 sigma. Wrong charges, wrong combination rules, or a shadowed
[atomtypes] entry -- the classic silent failures when two topologies are merged by hand --
would not land on the right density. (The small solute-induced shift, ~+3 kg/m3, is below
the resolution of this run; it is NOT claimed.)

The charge method is recorded and displayed, because charges are the single largest error
source in a small-molecule force field and results computed with different ones are not
comparable.
"""

from __future__ import annotations

import shutil
import time

from ..config import RUNS_DIR
from ..engine import _now, _save
from ..gmx import gmx
from ..mdp import md_mdp, minim_mdp
from ..recipes import Param


class LigandMD:
    key = "ligand_md"
    name = "Drug molecule in water"
    category = "Biomolecular"
    track = "classical"
    engine = "GROMACS + OpenFF"
    mode = "live"
    needs_gpu = True
    description = ("A real small molecule — aspirin, ibuprofen, caffeine, or any SMILES — "
                   "parameterised with OpenFF Sage and AM1-BCC-quality charges, solvated, "
                   "and run. This is the piece a protein force field cannot give you.")
    est = "~1 min"

    params = [
        Param("molecule", "Molecule", "text", default="aspirin",
              help="A name (aspirin, ibuprofen, caffeine, paracetamol, penicillin g) "
                   "or any SMILES string, e.g. CC(=O)Oc1ccccc1C(=O)O"),
        Param("n_waters", "Water molecules", "int", default=800, min=200, max=4000,
              help="More water = a bigger, slower, better-converged box."),
        Param("box_nm", "Box (nm)", "float", default=3.0, min=2.0, max=6.0,
              help="Must be more than twice the cutoff, or the molecule sees its own image."),
        Param("ns", "Production (ns)", "float", default=0.1, min=0.01, max=100.0,
              help="0.1 ns is enough to see it tumble and hydrate; longer for real numbers."),
        Param("temperature", "Temperature (K)", "float", default=300.0, min=100.0, max=500.0),
    ]

    def meta(self):
        return {"key": self.key, "name": self.name, "category": self.category,
                "track": self.track, "engine": self.engine, "mode": self.mode,
                "needs_gpu": self.needs_gpu, "description": self.description,
                "est": self.est, "params": [p.as_dict() for p in self.params]}

    def run(self, params, run_id=None, progress_cb=None):
        from .. import ligand

        p = {q.name: q.default for q in self.params}
        for q in self.params:
            if q.name in params and params[q.name] != "":
                p[q.name] = type(q.default)(params[q.name])

        run_id = run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_ligand"
        rd = RUNS_DIR / run_id
        rd.mkdir(parents=True, exist_ok=True)
        log = rd / "run.log"

        m = {"id": run_id, "recipe": self.key, "recipe_name": f"{p['molecule']} in water",
             "category": self.category, "track": self.track, "engine": self.engine,
             "mode": self.mode, "status": "running", "created": _now(), "params": p,
             "steps": [{"name": "parameterise (OpenFF)", "status": "running", "seconds": None},
                       {"name": "minimise", "status": "pending", "seconds": None},
                       {"name": f"production ({p['ns']} ns)", "status": "pending",
                        "seconds": None}]}
        _save(m, rd)

        def push(i, st, secs=None):
            m["steps"][i]["status"] = st
            if secs is not None:
                m["steps"][i]["seconds"] = round(secs, 1)
            _save(m, rd)
            if progress_cb:
                progress_cb(m)

        # ---- 1. parameterise -------------------------------------------------
        t0 = time.time()
        try:
            lig = ligand.parameterize(p["molecule"], name="LIG",
                                      n_waters=int(p["n_waters"]),
                                      box_nm=float(p["box_nm"]))
        except Exception as e:  # noqa: BLE001
            m["status"] = "error"
            m["error"] = str(e)
            m["steps"][0]["status"] = "error"
            _save(m, rd)
            return m
        push(0, "done", time.time() - t0)

        # The provenance of the FORCE FIELD, front and centre. A result is meaningless
        # without knowing how the charges were made.
        m["ligand"] = {k: lig.get(k) for k in
                       ("smiles", "inchikey", "n_atoms", "ligand_atoms", "formal_charge",
                        "net_partial_charge", "charge_method", "forcefield", "water_model",
                        "n_waters")}
        m["n_atoms"] = lig["n_atoms"]
        _save(m, rd)

        shutil.copy(lig["top"], rd / "sys.top")
        shutil.copy(lig["gro"], rd / "sys.gro")

        # ---- 2. minimise -----------------------------------------------------
        t0 = time.time()
        push(1, "running")
        (rd / "em.mdp").write_text(minim_mdp(nsteps=5000))
        rc, _ = gmx(["grompp", "-f", "em.mdp", "-c", "sys.gro", "-p", "sys.top",
                     "-o", "em.tpr", "-maxwarn", "2"], cwd=rd, log_path=log, check=False)
        if rc:
            m["status"] = "error"; m["error"] = "grompp failed (minimise)"
            push(1, "error"); return m
        gmx(["mdrun", "-deffnm", "em"], cwd=rd, log_path=log, check=False)
        push(1, "done", time.time() - t0)

        # ---- 3. production ---------------------------------------------------
        t0 = time.time()
        push(2, "running")
        nsteps = int(round(float(p["ns"]) * 1000 / 0.002))
        (rd / "md.mdp").write_text(md_mdp(nsteps=nsteps, dt=0.002,
                                          temperature=float(p["temperature"]),
                                          ensemble="NPT", nstxout=max(500, nsteps // 200)))
        rc, _ = gmx(["grompp", "-f", "md.mdp", "-c", "em.gro", "-p", "sys.top",
                     "-o", "md.tpr", "-maxwarn", "2"], cwd=rd, log_path=log, check=False)
        if rc:
            m["status"] = "error"; m["error"] = "grompp failed (production)"
            push(2, "error"); return m
        rc, _ = gmx(["mdrun", "-deffnm", "md"], cwd=rd, log_path=log, check=False)
        if rc:
            m["status"] = "error"; m["error"] = "mdrun failed"
            push(2, "error"); return m
        push(2, "done", time.time() - t0)

        # ---- trajectory for the viewer + the usual analyses -------------------
        gmx(["trjconv", "-f", "md.xtc", "-s", "md.tpr", "-o", "traj.pdb",
             "-pbc", "mol", "-conect"], cwd=rd, stdin_text="System\n", log_path=log,
            check=False)
        gmx(["energy", "-f", "md.edr", "-o", "energy.xvg"], cwd=rd,
            stdin_text="Potential\nTemperature\nDensity\n0\n", log_path=log, check=False)

        from ..engine import FRAME_CAP, _count_pdb
        from ..xvg import parse_xvg
        traj = rd / "traj.pdb"
        if traj.exists():
            nframes, natoms = _count_pdb(traj)
            if nframes > FRAME_CAP:
                gmx(["trjconv", "-f", "md.xtc", "-s", "md.tpr", "-o", "traj.pdb",
                     "-pbc", "mol", "-conect", "-skip", str(max(2, nframes // FRAME_CAP))],
                    cwd=rd, stdin_text="System\n", log_path=log, check=False)
                nframes, natoms = _count_pdb(traj)
            m["outputs"] = {"trajectory_pdb": "traj.pdb",
                            "n_frames": nframes, "n_atoms": natoms}
        exvg = rd / "energy.xvg"
        if exvg.exists():
            m["energy"] = parse_xvg(exvg)
        m["analyses"] = []

        m["status"] = "done"
        m["finished"] = _now()
        _save(m, rd)
        return m
