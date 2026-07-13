"""Reaction-profile track — a real QM bond scan (the chemistry enzymes catalyse).

Stretches one bond of a small molecule and computes the QM energy at each step
(PySCF), giving a dissociation / reaction-coordinate energy profile plus a
trajectory of the bond breaking. This is the electronic-structure core of an
enzymatic reaction; the same machinery embeds in a protein active site (QM/MM).
"""

from __future__ import annotations

from ..recipes import Param
from ..runbase import Run

# base near-equilibrium geometries (Angstrom) + the (i,j) bond to stretch
SYSTEMS = {
    "water (O–H)":     {"atoms": [("O", (0.00, 0.00, 0.00)), ("H", (0.96, 0.00, 0.00)),
                                  ("H", (-0.24, 0.93, 0.00))], "bond": (0, 1)},
    "HF (H–F)":        {"atoms": [("F", (0.00, 0.00, 0.00)), ("H", (0.92, 0.00, 0.00))],
                        "bond": (0, 1)},
    "ammonia (N–H)":   {"atoms": [("N", (0.00, 0.00, 0.00)), ("H", (0.00, 0.00, 1.01)),
                                  ("H", (0.95, 0.00, -0.33)), ("H", (-0.48, 0.82, -0.33))],
                        "bond": (0, 1)},
}


class ReactionScan:
    key = "reaction_scan"
    name = "Bond dissociation / reaction profile"
    category = "Quantum"
    classification = "reactions"
    track = "qmmm"
    engine = "PySCF"
    mode = "live"
    needs_gpu = False
    description = ("Stretch a chemical bond and watch the QM energy climb — a "
                   "reaction-coordinate profile. Bond making/breaking is exactly "
                   "what enzymes catalyse; the same QM core embeds in a protein "
                   "active site for QM/MM enzymology.")
    est = "seconds"

    params = [
        Param("system", "Molecule / bond", "choice", "water (O–H)",
              options=list(SYSTEMS.keys())),
        Param("method", "QM method", "choice", "HF", options=["HF", "B3LYP"]),
        Param("basis", "Basis set", "choice", "sto-3g", options=["sto-3g", "6-31g"]),
        Param("r_min", "Bond length start (Å)", "float", 0.75, 0.6, 1.0, 0.05),
        Param("r_max", "Bond length end (Å)", "float", 1.90, 1.4, 2.6, 0.05),
        Param("n_points", "Scan points", "int", 20, 6, 40, 1),
    ]

    def meta(self):
        return {"key": self.key, "name": self.name, "category": self.category,
                "track": self.track, "engine": self.engine, "mode": self.mode,
                "needs_gpu": self.needs_gpu, "classification": self.classification,
                "description": self.description, "est": self.est,
                "params": [p.as_dict() for p in self.params]}

    def run(self, params, run_id=None, progress_cb=None):
        p = {q.name: q.default for q in self.params}
        for q in self.params:
            if q.name in params and params[q.name] != "":
                p[q.name] = (params[q.name] if q.type == "choice"
                             else int(float(params[q.name])) if q.type == "int"
                             else float(params[q.name]))

        run = Run(self.key, self.name, p, track=self.track, engine="PySCF",
                  mode=self.mode, category=self.category,
                  step_names=["build molecule", "scan bond (QM energies)", "record profile"],
                  progress_cb=progress_cb, run_id=run_id)
        try:
            import numpy as np
            from pyscf import gto, scf, dft

            sysdef = SYSTEMS[p["system"]]
            symbols = [s for s, _ in sysdef["atoms"]]
            base = np.array([c for _, c in sysdef["atoms"]], float)
            i, j = sysdef["bond"]
            axis = base[j] - base[i]
            axis = axis / np.linalg.norm(axis)

            def energy(coords):
                mol = gto.M(atom=[[symbols[k], tuple(coords[k])] for k in range(len(symbols))],
                            basis=p["basis"], unit="Angstrom", verbose=0)
                if p["method"] == "B3LYP":
                    mf = dft.RKS(mol); mf.xc = "b3lyp"
                else:
                    mf = scf.RHF(mol)
                mf.verbose = 0; mf.max_cycle = 200
                return mf.kernel()

            run.step(0, "done")
            run.step(1, "running")
            rs = list(np.linspace(p["r_min"], p["r_max"], int(p["n_points"])))
            frames, energies = [], []
            for r in rs:
                coords = base.copy()
                coords[j] = base[i] + axis * r          # rigid stretch of bond i–j
                energies.append(energy(coords))
                frames.append(coords.copy())
            e0 = min(energies)
            rel = [(e - e0) * 627.509 for e in energies]   # Ha -> kcal/mol, rel. to minimum
            rmin_idx = int(np.argmin(energies))
            run.log(f"min energy at r={rs[rmin_idx]:.2f} A; "
                    f"rise to r={rs[-1]:.2f} A = {rel[-1]:.1f} kcal/mol")
            run.step(1, "done")

            run.step(2, "running")
            names = [f"{s}{k+1}" for k, s in enumerate(symbols)]
            run.set_traj(frames, symbols, names=names, resnames=["MOL"] * len(symbols))
            run.set_energy(rs, [rel], ["Relative energy"],
                           xaxis="bond length (Å)", yaxis="kcal/mol")
            run.add_analysis("profile", f"Reaction profile — {p['system']}", rs, [rel],
                             xaxis="bond length (Å)", yaxis="ΔE (kcal/mol)",
                             help="Energy vs bond length: the reaction-coordinate curve. "
                                  "The rise is the bond's dissociation energy.")
            run.m["summary"] = [
                ["System", p["system"]],
                ["Method", f"{p['method']} / {p['basis']}"],
                ["Equilibrium bond", f"{rs[rmin_idx]:.2f} Å"],
                ["ΔE over scan", f"{rel[-1]:.1f} kcal/mol"],
            ]
            run.step(2, "done")
            return run.finish("done")
        except Exception as e:  # noqa: BLE001
            import traceback
            (run.dir / "error.txt").write_text(traceback.format_exc())
            return run.finish("error", str(e))
