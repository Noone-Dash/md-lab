"""QM/MM track — real quantum chemistry with PySCF.

A small molecule is geometry-optimised at the QM level (HF or B3LYP) via ASE +
a PySCF energy/force calculator; the relaxation is captured as a trajectory and
an energy-convergence curve. Optionally an MM water (TIP3P point charges) is
added and a QM/MM electrostatic-embedding single point reports the interaction
energy and how the MM field polarises the molecule — genuine QM/MM, cheaply.
"""

from __future__ import annotations

from ..recipes import Param
from ..runbase import Run

# slightly distorted starting geometries (Angstrom) so the optimiser visibly relaxes
GEOMS = {
    "water":        [("O", (0.00, 0.00, 0.00)), ("H", (1.10, 0.00, 0.00)),
                     ("H", (-0.30, 1.00, 0.00))],
    "ammonia":      [("N", (0.00, 0.00, 0.00)), ("H", (0.00, 0.00, 1.12)),
                     ("H", (1.06, 0.00, -0.34)), ("H", (-0.53, 0.92, -0.34))],
    "formaldehyde": [("C", (0.00, 0.00, 0.00)), ("O", (0.00, 0.00, 1.35)),
                     ("H", (0.98, 0.00, -0.55)), ("H", (-0.98, 0.00, -0.55))],
    "methane":      [("C", (0.00, 0.00, 0.00)), ("H", (0.70, 0.70, 0.70)),
                     ("H", (-0.70, -0.70, 0.70)), ("H", (-0.70, 0.70, -0.70)),
                     ("H", (0.70, -0.70, -0.70))],
}
# an MM TIP3P water placed nearby (coords Angstrom, charges e)
MM_WATER = ([(3.1, 0.0, 0.0), (3.7, 0.55, 0.0), (3.7, -0.55, 0.0)],
            [-0.834, 0.417, 0.417])


class QMMM:
    key = "qmmm_opt"
    name = "QM / QM-MM (PySCF)"
    category = "Quantum"
    track = "qmmm"
    engine = "PySCF"
    mode = "live"
    needs_gpu = False
    description = ("Real quantum chemistry: optimise a small molecule with Hartree–"
                   "Fock or B3LYP, watch the energy converge to the minimum, then add "
                   "an MM water and measure the QM/MM electrostatic interaction. "
                   "Semi-empirical/DFT is the cheap route to near-QM accuracy.")
    est = "seconds"

    params = [
        Param("molecule", "Molecule", "choice", "water",
              options=["water", "ammonia", "formaldehyde", "methane"]),
        Param("method", "QM method", "choice", "HF", options=["HF", "B3LYP"]),
        Param("basis", "Basis set", "choice", "sto-3g", options=["sto-3g", "6-31g"]),
        Param("mm_embedding", "Add MM water (QM/MM)", "bool", True),
    ]

    def meta(self):
        return {"key": self.key, "name": self.name, "category": self.category,
                "track": self.track, "engine": self.engine, "mode": self.mode,
                "needs_gpu": self.needs_gpu, "description": self.description,
                "est": self.est, "params": [p.as_dict() for p in self.params]}

    def _mf(self, mol, method):
        from pyscf import scf, dft
        if method == "B3LYP":
            mf = dft.RKS(mol); mf.xc = "b3lyp"
        else:
            mf = scf.RHF(mol)
        mf.verbose = 0
        return mf

    def run(self, params, run_id=None, progress_cb=None):
        p = {q.name: q.default for q in self.params}
        for q in self.params:
            if q.name in params and params[q.name] != "":
                p[q.name] = (str(params[q.name]).lower() in ("1", "true", "yes", "on")
                             if q.type == "bool" else params[q.name])

        run = Run(self.key, self.name, p, track=self.track, engine="PySCF",
                  mode=self.mode, category=self.category,
                  step_names=["build molecule", "QM geometry optimisation",
                              "QM/MM embedding", "record"],
                  progress_cb=progress_cb, run_id=run_id)
        try:
            import numpy as np
            from ase import Atoms
            from ase.optimize import BFGS
            from ase.calculators.calculator import Calculator, all_changes
            from ase.units import Hartree, Bohr
            from pyscf import gto

            method, basis = p["method"], p["basis"]
            geom = GEOMS[p["molecule"]]
            symbols = [s for s, _ in geom]

            outer = self

            class PySCFCalc(Calculator):
                implemented_properties = ["energy", "forces"]

                def calculate(self, atoms=None, properties=("energy",),
                              system_changes=all_changes):
                    Calculator.calculate(self, atoms, properties, system_changes)
                    mol = gto.M(atom=[[s, tuple(pos)] for s, pos in
                                      zip(atoms.get_chemical_symbols(),
                                          atoms.get_positions())],
                                basis=basis, unit="Angstrom", verbose=0)
                    mf = outer._mf(mol, method)
                    e = mf.kernel()
                    g = mf.nuc_grad_method().kernel()
                    self.results["energy"] = e * Hartree
                    self.results["forces"] = -g * Hartree / Bohr

            run.step(0, "running")
            atoms = Atoms(symbols=symbols, positions=[list(c) for _, c in geom])
            atoms.calc = PySCFCalc()
            run.step(0, "done")

            # ---- QM optimisation, capturing every step ------------------------
            run.step(1, "running")
            frames, energies = [], []

            def snap():
                frames.append(atoms.get_positions().copy())
                energies.append(atoms.get_potential_energy())    # eV

            snap()
            opt = BFGS(atoms, logfile=None)
            opt.attach(snap, interval=1)
            opt.run(fmax=0.05, steps=40)
            e0 = energies[0]
            rel_kcal = [(e - e0) * 23.060548 for e in energies]   # eV -> kcal/mol
            run.log(f"optimised in {len(energies)-1} steps; "
                    f"ΔE = {rel_kcal[-1]:.2f} kcal/mol")
            run.step(1, "done")

            # ---- final QM properties + optional QM/MM -------------------------
            run.step(2, "running")
            final = atoms.get_positions()
            mol = gto.M(atom=[[s, tuple(c)] for s, c in zip(symbols, final)],
                        basis=basis, unit="Angstrom", verbose=0)
            mf = self._mf(mol, method)
            e_qm = mf.kernel()
            occ = mf.mo_occ > 0
            homo = mf.mo_energy[occ].max()
            lumo = mf.mo_energy[~occ].min()
            gap_ev = (lumo - homo) * 27.211386
            dip = float(np.linalg.norm(mf.dip_moment(unit="Debye", verbose=0)))

            summary = [
                ["Method", f"{method} / {basis}"],
                ["Final QM energy", f"{e_qm:.6f} Ha"],
                ["HOMO–LUMO gap", f"{gap_ev:.2f} eV"],
                ["Dipole moment", f"{dip:.2f} D"],
            ]
            if p["mm_embedding"]:
                from pyscf.qmmm import mm_charge
                coords, charges = MM_WATER
                mfq = mm_charge(self._mf(mol, method), coords, charges, unit="Angstrom")
                e_qmmm = mfq.kernel()
                dipq = float(np.linalg.norm(mfq.dip_moment(unit="Debye", verbose=0)))
                inter = (e_qmmm - e_qm) * 627.509    # Ha -> kcal/mol
                summary += [
                    ["QM/MM interaction", f"{inter:.2f} kcal/mol"],
                    ["Dipole with MM water", f"{dipq:.2f} D  (was {dip:.2f})"],
                ]
                run.log(f"QM/MM electrostatic interaction: {inter:.2f} kcal/mol")
            run.m["summary"] = summary
            run.step(2, "done")

            # ---- record -------------------------------------------------------
            run.step(3, "running")
            names = [f"{s}{i+1}" for i, s in enumerate(symbols)]
            run.set_traj(frames, symbols, names=names,
                         resnames=["QM"] * len(symbols))
            steps_x = list(range(len(rel_kcal)))
            run.set_energy(steps_x, [rel_kcal], ["Relative energy"],
                           xaxis="optimisation step", yaxis="kcal/mol")
            run.add_analysis("conv", "Geometry-optimisation energy", steps_x, [rel_kcal],
                             xaxis="optimisation step", yaxis="ΔE (kcal/mol)",
                             help="Energy falling to the relaxed minimum-energy structure.")
            run.step(3, "done")
            return run.finish("done")
        except Exception as e:  # noqa: BLE001
            import traceback
            (run.dir / "error.txt").write_text(traceback.format_exc())
            return run.finish("error", str(e))
