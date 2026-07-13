"""OpenMM track — real implicit-solvent MD of a small peptide.

Showcases OpenMM as a second, fully-programmable engine: amber14 + GBn2 implicit
solvent (no explicit water box, so it's cheap), Langevin dynamics, on whatever
platform is fastest (CUDA/OpenCL/CPU). Emits the standard manifest.
"""

from __future__ import annotations

import time
import urllib.request

from ..recipes import Param
from ..runbase import Run


class OpenMMImplicit:
    key = "openmm_implicit"
    name = "Peptide MD (OpenMM, implicit solvent)"
    category = "Biomolecular"
    track = "openmm"
    engine = "OpenMM"
    mode = "live"
    needs_gpu = True
    description = ("A small peptide simulated with OpenMM using the amber14 force "
                   "field and GBn2 implicit solvent — no explicit water, so it runs "
                   "cheaply. OpenMM is Python-programmable and the on-ramp for ML "
                   "potentials (MACE/ANI).")
    est = "~1–3 min (downloads a PDB)"

    params = [
        Param("pdb_id", "Peptide (PDB)", "choice", "1UAO",
              options=["1UAO", "1L2Y", "1VII"],
              help="1UAO chignolin (10 res, tiny) · 1L2Y Trp-cage · 1VII villin."),
        Param("temperature", "Temperature (K)", "float", 300.0, 250.0, 400.0, 10.0),
        Param("friction", "Langevin friction (1/ps)", "float", 1.0, 0.1, 5.0, 0.1),
        Param("steps", "MD steps (dt = 2 fs)", "int", 30000, 2000, 300000, 1000),
    ]

    def meta(self):
        return {"key": self.key, "name": self.name, "category": self.category,
                "track": self.track, "engine": self.engine, "mode": self.mode,
                "needs_gpu": self.needs_gpu, "description": self.description,
                "est": self.est, "params": [p.as_dict() for p in self.params]}

    def run(self, params, run_id=None, progress_cb=None):
        p = {q.name: q.default for q in self.params}
        for q in self.params:
            if q.name in params and params[q.name] != "":
                p[q.name] = (params[q.name] if q.type == "choice"
                             else int(float(params[q.name])) if q.type == "int"
                             else float(params[q.name]))

        run = Run(self.key, self.name, p, track=self.track, engine="OpenMM",
                  mode=self.mode, category=self.category,
                  step_names=["fetch & prepare (PDB, H, forcefield)",
                              "energy minimise", "Langevin MD", "analyse"],
                  progress_cb=progress_cb, run_id=run_id)
        try:
            import numpy as np
            from openmm import app, LangevinMiddleIntegrator, Platform, unit

            run.engine = self.engine
            # ---- prepare -------------------------------------------------------
            run.step(0, "running")
            pdb_id = str(p["pdb_id"]).upper()
            pdb_path = run.dir / "input.pdb"
            with urllib.request.urlopen(
                    f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30) as r:
                pdb_path.write_bytes(r.read())

            pdb = app.PDBFile(str(pdb_path))
            ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
            modeller = app.Modeller(pdb.topology, pdb.positions)
            modeller.deleteWater()
            modeller.addHydrogens(ff)
            system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                                     constraints=app.HBonds)
            integrator = LangevinMiddleIntegrator(
                p["temperature"] * unit.kelvin, p["friction"] / unit.picosecond,
                0.002 * unit.picoseconds)

            platform = None
            for name in ("CUDA", "OpenCL", "CPU"):
                try:
                    platform = Platform.getPlatformByName(name)
                    break
                except Exception:  # noqa: BLE001
                    continue
            sim = app.Simulation(modeller.topology, system, integrator, platform)
            sim.context.setPositions(modeller.positions)
            run.m["engine"] = f"OpenMM ({platform.getName()})"
            run.log(f"platform: {platform.getName()}  atoms: {system.getNumParticles()}")
            run.step(0, "done")

            # ---- minimise ------------------------------------------------------
            run.step(1, "running")
            sim.minimizeEnergy()
            run.step(1, "done")

            # topology metadata
            atoms = list(modeller.topology.atoms())
            symbols = [a.element.symbol if a.element else "C" for a in atoms]
            names = [a.name for a in atoms]
            resnames = [a.residue.name for a in atoms]
            resids = [a.residue.index + 1 for a in atoms]
            masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                               for i in range(system.getNumParticles())])
            ndof = 3 * system.getNumParticles() - system.getNumConstraints()
            kB = 0.00831446              # kJ/mol/K

            # ---- MD ------------------------------------------------------------
            run.step(2, "running")
            steps = int(p["steps"])
            n_frames = 60
            every = max(1, steps // n_frames)
            frames, t_ps, epot, ekin, temp, rg = [], [], [], [], [], []
            t0 = time.time()
            done = 0
            while done < steps:
                chunk = min(every, steps - done)
                sim.step(chunk)
                done += chunk
                st = sim.context.getState(getPositions=True, getEnergy=True)
                xyz = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
                frames.append(np.array(xyz))
                pe = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                epot.append(pe); ekin.append(ke)
                temp.append(2 * ke / (ndof * kB))
                t_ps.append(done * 0.002)
                cm = (masses[:, None] * xyz).sum(0) / masses.sum()
                rg.append(float(np.sqrt((masses * ((xyz - cm) ** 2).sum(1)).sum()
                                        / masses.sum())))
            run.log(f"MD wall time: {time.time() - t0:.1f}s")
            run.step(2, "done")

            # ---- record --------------------------------------------------------
            run.step(3, "running")
            run.set_traj(frames, symbols, names=names, resnames=resnames, resids=resids)
            run.set_energy(t_ps, [epot, ekin, temp],
                           ["Potential", "Kinetic", "Temperature"],
                           xaxis="Time (ps)", yaxis="kJ/mol  |  K")
            run.add_analysis("temperature", "Temperature", t_ps, [temp],
                             xaxis="Time (ps)", yaxis="K",
                             help="Langevin thermostat holding the target.")
            run.add_analysis("rg", "Radius of gyration (compactness)", t_ps, [rg],
                             xaxis="Time (ps)", yaxis="Å",
                             help="Peptide size over time — dips as it compacts/folds.")
            run.step(3, "done")
            return run.finish("done")
        except Exception as e:  # noqa: BLE001
            import traceback
            (run.dir / "error.txt").write_text(traceback.format_exc())
            return run.finish("error", str(e))
