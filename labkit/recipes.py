"""Simulation *recipes* — parametric definitions the engine can build & run.

Each recipe exposes:
  * ``params``      – UI form schema (sliders / dropdowns)
  * ``prepare()``   – write all input files, return the ordered list of gmx steps
  * ``analyses()``  – post-run gmx analyses that produce .xvg data for plots
  * ``viewer``      – how to turn the trajectory into something 3Dmol can show
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .mdp import minim_mdp, md_mdp, martini_em_mdp, martini_md_mdp
from .gro import lattice_gro, read_box

FF = "amber99sb-ildn.ff"  # ships every water model + ions we need
from .config import ASSETS_DIR as _ASSETS
MART_DIR = _ASSETS / "martini"


def _insane_bin():
    """Locate the `insane` console script. It is NOT always at sys.prefix/bin
    (pip --user, conda, system install all put it elsewhere)."""
    import shutil as _sh
    return (_sh.which("insane")
            or str(Path(sys.prefix) / "bin" / "insane"))


INSANE = _insane_bin()


# --------------------------------------------------------------------------- #
# small schema helpers
# --------------------------------------------------------------------------- #
@dataclass
class Param:
    name: str
    label: str
    type: str                       # int | float | choice | bool
    default: object
    min: float = None
    max: float = None
    step: float = None
    options: list = None
    help: str = ""

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Step:
    name: str
    argv: list
    stdin: str = None
    timeout: int = None
    func: object = None            # if set, a python callable(run_dir, log_path) instead of gmx


@dataclass
class Analysis:
    name: str
    label: str
    argv: list                      # gmx args; must write to *xvg*
    xvg: str
    stdin: str = None
    kind: str = "timeseries"        # timeseries | rdf | msd
    help: str = ""


def _coerce(params, schema):
    """Fill defaults + cast types from a raw params dict."""
    out = {}
    by_name = {p.name: p for p in schema}
    for p in schema:
        raw = params.get(p.name, p.default)
        if p.type == "int":
            out[p.name] = int(float(raw))
        elif p.type == "float":
            out[p.name] = float(raw)
        elif p.type == "bool":
            out[p.name] = str(raw).lower() in ("1", "true", "yes", "on")
        else:
            out[p.name] = str(raw)
    return out


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class Recipe:
    key = ""
    name = ""
    category = ""
    description = ""
    est = ""                        # human note on run time
    track = "classical"             # which UI page this belongs to
    engine = "GROMACS 2026.2"
    mode = "live"                   # live | model | unavailable
    needs_gpu = True                # GROMACS mdrun uses the GPU
    params: list = []

    # filenames produced by the final MD stage, consumed by the viewer/analysis
    outputs = {"tpr": "md.tpr", "xtc": "md.xtc", "gro": "md.gro", "edr": "md.edr"}
    # trjconv options for building the viewer trajectory
    viewer = {"pbc": "whole", "center": None, "select": "System"}
    energy_terms = ["Potential", "Kinetic-En.", "Total-Energy", "Temperature", "Pressure"]

    def coerce(self, params):
        return _coerce(params, self.params)

    def prepare(self, run_dir: Path, p: dict) -> list:
        raise NotImplementedError

    def analyses(self, run_dir: Path, p: dict) -> list:
        return []

    def derived(self, run_dir: Path, p: dict, manifest: dict) -> list:
        """Extra analyses computed in Python from the manifest (e.g. area/lipid)."""
        return []

    def viewer_opts(self, p: dict) -> dict:
        """trjconv options for the viewer; recipes may specialise per-parameter."""
        return self.viewer

    def meta(self):
        return {
            "key": self.key,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "est": self.est,
            "track": self.track,
            "engine": self.engine,
            "mode": self.mode,
            "needs_gpu": self.needs_gpu,
            "params": [p.as_dict() for p in self.params],
        }


# --------------------------------------------------------------------------- #
# 1. Lennard-Jones argon — the purest playground, zero external data
# --------------------------------------------------------------------------- #
class LennardJones(Recipe):
    key = "lj_argon"
    name = "Lennard-Jones argon fluid"
    category = "Fundamental"
    description = ("A box of argon atoms interacting through a pure Lennard-Jones "
                   "potential. Push temperature and density around to drive it "
                   "between gas, liquid and solid — no force field files needed.")
    est = "seconds"
    viewer = {"pbc": "atom", "center": None, "select": "System"}
    energy_terms = ["Potential", "Kinetic-En.", "Total-Energy", "Temperature",
                    "Pressure", "Volume"]

    params = [
        Param("n_atoms", "Number of atoms", "int", 500, 64, 4000, 1,
              help="More atoms = smoother statistics, slower run."),
        Param("density", "Density (atoms / nm³)", "float", 21.0, 0.5, 40.0, 0.5,
              help="~21 is liquid argon; ~2 behaves like a gas; >35 crowds toward solid."),
        Param("temperature", "Temperature (K)", "float", 120.0, 20.0, 400.0, 5.0,
              help="Argon boils ~87 K, critical point ~150 K."),
        Param("nsteps", "MD steps", "int", 50000, 1000, 500000, 1000),
        Param("ensemble", "Ensemble", "choice", "NVT", options=["NVT", "NPT"]),
    ]

    def prepare(self, run_dir, p):
        n = p["n_atoms"]
        box = (n / p["density"]) ** (1.0 / 3.0)
        rc = min(1.0, 0.49 * box)          # obey the minimum-image convention
        lattice_gro(run_dir / "conf.gro", n, box)

        (run_dir / "topol.top").write_text(
            "; Lennard-Jones argon — fully self-contained topology\n"
            "[ defaults ]\n"
            "; nbfunc  comb-rule  gen-pairs\n"
            "  1        2          no\n\n"
            "[ atomtypes ]\n"
            "; name  at.num  mass     charge  ptype  sigma(nm)  epsilon(kJ/mol)\n"
            "  Ar    18      39.948   0.000   A      0.34050    0.99600\n\n"
            "[ moleculetype ]\n"
            "; name  nrexcl\n"
            "  AR    1\n\n"
            "[ atoms ]\n"
            "; nr type resnr residue atom cgnr charge  mass\n"
            "  1  Ar   1     AR      Ar   1    0.000  39.948\n\n"
            "[ system ]\n Argon LJ fluid\n\n"
            "[ molecules ]\n"
            f" AR  {n}\n"
        )
        (run_dir / "em.mdp").write_text(
            minim_mdp(rvdw=rc, rcoulomb=rc, coulombtype="Cut-off"))
        (run_dir / "md.mdp").write_text(md_mdp(
            nsteps=p["nsteps"], dt=0.005, temperature=p["temperature"],
            ensemble=p["ensemble"], coulombtype="Cut-off", rvdw=rc, rcoulomb=rc,
            tc_grps="System", constraints="none", nstxout=max(200, p["nsteps"] // 60)))

        return [
            Step("grompp (minimise)", ["grompp", "-f", "em.mdp", "-c", "conf.gro",
                 "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "2"]),
            Step("energy minimisation", ["mdrun", "-deffnm", "em", "-v", "-ntmpi", "1"]),
            Step("grompp (MD)", ["grompp", "-f", "md.mdp", "-c", "em.gro",
                 "-p", "topol.top", "-o", "md.tpr", "-maxwarn", "2"]),
            Step("MD production", ["mdrun", "-deffnm", "md", "-v", "-ntmpi", "1"]),
        ]

    def analyses(self, run_dir, p):
        return [
            Analysis("rdf", "Radial distribution g(r)  Ar–Ar",
                     ["rdf", "-f", "md.xtc", "-s", "md.tpr", "-o", "rdf.xvg",
                      "-ref", "name Ar", "-sel", "name Ar", "-bin", "0.005"],
                     "rdf.xvg", kind="rdf",
                     help="Peaks = coordination shells. Sharp = solid, broad = liquid, flat = gas."),
            Analysis("msd", "Mean-square displacement (diffusion)",
                     ["msd", "-f", "md.xtc", "-s", "md.tpr", "-o", "msd.xvg",
                      "-sel", "name Ar"],
                     "msd.xvg", kind="msd",
                     help="Slope ∝ diffusion coefficient D."),
        ]


# --------------------------------------------------------------------------- #
# 2. SPC/E water box
# --------------------------------------------------------------------------- #
class WaterBox(Recipe):
    key = "water_box"
    name = "Water box (SPC/E)"
    category = "Solvent"
    description = ("A cubic box of water. The workhorse solvent for biomolecular "
                   "MD — watch it equilibrate to the right density and measure the "
                   "O–O radial distribution function.")
    est = "~1–3 min"
    viewer = {"pbc": "whole", "center": None, "select": "System"}
    energy_terms = ["Potential", "Kinetic-En.", "Total-Energy", "Temperature",
                    "Pressure", "Density", "Volume"]

    params = [
        Param("box_nm", "Box size (nm)", "float", 2.5, 1.8, 5.0, 0.1,
              help="Cubic edge length; bigger = more molecules, slower."),
        Param("water_model", "Water model", "choice", "spce", options=["spce", "tip3p"]),
        Param("temperature", "Temperature (K)", "float", 300.0, 250.0, 373.0, 5.0),
        Param("ensemble", "Ensemble", "choice", "NPT", options=["NPT", "NVT"],
              help="NPT lets the box relax to equilibrium density."),
        Param("nsteps", "MD steps", "int", 25000, 1000, 500000, 1000,
              help="dt = 2 fs, so 25000 steps = 50 ps."),
    ]

    def _topol(self, model):
        return (f'#include "{FF}/forcefield.itp"\n'
                f'#include "{FF}/{model}.itp"\n'
                f'#include "{FF}/ions.itp"\n\n'
                "[ system ]\n Water box\n\n"
                "[ molecules ]\n")

    def prepare(self, run_dir, p):
        box = p["box_nm"]
        rc = min(1.0, 0.49 * box)
        (run_dir / "topol.top").write_text(self._topol(p["water_model"]))
        (run_dir / "em.mdp").write_text(minim_mdp(rvdw=rc, rcoulomb=rc))
        (run_dir / "md.mdp").write_text(md_mdp(
            nsteps=p["nsteps"], dt=0.002, temperature=p["temperature"],
            ensemble=p["ensemble"], rvdw=rc, rcoulomb=rc, tc_grps="System",
            constraints="h-bonds", nstxout=max(200, p["nsteps"] // 50)))

        return [
            Step("solvate box", ["solvate", "-cs", "spc216.gro",
                 "-box", box, box, box, "-o", "conf.gro", "-p", "topol.top"]),
            Step("grompp (minimise)", ["grompp", "-f", "em.mdp", "-c", "conf.gro",
                 "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "2"]),
            Step("energy minimisation", ["mdrun", "-deffnm", "em", "-v", "-ntmpi", "1"]),
            Step("grompp (MD)", ["grompp", "-f", "md.mdp", "-c", "em.gro",
                 "-p", "topol.top", "-o", "md.tpr", "-maxwarn", "2"]),
            Step("MD production", ["mdrun", "-deffnm", "md", "-v", "-ntmpi", "1"]),
        ]

    def analyses(self, run_dir, p):
        return [
            Analysis("rdf", "Radial distribution g(r)  O–O",
                     ["rdf", "-f", "md.xtc", "-s", "md.tpr", "-o", "rdf.xvg",
                      "-ref", "name OW", "-sel", "name OW", "-bin", "0.002"],
                     "rdf.xvg", kind="rdf",
                     help="First peak ~0.28 nm is the water hydrogen-bond shell."),
            Analysis("msd", "Water self-diffusion (MSD)",
                     ["msd", "-f", "md.xtc", "-s", "md.tpr", "-o", "msd.xvg",
                      "-sel", "name OW"],
                     "msd.xvg", kind="msd"),
        ]


# --------------------------------------------------------------------------- #
# 3. NaCl in water
# --------------------------------------------------------------------------- #
class SaltWater(WaterBox):
    key = "nacl_water"
    name = "NaCl in water"
    category = "Solvent"
    description = ("Salt water: solvate a box, then replace waters with Na⁺ and Cl⁻ "
                   "ions at a chosen concentration and watch the solvation shells form.")
    est = "~1–3 min"

    params = [
        Param("box_nm", "Box size (nm)", "float", 3.0, 2.0, 5.0, 0.1),
        Param("conc", "Salt concentration (mol/L)", "float", 0.5, 0.05, 3.0, 0.05),
        Param("temperature", "Temperature (K)", "float", 300.0, 250.0, 373.0, 5.0),
        Param("ensemble", "Ensemble", "choice", "NPT", options=["NPT", "NVT"]),
        Param("nsteps", "MD steps", "int", 25000, 1000, 500000, 1000),
    ]
    energy_terms = ["Potential", "Kinetic-En.", "Total-Energy", "Temperature",
                    "Pressure", "Density", "Volume"]

    def prepare(self, run_dir, p):
        box = p["box_nm"]
        rc = min(1.0, 0.49 * box)
        (run_dir / "topol.top").write_text(self._topol("spce"))
        (run_dir / "ions.mdp").write_text(minim_mdp(rvdw=rc, rcoulomb=rc))
        (run_dir / "em.mdp").write_text(minim_mdp(rvdw=rc, rcoulomb=rc))
        (run_dir / "md.mdp").write_text(md_mdp(
            nsteps=p["nsteps"], dt=0.002, temperature=p["temperature"],
            ensemble=p["ensemble"], rvdw=rc, rcoulomb=rc, tc_grps="System",
            constraints="h-bonds", nstxout=max(200, p["nsteps"] // 50)))

        return [
            Step("solvate box", ["solvate", "-cs", "spc216.gro",
                 "-box", box, box, box, "-o", "conf.gro", "-p", "topol.top"]),
            Step("grompp (ions)", ["grompp", "-f", "ions.mdp", "-c", "conf.gro",
                 "-p", "topol.top", "-o", "ions.tpr", "-maxwarn", "2"]),
            Step("add Na⁺/Cl⁻ ions", ["genion", "-s", "ions.tpr", "-o", "conf.gro",
                 "-p", "topol.top", "-pname", "NA", "-nname", "CL",
                 "-conc", p["conc"], "-neutral"], stdin="SOL\n"),
            Step("grompp (minimise)", ["grompp", "-f", "em.mdp", "-c", "conf.gro",
                 "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "2"]),
            Step("energy minimisation", ["mdrun", "-deffnm", "em", "-v", "-ntmpi", "1"]),
            Step("grompp (MD)", ["grompp", "-f", "md.mdp", "-c", "em.gro",
                 "-p", "topol.top", "-o", "md.tpr", "-maxwarn", "2"]),
            Step("MD production", ["mdrun", "-deffnm", "md", "-v", "-ntmpi", "1"]),
        ]

    def analyses(self, run_dir, p):
        return [
            Analysis("rdf_na_ow", "g(r)  Na⁺ – water O",
                     ["rdf", "-f", "md.xtc", "-s", "md.tpr", "-o", "rdf.xvg",
                      "-ref", "name NA", "-sel", "name OW", "-bin", "0.002"],
                     "rdf.xvg", kind="rdf",
                     help="First peak = the Na⁺ first hydration shell (~0.24 nm)."),
            Analysis("rdf_na_cl", "g(r)  Na⁺ – Cl⁻",
                     ["rdf", "-f", "md.xtc", "-s", "md.tpr", "-o", "rdf_nacl.xvg",
                      "-ref", "name NA", "-sel", "name CL", "-bin", "0.005"],
                     "rdf_nacl.xvg", kind="rdf"),
        ]


# --------------------------------------------------------------------------- #
# 4. Protein in water
# --------------------------------------------------------------------------- #
class Protein(Recipe):
    key = "protein"
    name = "Protein in water"
    category = "Biomolecular"
    description = ("The real thing: fetch a PDB, build a topology with pdb2gmx, "
                   "solvate, add ions, minimise and run MD. Defaults to Trp-cage "
                   "(1L2Y, 20 residues) — try 1AKI for lysozyme.")
    est = "~2–8 min (downloads a PDB)"
    viewer = {"pbc": "mol", "center": "Protein", "select": "Protein"}
    energy_terms = ["Potential", "Kinetic-En.", "Total-Energy", "Temperature",
                    "Pressure", "Density"]

    params = [
        Param("pdb_id", "PDB ID", "choice", "1L2Y",
              options=["1L2Y", "1AKI", "1UBQ", "2MLT"],
              help="1L2Y Trp-cage · 1AKI lysozyme · 1UBQ ubiquitin · 2MLT melittin."),
        Param("water_model", "Water model", "choice", "tip3p", options=["tip3p", "spce"]),
        Param("pad_nm", "Box padding (nm)", "float", 1.0, 0.6, 2.0, 0.1,
              help="Minimum distance from protein to box edge."),
        Param("conc", "Salt concentration (mol/L)", "float", 0.15, 0.0, 1.0, 0.05),
        Param("temperature", "Temperature (K)", "float", 300.0, 250.0, 400.0, 5.0),
        Param("nsteps", "MD steps", "int", 25000, 1000, 500000, 1000),
    ]

    def prepare(self, run_dir, p):
        pdb_id = p["pdb_id"].upper()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                (run_dir / "protein.pdb").write_bytes(r.read())
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Could not download {pdb_id} from RCSB ({e}). "
                "Check internet access or place {pdb_id}.pdb in the run folder.")

        rc = 1.0
        (run_dir / "ions.mdp").write_text(minim_mdp(rvdw=rc, rcoulomb=rc))
        (run_dir / "em.mdp").write_text(minim_mdp(rvdw=rc, rcoulomb=rc))
        (run_dir / "md.mdp").write_text(md_mdp(
            nsteps=p["nsteps"], dt=0.002, temperature=p["temperature"],
            ensemble="NPT", rvdw=rc, rcoulomb=rc,
            tc_grps="Protein Non-Protein", constraints="h-bonds",
            nstxout=max(200, p["nsteps"] // 50)))

        steps = [
            Step("pdb2gmx (build topology)",
                 ["pdb2gmx", "-f", "protein.pdb", "-o", "processed.gro",
                  "-p", "topol.top", "-ff", "amber99sb-ildn",
                  "-water", p["water_model"], "-ignh"]),
            Step("editconf (define box)",
                 ["editconf", "-f", "processed.gro", "-o", "box.gro",
                  "-c", "-d", p["pad_nm"], "-bt", "cubic"]),
            Step("solvate", ["solvate", "-cp", "box.gro", "-cs", "spc216.gro",
                 "-o", "solv.gro", "-p", "topol.top"]),
            Step("grompp (ions)", ["grompp", "-f", "ions.mdp", "-c", "solv.gro",
                 "-p", "topol.top", "-o", "ions.tpr", "-maxwarn", "2"]),
        ]
        if p["conc"] > 0:
            steps.append(Step("add ions", ["genion", "-s", "ions.tpr", "-o", "conf.gro",
                 "-p", "topol.top", "-pname", "NA", "-nname", "CL",
                 "-conc", p["conc"], "-neutral"], stdin="SOL\n"))
        else:
            steps.append(Step("neutralise", ["genion", "-s", "ions.tpr", "-o", "conf.gro",
                 "-p", "topol.top", "-pname", "NA", "-nname", "CL",
                 "-neutral"], stdin="SOL\n"))
        steps += [
            Step("grompp (minimise)", ["grompp", "-f", "em.mdp", "-c", "conf.gro",
                 "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "2"]),
            Step("energy minimisation", ["mdrun", "-deffnm", "em", "-v", "-ntmpi", "1"]),
            Step("grompp (MD)", ["grompp", "-f", "md.mdp", "-c", "em.gro",
                 "-p", "topol.top", "-o", "md.tpr", "-maxwarn", "2"]),
            Step("MD production", ["mdrun", "-deffnm", "md", "-v", "-ntmpi", "1"]),
        ]
        return steps

    def analyses(self, run_dir, p):
        return [
            Analysis("rmsd", "Backbone RMSD vs start",
                     ["rms", "-s", "md.tpr", "-f", "md.xtc", "-o", "rmsd.xvg", "-tu", "ns"],
                     "rmsd.xvg", stdin="Backbone\nBackbone\n", kind="timeseries",
                     help="How far the structure drifts from the crystal/NMR pose."),
            Analysis("gyrate", "Radius of gyration (compactness)",
                     ["gyrate", "-s", "md.tpr", "-f", "md.xtc", "-o", "gyrate.xvg"],
                     "gyrate.xvg", stdin="Protein\n", kind="timeseries"),
            Analysis("rmsf", "Per-residue flexibility (RMSF)",
                     ["rmsf", "-s", "md.tpr", "-f", "md.xtc", "-o", "rmsf.xvg", "-res"],
                     "rmsf.xvg", stdin="Backbone\n", kind="timeseries",
                     help="Which residues wiggle most."),
        ]


# --------------------------------------------------------------------------- #
# 5. Coarse-grained lipid bilayer (Martini 3) — the bridge toward cell scale
# --------------------------------------------------------------------------- #
class Martini(Recipe):
    key = "martini_bilayer"
    name = "Lipid bilayer (Martini CG)"
    category = "Coarse-grained"
    track = "cg"
    description = ("A coarse-grained phospholipid membrane built with insane + "
                   "Martini 3 (~4 atoms per bead). Coarse-graining reaches "
                   "membrane length/time scales for ~1000× less compute — the "
                   "first real step from molecules toward cell-scale structures.")
    est = "~1–3 min"
    energy_terms = ["Potential", "Temperature", "Pressure", "Box-X", "Box-Y"]

    params = [
        Param("lipid", "Lipid type", "choice", "POPC",
              options=["POPC", "DPPC", "DOPC", "POPE", "DLPC"],
              help="POPC/DOPC are fluid; DPPC can gel; POPE is cone-shaped."),
        Param("box_xy", "Membrane patch X=Y (nm)", "float", 8.0, 5.0, 15.0, 0.5,
              help="Bigger patch = more lipids, slower."),
        Param("box_z", "Box height incl. water (nm)", "float", 10.0, 7.0, 16.0, 0.5),
        Param("temperature", "Temperature (K)", "float", 310.0, 270.0, 340.0, 5.0,
              help="310 K ≈ body temperature."),
        Param("nsteps", "MD steps", "int", 20000, 2000, 500000, 1000,
              help="dt = 20 fs (Martini's big step), so 20000 steps ≈ 0.4 µs effective."),
    ]

    def viewer_opts(self, p):
        # show only the lipids (drop water beads) for a clean membrane view
        return {"pbc": "mol", "center": None, "select": p["lipid"]}

    def prepare(self, run_dir, p):
        lipid, x, z = p["lipid"], p["box_xy"], p["box_z"]
        (run_dir / "em.mdp").write_text(martini_em_mdp())
        (run_dir / "md.mdp").write_text(martini_md_mdp(
            nsteps=p["nsteps"], dt=0.02, temperature=p["temperature"],
            nstxout=max(200, p["nsteps"] // 60)))

        def build(run_dir, log_path):
            with open(log_path, "a") as lg:
                lg.write(f"\n$ insane -l {lipid} -x {x} -y {x} -z {z} -sol W\n")
            r = subprocess.run(
                [str(INSANE), "-l", lipid, "-x", str(x), "-y", str(x), "-z", str(z),
                 "-sol", "W", "-o", "bilayer.gro", "-p", "insane.top"],
                cwd=run_dir, capture_output=True, text=True)
            with open(log_path, "a") as lg:
                lg.write((r.stdout or "") + (r.stderr or ""))
            if r.returncode != 0 or not (run_dir / "bilayer.gro").exists():
                raise RuntimeError("insane failed to build the bilayer (see log)")
            # collect the [ molecules ] block insane produced
            mols, inblock = [], False
            for line in (run_dir / "insane.top").read_text().splitlines():
                if line.strip().startswith("[ molecules ]"):
                    inblock = True; continue
                if inblock:
                    s = line.split()
                    if len(s) >= 2 and not s[0].startswith(";"):
                        mols.append((s[0], int(s[1])))
            incl = "\n".join(
                f'#include "{MART_DIR}/{n}"' for n in
                ["martini_v3.0.0.itp", "martini_v3.0.0_phospholipids_v1.itp",
                 "martini_v3.0.0_solvents_v1.itp", "martini_v3.0.0_ions_v1.itp"])
            molblock = "\n".join(f"{n:<6} {c}" for n, c in mols)
            (run_dir / "topol.top").write_text(
                f"{incl}\n\n[ system ]\nMartini {lipid} bilayer\n\n[ molecules ]\n{molblock}\n")
            n_lip = sum(c for n, c in mols if n == lipid)
            (run_dir / "mparams.json").write_text(json.dumps({"lipid": lipid, "n_lipids": n_lip}))

        return [
            Step("build bilayer (insane)", [], func=build),
            Step("grompp (minimise)", ["grompp", "-f", "em.mdp", "-c", "bilayer.gro",
                 "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "10"]),
            Step("energy minimisation", ["mdrun", "-deffnm", "em", "-v", "-ntmpi", "1"]),
            Step("grompp (MD)", ["grompp", "-f", "md.mdp", "-c", "em.gro",
                 "-p", "topol.top", "-o", "md.tpr", "-maxwarn", "10"]),
            Step("MD production", ["mdrun", "-deffnm", "md", "-v", "-ntmpi", "1"]),
        ]

    def analyses(self, run_dir, p):
        frame_dt = max(200, p["nsteps"] // 60) * 0.02   # ps between saved frames
        return [
            Analysis("msd", "Lipid lateral diffusion (MSD of PO4 beads)",
                     ["msd", "-f", "md.xtc", "-s", "md.tpr", "-o", "msd.xvg",
                      "-sel", "name PO4", "-lateral", "z",
                      "-trestart", f"{frame_dt:g}"],
                     "msd.xvg", kind="msd",
                     help="In-plane MSD; slope ∝ how fast lipids slide within the membrane."),
        ]

    def derived(self, run_dir, p, manifest):
        e = manifest.get("energy")
        if not e:
            return []
        try:
            n = json.loads((run_dir / "mparams.json").read_text())["n_lipids"]
        except Exception:  # noqa: BLE001
            return []
        find = lambda nm: next((s for l, s in zip(e["legends"], e["series"])
                                if l.replace(" ", "") == nm), None)
        bx, by = find("Box-X"), find("Box-Y")
        if not bx or not by or n < 2:
            return []
        per_leaflet = n / 2.0
        m = min(len(bx), len(by), len(e["x"]))
        apl = [bx[i] * by[i] / per_leaflet for i in range(m)]
        return [{
            "name": "apl", "label": "Area per lipid", "kind": "timeseries",
            "help": "Membrane area ÷ lipids per leaflet. Fluid POPC ≈ 0.64 nm².",
            "data": {"title": "", "xaxis": e.get("xaxis") or "Time (ps)",
                     "yaxis": "nm²", "legends": [], "x": e["x"][:m], "series": [apl]},
        }]


REGISTRY = {r.key: r for r in [LennardJones(), WaterBox(), SaltWater(), Protein(), Martini()]}


def list_recipes():
    return [r.meta() for r in REGISTRY.values()]


def get_recipe(key) -> Recipe:
    if key not in REGISTRY:
        raise KeyError(f"unknown recipe '{key}'")
    return REGISTRY[key]
