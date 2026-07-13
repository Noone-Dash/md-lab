"""ResolvedPlan -> input files + the ordered gmx Steps that execute it.

Reuses the existing Step/engine/scheduler machinery, so a Plan run produces the
same run.json manifest the viewer already understands.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from ..recipes import Step, Analysis
from .mdp_emit import emit_mdp
from .resolve import resolve

FF_WATER_ITP = {"spce": "spce", "spc": "spc", "tip3p": "tip3p",
                "tip4p": "tip4p", "tip4pew": "tip4pew", "tip5p": "tip5p"}

ANALYSES = {
    "rmsd": lambda o: Analysis("rmsd", "Backbone RMSD vs start",
        # -tu ps, NOT ns: every other series in the manifest (all energy terms) is in ps,
        # and metrics.uncertainty() infers dt from the x-axis. With -tu ns the RMSD dt came
        # out 1000x too small and tau_int_ps was reported 1000x too small with it.
        ["rms", "-s", o["tpr"], "-f", o["xtc"], "-o", "rmsd.xvg", "-tu", "ps"],
        "rmsd.xvg", stdin="Backbone\nBackbone\n",
        help="How far the structure drifts from the starting pose."),
    "gyrate": lambda o: Analysis("gyrate", "Radius of gyration (compactness)",
        ["gyrate", "-s", o["tpr"], "-f", o["xtc"], "-o", "gyrate.xvg"],
        "gyrate.xvg", stdin="Protein\n"),
    "rmsf": lambda o: Analysis("rmsf", "Per-residue flexibility (RMSF)",
        ["rmsf", "-s", o["tpr"], "-f", o["xtc"], "-o", "rmsf.xvg", "-res"],
        "rmsf.xvg", stdin="Backbone\n"),
    "rdf_ow": lambda o: Analysis("rdf", "Radial distribution g(r) O–O",
        ["rdf", "-f", o["xtc"], "-s", o["tpr"], "-o", "rdf.xvg",
         "-ref", "name OW", "-sel", "name OW", "-bin", "0.002"],
        "rdf.xvg", kind="rdf", help="First peak ≈0.28 nm is the water H-bond shell."),
    "msd_ow": lambda o: Analysis("msd", "Water self-diffusion (MSD)",
        ["msd", "-f", o["xtc"], "-s", o["tpr"], "-o", "msd.xvg", "-sel", "name OW"],
        "msd.xvg", kind="msd"),
}


class PlanRecipe:
    """Adapter so labkit.engine can post-process a Plan run like a Recipe."""

    def __init__(self, plan, outputs, viewer, energy_terms, analyses_keys):
        self.key = "plan"
        self.name = plan.name
        self.category = {"protein": "Biomolecular", "membrane": "Coarse-grained",
                         "solvent": "Solvent", "fluid": "Fundamental",
                         "qm": "Quantum"}.get(plan.system.kind, "Solvent")
        self.track = "plan"
        self.engine = "GROMACS 2026.2"
        self.mode = "live"
        self.needs_gpu = True
        self.outputs = outputs
        self.viewer = viewer
        self.energy_terms = energy_terms
        self._analyses = analyses_keys

    def viewer_opts(self, p):
        return self.viewer

    def analyses(self, run_dir, p):
        return [ANALYSES[a](self.outputs) for a in self._analyses if a in ANALYSES]

    def derived(self, run_dir, p, manifest):
        return []


def build(plan, run_dir: Path):
    """Write inputs, return (steps, PlanRecipe)."""
    rp = resolve(plan)
    s = plan.system
    run_dir.mkdir(parents=True, exist_ok=True)
    steps: list[Step] = []
    ff = s.forcefield
    water = s.water_model

    # ---------- system construction ---------------------------------------- #
    if s.kind == "protein":
        pdb_id = s.pdb_id.upper()
        url = (f"https://files.rcsb.org/download/{pdb_id}.pdb"
               if s.structure_source == "rcsb"
               else f"https://alphafold.ebi.ac.uk/files/AF-{pdb_id}-F1-model_v4.pdb")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                (run_dir / "input.pdb").write_bytes(r.read())
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Could not fetch {pdb_id} from {url} ({e}).\n"
                f"Compute nodes are frequently offline. Either fetch the PDB on a login "
                f"node and use system.structure_source='file' with a local input.pdb, "
                f"or run where the network is reachable.") from e

        p2g = ["pdb2gmx", "-f", "input.pdb", "-o", "processed.gro", "-p", "topol.top",
               "-ff", ff, "-water", water]
        if s.ignore_hydrogens:
            p2g.append("-ignh")
        steps.append(Step("pdb2gmx (build topology)", p2g))
        steps.append(Step("editconf (define box)",
                          ["editconf", "-f", "processed.gro", "-o", "box.gro", "-c",
                           "-d", s.box_padding_nm, "-bt", s.box_shape]))
        steps.append(Step("solvate", ["solvate", "-cp", "box.gro", "-cs", "spc216.gro",
                                      "-o", "solv.gro", "-p", "topol.top"]))
        start = "solv.gro"
    elif s.kind in ("solvent", "fluid"):
        itp = FF_WATER_ITP.get(water, "spce")
        (run_dir / "topol.top").write_text(
            f'#include "{ff}.ff/forcefield.itp"\n'
            f'#include "{ff}.ff/{itp}.itp"\n'
            f'#include "{ff}.ff/ions.itp"\n\n'
            f"[ system ]\n{plan.name}\n\n[ molecules ]\n")
        b = s.box_size_nm
        steps.append(Step("solvate box",
                          ["solvate", "-cs", "spc216.gro", "-box", b, b, b,
                           "-o", "solv.gro", "-p", "topol.top"]))
        start = "solv.gro"
    else:
        raise ValueError(f"plan build does not yet support system.kind={s.kind!r} "
                         f"(use the existing recipes for membrane/qm)")

    # ---------- ions -------------------------------------------------------- #
    needs_ions = s.neutralize or (s.salt_conc_M or 0) > 0
    if needs_ions:
        # a throwaway mdp just to make a .tpr for genion
        (run_dir / "ions.mdp").write_text(emit_mdp(
            {"name": "ions", "type": "minimize",
             "mdp": {**rp["stages"][0]["mdp"], "nsteps": 1}}))
        steps.append(Step("grompp (ions)",
                          ["grompp", "-f", "ions.mdp", "-c", start, "-p", "topol.top",
                           "-o", "ions.tpr", "-maxwarn", "2"]))
        gi = ["genion", "-s", "ions.tpr", "-o", "ionised.gro", "-p", "topol.top",
              "-pname", "NA", "-nname", "CL"]
        if (s.salt_conc_M or 0) > 0:
            gi += ["-conc", s.salt_conc_M]
        if s.neutralize:
            gi.append("-neutral")
        steps.append(Step("add ions", gi, stdin="SOL\n"))
        start = "ionised.gro"

    # ---------- one grompp+mdrun per stage --------------------------------- #
    prev_gro = start
    last = None
    for st in rp["stages"]:
        nm = st["name"]
        (run_dir / f"{nm}.mdp").write_text(emit_mdp(st))
        grompp = ["grompp", "-f", f"{nm}.mdp", "-c", prev_gro, "-p", "topol.top",
                  "-o", f"{nm}.tpr", "-maxwarn", "2"]
        if st["mdp"].get("define") == "-DPOSRES":
            grompp += ["-r", prev_gro]           # restraint reference coordinates
        steps.append(Step(f"grompp ({nm})", grompp))
        steps.append(Step(f"mdrun ({nm})",
                          ["mdrun", "-deffnm", nm, "-v"]))
        prev_gro = f"{nm}.gro"
        last = nm

    outputs = {"tpr": f"{last}.tpr", "xtc": f"{last}.xtc",
               "gro": f"{last}.gro", "edr": f"{last}.edr"}

    if s.kind == "protein":
        viewer = {"pbc": "mol", "center": "Protein", "select": "Protein"}
        terms = ["Potential", "Kinetic-En.", "Total-Energy", "Temperature", "Pressure", "Density"]
    else:
        viewer = {"pbc": "whole", "center": None, "select": "System"}
        terms = ["Potential", "Kinetic-En.", "Total-Energy", "Temperature",
                 "Pressure", "Density", "Volume"]

    return steps, PlanRecipe(plan, outputs, viewer, terms, plan.analyses)
