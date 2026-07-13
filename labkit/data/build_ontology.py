#!/usr/bin/env python3
"""Assemble labkit/data/ontology.json.

Sources:
  * 130 params produced by the design workflow (4 areas)
  * the system_build_and_forcefield area, WRITTEN BY HAND because that
    workflow agent died mid-response (API error). Nothing here is invented:
    every force field / water model listed is one actually installed under
    $GMX_ROOT/share/gromacs/top, verified at build time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
def _gmx_top():
    """Locate GROMACS' share/top. A SILENT fallback here is dangerous: it would
    generate an ontology advertising force fields this machine does not have."""
    import sys as _s
    _s.path.insert(0, str(HERE.parent.parent))
    from labkit import config as _cfg
    b = Path(_cfg.gmx_binary())               # .../bin/gmx
    return b.parent.parent / "share/gromacs/top"


GMX_TOP = _gmx_top()


def installed_forcefields():
    if not GMX_TOP.exists():
        raise SystemExit(
            f"Cannot read GROMACS force fields at {GMX_TOP}.\n"
            f"The ontology must reflect what is ACTUALLY installed — refusing to "
            f"guess.\nSet GMX_ROOT or `module load gromacs`, then re-run.")
    return sorted(p.name[:-3] for p in GMX_TOP.glob("*.ff"))


def installed_water_models():
    ff = GMX_TOP / "amber99sb-ildn.ff"
    if not ff.exists():
        raise SystemExit(f"No amber99sb-ildn.ff under {GMX_TOP} — cannot enumerate "
                         f"water models. Refusing to guess.")
    return sorted(p.stem for p in ff.glob("*.itp")
                  if p.stem in {"spce", "spc", "tip3p", "tip4p", "tip5p", "tip4pew"})


def P(key, label, meaning, type, unit, default, *, mdp_key=None, options=None,
      range="", applies_to=(), depends_on=(), cost_impact="", agent_guidance="",
      stage_scope="build"):
    return {
        "key": key, "area": "system_build_and_forcefield", "label": label,
        "meaning": meaning, "mdp_key": mdp_key, "type": type, "unit": unit,
        "default": default, "options": list(options) if options else None,
        "range": range, "applies_to": list(applies_to), "depends_on": list(depends_on),
        "cost_impact": cost_impact, "agent_guidance": agent_guidance,
        "stage_scope": stage_scope,
    }


def system_area():
    ffs = installed_forcefields()
    waters = installed_water_models()
    return [
        P("forcefield", "Force field",
          "The parameter set giving every atom its charge, van der Waals radius and bonded terms. "
          "It is the single biggest determinant of whether the physics is right.",
          "choice", "dimensionless", "amber99sb-ildn", options=ffs,
          applies_to=["protein", "solvent", "membrane"],
          agent_guidance="amber99sb-ildn or amber14sb for proteins; charmm27 if the user asks for CHARMM; "
                         "martini3 ONLY for coarse-grained membranes (it needs a totally different mdp regime). "
                         "Never mix a Martini force field with atomistic cutoffs."),
        P("water_model", "Water model",
          "Which rigid water geometry/charge set is used for the solvent. Sets the density you should expect.",
          "choice", "dimensionless", "tip3p", options=waters,
          applies_to=["solvent", "protein", "membrane"], depends_on=["forcefield"],
          agent_guidance="tip3p pairs with amber/charmm; spce gives a better bulk density (~998). "
                         "Expect ~985-1010 kg/m3 in NPT; if you get 1100+, the box or ion count is wrong."),
        P("structure_source", "Structure source",
          "Where the starting coordinates come from: an experimental PDB entry, an AlphaFold prediction, or a local file.",
          "choice", "dimensionless", "rcsb", options=["rcsb", "alphafold", "file", "none"],
          applies_to=["protein"],
          agent_guidance="Use rcsb with a 4-character PDB id (1AKI). Use alphafold with a UniProt id (P69905) "
                         "when no experimental structure exists."),
        P("pdb_id", "PDB / UniProt id",
          "The identifier fetched from the chosen database.",
          "string", "dimensionless", "1AKI", applies_to=["protein"], depends_on=["structure_source"],
          agent_guidance="1AKI lysozyme, 1UBQ ubiquitin, 1L2Y Trp-cage, 6LU7 SARS-CoV-2 protease."),
        P("box_shape", "Box shape",
          "Periodic cell geometry. A dodecahedron holds the same solute in ~29% less solvent than a cube, "
          "so it is ~29% cheaper for the same minimum-image distance.",
          "choice", "dimensionless", "dodecahedron",
          options=["cubic", "dodecahedron", "octahedron", "triclinic"],
          applies_to=["protein", "solvent"], cost_impact="dodecahedron ~0.71x the atoms of cubic",
          agent_guidance="Prefer dodecahedron for a globular protein — it is strictly cheaper for the same "
                         "physics. Use cubic for a plain solvent box or a membrane."),
        P("box_padding_nm", "Solute-to-edge distance",
          "Minimum distance from any solute atom to the box wall. Must exceed the cutoff, or the protein "
          "sees its own periodic image.",
          "float", "nm", 1.2, range="0.8 - 2.0", applies_to=["protein"],
          depends_on=["rvdw", "rcoulomb"],
          cost_impact="atoms grow ~ (L+2d)^3; going 1.0 -> 1.5 nm can nearly double the system",
          agent_guidance="Must be >= rcoulomb (typically 1.0 nm). 1.2 nm is a safe default. Below 1.0 nm you "
                         "risk self-interaction artefacts."),
        P("box_size_nm", "Explicit box edge",
          "Edge length for a solvent-only box, when there is no solute to pad around.",
          "float", "nm", 3.0, range="1.8 - 12.0", applies_to=["solvent", "fluid"],
          cost_impact="atoms ~ L^3", agent_guidance="Must be > 2 x cutoff (so > 2.0 nm for a 1.0 nm cutoff)."),
        P("salt_conc_M", "Salt concentration",
          "Molar concentration of added NaCl-type salt, on top of any counter-ions needed for neutrality.",
          "float", "mol/L", 0.15, range="0.0 - 2.0", applies_to=["protein", "solvent"],
          agent_guidance="0.15 M is physiological. Ion count = conc * V_box(nm^3) * 0.6022 "
                         "(NOT 6.022e-4 — that error is 1000x and would make every system look unsalted)."),
        P("neutralize", "Neutralise net charge",
          "Add counter-ions until the system's total charge is exactly zero. PME requires this; a charged "
          "box under PME silently adds a neutralising background jelly and distorts energetics.",
          "bool", "dimensionless", True, applies_to=["protein", "solvent"],
          depends_on=["coulombtype"],
          agent_guidance="Always true when coulombtype=PME. The net charge must be MEASURED from pdb2gmx's "
                         "reported qtot, never guessed from sequence."),
        P("ion_positive", "Cation", "Species used for positive ions.", "choice", "dimensionless", "NA",
          options=["NA", "K", "CA", "MG"], applies_to=["protein", "solvent"]),
        P("ion_negative", "Anion", "Species used for negative ions.", "choice", "dimensionless", "CL",
          options=["CL"], applies_to=["protein", "solvent"]),
        P("ignore_hydrogens", "Rebuild hydrogens",
          "Discard hydrogens present in the input file and let pdb2gmx add them per the force field. "
          "Crystal structures rarely have usable hydrogens.",
          "bool", "dimensionless", True, applies_to=["protein"],
          agent_guidance="Keep true for X-ray structures. Set false only if the input already has correct, "
                         "force-field-consistent hydrogens (e.g. a previous GROMACS output)."),
        P("posres_fc_kj", "Position-restraint force constant",
          "Spring constant tethering restrained atoms to their starting positions during equilibration, "
          "so the solvent relaxes around a solute that is not yet allowed to move.",
          "float", "kJ/mol/nm^2", 1000.0, range="0 - 10000", applies_to=["protein", "membrane"],
          mdp_key="define", stage_scope="stage",
          agent_guidance="1000 during NVT/NPT equilibration; 0 (no restraints) in production. Ramping "
                         "1000 -> 500 -> 0 over successive stages is gentler for fragile systems."),
        P("define", "Preprocessor defines",
          "C-preprocessor flags passed to grompp, which switch on #ifdef blocks in the topology. "
          "-DPOSRES activates the position-restraint itp that pdb2gmx wrote.",
          "string", "dimensionless", "", mdp_key="define",
          applies_to=["protein", "membrane", "solvent"], depends_on=["posres_fc_kj"],
          stage_scope="stage",
          agent_guidance="Set automatically to '-DPOSRES' when a stage has posres_fc_kj > 0. "
                         "grompp then also needs -r <reference.gro>."),
        P("posres_selection", "Restrained atom set",
          "Which atoms feel the position restraint.",
          "choice", "dimensionless", "Protein-H",
          options=["Protein-H", "Backbone", "CA", "Protein"], applies_to=["protein"],
          depends_on=["posres_fc_kj"], stage_scope="stage",
          agent_guidance="Protein-H (all heavy atoms) is standard for equilibration. Backbone/CA are looser "
                         "and let side chains relax."),
        P("lipid", "Lipid type", "Which phospholipid builds the coarse-grained bilayer.",
          "choice", "dimensionless", "POPC", options=["POPC", "DPPC", "DOPC", "POPE", "DLPC"],
          applies_to=["membrane"], depends_on=["forcefield"],
          agent_guidance="Only valid with forcefield=martini3. Expect area-per-lipid ~0.64 nm2 for POPC."),
        P("qm_method", "QM method", "Level of quantum theory for the QM region.",
          "choice", "dimensionless", "HF", options=["HF", "B3LYP", "PBE", "MP2"], applies_to=["qm"],
          cost_impact="B3LYP/MP2 are far more expensive than HF; MP2 scales ~N^5",
          agent_guidance="HF/sto-3g for a fast sanity check; B3LYP/6-31g for anything quantitative."),
        P("qm_basis", "Basis set", "Size of the atomic-orbital basis; the main QM accuracy/cost dial.",
          "choice", "dimensionless", "sto-3g", options=["sto-3g", "6-31g", "6-31g*", "cc-pvdz"],
          applies_to=["qm"], cost_impact="cost ~ N_basis^4 (HF); sto-3g is a toy, 6-31g* is a minimum for real numbers",
          agent_guidance="sto-3g overestimates HOMO-LUMO gaps badly. Use 6-31g* if the user cares about energies."),
        P("qm_charge", "QM total charge", "Net charge of the quantum region.",
          "int", "e", 0, range="-4 - 4", applies_to=["qm"]),
        P("qm_spin", "QM spin multiplicity", "2S+1 for the quantum region; 1 = closed shell.",
          "int", "dimensionless", 1, range="1 - 5", applies_to=["qm"],
          agent_guidance="Must be 1 for RHF/RKS. A radical or O2 needs >1 and an unrestricted method."),
    ]


def main():
    workflow = json.loads((HERE / "ontology_source.json").read_text())
    for p in workflow:
        p.setdefault("stage_scope", "stage")
    onto = workflow + system_area()
    out = HERE / "ontology.json"
    out.write_text(json.dumps({
        "schema_version": "ontology/1",
        "note": "Machine-readable MD parameter ontology. Every entry an agent may set.",
        "parameters": onto,
    }, indent=1))
    areas = {}
    for p in onto:
        areas[p["area"]] = areas.get(p["area"], 0) + 1
    print(f"wrote {out}  ({len(onto)} parameters)")
    for a, n in sorted(areas.items()):
        print(f"   {a:<34} {n}")
    print(f"   force fields detected on disk: {', '.join(installed_forcefields())}")
    print(f"   water models detected on disk: {', '.join(installed_water_models())}")


if __name__ == "__main__":
    main()
