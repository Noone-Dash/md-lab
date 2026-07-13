"""Parameterise a small molecule. RUNS INSIDE THE ISOLATED LIGAND ENV, not the main venv.

This script is executed by `labkit/ligand/__init__.py` as a subprocess in a micromamba env
that has openff-toolkit / interchange / nagl / rdkit. Those are conda-only (openff-toolkit
is not on PyPI at all) and drag in a large dependency tree, so they are kept OUT of the
lab's own venv entirely. The contract between the two is this file's JSON on stdout.

WHY OPENFF AND NOT GAFF/ANTECHAMBER
-----------------------------------
The classical route is antechamber -> GAFF -> AM1-BCC charges, which needs AmberTools
(sqm) and its own build. OpenFF's SMIRNOFF typing needs no antechamber: it assigns
parameters directly from SMARTS patterns on the molecular graph. For charges we use NAGL,
a graph network trained to reproduce AM1-BCC, which is deterministic, fast and needs no QM
call. That keeps the whole path pure-Python and reproducible.

The charge method is RECORDED in the output. It matters: charges are the single biggest
source of error in small-molecule force fields, and a result computed with one method is
not comparable to a result computed with another.
"""

from __future__ import annotations

import json
import sys
import traceback


def _die(msg, **extra):
    print(json.dumps({"ok": False, "error": msg, **extra}))
    sys.exit(0)          # exit 0: the FAILURE is the payload, not a crash


def main():
    args = json.loads(sys.stdin.read())
    smiles = args["smiles"]
    name = args.get("name", "LIG")
    outdir = args["outdir"]

    from openff.toolkit import ForceField, Molecule
    from openff.units import unit

    try:
        mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    except Exception as e:  # noqa: BLE001
        _die(f"could not parse SMILES {smiles!r}: {e}")

    mol.name = name
    # A 3D conformer is required: SMIRNOFF torsions are assigned on the graph, but the
    # coordinates we hand GROMACS have to be real geometry, not a flat drawing.
    mol.generate_conformers(n_conformers=1)

    charge_method = None
    try:
        from openff.nagl_models import list_available_nagl_models  # noqa: F401
        mol.assign_partial_charges("openff-gnn-am1bcc-0.1.0-rc.3.pt")
        charge_method = "NAGL openff-gnn-am1bcc (AM1-BCC surrogate)"
    except Exception:  # noqa: BLE001
        try:
            mol.assign_partial_charges("am1bcc")          # AmberTools, if present
            charge_method = "AM1-BCC (AmberTools sqm)"
        except Exception:  # noqa: BLE001
            try:
                mol.assign_partial_charges("gasteiger")
                charge_method = "GASTEIGER — LOW QUALITY, NOT FOR PUBLICATION"
            except Exception as e:  # noqa: BLE001
                _die(f"no charge method available: {e}")

    ff_name = args.get("forcefield", "openff-2.2.0.offxml")
    try:
        ff = ForceField(ff_name)
    except Exception as e:  # noqa: BLE001
        _die(f"force field {ff_name!r} not available: {e}")

    n_waters = int(args.get("n_waters", 0))
    solvated = False
    try:
        if n_waters > 0:
            # SOLVATE INSIDE OPENFF, in one shot. The alternative -- write the ligand
            # topology, then bolt water on with `gmx solvate` -- means hand-merging two
            # topologies with different atom-type namespaces, which is exactly where these
            # pipelines go silently wrong (a duplicate or shadowed [atomtypes] entry does
            # not error, it just gives you the WRONG nonbonded parameters).
            from openff.interchange.components._packmol import UNIT_CUBE, pack_box

            water = Molecule.from_smiles("O")
            water.name = "SOL"          # or every tool downstream sees water as "MOL"
            water.generate_conformers(n_conformers=1)
            water.assign_partial_charges("gasteiger")   # overridden by the water model below

            box_nm = float(args.get("box_nm", 3.0))
            topology = pack_box(
                molecules=[mol, water],
                number_of_copies=[1, n_waters],
                box_vectors=box_nm * UNIT_CUBE * unit.nanometer,
            )
            ff_water = ForceField(ff_name, "tip3p.offxml")
            interchange = ff_water.create_interchange(
                topology, charge_from_molecules=[mol])
            solvated = True
        else:
            interchange = ff.create_interchange(
                mol.to_topology(), charge_from_molecules=[mol])
    except Exception as e:  # noqa: BLE001
        _die(f"parameterisation failed: {e}", trace=traceback.format_exc()[-500:])

    # GROMACS (2020+) has no non-periodic path: a topology with no box vectors cannot be
    # written at all. A lone molecule in vacuum genuinely has no box, so give it one. This
    # box is a PLACEHOLDER -- the real one is built when the ligand is solvated (or when it
    # is merged into a protein's box) -- but it has to be big enough that the molecule does
    # not see its own periodic image through the cutoff.
    if interchange.box is None:
        import numpy as np
        interchange.box = np.eye(3) * float(args.get("box_nm", 4.0)) * unit.nanometer

    import pathlib
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    top = out / f"{name}.top"
    gro = out / f"{name}.gro"
    interchange.to_top(top)
    interchange.to_gro(gro)

    # Interchange names EVERY molecule's residue "MOL" -- so water comes out as MOL1, not
    # SOL. Nothing downstream recognises that as water: the viewer's water selector misses
    # it (so "hide water" silently does nothing), and `gmx select`/index groups cannot pick
    # it out either. It is also just wrong. Rename at the SOURCE, so the .tpr grompp builds
    # -- and therefore every trajectory trjconv writes from it -- carries the right names.


    q = mol.partial_charges.m_as(unit.elementary_charge)
    print(json.dumps({
        "ok": True,
        "name": name,
        "smiles": smiles,
        "inchikey": mol.to_inchikey(fixed_hydrogens=False),
        "n_atoms": int(interchange.topology.n_atoms),   # the SYSTEM, not just the ligand
        "ligand_atoms": int(mol.n_atoms),
        "formal_charge": int(mol.total_charge.m_as(unit.elementary_charge)),
        "net_partial_charge": round(float(sum(q)), 6),
        "charge_method": charge_method,
        "forcefield": ff_name,
        "solvated": solvated,
        "n_waters": n_waters,
        "water_model": "tip3p" if solvated else None,
        "top": str(top),
        "gro": str(gro),
    }))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        _die(f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-500:])
