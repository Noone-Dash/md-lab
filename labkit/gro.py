"""Helpers for building/reading GROMACS ``.gro`` coordinate files.

Used mainly by the Lennard-Jones recipe, which places atoms on a lattice with no
need for any external structure file.
"""

from __future__ import annotations

import math
from pathlib import Path


def lattice_gro(path, n_atoms: int, box_nm: float, resname="AR", atomname="Ar"):
    """Write *n_atoms* on a simple-cubic lattice inside a cubic box of side *box_nm*.

    Energy minimisation later relaxes the lattice, so a regular grid is fine as a
    starting point and is fully deterministic (no RNG needed).
    """
    path = Path(path)
    per_side = math.ceil(n_atoms ** (1.0 / 3.0))
    spacing = box_nm / per_side
    # keep atoms off the box face so periodic images don't sit on top of each other
    offset = spacing / 2.0

    coords = []
    for i in range(per_side):
        for j in range(per_side):
            for k in range(per_side):
                if len(coords) >= n_atoms:
                    break
                coords.append((
                    offset + i * spacing,
                    offset + j * spacing,
                    offset + k * spacing,
                ))
    coords = coords[:n_atoms]

    lines = [f"Lennard-Jones fluid: {n_atoms} atoms", f"{n_atoms:5d}"]
    for idx, (x, y, z) in enumerate(coords, start=1):
        resid = idx
        lines.append(
            f"{resid % 100000:5d}{resname:<5s}{atomname:>5s}{idx % 100000:5d}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    lines.append(f"{box_nm:10.5f}{box_nm:10.5f}{box_nm:10.5f}")
    path.write_text("\n".join(lines) + "\n")
    return len(coords)


def read_box(gro_path):
    """Return the (x, y, z) box vector lengths (nm) from the last line of a .gro."""
    lines = Path(gro_path).read_text().splitlines()
    parts = lines[-1].split()
    return tuple(float(v) for v in parts[:3])


def count_atoms(gro_path) -> int:
    return int(Path(gro_path).read_text().splitlines()[1].strip())
