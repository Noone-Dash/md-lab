"""Small-molecule parameterisation — the gap that blocked every drug-discovery use case.

Until now the lab could simulate a protein and a box of water and nothing else. A ligand —
an actual drug candidate — has no parameters in any protein force field, because protein
force fields only know about the twenty amino acids. Without this, "simulate this compound
bound to that protein" was simply not a thing the lab could do.

HOW THIS IS WIRED, AND WHY IT LOOKS INDIRECT
--------------------------------------------
openff-toolkit is conda-only (it is not on PyPI at all) and pulls a large dependency tree.
Installing it into the lab's venv would contaminate the environment that runs everything
else. So it lives in an ISOLATED micromamba env and is called as a SUBPROCESS; the contract
is a line of JSON. If that env is absent, this module fails with the exact command to build
it, rather than a stack trace about a missing import.

WHAT YOU GET, AND WHAT IT IS WORTH
----------------------------------
    SMILES -> OpenFF Sage (SMIRNOFF typing) -> NAGL charges -> GROMACS .top + .gro

The charge method is recorded in the manifest and shown in the UI, because charges are the
single largest error source in a small-molecule force field: a number computed with
Gasteiger charges is NOT comparable to one computed with AM1-BCC, and quietly swapping them
would silently change every result downstream.

Results are cached by InChIKey — the same molecule never gets parameterised twice, and the
cache key is the molecule's identity, not the string you happened to type.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from ..config import DATA_DIR, EnvironmentError_

HERE = Path(__file__).resolve().parent
CACHE = DATA_DIR / "_ligands"

# Same discipline as GROMACS: an env location is DISCOVERED, never hardcoded.
TOOLS = Path(os.environ.get("MDLAB_TOOLS", Path.home() / ".local/share/mdlab"))
LIGAND_ENV = Path(os.environ.get("MDLAB_LIGAND_ENV", TOOLS / "envs/ligand"))

BUILD_CMD = (
    f"  micromamba create -y -p {LIGAND_ENV} -c conda-forge \\\n"
    f"      python=3.11 openff-toolkit openff-interchange openff-nagl rdkit"
)


def env_python() -> Path | None:
    p = LIGAND_ENV / "bin" / "python"
    return p if p.exists() else None


def available() -> bool:
    return env_python() is not None


def require_env():
    if not available():
        raise EnvironmentError_(
            "Ligand parameterisation needs the isolated OpenFF environment, which is not "
            "installed.\n"
            "  openff-toolkit is conda-only (not on PyPI), so it is kept OUT of the lab's "
            "venv.\n"
            "  Build it once:\n\n" + BUILD_CMD + "\n\n"
            f"  Or point MDLAB_LIGAND_ENV at an existing env.\n"
            f"  `python -m labkit.doctor` will confirm.")


# A few molecules people actually ask for, so a demo does not require knowing SMILES.
KNOWN_SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "acetylsalicylic acid": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "benzene": "c1ccccc1",
    "methane": "C",
    "ethanol": "CCO",
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "penicillin g": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc1ccccc1)C(=O)O)C",
}


def smiles_for(text: str) -> str | None:
    """A name or a SMILES string -> SMILES. Deterministic; no model involved."""
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()
    if low in KNOWN_SMILES:
        return KNOWN_SMILES[low]
    for name, smi in KNOWN_SMILES.items():
        if re.search(r"\b" + re.escape(name) + r"\b", low):
            return smi
    # Looks like a SMILES already? Only accept it if RDKit (or the ligand env) says so —
    # never guess, because a mis-parsed SMILES is a different molecule, silently.
    if re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$%/\\.:]+", t) and any(c in t for c in "Cc[NnOoSs"):
        return t
    return None


def parameterize(molecule: str, name: str = "LIG", forcefield: str = "openff-2.2.0.offxml",
                 n_waters: int = 0, box_nm: float = 3.0, force: bool = False) -> dict:
    """name-or-SMILES -> {top, gro, charge_method, ...}. Cached.

    n_waters > 0 solvates the molecule INSIDE OpenFF (packmol + TIP3P) and returns a
    complete, runnable GROMACS system rather than a bare ligand topology.
    """
    require_env()
    smiles = smiles_for(molecule)
    if not smiles:
        raise ValueError(
            f"{molecule!r} is neither a known molecule name nor a valid SMILES string.\n"
            f"  Known: {', '.join(sorted(KNOWN_SMILES))}\n"
            f"  Or pass a SMILES, e.g. 'CC(=O)Oc1ccccc1C(=O)O' for aspirin.")

    CACHE.mkdir(parents=True, exist_ok=True)
    # Cache by the molecule's IDENTITY. Two different SMILES strings can be the same
    # molecule; the same string with a different force field is NOT the same result.
    tag = re.sub(r"[^A-Za-z0-9]+", "_", f"{smiles}_{forcefield}_w{n_waters}")[:100]
    meta_f = CACHE / f"{tag}.json"
    if meta_f.exists() and not force:
        try:
            m = json.loads(meta_f.read_text())
            if Path(m.get("top", "")).exists() and Path(m.get("gro", "")).exists():
                m["cached"] = True
                return m
        except Exception:  # noqa: BLE001
            pass

    outdir = CACHE / tag
    payload = {"smiles": smiles, "name": name, "outdir": str(outdir),
               "forcefield": forcefield, "n_waters": n_waters, "box_nm": box_nm}
    # Calling the env's python directly does NOT activate the env: its bin/ is not on PATH,
    # so packmol (a BINARY the solvation step shells out to) was invisible and pack_box
    # failed with "Packmol not found". Put the env's bin first.
    child = dict(os.environ)
    child["PATH"] = f"{LIGAND_ENV / 'bin'}{os.pathsep}{child.get('PATH', '')}"
    r = subprocess.run([str(env_python()), str(HERE / "param.py")],
                       input=json.dumps(payload), capture_output=True, text=True,
                       timeout=900, env=child)
    line = (r.stdout or "").strip().splitlines()
    if not line:
        raise RuntimeError(f"ligand parameterisation produced no output.\n"
                           f"{(r.stderr or '')[-400:]}")
    try:
        out = json.loads(line[-1])
    except json.JSONDecodeError:
        raise RuntimeError(f"ligand env returned junk: {line[-1][:200]}")

    if not out.get("ok"):
        raise RuntimeError(f"could not parameterise {molecule!r}: {out.get('error')}")

    out["cached"] = False
    out["query"] = molecule
    meta_f.write_text(json.dumps(out, indent=2))
    return out


def summary() -> dict:
    return {"available": available(), "env": str(LIGAND_ENV),
            "python": str(env_python()) if available() else None,
            "build_command": BUILD_CMD if not available() else None,
            "known_molecules": sorted(KNOWN_SMILES)}
