"""Preflight: `python -m labkit.doctor`

Tells you exactly what is present, what is missing, and what to do about it —
instead of letting you find out via two pages of stack traces.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

from . import config as C

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def _p(status, name, detail=""):
    print(f"[{status}] {name:<26} {detail}")


def main() -> int:
    print(f"labkit doctor — repo at {C.REPO_ROOT}\n")
    fatal = 0
    warn = 0

    # ---- ligand parameterisation (isolated env; openff is conda-only) -------
    def _ligand_check():
        from . import ligand
        if ligand.available():
            _p(OK, "ligand params (OpenFF)", f"env: {ligand.LIGAND_ENV}")
            return 0
        _p(WARN, "ligand params (OpenFF)",
           "small molecules / drug candidates CANNOT be simulated without this.")
        print(f"         openff-toolkit is conda-only, so it lives in an isolated env:")
        print(ligand.BUILD_CMD)
        return 1

    # ---- python deps -------------------------------------------------------
    required = {"flask": "web UI", "numpy": "everything", "psutil": "scheduler telemetry"}
    optional = {"openmm": "OpenMM track", "pyscf": "QM/QM-MM track",
                "ase": "QM optimiser", "insane": "Martini bilayer builder"}
    for mod, why in required.items():
        if importlib.util.find_spec(mod):
            _p(OK, f"python: {mod}", why)
        else:
            _p(BAD, f"python: {mod}", f"MISSING ({why}) -> pip install -r requirements.txt")
            fatal += 1
    for mod, why in optional.items():
        if importlib.util.find_spec(mod):
            _p(OK, f"python: {mod}", why)
        else:
            _p(WARN, f"python: {mod}", f"absent — {why} disabled")
            warn += 1

    # ---- GROMACS -----------------------------------------------------------
    try:
        g = C.find_gromacs()
        _p(OK, "GROMACS", f"{g['version']}  ({g['how']})")
        print(f"        binary: {g['binary']}")
        if g["gmxrc"]:
            print(f"        GMXRC : {g['gmxrc']}")
    except C.EnvironmentError_ as e:
        _p(BAD, "GROMACS", "NOT FOUND")
        for line in str(e).splitlines()[1:]:
            print("       " + line)
        fatal += 1

    # ---- GPU ---------------------------------------------------------------
    if C.has_gpu():
        _p(OK, "GPU", C.gpu_name())
    else:
        _p(WARN, "GPU", "none detected — MD will run on CPU (much slower, still correct)")
        warn += 1

    # ---- process isolation -------------------------------------------------
    if C.has_systemd_user():
        _p(OK, "systemd --user", "cgroup CPU/memory caps enforced")
    else:
        _p(WARN, "systemd --user", "absent (normal on HPC nodes) -> "
                                   "scheduler falls back to plain subprocesses")
        warn += 1

    # ---- network -----------------------------------------------------------
    if C.has_internet():
        _p(OK, "internet", "PDB/AlphaFold fetches will work")
    else:
        _p(WARN, "internet", "OFFLINE — structure_source='rcsb' will fail. "
                             "Use structure_source='file' with a local PDB.")
        warn += 1

    # ---- writable output ---------------------------------------------------
    try:
        C.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        t = C.RUNS_DIR / ".writetest"
        t.write_text("x")
        t.unlink()
        _p(OK, "output dir", str(C.DATA_DIR))
    except Exception as e:  # noqa: BLE001
        _p(BAD, "output dir", f"NOT WRITABLE: {C.DATA_DIR} ({e}) -> set MDLAB_DATA=/scratch/...")
        fatal += 1

    # ---- assets ------------------------------------------------------------
    martini = C.ASSETS_DIR / "martini" / "martini_v3.0.0.itp"
    if martini.exists():
        _p(OK, "Martini force field", str(martini.parent))
    else:
        _p(WARN, "Martini force field", "absent -> ./scripts/fetch_assets.sh "
                                        "(coarse-grained track disabled)")
        warn += 1

    # ---- local LLM (optional) ---------------------------------------------
    models = C.ollama_models()
    if models:
        chat = C.pick_model(C.CHAT_MODEL_PREF)
        tr = C.pick_model(C.TRANSLATE_MODEL_PREF)
        _p(OK, "Ollama", f"{len(models)} model(s) at {C.OLLAMA_HOST}")
        print(f"        chat      : {chat or 'none suitable'}")
        print(f"        translator: {tr or 'none suitable'}")
    else:
        _p(WARN, "Ollama", f"no models at {C.OLLAMA_HOST} — chat/NL disabled. "
                           f"The plan pipeline does NOT need a model.")
        warn += 1

    warn += _ligand_check()

    print()
    if fatal:
        print(f"{fatal} fatal problem(s). Fix those first; warnings only disable optional features.")
    else:
        print(f"Ready. ({warn} warning(s) — optional features only.)")
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
