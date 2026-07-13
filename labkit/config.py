"""The ONE place the environment is resolved. Nothing else may hardcode a path.

Every previous module baked in the developer's own machine
(`/home/v_u/Documents/tools/opt/gromacs-2026.2`), so a fresh clone on any other box —
a cluster, a laptop — produced a wall of errors. This module replaces that with
discovery + explicit failure.

Resolution order for GROMACS (first hit wins):
    1. $GMX_ROOT/bin/gmx            — explicit override
    2. `gmx` already on PATH        — the `module load gromacs` case (no GMXRC needed)
    3. `gmx_mpi` on PATH            — MPI builds
    4. conventional install prefixes
    5. give up LOUDLY with an actionable message (never a silent fallback)

Everything is overridable by environment variable, and `python -m labkit.doctor`
reports exactly what is present and what is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------- repo layout
REPO_ROOT = Path(__file__).resolve().parent.parent

# Writable output. On a cluster $HOME is often quota'd/read-only-ish and you are
# told to use scratch, so allow an override.
DATA_DIR = Path(os.environ.get("MDLAB_DATA", REPO_ROOT / "simulations")).resolve()
RUNS_DIR = DATA_DIR / "runs"
JOBS_DIR = DATA_DIR / "_jobs"
ASSETS_DIR = Path(os.environ.get("MDLAB_ASSETS", REPO_ROOT / "assets")).resolve()

# ---------------------------------------------------------------- web UI
UI_HOST = os.environ.get("MDLAB_HOST", "127.0.0.1")   # loopback by DEFAULT.
UI_PORT = int(os.environ.get("MDLAB_PORT", "5057"))   # a shared cluster must not be
#                                                       exposed on 0.0.0.0 by accident.

# ---------------------------------------------------------------- local LLM
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
CHAT_MODEL_PREF = [m for m in (os.environ.get("MDLAB_LOCAL_MODEL"),) if m] + [
    "gpt-oss:20b", "qwen3:14b", "qwen3:8b", "llama3.1:8b",
]
TRANSLATE_MODEL_PREF = [m for m in (os.environ.get("MDLAB_TRANSLATOR"),) if m] + [
    "qwen3:8b", "qwen3:14b", "qwen2.5-coder:7b", "qwen2.5-coder:32b", "gpt-oss:20b",
]


class EnvironmentError_(RuntimeError):
    """Raised with an actionable message, never a bare stack trace."""


# ---------------------------------------------------------------- GROMACS
GMX_CANDIDATE_PREFIXES = [
    "/usr/local/gromacs", "/opt/gromacs", "/usr/lib/gromacs",
    str(Path.home() / "gromacs"), str(Path.home() / "opt/gromacs"),
]


@lru_cache(maxsize=1)
def find_gromacs() -> dict:
    """-> {'binary', 'gmxrc'|None, 'how', 'version'} or raises with instructions."""
    # 1. explicit override
    root = os.environ.get("GMX_ROOT")
    if root:
        b = Path(root) / "bin" / "gmx"
        if b.exists():
            return _gmx_info(str(b), Path(root) / "bin" / "GMXRC", "GMX_ROOT")
        b = Path(root) / "bin" / "gmx_mpi"
        if b.exists():
            return _gmx_info(str(b), Path(root) / "bin" / "GMXRC", "GMX_ROOT (gmx_mpi)")
        raise EnvironmentError_(
            f"GMX_ROOT is set to {root!r} but {root}/bin/gmx does not exist.\n"
            f"Point GMX_ROOT at the GROMACS install prefix (the dir containing bin/gmx).")

    # 2. already on PATH — this is the `module load gromacs` case; no GMXRC needed
    for exe in ("gmx", "gmx_mpi"):
        p = shutil.which(exe)
        if p:
            return _gmx_info(p, None, f"{exe} on PATH")

    # 3. conventional prefixes (and versioned siblings like gromacs-2026.2)
    for prefix in GMX_CANDIDATE_PREFIXES:
        pp = Path(prefix)
        for cand in ([pp] + sorted(pp.parent.glob(pp.name + "*"), reverse=True)
                     if pp.parent.exists() else [pp]):
            b = cand / "bin" / "gmx"
            if b.exists():
                return _gmx_info(str(b), cand / "bin" / "GMXRC", f"found at {cand}")

    raise EnvironmentError_(
        "GROMACS not found.\n"
        "  Fix one of these:\n"
        "    export GMX_ROOT=/path/to/gromacs      # install prefix containing bin/gmx\n"
        "    module load gromacs                   # on an HPC cluster\n"
        "    source /path/to/gromacs/bin/GMXRC\n"
        "  Then re-run.  `python -m labkit.doctor` will confirm.")


def _gmx_info(binary: str, gmxrc: Path | None, how: str) -> dict:
    ver, mpi, gpu = "unknown", "none", "none"
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True,
                             timeout=30, env=os.environ.copy()).stdout
        for line in out.splitlines():
            low = line.lower()
            if "gromacs version" in low:
                ver = line.split(":", 1)[1].strip()
            elif low.startswith("mpi library"):
                v = line.split(":", 1)[1].strip().lower()
                mpi = "thread_mpi" if "thread" in v else ("library" if "mpi" in v else "none")
            elif low.startswith("gpu support"):
                gpu = line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return {"binary": binary,
            "gmxrc": str(gmxrc) if gmxrc and Path(gmxrc).exists() else None,
            "how": how, "version": ver, "mpi": mpi, "gpu": gpu}


def mdrun_flags() -> list:
    """The ONE place mdrun parallelism is decided.

    `-ntmpi` is a THREAD-MPI-only flag. On a library-MPI build (gmx_mpi, i.e. what
    `module load gromacs` gives you at most HPC sites) mdrun aborts:
        'Setting the number of thread-MPI ranks is only supported with thread-MPI'
    So it is emitted ONLY when the build is thread-MPI.

    Thread count comes from the CPUs we are actually allowed to use (a Slurm cpuset
    is usually a subset of the machine), never from the host CPU count.
    """
    info = find_gromacs()
    flags = []
    if info.get("mpi") == "thread_mpi":
        flags += ["-ntmpi", "1"]         # thread-MPI builds only
    flags += ["-ntomp", str(allowed_cores())]
    return flags


def allowed_cores() -> int:
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n and n.isdigit():
        return max(1, min(int(n), 64))
    try:
        return max(1, min(len(os.sched_getaffinity(0)), 64))
    except AttributeError:          # macOS / Windows
        return max(1, min(os.cpu_count() or 1, 64))


def allowed_mem_gb() -> float:
    """Memory we are actually allowed to use — NOT the memory the node happens to have.

    Inside a Slurm allocation or a cgroup, the node's physical RAM is a fiction: touch
    more than the limit and the OOM killer takes the job with no diagnostic. Order:
    Slurm's own accounting -> the cgroup v2 limit -> physical RAM.
    """
    per_node = os.environ.get("SLURM_MEM_PER_NODE")          # MB
    if per_node and per_node.isdigit():
        return int(per_node) / 1024.0
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")            # MB
    if per_cpu and per_cpu.isdigit():
        return int(per_cpu) * allowed_cores() / 1024.0
    try:                                                     # cgroup v2
        v = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if v != "max":
            return int(v) / 1024**3
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil
        return psutil.virtual_memory().total / 1024**3
    except Exception:  # noqa: BLE001
        return 8.0


def child_env(**overrides) -> dict:
    """Environment for a child process.

    Must be a COPY of os.environ, not a hand-picked whitelist: dropping
    LD_LIBRARY_PATH means a module-loaded gmx cannot link its own libs, and dropping
    MDLAB_DATA means the child writes its results where the parent will never look.
    """
    env = dict(os.environ)
    env["MDLAB_DATA"] = str(DATA_DIR)
    env["MDLAB_ASSETS"] = str(ASSETS_DIR)
    env.setdefault("OMP_NUM_THREADS", str(allowed_cores()))
    for k, v in overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return env


def default_budget() -> dict:
    """Compute budget derived from the ALLOCATION, not from constants.

    6 cores / 24 GB was hardcoded. On a 4-core cluster allocation that oversubscribes
    and thrashes; on a 72-core node it wastes most of the machine. Both are wrong.
    """
    cores, mem = allowed_cores(), allowed_mem_gb()
    conc = 1 if cores <= 4 else 2
    return {
        "max_concurrent": conc,
        "max_gpu_jobs": 1 if has_gpu() else 0,
        "cores_per_job": max(1, (cores - 1) // conc),
        "mem_per_job_gb": max(2, int(mem * 0.8 / conc)),
    }


def gmx_binary() -> str:
    return find_gromacs()["binary"]


def gmx_available() -> bool:
    try:
        find_gromacs()
        return True
    except EnvironmentError_:
        return False


# ---------------------------------------------------------------- GPU
@lru_cache(maxsize=1)
def has_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=1)
def gpu_name() -> str:
    if not has_gpu():
        return "none"
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------- systemd
@lru_cache(maxsize=1)
def has_systemd_user() -> bool:
    """A user systemd session with cgroup delegation. Absent on most HPC compute nodes."""
    if not shutil.which("systemd-run"):
        return False
    try:
        r = subprocess.run(["systemd-run", "--user", "--quiet", "--scope", "true"],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- internet
@lru_cache(maxsize=1)
def has_internet() -> bool:
    """Compute nodes are frequently offline. PDB fetches must fail with a clear message."""
    import urllib.request
    try:
        urllib.request.urlopen("https://files.rcsb.org", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=1)
def ollama_models() -> list:
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def pick_model(prefs: list) -> str | None:
    """First preferred model that is ACTUALLY pulled. No hardcoded assumption."""
    have = ollama_models()
    for m in prefs:
        if m in have:
            return m
    base = {h.split(":")[0]: h for h in have}
    for m in prefs:
        if m.split(":")[0] in base:
            return base[m.split(":")[0]]
    return have[0] if have else None


def summary() -> dict:
    try:
        g = find_gromacs()
    except EnvironmentError_ as e:
        g = {"error": str(e).splitlines()[0]}
    return {
        "repo_root": str(REPO_ROOT),
        "data_dir": str(DATA_DIR),
        "gromacs": g,
        "gpu": gpu_name(),
        "systemd_user": has_systemd_user(),
        "internet": has_internet(),
        "ollama": {"host": OLLAMA_HOST, "models": ollama_models()},
        "ui": f"{UI_HOST}:{UI_PORT}",
    }
