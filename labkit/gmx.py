"""Thin wrapper around the ``gmx`` executable.

Every call sources GMXRC in a login-ish shell so that the correct libraries and
data paths are set, then runs a single ``gmx`` subcommand.  stdout+stderr are
merged and appended to a per-run log file so the UI can tail progress live.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from . import config as _cfg   # the ONE place the environment is resolved


class GmxError(RuntimeError):
    """Raised when a gmx subcommand exits non-zero."""

    def __init__(self, argv, returncode, tail):
        self.argv = argv
        self.returncode = returncode
        self.tail = tail
        super().__init__(
            f"gmx {' '.join(argv)} failed (exit {returncode}).\n--- last output ---\n{tail}"
        )


def _shell(cmdline: str, cwd, log_path=None, stdin_text=None, timeout=None):
    """Run *cmdline* under a shell, streaming to a log file.

    GMXRC is sourced only if we actually have one (a source/prefix install). When gmx
    came from PATH (`module load gromacs`) there is nothing to source. We use `bash -c`,
    NOT `bash -lc`: a login shell re-reads the user's profile, which on a cluster can
    reset PATH, unload modules, or emit banners into our parsed output.
    """
    info = _cfg.find_gromacs()
    setup = os.environ.get("MDLAB_GMX_SETUP", "")   # e.g. "module load gromacs/2023.3"
    if info["gmxrc"]:
        full = f"source {shlex.quote(info['gmxrc'])} >/dev/null 2>&1 && {cmdline}"
    else:
        full = cmdline
    if setup:
        full = f"{setup} && {full}"
    log_f = open(log_path, "a") if log_path else None
    try:
        if log_f:
            log_f.write(f"\n$ {cmdline}\n")
            log_f.flush()
        proc = subprocess.run(
            ["bash", "-c", full],
            cwd=str(cwd),
            input=stdin_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        if log_f:
            log_f.write(proc.stdout or "")
            log_f.flush()
        return proc.returncode, (proc.stdout or "")
    finally:
        if log_f:
            log_f.close()


def gmx(args, cwd, log_path=None, stdin_text=None, timeout=None, check=True):
    """Run ``gmx <args...>`` in *cwd*.

    args : list[str]        subcommand + flags, e.g. ["grompp", "-f", "md.mdp", ...]
    stdin_text : str|None   fed to interactive prompts (group selection, etc.)
    Returns (returncode, combined_output).
    """
    argv = [str(a) for a in args]
    # mdrun parallelism is decided in ONE place (config.mdrun_flags): -ntmpi only on
    # thread-MPI builds, -ntomp from the CPUs we are actually allowed to use.
    if argv and argv[0] == "mdrun":
        have = set(argv)
        for fl in _cfg.mdrun_flags():
            if fl.startswith("-") and fl in have:
                break
        else:
            argv = argv + _cfg.mdrun_flags()
    # GROMACS defaults to prompting before overwriting; -quiet+backups off keeps runs clean.
    gmx_bin = _cfg.gmx_binary()          # discovered: $GMX_ROOT, PATH, or a known prefix
    cmdline = (shlex.quote(gmx_bin) + " -quiet -nobackup "
               + " ".join(shlex.quote(a) for a in argv))
    rc, out = _shell(cmdline, cwd, log_path=log_path, stdin_text=stdin_text, timeout=timeout)
    if check and rc != 0:
        tail = "\n".join(out.splitlines()[-40:])
        raise GmxError(argv, rc, tail)
    return rc, out


def gmx_version() -> str:
    _, out = gmx(["--version"], cwd=Path.cwd(), check=False)
    for line in out.splitlines():
        if "GROMACS version" in line:
            return line.split(":", 1)[1].strip()
    return "unknown"
