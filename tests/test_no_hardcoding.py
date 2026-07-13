"""This bug class cost the user a working cluster deploy. It does not come back.

Greps the tree for machine-specific constants. Anything environment-dependent must
live in labkit/config.py and nowhere else.

    python tests/test_no_hardcoding.py
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# pattern -> why it is forbidden
FORBIDDEN = {
    r"/home/[a-z_]+/": "an absolute path into someone's home directory",
    r"spark-8d6e": "the developer's hostname",
    r"gromacs-2026\.2": "a pinned GROMACS version/path",
    r"Documents/tools": "the developer's directory layout",
    r'"-ntmpi"': "mdrun parallelism must come from config.mdrun_flags() "
                 "(-ntmpi is thread-MPI only and aborts a library-MPI build)",
    r"sys\.prefix": "assumes a venv layout; use shutil.which()",
    r"100\.109\.29\.55": "the developer's Tailscale IP",
}

# config.py is ALLOWED to know about the environment — that is its job. The two test files
# are allowed to NAME the forbidden constants, because asserting on them is the point.
ALLOWED = {"labkit/config.py", "tests/test_no_hardcoding.py",
           "tests/test_audit_regressions.py", "README.md", "SETUP.md"}


def main() -> int:
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                           text=True).stdout.split()
    bad = []
    for f in files:
        if f in ALLOWED or not f.endswith((".py", ".js", ".sh", ".toml", ".html")):
            continue
        try:
            text = (ROOT / f).read_text(errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for pat, why in FORBIDDEN.items():
            for m in re.finditer(pat, text):
                line = text[:m.start()].count("\n") + 1
                bad.append((f, line, m.group()[:40], why))

    if bad:
        print(f"{len(bad)} hardcoded machine-specific value(s):\n")
        for f, line, hit, why in bad:
            print(f"  {f}:{line}\n     found: {hit!r}\n     why:   {why}\n")
        print("Move it to labkit/config.py.")
        return 1

    print(f"ok — {len(files)} tracked files, no hardcoded paths/hosts/versions")
    print("     (environment lives in labkit/config.py, as it must)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
