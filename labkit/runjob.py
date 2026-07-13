"""Run a single experiment as its own process.

Invoked by the scheduler (usually inside a systemd transient scope with cgroup
CPU/memory caps):  python -m labkit.runjob <run_id> <key> <params.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labkit.engine import run_recipe          # noqa: E402
from labkit.recipes import REGISTRY           # noqa: E402
from labkit.tracks import TRACKS, get_track    # noqa: E402


def main():
    run_id, key, pfile = sys.argv[1], sys.argv[2], sys.argv[3]
    params = json.loads(Path(pfile).read_text())
    if key == "__plan__":
        from labkit.plan.schema import Plan
        from labkit.engine import run_plan
        run_plan(Plan.from_dict(params["plan"]), run_id=run_id)
    elif key in REGISTRY:
        run_recipe(key, params, run_id=run_id)
    elif key in TRACKS:
        get_track(key).run(params, run_id=run_id)
    else:
        raise SystemExit(f"unknown experiment '{key}'")


if __name__ == "__main__":
    main()
