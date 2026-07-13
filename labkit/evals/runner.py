"""Run the physics benchmarks and grade them against known reference values.

    ./.venv/bin/python -m labkit.evals.runner            # run all
    ./.venv/bin/python -m labkit.evals.runner water_density lj_liquid_is_structured
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "simulations" / "eval_results.json"
BENCH = json.loads((HERE / "benchmarks.json").read_text())["benchmarks"]

from .metrics import extract           # noqa: E402


def _run_one(b, progress=None):
    from ..engine import run_recipe
    from ..tracks import get_track

    spec = b["run"]
    t0 = time.time()
    if spec["kind"] == "recipe":
        m = run_recipe(spec["key"], spec["params"])
    else:
        m = get_track(spec["key"]).run(spec["params"])
    secs = time.time() - t0

    if m["status"] != "done":
        return {"id": b["id"], "title": b["title"], "status": "ERROR",
                "detail": m.get("error", "run failed"), "seconds": round(secs, 1),
                "run_id": m["id"], "why": b["why"]}

    value = extract(m, b["metric"])
    exp = b["expect"]
    if value is None:
        verdict = "NO DATA"
    elif exp["min"] <= value <= exp["max"]:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "id": b["id"], "title": b["title"], "why": b["why"],
        "status": verdict,
        "measured": None if value is None else round(float(value), 4),
        "unit": b["unit"],
        "expected": f"{exp['min']} … {exp['max']}",
        "reference": b["reference"],
        "seconds": round(secs, 1),
        "run_id": m["id"],
    }


def run(ids=None, progress=None):
    todo = [b for b in BENCH if not ids or b["id"] in ids]
    out = []
    for i, b in enumerate(todo, 1):
        if progress:
            progress({"phase": "running", "i": i, "n": len(todo), "id": b["id"]})
        r = _run_one(b)
        out.append(r)
        if progress:
            progress({"phase": "done_one", "i": i, "n": len(todo), "result": r})
    summary = {
        "total": len(out),
        "passed": sum(1 for r in out if r["status"] == "PASS"),
        "failed": sum(1 for r in out if r["status"] == "FAIL"),
        "errored": sum(1 for r in out if r["status"] in ("ERROR", "NO DATA")),
        "results": out,
    }
    RESULTS.write_text(json.dumps(summary, indent=2))
    return summary


def load_last():
    if RESULTS.exists():
        try:
            return json.loads(RESULTS.read_text())
        except Exception:  # noqa: BLE001
            pass
    return None


if __name__ == "__main__":
    ids = sys.argv[1:] or None
    print(f"running {len(ids or BENCH)} physics benchmark(s) — these are REAL simulations\n")
    s = run(ids, progress=lambda p: (
        print(f"  [{p['i']}/{p['n']}] {p['id']} …", flush=True)
        if p["phase"] == "running" else None))
    print()
    for r in s["results"]:
        mark = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERR ", "NO DATA": "NODATA"}[r["status"]]
        meas = f"{r['measured']} {r['unit']}" if r.get("measured") is not None else r.get("detail", "-")
        print(f"  {mark}  {r['title']:<42} measured {str(meas):<22} expect {r.get('expected','-')}  ({r['seconds']}s)")
    print(f"\n  {s['passed']}/{s['total']} passed"
          + (f", {s['failed']} FAILED" if s["failed"] else "")
          + (f", {s['errored']} errored" if s["errored"] else ""))
    sys.exit(0 if s["failed"] == 0 and s["errored"] == 0 else 1)
