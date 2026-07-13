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
from ..config import DATA_DIR as _DATA
RESULTS = _DATA / "eval_results.json"
BENCH = json.loads((HERE / "benchmarks.json").read_text())["benchmarks"]

from .metrics import extract, uncertainty           # noqa: E402


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

    # The error bar on the measurement, where the measurement IS a time average.
    # A PASS with an error bar so wide it also covers FAIL is not a PASS, it is a run
    # too short to decide — and it must say so rather than reporting a bare number.
    unc = uncertainty(m, b["metric"])
    if unc and verdict in ("PASS", "FAIL"):
        if not unc.get("resolvable"):
            # We could not put a defensible error bar on this. Say so; do not dress a
            # point estimate up as a verified one.
            verdict += "*"                 # e.g. PASS* — unquantified uncertainty
        else:
            lo, hi = unc["ci95"]
            inside = exp["min"] <= lo and hi <= exp["max"]
            outside = hi < exp["min"] or lo > exp["max"]
            if not (inside or outside):
                verdict = "INCONCLUSIVE"   # the CI straddles the acceptance boundary

    res = {
        "id": b["id"], "title": b["title"], "why": b["why"],
        "status": verdict,
        "measured": None if value is None else round(float(value), 4),
        "unit": b["unit"],
        "expected": f"{exp['min']} … {exp['max']}",
        "reference": b["reference"],
        "seconds": round(secs, 1),
        "run_id": m["id"],
    }
    if unc:
        res["uncertainty"] = {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in unc.items() if k != "ci95"}
        if unc.get("resolvable"):
            res["ci95"] = [round(unc["ci95"][0], 4), round(unc["ci95"][1], 4)]
            res["sem"] = round(unc["sem"], 4)
    return res


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
    # PASS* / FAIL* = the verdict holds, but the run was too short to put a defensible
    # error bar on it. Match on the stem, or a starred pass silently counts as not-passed.
    summary = {
        "total": len(out),
        "passed": sum(1 for r in out if r["status"].rstrip("*") == "PASS"),
        "failed": sum(1 for r in out if r["status"].rstrip("*") == "FAIL"),
        "inconclusive": sum(1 for r in out if r["status"] == "INCONCLUSIVE"),
        "unquantified": sum(1 for r in out if r["status"].endswith("*")),
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
        # .get, not [] — an unlisted verdict used to raise KeyError and crash the report
        # AFTER every real simulation had already been run.
        mark = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERR ", "NO DATA": "NODATA",
                "INCONCLUSIVE": "INCONC", "PASS*": "PASS*", "FAIL*": "FAIL*",
                }.get(r["status"], r["status"][:6])
        if r.get("sem") is not None:
            meas = f"{r['measured']} ± {r['sem']} {r['unit']}"
        elif r.get("measured") is not None:
            meas = f"{r['measured']} {r['unit']} (no error bar)"
        else:
            meas = r.get("detail", "-")
        print(f"  {mark:<7}{r['title']:<42} measured {str(meas):<28} "
              f"expect {r.get('expected','-')}  ({r['seconds']}s)")
    print("\n  * = the run was too short relative to its own correlation time to put a\n"
          "      defensible error bar on the number. The verdict is a point estimate.")
    print(f"\n  {s['passed']}/{s['total']} passed"
          + (f", {s['failed']} FAILED" if s["failed"] else "")
          + (f", {s['errored']} errored" if s["errored"] else ""))
    sys.exit(0 if s["failed"] == 0 and s["errored"] == 0 else 1)
