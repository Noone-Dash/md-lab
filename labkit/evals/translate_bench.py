"""Which local model is good enough to be the TRANSLATOR?

The pipeline is:  request --extract--> Intent      (pure code)
                  request --translate--> Plan_raw  (the model)
                  (Plan_raw, Intent) --enforce--> Plan  (pure code, overwrites the model)

So the model can ONLY get wrong the things the intent contract does not cover. This
benchmark measures exactly that, with an ablation that separates the two contributions:

    RAW  = does the model's own output match the ground truth?      (model alone)
    FINAL= does the plan match after enforce()?                     (model + guardrails)

FINAL - RAW is what the deterministic layer buys you. If FINAL is ~1.0 for every model,
then model choice is a LATENCY decision, not a correctness one — and you should run the
smallest, fastest model that clears the bar.

Cases deliberately mix:
  * covered intents      (temperature, salt, force field, molecule, duration, protocol)
  * UNCOVERED intents    (explicit box size / padding) — here the model is on its own,
                          and this is where a better model can actually pay for itself.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from labkit.agent.intent import extract, enforce          # noqa: E402
from labkit.agent.translate import translate              # noqa: E402
from labkit.config import DATA_DIR                        # noqa: E402
from labkit.plan import Plan, validate                    # noqa: E402

RESULTS = DATA_DIR / "translate_bench.json"


def _get(plan, path, default=None):
    cur = plan
    for k in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(k)]
            except Exception:  # noqa: BLE001
                return default
        elif isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
        if cur is None:
            return default
    return cur


def _prod_temp(plan):
    dyn = [s for s in (plan.get("stages") or []) if s.get("type") == "dynamics"]
    if not dyn:
        return None
    return (dyn[-1].get("params") or {}).get("temperature")


def _prod_ns(plan):
    dyn = [s for s in (plan.get("stages") or []) if s.get("type") == "dynamics"]
    return dyn[-1].get("sim_time_ns") if dyn else None


def _n_stages(plan):
    return len(plan.get("stages") or [])


# (request, {check_name: (getter, expected, tolerance)}, covered_by_intent?)
CASES = [
    ("Lysozyme at body temperature in physiological salt, proper equilibration, 50 ps production",
     {"pdb": (lambda p: _get(p, "system.pdb_id"), "1AKI", None),
      "temp": (_prod_temp, 310.0, 1.0),
      "salt": (lambda p: _get(p, "system.salt_conc_M"), 0.15, 0.02),
      "stages": (_n_stages, 4, 0)}, True),

    ("Water box at 300 K, NPT, for 20 ps",
     {"kind": (lambda p: _get(p, "system.kind"), "solvent", None),
      "temp": (_prod_temp, 300.0, 1.0),
      "ns": (_prod_ns, 0.02, 1e-6)}, True),

    ("Ubiquitin with a CHARMM force field at 310 K for 100 ps",
     {"pdb": (lambda p: _get(p, "system.pdb_id"), "1UBQ", None),
      "ff": (lambda p: _get(p, "system.forcefield"), "charmm27", None),
      "temp": (_prod_temp, 310.0, 1.0),
      "ns": (_prod_ns, 0.1, 1e-6)}, True),

    ("Simulate 1UBQ in 0.15 M NaCl at 300 K for 50 ps",
     {"pdb": (lambda p: _get(p, "system.pdb_id"), "1UBQ", None),
      "salt": (lambda p: _get(p, "system.salt_conc_M"), 0.15, 0.02),
      "temp": (_prod_temp, 300.0, 1.0)}, True),

    ("Trp-cage at 300 K for 100 ps with no salt",
     {"pdb": (lambda p: _get(p, "system.pdb_id"), "1L2Y", None),
      "salt": (lambda p: _get(p, "system.salt_conc_M"), 0.0, 0.001),
      "temp": (_prod_temp, 300.0, 1.0)}, True),

    # --- the intent contract does NOT parse an explicit box size / padding ---
    ("A 4 nm box of SPC/E water at 350 K, constant volume, 30 ps",
     {"kind": (lambda p: _get(p, "system.kind"), "solvent", None),
      "box": (lambda p: _get(p, "system.box_size_nm"), 4.0, 0.01),   # UNCOVERED
      "temp": (_prod_temp, 350.0, 1.0),
      "ens": (lambda p: (p["stages"][-1].get("params") or {}).get("ensemble"), "NVT", None)},
     False),

    ("Lysozyme at 300 K for 50 ps, use 2.0 nm of padding around the protein",
     {"pdb": (lambda p: _get(p, "system.pdb_id"), "1AKI", None),
      "pad": (lambda p: _get(p, "system.box_padding_nm"), 2.0, 0.01),  # UNCOVERED
      "temp": (_prod_temp, 300.0, 1.0)}, False),
]


def wilson(s, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def _grade(plan, checks):
    if not plan:
        return 0, len(checks), ["no plan"]
    ok, fails = 0, []
    for name, (getter, expected, tol) in checks.items():
        try:
            got = getter(plan)
        except Exception:  # noqa: BLE001
            got = None
        if got is None:
            fails.append(f"{name}=missing")
            continue
        if tol is None:
            good = str(got).lower() == str(expected).lower()
        else:
            try:
                good = abs(float(got) - float(expected)) <= tol
            except Exception:  # noqa: BLE001
                good = False
        if good:
            ok += 1
        else:
            fails.append(f"{name}={got}!={expected}")
    return ok, len(checks), fails


def run_model(model, k=2, verbose=True):
    raw_s = raw_n = fin_s = fin_n = 0
    unc_s = unc_n = 0
    lats, invalid = [], 0
    for req, checks, covered in CASES:
        for _ in range(k):
            t0 = time.time()
            try:
                r = translate(req, model=model)
            except Exception as e:  # noqa: BLE001
                r = {"plan": None, "error": str(e)[:60]}
            dt = time.time() - t0
            lats.append(dt)

            raw = r.get("plan")
            a, b, _f = _grade(raw, checks)
            raw_s += a; raw_n += b

            it = extract(req)
            fin = enforce(raw, it) if raw else None
            c, d, fails = _grade(fin, checks)
            fin_s += c; fin_n += d
            if not covered:
                unc_s += c; unc_n += d
            if fin:
                v = validate(Plan.from_dict(fin))
                if not v["ok"]:
                    invalid += 1
            if verbose:
                print(f"    {'OK ' if c == d else 'BAD'} {req[:44]:<46} "
                      f"raw {a}/{b}  final {c}/{d}  ({dt:.0f}s) {';'.join(fails[:2])}",
                      flush=True)

    p_raw, _, _ = wilson(raw_s, raw_n)
    p_fin, lo, hi = wilson(fin_s, fin_n)
    p_unc, ulo, uhi = wilson(unc_s, unc_n)
    med = statistics.median(lats) if lats else 0
    return {
        "model": model, "k": k,
        "raw_accuracy": round(p_raw, 3),
        "final_accuracy": round(p_fin, 3), "final_ci95": [round(lo, 3), round(hi, 3)],
        "uncovered_accuracy": round(p_unc, 3), "uncovered_ci95": [round(ulo, 3), round(uhi, 3)],
        "guardrail_gain": round(p_fin - p_raw, 3),
        "median_latency_s": round(med, 1),
        "invalid_plans": invalid,
        "n_checks": fin_n,
    }


def main(models, k=2):
    out = []
    for m in models:
        print(f"\n=== {m} ===", flush=True)
        try:
            out.append(run_model(m, k))
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            out.append({"model": m, "error": str(e)[:80], "final_accuracy": 0})
    out.sort(key=lambda r: (-r.get("final_accuracy", 0), r.get("median_latency_s", 1e9)))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 100)
    print(f"{'model':<22}{'RAW':>7}{'FINAL':>8}{'95% CI':>16}{'UNCOVERED':>12}"
          f"{'gain':>7}{'lat':>7}{'bad':>5}")
    print("-" * 100)
    for r in out:
        if "error" in r:
            print(f"{r['model']:<22}  FAILED: {r['error'][:50]}")
            continue
        ci = r["final_ci95"]
        print(f"{r['model']:<22}{r['raw_accuracy']:>7.2f}{r['final_accuracy']:>8.2f}"
              f"{f'[{ci[0]:.2f},{ci[1]:.2f}]':>16}{r['uncovered_accuracy']:>12.2f}"
              f"{r['guardrail_gain']:>+7.2f}{r['median_latency_s']:>6.0f}s"
              f"{r['invalid_plans']:>5}")
    print("=" * 100)
    print("RAW       = the model's own output vs ground truth")
    print("FINAL     = after the deterministic intent contract overwrites it")
    print("UNCOVERED = cases the contract does NOT parse — the model is on its own here")
    print("gain      = FINAL - RAW, i.e. what the guardrails buy you")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-k")]
    kk = next((int(a[2:]) for a in sys.argv[1:] if a.startswith("-k")), 2)
    main(args or ["qwen3:8b"], k=kk)
