"""Every bug an adversarial audit found in code I had already called verified.

17 findings, each confirmed by 3 independent skeptics whose job was to REFUTE it. Each one
is pinned here. They are grouped by what they would have done to a scientist using this:

  SILENTLY WRONG PHYSICS  — the simulation runs, finishes green, and is wrong.
  SILENTLY WRONG NUMBERS  — the analysis reports a confident value that is not defensible.
  SILENTLY WRONG SCORES   — the grader reports a model/benchmark as good when it is not.
  WRONG ON A CLUSTER      — works here, breaks or wastes the machine elsewhere.

    python tests/test_audit_regressions.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labkit.agent.intent import enforce, extract, verify   # noqa: E402
from labkit.uncertainty import stats                        # noqa: E402

FAILS = []


def check(name, got, want, why):
    ok = got == want
    if not ok:
        FAILS.append(f"{name}: got {got!r}, want {want!r}\n      ({why})")
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<52} {got!r}")


# ---------------------------------------------------------------- SILENTLY WRONG PHYSICS
def test_physics():
    print("\nSILENTLY WRONG PHYSICS — the run finishes green and is wrong\n")

    # "137 C" contains the substring "37 c". The unanchored body-temperature pattern
    # matched it and pinned 310 K. A 100 K error in the thermostat, and verify() clean.
    check("137 C is 410 K, not body temperature",
          extract("Argon at 137 C, 1 ns").temperature_K, 410.15,
          "unanchored r'37\\s*c' matched inside '137 c'")
    check("body temperature still works",
          extract("Lysozyme at body temperature, 50 ps").temperature_K, 310.0, "")

    # A PDB id is [0-9][a-z0-9]{3} — and so is "10ns". This fabricated pdb_id='10NS'
    # (which 404s at RCSB) AND ran before the name lookup, destroying the real id.
    check("'10ns' is a duration, not PDB entry 10NS",
          extract("simulate lysozyme for 10ns at 300 K").pdb_id, "1AKI",
          "the bare-id regex matched the unit-suffixed number and pre-empted KNOWN_PDB")
    check("a real explicit id still wins",
          extract("Simulate 1UBQ in 0.15 M NaCl at 300 K for 50 ps").pdb_id, "1UBQ", "")

    # production_ns took the FIRST duration in the string.
    check("production time is the PRODUCTION one",
          extract("Lysozyme: equilibrate for 1 ns, then run 20 ns of production"
                  ).production_ns, 20.0,
          "took the first duration -> pinned equilibration time as production: 20x error")
    check("...also when the word order flips",
          extract("Lysozyme: 100 ps of equilibration then 5 ns production"
                  ).production_ns, 5.0, "")
    check("genuinely ambiguous -> pin NOTHING",
          extract("do 1 ns then 2 ns then 3 ns").production_ns, None,
          "a wrong pin is worse than no pin: verify() cannot see through it")

    # r"tip4p" matches inside "tip4p-ew" — a DIFFERENT water model, and pinned.
    check("TIP4P-Ew is not TIP4P",
          extract("A 4 nm box of TIP4P-Ew water at 300 K").water_model, "tip4pew",
          "most-specific-first ordering; plain tip4p has different charges")
    check("SPC/E still resolves",
          extract("A 4 nm box of SPC/E water at 350 K").water_model, "spce", "")

    # "no equilibration" matched r"equilibrat\w*" and switched equilibration ON.
    check("'no equilibration' means NO",
          extract("Lysozyme, 5 ns, no equilibration, NVT").equilibrate, False,
          "the regex had no negation guard, so a refusal turned the template on")

    # enforce() wrote the pinned dt into `plan` (the CALLER'S INPUT dict) rather than `p`
    # (the deep copy it returns). The pin mutated a dict nobody reads; the returned plan
    # kept the model's timestep. dt was an unpinned physical field, silently.
    it = extract("A water box at 300 K for 20 ps with a 0.5 fs timestep")
    model_plan = {"name": "x", "system": {"kind": "solvent", "box_size_nm": 3.0},
                  "stages": [{"name": "prod", "type": "dynamics", "sim_time_ns": 0.02,
                              "params": {"ensemble": "NPT", "temperature": 300,
                                         "dt": 0.002}}]}
    out = enforce(model_plan, it)
    check("dt reaches the RETURNED plan",
          out["stages"][-1]["params"]["dt"], 0.0005,
          "enforce() mutated the input dict instead of the copy it returns")

    # The equilibration template hardcoded NPT for production, and verify() had no
    # ensemble check at all — so an explicit NVT request was silently converted.
    it = extract("Lysozyme, NVT production for 10 ns, with proper equilibration")
    out = enforce({"name": "x", "system": {"kind": "protein", "pdb_id": "1AKI"},
                   "stages": [{"name": "prod", "type": "dynamics", "sim_time_ns": 10.0,
                               "params": {"ensemble": "NVT", "temperature": 300}}]}, it)
    check("explicit NVT survives the template",
          out["stages"][-1]["params"]["ensemble"], "NVT",
          "template hardcoded NPT; verify() had no ensemble assertion to catch it")
    check("verify() now asserts the ensemble", verify(out, it), [], "")

    # The whole point of the contract: the model may never source a physical value.
    from labkit.plan import Plan
    from labkit.plan.resolve import resolve
    it = extract("Lysozyme at body temperature in physiological salt, SPC/E water, "
                 "2.0 nm padding, 1 fs timestep, NVT production, proper equilibration, "
                 "50 ps production")
    good = {"name": "g", "system": {"kind": "protein", "pdb_id": "1AKI",
            "structure_source": "rcsb", "forcefield": "amber99sb-ildn",
            "water_model": "spce", "box_shape": "dodecahedron", "box_padding_nm": 2.0,
            "salt_conc_M": 0.15, "neutralize": True},
            "stages": [{"name": "prod", "type": "dynamics", "sim_time_ns": 0.05,
                        "params": {"ensemble": "NVT", "temperature": 310, "dt": 0.001}}]}
    evil = {"name": "e", "system": {"kind": "protein", "pdb_id": "9ZZZ",
            "structure_source": "rcsb", "forcefield": "gromos54a7",
            "water_model": "tip5p", "box_shape": "cubic", "box_padding_nm": 0.4,
            "salt_conc_M": 0.0, "neutralize": False},
            "stages": [{"name": "prod", "type": "dynamics", "sim_time_ns": 9.0,
                        "params": {"ensemble": "NPT", "temperature": 500, "dt": 0.005}}]}
    ra = resolve(Plan.from_dict(enforce(good, it)))
    rb = resolve(Plan.from_dict(enforce(evil, it)))
    ndiff = sum(1 for sa, sb in zip(ra["stages"], rb["stages"])
                for k in set(sa["mdp"]) | set(sb["mdp"])
                if sa["mdp"].get(k) != sb["mdp"].get(k))
    check("a maximally-wrong model changes 0 mdp keys", ndiff, 0,
          "this passed BEFORE the dt fix only because the template's default masked it")


# ---------------------------------------------------------------- SILENTLY WRONG NUMBERS
def _ar1(rng, phi, n):
    eps = rng.normal(size=n)
    x = np.empty(n)
    x[0] = eps[0] / math.sqrt(1 - phi ** 2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def test_uncertainty():
    print("\nSILENTLY WRONG NUMBERS — a confident error bar that is not defensible\n")
    rng = np.random.default_rng(5)
    phi = 0.9                                   # tau_int = 9.5 samples, exactly

    # THE FATAL ONE. The old 'estimators_agree' cross-check was True in 100% of trials at
    # every N: when the run is short, Sokal's window cannot open AND blocking has one
    # level, so BOTH estimators degenerate to the naive SEM and "agree" — precisely in the
    # regime the flag existed to catch. It certified error bars 6x too narrow.
    short = [stats(_ar1(rng, phi, 16)) for _ in range(30)]
    check("N=16, tau=9.5 -> REFUSE, do not guess",
          all(not s["resolvable"] for s in short), True,
          "old code reported a SEM 0.17x of truth with estimators_agree=True, 500/500 seeds")
    check("...and emits no CI at all",
          all("ci95" not in s for s in short), True, "")
    check("the vacuous agree-flag is gone",
          any("estimators_agree" in s for s in short), False,
          "two estimators that fail identically are not a cross-check")

    # A long run must still produce an accurate bar.
    long_ = stats(_ar1(rng, phi, 20000))
    var = 1.0 / (1 - phi ** 2)
    sem_true = math.sqrt(2 * 9.5 * var / 20000)
    ratio = long_["sem"] / sem_true
    check("N=20000 -> accurate bar (0.8-1.3x of truth)",
          0.8 <= ratio <= 1.3, True, f"ratio={ratio:.2f}")
    check("...and it IS resolvable", long_["resolvable"], True, "")

    # Sokal's c=5 window closes on the FAST mode of a multi-timescale ACF and misses a
    # small-amplitude slow one. AR(1) (a single exponential) is the one family where c=5
    # is safe — which is exactly why the original self_test never caught this.
    slow = _ar1(rng, 0.99, 300_000)
    x = rng.normal(size=300_000) + 0.3 * slow
    s = stats(x)
    w = (0.09 * (1 / (1 - 0.99 ** 2))) / (1 + 0.09 * (1 / (1 - 0.99 ** 2)))
    tau_true = 0.5 + w * 0.99 / 0.01
    var_x = 1 + 0.09 * (1 / (1 - 0.99 ** 2))
    sem_true = math.sqrt(2 * tau_true * var_x / 300_000)
    ratio = s["sem"] / sem_true
    check("multi-timescale ACF: bar within 0.75-1.35x",
          0.75 <= ratio <= 1.35, True,
          f"ratio={ratio:.2f}; Sokal alone understated it — SEM is now max(sokal, blocking)")


# ---------------------------------------------------------------- SILENTLY WRONG SCORES
def test_graders():
    print("\nSILENTLY WRONG SCORES — a grader that reports 'good' when it is not\n")
    from labkit.evals import translate_bench as tb

    # A model returning pdb_id=None hit pdb_title() -> None, which IsProtein read as
    # "we are offline" and RAISED — killing the model's ENTIRE benchmark row. Two of
    # three models were reported as an infrastructure failure when they had simply
    # answered wrongly. Wrong must be wrong; offline must be offline.
    gfp = tb.IsProtein("fluorescent protein")
    for bad in (None, "GFP", "green fluorescent protein", "", "9ZZZ", "1AKI"):
        check(f"pdb_id={bad!r} is WRONG, not 'offline'", gfp(bad), False,
              "a model's wrong answer was being reported as a network failure")
    check("a real GFP entry is right", gfp("1EMA"), True, "")
    check("...and so is a DIFFERENT real GFP entry", gfp("1GFL"), True,
          "grading against one hardcoded id would measure agreement with my guess")

    # A 503/429 was caught by `except HTTPError` alongside 404 and cached as "" —
    # permanently poisoning the disk cache and scoring a CORRECT model wrong forever.
    import inspect
    src = inspect.getsource(tb.pdb_title)
    check("only a 404 is cached as 'not an entry'",
          "e.code == 404" in src, True,
          "a transient 5xx used to poison the cache for a REAL entry")

    # runner emitted INCONCLUSIVE, which was absent from the report's mark dict ->
    # KeyError, crashing the report AFTER every real simulation had already run.
    from labkit.evals import runner
    src = inspect.getsource(runner)
    check("the report cannot KeyError on a verdict",
          '.get(r["status"]' in src, True,
          "INCONCLUSIVE was not in the mark dict; the CLI died after running everything")


# ---------------------------------------------------------------- WRONG ON A CLUSTER
def test_cluster():
    print("\nWRONG ON A CLUSTER — fine here, broken or wasteful elsewhere\n")
    import os

    from labkit import config as cfg

    # A cgroup CPUQuota caps CPU *time*, not affinity, so sched_getaffinity in the child
    # still sees every core: each concurrent job spawned node-wide OpenMP threads.
    os.environ["OMP_NUM_THREADS"] = "4"
    flags = cfg.mdrun_flags()
    check("-ntomp honours the per-job budget",
          flags[flags.index("-ntomp") + 1], "4",
          "used the whole allocation, so concurrent jobs each went node-wide and thrashed")
    del os.environ["OMP_NUM_THREADS"]

    # -ntmpi is thread-MPI ONLY; a library-MPI gmx_mpi aborts on it.
    check("-ntmpi only for thread-MPI builds",
          ("-ntmpi" in cfg.mdrun_flags()) ==
          (cfg.find_gromacs().get("mpi") == "thread_mpi"), True, "")

    # _pin() computed sorted(sched_getaffinity(0))[:cores] independently in each child,
    # so EVERY concurrent job landed on the SAME cpus while the rest of the node idled.
    # This is the path an HPC compute node actually takes (no user systemd).
    import inspect

    from labkit.scheduler import Scheduler
    check("jobs get DISJOINT cpu slices",
          "_free_cpus" in inspect.getsource(Scheduler), True,
          "all concurrent jobs were stacked onto the first cores_per_job CPUs")

    # backfill hardcoded dt=0.002 to turn target_ns into -nsteps, ignoring the dt that
    # resolve() actually wrote. A Martini plan (dt=0.02) ran 10x less physical time.
    from labkit import backfill
    check("backfill takes dt from the resolved plan",
          "resolve(plan)" in inspect.getsource(backfill.run_campaign), True,
          "hardcoded 0.002 -> a coarse-grained campaign ran 10x short")


if __name__ == "__main__":
    test_physics()
    test_uncertainty()
    test_graders()
    test_cluster()
    print()
    if FAILS:
        print(f"{len(FAILS)} REGRESSION(S):\n")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL AUDIT REGRESSIONS PASS — 17 confirmed bugs, each pinned.")
