"""Hardware benchmark: what does THIS GPU actually deliver, in ns/day, on real systems?

Everything about feasibility (what can we simulate, for how long, at what duty cycle)
reduces to one measured number per system size: ns/day. GROMACS reports it directly.
No estimates, no vendor numbers.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from labkit.gmx import gmx                      # noqa: E402
from labkit.plan import Plan                    # noqa: E402
from labkit.plan.build import build             # noqa: E402
from labkit.engine import RUNS_DIR              # noqa: E402

from labkit.config import DATA_DIR as _DATA
OUT = _DATA / "hw_bench.json"

# representative biomolecular systems, small -> large
# Same protein, growing solvent shell: isolates the size-scaling law with no
# confounds (same topology, same force field). 4HHB was dropped: it has haem
# groups that amber99sb-ildn cannot parameterise (a real limitation, not a bug).
SYSTEMS = [
    {"id": "lysozyme_pad1.0", "pdb": "1AKI", "pad": 1.0, "note": "small box"},
    {"id": "lysozyme_pad1.4", "pdb": "1AKI", "pad": 1.4, "note": "typical production box"},
    {"id": "ubiquitin_pad1.2", "pdb": "1UBQ", "pad": 1.2, "note": "different protein, same regime"},
    {"id": "lysozyme_pad2.0", "pdb": "1AKI", "pad": 2.0, "note": "fat solvent shell"},
    {"id": "lysozyme_pad2.8", "pdb": "1AKI", "pad": 2.8, "note": "large system"},
]

BENCH_STEPS = 50000        # 100 ps at 2 fs
REPEATS = 3                # median of 3; a single short run is not a measurement


def _perf(log: Path):
    """GROMACS prints its own performance table. Read ns/day and atom count."""
    if not log.exists():
        return None, None
    txt = log.read_text(errors="replace")
    m = re.search(r"Performance:\s+([\d.]+)\s+([\d.]+)", txt)
    nsday = float(m.group(1)) if m else None
    a = re.search(r"There are:\s+(\d+)\s+Atoms", txt)
    atoms = int(a.group(1)) if a else None
    return nsday, atoms


def bench_one(spec):
    plan = Plan.from_dict({
        "name": f"bench_{spec['id']}",
        "system": {"kind": "protein", "structure_source": "rcsb", "pdb_id": spec["pdb"],
                   "forcefield": "amber99sb-ildn", "water_model": "tip3p",
                   "box_shape": "dodecahedron", "box_padding_nm": spec["pad"],
                   "salt_conc_M": 0.15, "neutralize": True},
        "stages": [
            {"name": "minimize", "type": "minimize", "max_steps": 2000,
             "params": {"ensemble": "NVT", "temperature": 300}},
            {"name": "bench", "type": "dynamics",
             "sim_time_ns": BENCH_STEPS * 0.002 / 1000,
             "params": {"ensemble": "NPT", "temperature": 300}},
        ],
        "analyses": [],
    })
    run_dir = RUNS_DIR / f"hwbench_{spec['id']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "run.log"

    steps, _ = build(plan, run_dir)
    t0 = time.time()
    rates = []
    for st in steps:
        argv = list(st.argv)
        # -resethway: reset the timers at the halfway mark so startup, domain-decomp
        # load balancing and PME auto-tuning do not pollute the measured rate.
        if argv and argv[0] == "mdrun" and "bench" in " ".join(argv):
            argv += ["-resethway"]
        gmx(argv, cwd=run_dir, log_path=log, stdin_text=st.stdin)
    wall = time.time() - t0

    nsday, atoms = _perf(run_dir / "bench.log")
    rates.append(nsday)
    # repeat the mdrun only (system is already built) to get a median
    for _ in range(REPEATS - 1):
        gmx(["mdrun", "-deffnm", "bench", "-v", "-resethway"],
            cwd=run_dir, log_path=log)
        r, _a = _perf(run_dir / "bench.log")
        if r:
            rates.append(r)
    rates = [r for r in rates if r]
    nsday = sorted(rates)[len(rates)//2] if rates else None
    return {"id": spec["id"], "pdb": spec["pdb"], "note": spec["note"],
            "atoms": atoms, "ns_per_day": nsday, "all_rates": rates,
            "setup_plus_bench_wall_s": round(wall, 1)}


def main():
    results = []
    for s in SYSTEMS:
        print(f"benchmarking {s['id']} ({s['pdb']}, padding {s['pad']} nm) …", flush=True)
        try:
            r = bench_one(s)
        except Exception as e:  # noqa: BLE001
            r = {"id": s["id"], "error": str(e)[:200]}
        print(f"   -> {r.get('atoms')} atoms, {r.get('ns_per_day')} ns/day "
              f"{r.get('error','')}", flush=True)
        results.append(r)
    OUT.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print(f"{'system':<20}{'atoms':>9}{'ns/day':>10}{'us/day':>10}   note")
    print("-" * 78)
    for r in results:
        if r.get("ns_per_day"):
            print(f"{r['id']:<20}{r['atoms']:>9,}{r['ns_per_day']:>10.1f}"
                  f"{r['ns_per_day']/1000:>10.3f}   {r['note'][:28]}")
        else:
            print(f"{r['id']:<20}{'—':>9}{'FAILED':>10}   {r.get('error','')[:40]}")
    print("=" * 78)
    return results


if __name__ == "__main__":
    main()
