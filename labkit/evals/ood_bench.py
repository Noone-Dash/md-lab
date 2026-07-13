"""Where does the deterministic contract BREAK, and does the LLM actually rescue it?

THE PROBLEM WITH THE MAIN BENCHMARK
-----------------------------------
translate_bench measures the request space I wrote the regexes FOR. That is teaching to
the test. On it, a null model produces a .mdp identical to gpt-oss:20b on all 10 cases --
which proves the contract is airtight on its own turf and says nothing at all about
anything else.

This is the honest complement: phrasings deliberately chosen to be OUT of the regexes'
distribution. "blood heat" for 310 K. "half a nanosecond". "a cube 5 nm on a side".
Semantic identification ("the enzyme that digests starch") instead of a name.

Three columns, and the third is the only one that can justify running a model at all:

    INTENT   the deterministic contract parses it            -> the LLM is irrelevant
    LLM      the contract misses it, the model gets it right -> the LLM EARNS its place
    NEITHER  both miss -> a silent DEFAULT is substituted    -> the real failure mode

NEITHER is the dangerous column. A request that nothing parses does not error; it gets a
plausible default (300 K, NPT, 0.1 ns) and runs, and the user is never told that the thing
they asked for was quietly dropped. Every case here that lands in NEITHER is a coverage bug
with a known address.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from labkit.agent.intent import extract, resolve_structure   # noqa: E402
from labkit.agent.translate import translate                 # noqa: E402

# (request, field, expected, how to read it out of a raw model plan)
def _t(p):
    dyn = [s for s in (p.get("stages") or []) if s.get("type") == "dynamics"]
    return (dyn[-1].get("params") or {}).get("temperature") if dyn else None


def _ns(p):
    dyn = [s for s in (p.get("stages") or []) if s.get("type") == "dynamics"]
    return dyn[-1].get("sim_time_ns") if dyn else None


def _ens(p):
    dyn = [s for s in (p.get("stages") or []) if s.get("type") == "dynamics"]
    return (dyn[-1].get("params") or {}).get("ensemble") if dyn else None


def _sysf(f):
    return lambda p: (p.get("system") or {}).get(f)


CASES = [
    # --- temperature, phrased the way people actually talk ------------------------
    ("Simulate lysozyme at blood heat for 50 ps",
     "temperature_K", 310.0, _t),
    ("Simulate lysozyme, warm it up to 350 kelvin, 50 ps",
     "temperature_K", 350.0, _t),
    ("Simulate lysozyme just below the boiling point of water, 50 ps",
     "temperature_K", 370.0, _t),          # ~370 K; anything 363-373 is defensible

    # --- duration, spelled out ----------------------------------------------------
    ("Simulate lysozyme at 300 K for half a nanosecond",
     "production_ns", 0.5, _ns),
    ("Simulate lysozyme at 300 K for two hundred picoseconds",
     "production_ns", 0.2, _ns),

    # --- box, phrased geometrically ----------------------------------------------
    ("A cube of water 5 nanometres on a side at 300 K, 20 ps",
     "box_size_nm", 5.0, _sysf("box_size_nm")),
    ("Water at 300 K for 20 ps, leave 15 angstroms around the solute",
     "box_padding_nm", 1.5, _sysf("box_padding_nm")),

    # --- ensemble, phrased physically --------------------------------------------
    ("Simulate water at 300 K for 20 ps, keep the volume fixed",
     "ensemble", "NVT", _ens),
    ("Simulate water at 300 K for 20 ps, let the box breathe at 1 bar",
     "ensemble", "NPT", _ens),

    # --- salt ---------------------------------------------------------------------
    ("Simulate lysozyme at 300 K for 50 ps in 0.9% saline",
     "salt_M", 0.15, _sysf("salt_conc_M")),   # 0.9% NaCl IS physiological ~0.154 M
    ("Simulate lysozyme at 300 K for 50 ps with no counterions at all",
     "salt_M", 0.0, _sysf("salt_conc_M")),

    # --- the molecule, identified SEMANTICALLY rather than named ------------------
    ("Simulate the enzyme that digests starch, at 300 K for 50 ps",
     "pdb_id", "AMYLASE", _sysf("pdb_id")),   # graded loosely: must be an amylase
    ("Simulate the protein in PDB entry 4HHB at 300 K",
     "pdb_id", "4HHB", _sysf("pdb_id")),
]


def _close(got, want, field):
    if got is None:
        return False
    if field == "pdb_id" and want == "AMYLASE":
        from labkit.structures import _title
        t = _title(str(got)) or ""
        return "amylase" in t
    if isinstance(want, float):
        try:
            if field == "temperature_K" and want == 370.0:
                return 363.0 <= float(got) <= 373.15      # "just below boiling"
            tol = max(0.02 * abs(want), 1e-6)
            return abs(float(got) - float(want)) <= tol
        except (TypeError, ValueError):
            return False
    return str(got).lower() == str(want).lower()


def main(model="gpt-oss:20b"):
    print(f"OUT-OF-DISTRIBUTION requests — phrasings the regexes were NOT written for\n"
          f"model under test: {model}\n")
    print(f"  {'request':<56}{'INTENT':>8}{'LLM':>6}   verdict")
    print("  " + "-" * 84)
    tally = {"INTENT": 0, "LLM": 0, "NEITHER": 0}
    misses = []
    for req, field, want, getter in CASES:
        it = extract(req)
        if field == "pdb_id":
            resolve_structure(it, req)
        got_i = getattr(it, field, None)
        by_intent = _close(got_i, want, field)

        # Grade the FINAL plan -- what actually reaches GROMACS -- not the raw model
        # output. Grading raw output was itself misleading: it credited the model for
        # answers that unconditional enforcement then threw away.
        from labkit.agent.intent import plan_from_request
        r = plan_from_request(req, model=model)
        fin = r["plan"]
        got_f = getter(fin) if fin else None
        prov = (fin or {}).get("_provenance", {})
        by_final = _close(got_f, want, field)
        src = prov.get({"temperature_K": "temperature", "production_ns": "sim_time_ns",
                        "ensemble": "ensemble"}.get(field, field), "-")

        if by_intent and by_final:
            v, tag = "INTENT", "contract parsed it — the model is irrelevant"
        elif by_final:
            v, tag = "LLM", f"contract missed it; the MODEL got it right [{src}]"
        else:
            v, tag = "NEITHER", f"both missed -> silent default (final={got_f} [{src}])"
            misses.append((req, field, want, got_i, got_f))
        by_llm = by_final and not by_intent
        tally[v] += 1
        print(f"  {req[:54]:<56}{'ok' if by_intent else '—':>8}"
              f"{('ok' if by_llm else '—') if not by_intent else '':>6}   {v:<8} {tag}")

    n = len(CASES)
    print("  " + "-" * 84)
    print(f"  covered by the deterministic contract : {tally['INTENT']}/{n}")
    print(f"  rescued by the LLM (it earns its place): {tally['LLM']}/{n}")
    print(f"  SILENTLY DEFAULTED (the failure mode)  : {tally['NEITHER']}/{n}")
    if misses:
        print("\n  Every line below is a request whose meaning was DROPPED without a word:\n")
        for req, field, want, gi, gl in misses:
            print(f"    {field:<16} wanted {str(want):<10} got intent={gi} llm={gl}")
            print(f"      {req}")
    return tally


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gpt-oss:20b")
