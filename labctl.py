#!/usr/bin/env python3
"""labctl — command-line front-end for the GROMACS lab.

Examples
--------
  ./labctl.py list
  ./labctl.py info lj_argon
  ./labctl.py run lj_argon --n_atoms 800 --temperature 90 --density 21
  ./labctl.py run water_box --box_nm 2.5 --nsteps 20000
  ./labctl.py runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from labkit import list_recipes, get_recipe, run_recipe, list_runs  # noqa: E402


def _print_progress(m):
    done = sum(1 for s in m["steps"] if s["status"] == "done")
    total = len(m["steps"]) or 1
    cur = next((s["name"] for s in m["steps"] if s["status"] == "running"), m["status"])
    sys.stdout.write(f"\r[{done}/{total}] {m['status']:>10} · {cur:<28}")
    sys.stdout.flush()


def cmd_list(_):
    for r in list_recipes():
        print(f"  {r['key']:<12} {r['name']:<32} [{r['category']}]  ~{r['est']}")


def cmd_info(a):
    r = get_recipe(a.recipe)
    print(f"{r.name}  ({r.key})\n{'-'*60}\n{r.description}\n")
    print("Parameters:")
    for p in r.params:
        rng = ""
        if p.options:
            rng = "  choices: " + ", ".join(map(str, p.options))
        elif p.min is not None:
            rng = f"  range: {p.min}..{p.max}"
        print(f"  --{p.name:<14} default={p.default!s:<8}{rng}")
        if p.help:
            print(f"      {p.help}")


def cmd_run(a):
    r = get_recipe(a.recipe)
    params = {}
    for p in r.params:
        val = getattr(a, p.name, None)
        if val is not None:
            params[p.name] = val
    print(f"Running {r.name} with {params or 'defaults'} ...")
    m = run_recipe(a.recipe, params, progress_cb=_print_progress)
    print()
    if m["status"] == "done":
        o = m.get("outputs", {})
        print(f"✓ done · run id: {m['id']}")
        print(f"  trajectory: {o.get('n_frames')} frames × {o.get('n_atoms')} atoms")
        print(f"  analyses:   {', '.join(x['name'] for x in m['analyses']) or 'none'}")
        print(f"  folder:     simulations/runs/{m['id']}")
    else:
        print(f"✗ {m['status']}: {m['error']}")
        sys.exit(1)


def cmd_runs(_):
    for r in list_runs():
        print(f"  {r['id']:<32} {r['status']:<10} {r['recipe_name']}")


def main():
    ap = argparse.ArgumentParser(description="GROMACS lab controller")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list recipes").set_defaults(func=cmd_list)
    sub.add_parser("runs", help="list past runs").set_defaults(func=cmd_runs)

    pi = sub.add_parser("info", help="show a recipe's parameters")
    pi.add_argument("recipe")
    pi.set_defaults(func=cmd_info)

    pr = sub.add_parser("run", help="run a recipe")
    pr.add_argument("recipe")
    # attach every recipe's params as optional flags
    seen = set()
    for r in list_recipes():
        for p in r["params"]:
            if p["name"] not in seen:
                pr.add_argument(f"--{p['name']}")
                seen.add(p["name"])
    pr.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
