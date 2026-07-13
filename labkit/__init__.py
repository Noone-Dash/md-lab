"""labkit — a small engine for building, running, and analysing GROMACS simulations.

The lab is organised around *recipes* (parametric simulation definitions) and an
*engine* that materialises a recipe into a run directory, executes the GROMACS
pipeline, and extracts analysis data for the viewer UI.
"""

from .gmx import GMX_ROOT, gmx, gmx_version
from .recipes import REGISTRY, get_recipe, list_recipes
from .engine import run_recipe, load_run, list_runs

__all__ = [
    "GMX_ROOT",
    "gmx",
    "gmx_version",
    "REGISTRY",
    "get_recipe",
    "list_recipes",
    "run_recipe",
    "load_run",
    "list_runs",
]
