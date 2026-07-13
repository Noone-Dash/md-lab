"""Generators for GROMACS ``.mdp`` parameter files.

Rather than shipping static .mdp files, we build them from parameters so the UI
can expose temperature / length / ensemble as sliders and have them flow through.
"""

from __future__ import annotations


def _fmt(d: dict) -> str:
    width = max(len(k) for k in d)
    return "\n".join(f"{k.ljust(width)} = {v}" for k, v in d.items()) + "\n"


def minim_mdp(emtol: float = 1000.0, nsteps: int = 50000, rvdw: float = 1.0,
              rcoulomb: float = 1.0, coulombtype: str = "PME") -> str:
    """Steepest-descent energy minimisation."""
    return _fmt({
        "; Energy minimisation": "",
        "integrator": "steep",
        "emtol": emtol,
        "emstep": 0.01,
        "nsteps": nsteps,
        "nstlist": 10,
        "cutoff-scheme": "Verlet",
        "coulombtype": coulombtype,
        "rcoulomb": rcoulomb,
        "vdwtype": "Cut-off",
        "rvdw": rvdw,
        "rlist": max(rvdw, rcoulomb),
        "pbc": "xyz",
    })


MAX_ENERGY_FRAMES = 100_000   # ~10 MB of .edr even for a microsecond run, and enough
#                               samples that tau_int is actually resolvable.


def md_mdp(*, nsteps: int, dt: float, temperature: float, ensemble: str = "NVT",
           coulombtype: str = "PME", rvdw: float = 1.0, rcoulomb: float = 1.0,
           tc_grps: str = "System", constraints: str = "none",
           nstxout: int = 1000, gen_vel: bool = True, tau_t: float = 0.5,
           pcoupl: str = "C-rescale", ref_p: float = 1.0,
           compressibility: float = 4.5e-5) -> str:
    """A general MD run (NVT or NPT) with configurable thermostat/barostat."""
    p = {
        "; MD run": "",
        "integrator": "md",
        "dt": dt,
        "nsteps": nsteps,
        "; output control": "",
        "nstxout-compressed": nstxout,     # .xtc trajectory for the viewer
        "compressed-x-precision": 1000,
        # ENERGY sampling gets its OWN budget, not nstxout//2. Trajectory frames are
        # all-atom coordinates and are budgeted for the VIEWER; energy frames are ~100
        # bytes and are what every reported mean and error bar is computed from. Tying
        # them together meant the physics evals sampled thermodynamics ~50 times per run
        # -- far coarser than the correlation time -- so tau_int was unmeasurable and
        # uncertainty.stats() correctly REFUSED to put an error bar on any of them.
        # (Fixed in the plan path first; this is the legacy recipe path, same bug.)
        "nstenergy": max(10, nsteps // MAX_ENERGY_FRAMES) if nsteps else 100,
        "nstlog": max(100, nstxout // 2),
        "; neighbour search": "",
        "cutoff-scheme": "Verlet",
        "nstlist": 20,
        "pbc": "xyz",
        "; electrostatics & vdw": "",
        "coulombtype": coulombtype,
        "rcoulomb": rcoulomb,
        "vdwtype": "Cut-off",
        "rvdw": rvdw,
        "rlist": max(rvdw, rcoulomb),
        "DispCorr": "EnerPres",
        "; constraints": "",
        "constraints": constraints,
        "constraint-algorithm": "lincs",
        "; temperature coupling": "",
        "tcoupl": "V-rescale",
        "tc-grps": tc_grps,
        "tau-t": " ".join([str(tau_t)] * len(tc_grps.split())),
        "ref-t": " ".join([str(temperature)] * len(tc_grps.split())),
    }
    if ensemble.upper() == "NPT":
        p.update({
            "; pressure coupling": "",
            "pcoupl": pcoupl,
            "pcoupltype": "isotropic",
            "tau-p": 2.0,
            "ref-p": ref_p,
            "compressibility": compressibility,
        })
    else:
        p["pcoupl"] = "no"
    p["; velocity generation"] = ""
    p["gen-vel"] = "yes" if gen_vel else "no"
    if gen_vel:
        p["gen-temp"] = temperature
        p["gen-seed"] = -1
    return _fmt(p)


# --------------------------------------------------------------------------- #
# Martini coarse-grained settings are quite different from atomistic MD:
# reaction-field electrostatics with epsilon_r = 15, shifted LJ, big timestep.
# --------------------------------------------------------------------------- #
def martini_em_mdp() -> str:
    return _fmt({
        "; Martini energy minimisation": "",
        "integrator": "steep",
        "emtol": 100.0,
        "emstep": 0.02,
        "nsteps": 5000,
        "nstlist": 20,
        "cutoff-scheme": "Verlet",
        "coulombtype": "reaction-field",
        "rcoulomb": 1.1,
        "epsilon_r": 15,
        "epsilon_rf": 0,
        "vdwtype": "cutoff",
        "vdw-modifier": "Potential-shift-verlet",
        "rvdw": 1.1,
        "verlet-buffer-tolerance": 0.005,
        "pbc": "xyz",
    })


def martini_md_mdp(*, nsteps: int, dt: float, temperature: float,
                   nstxout: int = 1000, pressure: str = "semiisotropic") -> str:
    p = {
        "; Martini MD": "",
        "integrator": "md",
        "dt": dt,
        "nsteps": nsteps,
        "nstxout-compressed": nstxout,
        "compressed-x-precision": 1000,
        "nstenergy": max(100, nstxout // 2),
        "nstlog": max(100, nstxout // 2),
        "cutoff-scheme": "Verlet",
        "nstlist": 20,
        "pbc": "xyz",
        "coulombtype": "reaction-field",
        "rcoulomb": 1.1,
        "epsilon_r": 15,
        "epsilon_rf": 0,
        "vdwtype": "cutoff",
        "vdw-modifier": "Potential-shift-verlet",
        "rvdw": 1.1,
        "verlet-buffer-tolerance": 0.005,
        "tcoupl": "v-rescale",
        "tc-grps": "System",
        "tau-t": 1.0,
        "ref-t": temperature,
        "gen-vel": "yes",
        "gen-temp": temperature,
        "gen-seed": -1,
        "constraints": "none",
    }
    if pressure == "semiisotropic":     # membranes: couple bilayer plane and normal separately
        p.update({
            "pcoupl": "c-rescale", "pcoupltype": "semiisotropic",
            "tau-p": 12.0, "compressibility": "3e-4 3e-4",
            "ref-p": "1.0 1.0",
        })
    elif pressure == "isotropic":
        p.update({
            "pcoupl": "c-rescale", "pcoupltype": "isotropic",
            "tau-p": 12.0, "compressibility": "3e-4", "ref-p": "1.0",
        })
    else:
        p["pcoupl"] = "no"
    return _fmt(p)
