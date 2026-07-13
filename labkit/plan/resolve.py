"""Plan -> ResolvedPlan. Pure: no I/O, no GROMACS.

Fills defaults, applies the force-field regime preset, and does the derivations an
agent should NOT have to do by hand (nsteps from physical time, output frequencies
from a target frame count, gen-vel/continuation from stage position).
"""

from __future__ import annotations

from .schema import Plan, Stage

TARGET_FRAMES = 100          # trajectory frames we aim to write per dynamics stage
MAX_ENERGY_FRAMES = 100_000   # ~10 MB of .edr even for a microsecond run, and
#                               enough samples that tau_int is actually resolvable.


def regime(system) -> str:
    return "martini" if str(system.forcefield).lower().startswith("martini") else "atomistic"


ATOMISTIC = {
    "cutoff-scheme": "Verlet", "nstlist": 20, "pbc": "xyz",
    "verlet-buffer-tolerance": 0.005,
    "coulombtype": "PME", "rcoulomb": 1.0, "fourierspacing": 0.12, "pme-order": 4,
    "vdwtype": "Cut-off", "vdw-modifier": "Potential-shift-verlet", "rvdw": 1.0,
    "DispCorr": "EnerPres",
    "constraints": "h-bonds", "constraint-algorithm": "LINCS",
    "lincs-order": 4, "lincs-iter": 1,
}

MARTINI = {
    "cutoff-scheme": "Verlet", "nstlist": 20, "pbc": "xyz",
    "verlet-buffer-tolerance": 0.005,
    "coulombtype": "reaction-field", "rcoulomb": 1.1,
    "epsilon_r": 15, "epsilon_rf": 0,
    "vdwtype": "cutoff", "vdw-modifier": "Potential-shift-verlet", "rvdw": 1.1,
    "constraints": "none",
}

DT_DEFAULT = {"atomistic": 0.002, "martini": 0.02}
EM_DEFAULT = {"atomistic": {"emtol": 1000.0, "emstep": 0.01},
              "martini": {"emtol": 100.0, "emstep": 0.02}}


def _tc_grps(system) -> str:
    if system.kind == "protein":
        return "Protein Non-Protein"
    if system.kind == "membrane":
        return "System"
    return "System"


def resolve(plan: Plan) -> dict:
    """-> {'name', 'system', 'regime', 'stages': [ {name,type,mdp,posres_fc_kj,...} ]}"""
    sysm = plan.system
    reg = regime(sysm)
    base = dict(MARTINI if reg == "martini" else ATOMISTIC)

    from .ontology import known as _known

    def _canon(d: dict) -> dict:
        """LLMs write tau_t; GROMACS wants tau-t. Accept both, emit one."""
        out = {}
        for k, v in d.items():
            if not _known(k) and _known(k.replace("_", "-")):
                k = k.replace("_", "-")
            out[k] = v
        return out

    resolved = []
    seen_dynamics = False
    for st in plan.stages:
        mdp = dict(base)
        p = _canon(dict(st.params or {}))

        # friendly aliases an LLM is likely to emit
        if "temperature" in p:
            p["ref-t"] = p.pop("temperature")
        if "pressure" in p:
            p["ref-p"] = p.pop("pressure")
        ensemble = str(p.pop("ensemble", "NVT" if st.type == "dynamics" else "NONE")).upper()

        dt = float(p.get("dt", DT_DEFAULT[reg]))

        if st.type == "minimize":
            mdp["integrator"] = p.pop("integrator", "steep")
            mdp["emtol"] = float(p.pop("emtol", EM_DEFAULT[reg]["emtol"]))
            mdp["emstep"] = float(p.pop("emstep", EM_DEFAULT[reg]["emstep"]))
            mdp["nsteps"] = int(st.max_steps)
            mdp["tcoupl"] = "no"
            mdp["pcoupl"] = "no"
            mdp["gen-vel"] = False
            # An EM stage has no temperature/pressure. The schema now REQUIRES a
            # params block on every stage, so temperature arrives here and would
            # otherwise leak 'ref-t' into the mdp -> grompp aborts (tcoupl=no + ref-t).
            for k in ("ref-t", "tau-t", "tc-grps", "ref-p", "tau-p", "pcoupltype",
                      "compressibility", "gen-temp", "gen-seed", "continuation",
                      "nsttcouple", "nstpcouple"):
                p.pop(k, None)
        else:
            seen_first = not seen_dynamics
            seen_dynamics = True
            mdp["integrator"] = p.pop("integrator", "md")
            mdp["dt"] = dt
            p.pop("dt", None)
            # nsteps is DERIVED from physical time — the agent never counts steps
            nsteps = int(round(float(st.sim_time_ns) * 1000.0 / dt))
            mdp["nsteps"] = nsteps

            # thermostat — ref-t/tau-t must be ONE VALUE PER GROUP or grompp aborts
            mdp["tcoupl"] = p.pop("tcoupl", "V-rescale")
            grps = p.pop("tc-grps", _tc_grps(sysm))
            ngrp = len(str(grps).split())
            mdp["tc-grps"] = grps
            reft = float(p.pop("ref-t", 300.0))
            taut = float(p.pop("tau-t", 1.0 if reg == "martini" else 0.1))
            mdp["ref-t"] = [reft] * ngrp
            mdp["tau-t"] = [taut] * ngrp

            # barostat
            if ensemble == "NPT":
                mdp["pcoupl"] = p.pop("pcoupl", "C-rescale")
                ptype = p.pop("pcoupltype",
                              "semiisotropic" if sysm.kind == "membrane" else "isotropic")
                mdp["pcoupltype"] = ptype
                mdp["tau-p"] = float(p.pop("tau-p", 12.0 if reg == "martini" else 2.0))
                refp = float(p.pop("ref-p", 1.0))
                comp = float(p.pop("compressibility",
                                   4.5e-5 if reg == "atomistic" else 3e-4))
                n = 2 if ptype == "semiisotropic" else 1
                mdp["ref-p"] = [refp] * n if n > 1 else refp
                mdp["compressibility"] = [comp] * n if n > 1 else comp
            else:
                mdp["pcoupl"] = "no"
                p.pop("pcoupl", None); p.pop("pcoupltype", None)
                p.pop("tau-p", None); p.pop("ref-p", None); p.pop("compressibility", None)

            # velocities: generate once, then continue from the previous stage
            gen = bool(p.pop("gen-vel", seen_first))
            mdp["gen-vel"] = gen
            mdp["continuation"] = not gen
            if gen:
                mdp["gen-temp"] = reft
                mdp["gen-seed"] = int(p.pop("gen-seed", -1))
            else:
                p.pop("gen-seed", None)

            # TRAJECTORY frames: budgeted, because each one is all-atom coordinates.
            # This number exists to make the viewer smooth.
            nst = max(100, nsteps // TARGET_FRAMES) if nsteps else 100
            mdp["nstxout-compressed"] = int(p.pop("nstxout-compressed", nst))
            mdp["compressed-x-precision"] = 1000

            # ENERGY frames: a DIFFERENT budget, because they are ~100 bytes each and
            # they are what every reported mean is computed from. Tying nstenergy to
            # the viewer's frame count (which is what this used to do) meant we sampled
            # thermodynamics ~120 times per run, at intervals of ~10 ps — far coarser
            # than the correlation time of the observables. tau_int was then
            # unmeasurable (it pinned to its floor of 0.5) and every error bar was
            # unfalsifiable. Sampling density must be set by the statistics, not by the
            # animation.
            nste = max(50, nsteps // MAX_ENERGY_FRAMES) if nsteps else 50
            mdp["nstenergy"] = int(p.pop("nstenergy", nste))
            mdp["nstlog"] = int(p.pop("nstlog", nst))

        # position restraints -> the -DPOSRES define + refcoord scaling
        if st.posres_fc_kj and st.posres_fc_kj > 0:
            mdp["define"] = "-DPOSRES"
            if mdp.get("pcoupl", "no") != "no":
                mdp["refcoord-scaling"] = "com"

        # whatever the agent set that we did NOT consume above is a raw mdp override
        for k, v in p.items():
            mdp[k] = v

        resolved.append({
            "name": st.name, "type": st.type, "mdp": mdp,
            "posres_fc_kj": st.posres_fc_kj,
            "sim_time_ns": st.sim_time_ns,
        })

    return {"name": plan.name, "system": sysm, "regime": reg,
            "stages": resolved, "analyses": list(plan.analyses)}
