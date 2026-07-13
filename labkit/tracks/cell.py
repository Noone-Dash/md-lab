"""Reaction–diffusion inside a cell — a real (if simplified) particle model.

Particles of species A and B diffuse by Brownian motion inside a spherical
'cell' and react on contact:  A + B -> C  (the B partner is marked spent).
This is the same *class* of spatial-stochastic model as Smoldyn / ReaDDy /
Lattice Microbes (which have no ARM build here), implemented in NumPy so the
track is genuinely live rather than a canned example. Labelled mode='model'
so the UI is honest about it being a teaching-grade engine.
"""

from __future__ import annotations

import numpy as np

from ..recipes import Param
from ..runbase import Run

# species -> (element for 3Dmol colour, display name)
SPECIES = {0: ("N", "A"), 1: ("O", "B"), 2: ("S", "C"), 3: ("C", "spent")}


class CellRD:
    key = "cell_rd"
    name = "Reaction–diffusion in a cell"
    category = "Cell-scale"
    track = "cell"
    engine = "NumPy particle model"
    mode = "model"                    # honest label: simplified, swap for Smoldyn/ReaDDy later
    needs_gpu = False
    description = ("Particles diffuse inside a spherical cell and react on contact "
                  "(A + B → C). Spatial stochastic reaction–diffusion — the same "
                  "class of model as Smoldyn / Lattice Microbes used for whole-cell "
                  "simulations, here in NumPy so it actually runs.")
    est = "seconds"

    params = [
        Param("n_a", "Molecules of A (blue)", "int", 150, 20, 500, 10),
        Param("n_b", "Molecules of B (red)", "int", 150, 20, 500, 10),
        Param("mobility", "Diffusion speed (Å/step)", "float", 1.6, 0.2, 4.0, 0.1,
              help="How far a molecule jumps per step — bigger = faster mixing."),
        Param("react_prob", "Reaction probability on contact", "float", 1.0, 0.05, 1.0, 0.05),
        Param("cell_radius", "Cell radius (Å)", "float", 42.0, 20.0, 80.0, 2.0),
        Param("steps", "Simulation steps", "int", 1500, 200, 6000, 100),
    ]

    def meta(self):
        return {"key": self.key, "name": self.name, "category": self.category,
                "track": self.track, "engine": self.engine, "mode": self.mode,
                "needs_gpu": self.needs_gpu, "description": self.description,
                "est": self.est, "params": [p.as_dict() for p in self.params]}

    def run(self, params, run_id=None, progress_cb=None):
        p = {q.name: q.default for q in self.params}
        for q in self.params:
            if q.name in params and params[q.name] != "":
                p[q.name] = (int(float(params[q.name])) if q.type == "int"
                             else float(params[q.name]))

        run = Run(self.key, self.name, p, track=self.track, engine=self.engine,
                  mode=self.mode, category=self.category,
                  step_names=["seed molecules", "diffuse & react", "record kinetics"],
                  progress_cb=progress_cb, run_id=run_id)
        try:
            rng = np.random.default_rng(1234)
            na, nb = p["n_a"], p["n_b"]
            R = p["cell_radius"]
            sigma = p["mobility"]
            rreact = 3.0
            n = na + nb

            run.step(0, "running")
            # seed uniformly inside the sphere
            def sample_sphere(k):
                v = rng.normal(size=(k, 3))
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                r = R * rng.random(k) ** (1 / 3)
                return v * r[:, None]
            pos = np.vstack([sample_sphere(na), sample_sphere(nb)])
            species = np.array([0] * na + [1] * nb)
            run.step(0, "done")

            # simulate, recording ~60 frames
            run.step(1, "running")
            steps = p["steps"]
            n_frames = 60
            every = max(1, steps // n_frames)
            frames, counts_t = [], []
            for s in range(steps):
                pos = pos + rng.normal(scale=sigma, size=pos.shape)
                # reflect back into the sphere
                d = np.linalg.norm(pos, axis=1)
                out = d > R
                if out.any():
                    pos[out] *= (R / d[out])[:, None]

                # reactions: active A vs active B within rreact
                ai = np.where(species == 0)[0]
                bi = np.where(species == 1)[0]
                if len(ai) and len(bi):
                    diff = pos[ai][:, None, :] - pos[bi][None, :, :]
                    dist = np.linalg.norm(diff, axis=2)
                    hits = np.argwhere(dist < rreact)
                    used_a, used_b = set(), set()
                    for a_idx, b_idx in hits:
                        a, b = ai[a_idx], bi[b_idx]
                        if a in used_a or b in used_b:
                            continue
                        if rng.random() <= p["react_prob"]:
                            species[a] = 2      # becomes C
                            species[b] = 3      # spent partner
                            used_a.add(a); used_b.add(b)

                if s % every == 0 or s == steps - 1:
                    frames.append(pos.copy())
                    counts_t.append([int((species == 0).sum()),
                                     int((species == 1).sum()),
                                     int((species == 2).sum())])
            run.step(1, "done")

            # trajectory + kinetics
            run.step(2, "running")
            symbols = [SPECIES[int(c)][0] for c in species]
            names = [SPECIES[int(c)][1] for c in species]
            run.set_traj(frames, symbols, names=names,
                         resnames=names, box=[2 * R + 20] * 3)
            xs = list(range(len(counts_t)))
            series = [[c[k] for c in counts_t] for k in range(3)]
            run.set_energy(xs, series, ["A", "B", "C"],
                           xaxis="frame", yaxis="molecule count")
            run.add_analysis("kinetics", "Reaction kinetics  A + B → C",
                             xs, series, legends=["A (reactant)", "B (reactant)", "C (product)"],
                             xaxis="frame", yaxis="count",
                             help="Second-order, diffusion-limited: A and B decay as C rises.")
            run.step(2, "done")
            return run.finish("done")
        except Exception as e:  # noqa: BLE001
            import traceback
            (run.dir / "error.txt").write_text(traceback.format_exc())
            return run.finish("error", str(e))
