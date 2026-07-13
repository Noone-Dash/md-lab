# MD Lab

A molecular-dynamics workbench built around one principle:

> **A language model must never be the source of a physical value.**

Simulations are described as data (a `Plan`), the physics is fixed by deterministic code,
and every claim in this README is a number that was measured on the machine, not an estimate.

Engines: **GROMACS 2026.2** (CUDA), **OpenMM 8.5**, **PySCF 2.13**, plus a NumPy
reaction–diffusion model. Hardware: NVIDIA **GB10** (Grace-Blackwell, unified memory).

---

## The architecture (and why)

```
request ──X──▶ Intent          (pure code: regex/lookup — NO model)
        └─T──▶ Plan_raw        (small LLM, JSON-Schema-constrained decoding)
(Plan_raw, Intent) ──E──▶ Plan (pure code: projection — OVERWRITES the model)
Plan ──D──▶ .mdp / topology    (pure code)  ──▶ GROMACS
```

`X`, `E`, `D` are pure functions. Let `F` be the set of plan fields with physical
consequence (16 of them). The system guarantees

```
∀ f ∈ F :  Plan[f] = g_f(Intent, defaults)      — independent of the model's output
```

**Residual model-decided physical fields: 0 / 16.** Verified adversarially: feeding a
maximally-wrong model output (`pdb=9ZZZ`, wrong force field, wrong water model, 500 K,
9 ns, 1 stage) produces a plan *identical on all of F* to the correct one. The model's
output on `F` is discarded by construction, not merely validated.

Two independent guards, because they catch different things:

| Guard | Answers | Implementation |
|---|---|---|
| **Physics validator** | "is this plan legal?" | 26 rules, `labkit/data/rules.json` |
| **Intent contract** | "is this what was asked for?" | `labkit/agent/intent.py` |

A plan with `salt = 0.0 M` is *legal physics* and *completely wrong* if you asked for
physiological salt. No physics checker will ever catch that — hence the intent contract.

Additional invariant: `mdp_emit` is the only module permitted to write a `.mdp` file, and
it **raises** on any key not declared in `labkit/data/ontology.json` (150 documented
parameters). An undocumented knob cannot reach a simulation.

---

## Measured performance (not vendor numbers)

MD cost is linear in atom count, so `atoms × ns/day` should be constant. Measured with
`-resethway` (excludes startup/PME tuning), median of 3, exclusive GPU:

| system | atoms | ns/day |
|---|---|---|
| ubiquitin | 22,399 | 670 |
| lysozyme (1.0 nm pad) | 27,242 | 618 |
| lysozyme (1.4 nm pad) | 37,580 | 295 |
| lysozyme (2.0 nm pad) | 55,976 | 197 |
| lysozyme (2.8 nm pad) | 90,324 | 142 |

```
ns/day(N) ≈ K / N        K = 1.28e7 atom·ns/day  (±25%: PME grid + protein/water ratio)
```

**What that buys** (50k-atom protein+ligand, 20% duty cycle ⇒ ~51 ns/day ⇒ 18.7 µs/year):

| protocol | cost | time at 20% duty |
|---|---|---|
| pose stability (3 × 100 ns) | 300 ns | 6 days |
| MM-GBSA rescoring | 25 ns | ~2 ligands/day |
| RBFE / FEP, lean (12λ × 5 ns × 2 legs) | 120 ns | 2.4 days/pair |
| RBFE, publication-grade (20λ × 10 ns × 2 × 3 reps) | 1.2 µs | 24 days/pair |
| Trp-cage folding (10k atoms, ~4 µs) | 4 µs | 16 days |
| absolute BFE | 5–10 µs | months |

Ligand unbinding, allostery and folding of anything real are **µs–ms** and are out of
reach on one GPU without enhanced sampling. This is stated up front because pretending
otherwise is how MD demos lie.

---

## Physics validation

`labkit/evals/` runs **real simulations** and grades them against independent references.

| benchmark | measured | reference |
|---|---|---|
| water density (SPC/E, NPT) | 989–998 kg/m³ | 997 (experiment) |
| water O–O first shell | 0.274 nm | 0.28 (neutron diffraction) |
| thermostat holds setpoint | 299.0 K | 300 K |
| QM water energy (HF/STO-3G) | −74.9659 Ha | −74.9659 (literature) |
| O–H bond length | 0.99 Å | 0.96 (experiment) |
| ubiquitin backbone RMSD | 0.10 nm | stays folded |
| liquid argon g(r) peak | 2.90 | structured |
| argon **gas** g(r) peak | 1.41 | `exp(ε/kT)` = **1.49** (analytic) |

The gas test initially *failed* (2.19). The simulation was right; the **metric** was wrong —
it was taking the max of a noisy 624-bin histogram and grabbing a single-bin spike. Fixed
the metric, not the threshold. That is what the suite is for.

**Known deficiency:** these are point estimates with **no error bars**. For a
time-correlated series, `Var(mean) = σ²/N · (2τ_int/Δt)`, so the naive SEM is too small by
`√(2τ_int/Δt)` — often 3–10×. Block averaging + equilibration detection is **not yet
implemented**, so "agrees with experiment" is currently not a defensible claim.

---

## Layout

```
labkit/
  plan/          schema · jsonschema (grammar) · resolve · mdp_emit · validate · defaults · cost
  agent/         intent (deterministic contract) · translate (constrained LLM) · tools · chat
  evals/         physics benchmarks · agent benchmark · hardware benchmark · metrics
  data/          ontology.json (150 params) · rules.json (26 rules) · benchmarks.json
  scheduler.py   single-node queue: systemd + cgroups v2, GPU serialisation, pause/resume
  engine.py      run pipeline → uniform run.json manifest
viewer/          Flask UI: plan builder · 3D viewer · monitor · evals · chat
uicheck/         headless-browser screenshot + console check (Playwright)
tests/           regression tests pinning every bug that actually happened
```

## Running it

```bash
source env.sh                       # GROMACS + venv
python viewer/app.py 5057           # UI

python -m labkit.evals.runner       # physics benchmarks (real simulations)
python -m labkit.evals.hw_bench     # measure ns/day on this machine
python tests/test_plan.py           # regression tests
```

The chat/translator runs **locally** via Ollama (no API key). Setup, however, does not
depend on it: the plan pipeline is deterministic with or without a model.

## Not done (stated explicitly)

- **Uncertainty quantification** — block averaging, τ_int, equilibration detection. Until
  this lands, every reported mean is a point estimate.
- **Ligand/small-molecule parameterization** (GAFF/OpenFF). Without it, drug-discovery
  workflows are blocked: you cannot simulate an arbitrary compound. Biggest functional gap.
- Cofactors: `4HHB` fails — amber99sb-ildn has no haem parameters.
- Free-energy protocols (FEP/TI, umbrella, metadynamics) — Plumed is compiled in, unused.
- Membrane/QM systems still go through legacy presets, not the Plan path.

---

## Adaptive GPU scheduling (opportunistic backfill)

MD throughput is linear in GPU time (`work = duty × K`) — there is no batching gain. So a
fixed "use 20% of the GPU" cap is strictly wasteful: it idles the other 80% whenever you
are not using the machine. The policy implemented instead:

> **Run long campaigns whenever the GPU is idle; yield instantly when it is wanted.**

Expected utilisation → `1 − your_usage`, far above any fixed cap.

Preemption is safe because it is a *checkpointed continuation*, not a restart. Verified
from GROMACS's own log during a live campaign:

```
Writing checkpoint, step 181200
Received the TERM signal, stopping within 100 steps    <- interactive job arrived
Started mdrun on rank 0 ... (4 s later)
Restarting from checkpoint, appending to previous log file
Writing checkpoint, step 413900                        <- continued, not restarted
```

GPU released in ~4 s. Positions, velocities, thermostat/barostat state and the RNG stream
are all restored, so the trajectory is continuous. Worst-case loss = work since the last
checkpoint (`-cpt 1`, i.e. one minute).

```python
SCHED.submit_backfill(plan, target_ns=500)   # runs for weeks, in the gaps
```

Progress (`done_ns / target_ns`) is published live from the checkpoint, so the scheduler
can observe and act on it.
