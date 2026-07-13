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
| water density (SPC/E, NPT) | **996.9 ± 2.0** kg/m³ | 998 (SPC/E lit.); experiment 997 |
| thermostat holds setpoint | **299.1 ± 1.1** K | 300 K |
| water O–O first shell | 0.274 nm | 0.28 (neutron diffraction) |
| QM water energy (HF/STO-3G) | −74.9659 Ha | −74.9659 (literature) |
| O–H bond length | 0.99 Å | 0.96 (experiment) |
| liquid argon g(r) peak | 2.91 | structured |
| argon **gas** g(r) peak | 1.42 | `exp(ε/kT)` = **1.49** (analytic) |
| ubiquitin backbone RMSD | 0.098 nm* | stays folded |

`±` is the autocorrelation-corrected 95% SEM (below). Entries without one are not time
averages — an RDF peak and a converged SCF energy have no τ_int, so a bar would be
meaningless. `*` marks a number the run was too short to put a defensible bar on: RMSD comes
from *trajectory* frames, which are budgeted for the viewer, not for statistics.

The density reference is the **water model's own** published value, not experiment —
reproducing experiment is a property of SPC/E, not of this code. What the benchmark tests is
whether *we* set the simulation up correctly. Its window used to be 975–1025 against
"997 (experiment)", which is ±25 wide: it accepted SPC/E (999.7 ± 1.1, measured) and TIP3P
(986.0 ± 1.3, measured) **equally**, though they are 8σ apart. It could not detect a run
using the wrong water model. It was passing for the wrong reason.

The gas test initially *failed* (2.19). The simulation was right; the **metric** was wrong —
it was taking the max of a noisy 624-bin histogram and grabbing a single-bin spike. Fixed
the metric, not the threshold. That is what the suite is for.

### Error bars (`labkit/uncertainty.py`)

These were point estimates. They no longer are.

For a correlated series, `Var(x̄)` is a double sum over **all sample pairs**, not a single
sum, which gives `SEM = σ·√(2·τ_int/N)` and an effective sample size `N_eff = N/(2·τ_int)`.
The textbook `σ/√N` assumes independent samples; consecutive MD frames are not, and it is
too small by `√(2·τ_int)` — 3–10× in practice.

`τ_int` is estimated **two independent ways** (Sokal automatic windowing; Flyvbjerg–
Petersen blocking) and cross-checked, because an estimator you have not checked against
another one is an assumption. Both are validated against an **AR(1) process whose τ_int is
known in closed form**, `τ = (1+φ)/(2(1−φ))` — ground truth, not a plausibility check:

| φ | τ_true | τ_estimated | error | naive SEM is too small by |
|---|---|---|---|---|
| 0.00 | 0.50 | 0.50 | 0.6% | 1.0× |
| 0.80 | 4.50 | 4.46 | 0.8% | 3.0× |
| 0.95 | 19.50 | 20.23 | 3.8% | 6.4× |

**The sampling bug this exposed.** `nstenergy` was derived from `TARGET_FRAMES` — the
*viewer's* frame budget. The rate at which we sampled thermodynamics was set by what makes
a 3D animation look smooth: ~120 samples, ~10 ps apart. That is coarser than the
correlation time of the observables, so `τ_int` was **unmeasurable** (it pinned to its
0.5-sample floor) and every error bar was silently unfalsifiable. Energy frames are ~100
bytes; trajectory frames are all-atom coordinates. They now have separate budgets.

Sampled properly (TIP3P, 3 nm box, NPT 300 K, 0.1 ps), the measured correlation times
reproduce the coupling constants set in the `.mdp`:

| observable | measured τ_int | `.mdp` coupling constant |
|---|---|---|
| Temperature | 0.10 ps | `tau-t = 0.1` |
| Density | 2.19 ps | `tau-p = 2.0` |
| Pressure | 0.09 ps | (instantaneous virial — fast, as expected) |

The density autocorrelation time **is** the barostat coupling time. Nothing was fitted.
That correspondence is what makes the machinery trustworthy rather than decorative.

Consequences: the eval runner reports a 95% CI and returns **INCONCLUSIVE** — not PASS —
when the CI straddles the acceptance boundary. A PASS whose error bar also covers FAIL is
not a pass; it is a run too short to decide. And `time_for_precision()` inverts the
relation: water density to ±1.0 kg/m³ needs 0.33 ns; to ±0.1 kg/m³ needs 33 ns. Cost goes
as `1/SEM²`, so a 10× tighter bar costs 100× the compute — "just run it longer" stops
working fast.

---

## Layout

```
labkit/
  plan/          schema · jsonschema (grammar) · resolve · mdp_emit · validate · defaults · cost
  agent/         intent (deterministic contract) · translate (constrained LLM) · tools · chat
  evals/         physics benchmarks · agent benchmark · hardware benchmark · metrics
  data/          ontology.json (150 params) · rules.json (26 rules) · benchmarks.json
  uncertainty.py τ_int (Sokal + blocking), N_eff, honest SEM, time-to-precision
  config.py      the ONE place the environment is resolved — no path is hardcoded anywhere else
  scheduler.py   single-node queue: systemd + cgroups v2, GPU serialisation, pause/resume
  engine.py      run pipeline → uniform run.json manifest
viewer/          Flask UI: plan builder · 3D viewer · monitor · evals · chat
uicheck/         headless-browser screenshot + console check (Playwright)
tests/           regression tests pinning every bug that actually happened
```

## Running it

```bash
python -m labkit.doctor             # preflight: what is present, what is missing, how to fix it
python viewer/app.py 5057           # UI

python -m labkit.evals.runner       # physics benchmarks (real simulations)
python -m labkit.evals.hw_bench     # measure ns/day on this machine
python -m labkit.uncertainty        # validate the error-bar estimators against AR(1)
python tests/test_plan.py           # regression tests
python tests/test_no_hardcoding.py  # no machine-specific constants anywhere but config.py
```

**Portability.** Nothing is hardcoded to the machine it was written on. GROMACS is
discovered via `$GMX_ROOT` → `gmx` on `PATH` (the `module load` case) → conventional
prefixes → a **loud, actionable failure**, never a silent fallback. Thread counts come from
the cpuset you were actually given (`SLURM_CPUS_PER_TASK`), memory limits from
`SLURM_MEM_PER_NODE` or the cgroup, and `-ntmpi` is emitted **only** for thread-MPI builds
(it aborts a library-MPI `gmx_mpi`). `tests/test_no_hardcoding.py` greps the tree so this
bug class cannot come back.

The chat/translator runs **locally** via Ollama (no API key). Setup, however, does not
depend on it: the plan pipeline is deterministic with or without a model.

## Not done (stated explicitly)

- **Automatic equilibration detection.** τ_int and error bars now exist, but the
  discarded-transient fraction is still a fixed `last_frac`, not detected per-run
  (Chodera's marginal-`N_eff` criterion).
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
