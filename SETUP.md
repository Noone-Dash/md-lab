# Setup / reproduction

Everything below was executed on the machine this repo was developed on
(NVIDIA GB10, aarch64, Ubuntu). Steps are ordered; nothing is implied.

## 1. External prerequisite: GROMACS

Not vendored (it is a large C++ build). Any GROMACS >= 2023 with GPU support works;
this repo was developed against **2026.2, CUDA**.

```bash
# expected at $GMX_ROOT; override with the env var if yours lives elsewhere
export GMX_ROOT=/path/to/gromacs
source "$GMX_ROOT/bin/GMXRC"
gmx --version          # must report GPU support: CUDA
```

`labkit/gmx.py` sources GMXRC for every call, so nothing else needs to be on PATH.

## 2. Python environment

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## 3. External assets (force fields)

The Martini 3 force field is not vendored (size/licence). Fetch it:

```bash
./scripts/fetch_assets.sh
```

## 4. Optional: the local LLM (chat + NL translation)

The **deterministic plan pipeline does not need a model at all.** A model is only
used to parse free-text requests and to narrate results.

```bash
ollama serve &
ollama pull gpt-oss:20b      # tool-calling chat backend
ollama pull qwen3:8b         # smaller/faster translator
```

Set `MDLAB_LOCAL_MODEL` / `MDLAB_TRANSLATOR` to choose. If `ANTHROPIC_API_KEY` is set,
the cloud backend is used instead (requires `pip install anthropic`).

## 5. Run

```bash
source env.sh
python viewer/app.py 5057            # UI on http://127.0.0.1:5057
```

## 6. Reproduce every claim in the README

| claim | command | runtime |
|---|---|---|
| physics benchmarks (8/8 vs experiment/literature) | `python -m labkit.evals.runner` | ~1 min |
| `ns/day = 1.28e7 / N_atoms` on your GPU | `python -m labkit.evals.hw_bench` | ~30 min |
| plan-layer regression tests | `python tests/test_plan.py` | seconds |
| the UI actually renders (no blank canvases) | `cd uicheck && npm i playwright && npx playwright install chromium && node sweep.js` | ~3 min |
| LLM model comparison (Wilson intervals) | `python -m labkit.evals.agent_bench gpt-oss:20b qwen3:8b -k3` | ~30 min |

`hw_bench` numbers are hardware-specific — on a different GPU you get a different `K`,
which is the point: the cost model calibrates itself from measurement.

## Notes on determinism

- The plan pipeline (`intent -> translate -> enforce -> resolve -> mdp_emit`) is a pure
  function of the request at temperature 0. Verified: 5 identical runs -> 1 distinct plan.
- GROMACS itself is **not** bitwise reproducible on GPU across runs (non-deterministic
  reduction order). Trajectories differ; ensemble averages do not. Setup is deterministic;
  dynamics is not, and no MD code claims otherwise.
