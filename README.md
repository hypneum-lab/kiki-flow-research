# kiki-flow-research

**A Wasserstein gradient flow engine for LLM consolidation, grounded in the
Levelt–Baddeley model of language production.**

Upstream numerical engine of the *dreamOfkiki* research program, part of
[Hypneum Lab](https://github.com/hypneum-lab). Author: Clément Saillant
(L'Electron Rare).

[![CI](https://github.com/hypneum-lab/kiki-flow-research/actions/workflows/ci.yml/badge.svg)](https://github.com/hypneum-lab/kiki-flow-research/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## What is this?

`kiki-flow-research` models the state of a large language model as four
psycholinguistic activation densities (phono / lex / syntax / sem) that
evolve under a Wasserstein-regularized gradient flow. The implementation
solves the JKO scheme numerically on commodity Apple Silicon, extracts
a compact neural surrogate for streaming inference, and integrates with
existing MoE-LoRA routers through an advisory-only callback.

See the [design spec](docs/superpowers/specs/2026-04-17-kiki-flow-core-design.md)
for the full formalism and the [draft paper](paper/main.pdf) for the
quantitative results over five seeds.

## Three parallel tracks

| Track | Purpose | Performance | Status |
|---|---|---|---|
| `track1_perf` | Nightly offline consolidation over the LoRA stack tensor (40 hybrid species). | ~30 min on Studio M3 Ultra (host powered down since 2026-10 ; not re-run since). | tag `track1-v0.1` |
| `track2_paper` | N-particle Langevin × JKO entropic-regularized proxy for paper figures (4 Levelt-Baddeley species). | 2 min 31 s fast (GrosMac M5, GPU); 12 min 35 s to 12 min 45 s rigorous with the corrected solver over the three 2026-10-24 reruns (M1 Max, CPU). The old 84 min timed the defective prox, see `ERRATUM.md`. | tag `track2-v0.1` |
| `track3_deploy` | Pure-NumPy streaming surrogate, MiniLM query encoder, advisory-only routing. | p50 = 0.04 ms on GrosMac M5. | tag `track3-v0.1` |

T1 runs are gated by `DriftTrigger`: it fires when the new-concept rate in
the last 24 h of Aeon episodes (against `known_concepts` in
`bench/runs/T1/latest.json`) exceeds `threshold_on` = 0.07, then stays
disarmed until the rate drops below `threshold_off` = 0.05 (hysteresis, no
cooldown). These defaults come from a synthetic sweep (about 6 fires per
year instead of 121, recall 0.99) and should be recalibrated on a recorded
Aeon trace. A per-tick cron must pass `state_path` to persist the armed
flag; `threshold_off=None` restores the old memoryless trigger. Details in
`kiki_flow_core/track1_perf/CLAUDE.md`; sweep in
`scripts/drift_trigger_sweep.py`.

All three tracks share `kiki_flow_core/` (state, master equation,
Wasserstein ops, species, modules, hooks, telemetry). Strict ruff +
mypy; golden NPZ regressions in `tests/golden/`. Test count and
coverage are whatever CI measures (`uv run pytest`), not a fixed
number: on the `main` run of 2026-10-25 (`175551c`, GitHub Actions run
40198407183) pytest reported 333 tests collected (332 passed,
1 deselected) and 87 % line coverage of `kiki_flow_core`. The CI badge
above tracks the current state.

## Install

```bash
git clone https://github.com/hypneum-lab/kiki-flow-research.git
cd kiki-flow-research
uv sync --all-extras
uv run pre-commit install
```

Tested on macOS 26.3 (arm64) with Python 3.14, MLX 0.31, JAX 0.10,
POT 0.9. Other platforms may work but have not been validated.

## Reproduce in one command

```bash
git clone https://github.com/hypneum-lab/kiki-flow-research.git && cd kiki-flow-research
uv sync --locked --all-extras   # Linux x86_64 only: then `uv pip install mlx-cpu==0.31.1` (the lock ships the Metal backend only)
PYTHONPATH=. uv run python -c "from pathlib import Path; from kiki_flow_core.track2_paper.paper_run import run_paper; \
  run_paper(seeds=[0,1,2,3,4], out_dir=Path('paper_repro'), make_all_figures=True)"   # CPU-only pilot: seeds=[0], use_mlx=False
```

`out_dir='paper_repro'` keeps the committed `paper/` artefacts intact (writing to `paper/` overwrites `stats.json`, `figures/` and `trajectories/`).
Expected output: `paper_repro/stats.json` with `"n_seeds": 5`, every `per_seed[*].n_slow_completed == 100` and
`aggregate.<species>.entropy_bits_mean ≈ 5.99998` (flat potentials, near-maximal 6-bit entropy);
`paper_repro/figures/fig1_phase_seed{0..4}`, `fig2_f_decay`, `fig3_turing` (+ `fig4`/`fig5` placeholders);
`paper_repro/trajectories/seed{0..4}.safetensors` (MLX `seed0` sha256 `5a08487f…3493b`, same as the committed `paper/trajectories/seed0.safetensors`).
Duration: 2 min 31 s (GrosMac M5, GPU) or 3 min 37 s (M1 Max, MLX on CPU) for 5 seeds; Linux x86-64 CPU with the
NumPy simulator: ~6 min per seed (5 min 57 s measured for `seeds=[0]` on a Xeon X5680), bit-identical across runs.
Full reproduction (ε-sweep, CL benchmarks, dense sweep, fast-vs-rigorous KL): see "Reproduce the paper" below and `.github/workflows/reproducibility.yml`; rigorous path (`use_w2_prox=True`): 12 min 42 s on M1 Max (`PERFORMANCE.md`).

## Reproduce the paper

The fast-path figures and statistics are produced by the scripts
below; each is deterministic under the given seeds. The rigorous-path
reruns and the regenerated Fig. 1 (per-species moments, both paths)
and Fig. 4 (ε as Sinkhorn regularisation only) are driven by the
erratum scripts listed in `ERRATUM.md` §7, §9 and §10
(`experiments/erratum_reruns_2026_10_24.py`,
`experiments/erratum_reg_reruns_2026_10_24.py`,
`experiments/erratum_maxiter_reruns_2026_10_25.py`); they write to new
directories and leave the historical outputs untouched.

```bash
# Figures 2-3: F-decay + Turing profiles, fast-path stats (5 seeds, ~2 min 31 s on M5)
PYTHONPATH=. uv run python -c "
from pathlib import Path
from kiki_flow_core.track2_paper.paper_run import run_paper
run_paper(seeds=[0,1,2,3,4], out_dir=Path('paper'), make_all_figures=True)
"

# Historical epsilon sweep (conflates epsilon with the prox step; superseded, ERRATUM.md §9)
PYTHONPATH=. uv run python scripts/epsilon_sweep.py

# Figure 5: measured continual-learning proxy (5 seeds)
PYTHONPATH=. uv run python scripts/cl_benchmark.py

# Hyperparameter sweeps, fast path: 27-config coarse grid (paper/hyperparam_sweep.json)
# and 72-config dense grid (paper/hyperparam_sweep_dense.json, the specialization finding)
PYTHONPATH=. uv run python scripts/hyperparam_sweep.py
PYTHONPATH=. uv run python scripts/hyperparam_sweep_dense.py

# Compile the LaTeX paper (requires tectonic or a TeX distribution)
cd paper && tectonic main.tex
```

### Real-LLM CL benchmark (§3.4, figure 7)

The §3.4 table and fig7 come from a real LoRA fine-tuning run on kxkm-ai
(RTX 4090). The orchestrator has three modes:

```bash
# Stub — deterministic summary, runs in CI (no network, no GPU)
PYTHONPATH=. uv run python -m scripts.cl_llm_bench.run_cl_bench \
  --mode stub --tasks phono_sst2,lex_cola,syn_boolq \
  --output bench/cl_llm/runs/stub_0 --seed 0

# Preflight — read-only SSH probe of the remote host
PYTHONPATH=. uv run python -m scripts.cl_llm_bench.run_cl_bench \
  --mode preflight --ssh-host kxkm-ai \
  --tasks phono_sst2,lex_cola,syn_boolq \
  --output bench/cl_llm/runs/preflight_check --seed 0

# Real — gated by --i-confirm-heavy-training; see runbook for prereqs
PYTHONPATH=. uv run python -m scripts.cl_llm_bench.run_cl_bench \
  --mode real --i-confirm-heavy-training --ssh-host kxkm-ai \
  --tasks phono_sst2,lex_cola,syn_boolq \
  --output bench/cl_llm/runs/my_run --seed 0 \
  --max-samples 5000 --n-steps 1500 --base-model Qwen/Qwen3-4B
```

Full prerequisites, GPU-time budget, and per-seed sweep recipe are in
[`docs/superpowers/runbooks/real-cl-bench.md`](docs/superpowers/runbooks/real-cl-bench.md).

Run the full test suite:

```bash
uv run pytest --cov=kiki_flow_core
```

## Quantitative results (paper draft on `main`, last tag `paper-v0.13-draft`)

> **Erratum (2026-10-24/25).** Four defects affected the rigorous
> (entropic-W2) path: the POT and MLX `prox_w2` ignored their reference
> (`ERRATUM.md` §1), `FullJKOSolver.epsilon` was never forwarded, so
> Sinkhorn ran at reg = h = 0.05 where the paper states ε = 0.01 (§9),
> and `max_iter` was not forwarded (§10; no reported number changes).
> Withdrawn: the 5× / 5.13× MLX speedup, the fast-vs-rigorous
> "agreement", the old ε-sweep and the old Fig. 1. Only runs that call
> this prox are affected; fast-path results (Figs. 2–3, sweeps, CL
> benchmarks) never did and describe the projected-gradient scheme. An
> ε sweep (§11) shows the Sinkhorn temperature does not explain the
> fast/rigorous discrepancy; its residual cause is not isolated. Public
> record: GitHub releases `paper-v0.12-draft` and `paper-v0.13-draft`,
> Zenodo [10.5281/zenodo.23619954](https://doi.org/10.5281/zenodo.23619954)
> (v0.12) and [10.5281/zenodo.23636477](https://doi.org/10.5281/zenodo.23636477)
> (v0.13). The rows below give the values re-measured with the
> corrected solver. See [`ERRATUM.md`](ERRATUM.md).

| Claim | Measured value | Source |
|---|---|---|
| Flow tractable at scale | 5 seeds × 100 slow × 10k particles: fast 2 min 31 s (M5, GPU) or 3 min 37 s (M1 Max, CPU); rigorous with the corrected solver 12 min 35 s to 12 min 45 s over the three 2026-10-24 reruns (M1 Max, CPU; latest: ε = 0.01, `max_iter` = 1000, 765 s). The old 84 min timed the defective prox | `paper/stats.json`, `paper_rigorous_fixedprox/stats.json`, `paper_rigorous_fixedeps/stats.json`, `paper_rigorous_maxiter/stats.json` |
| Fast vs rigorous agreement (corrected solver) | **They do not agree**: KL(rigorous‖fast) = 0.218 ± 0.026 nats (reduced setup, ε = h = 0.05, 5 seeds); 0.272 ± 0.014 nats at paper scale with ε = 0.01 (0.262 before the §9 fix); previously reported 0.000638 with the defective prox | `paper/erratum_2026_10_24/fast_vs_rigorous_kl_fixed.json`, `paper/erratum_2026_10_24_reg/fig1_species_moments_summary_reg.json` |
| Specialization under asymmetric potentials, as **excess over the Gibbs equilibrium of the potentials** (primary statistic since 2026-10-25) | **No emergent specialization in the regime studied.** Corrected entropic JKO step (reg = h², reinjection; 33 (α, γ) × 10 seeds): raw gap 2.72 ± 0.05 bits at (10, ·, 0), 3.53 ± 0.05 at α = 30, but excess gap − gap_Gibbs < 0 at every α ≤ 20, γ ≤ 10 (−0.39 [−0.42, −0.36] bits at (10, ·, 0), 95 % bootstrap CI over seeds); positive only with γ ≥ 20 (up to +0.53 [+0.49, +0.57] at (5, ·, 50)) and marginally at α = 30 (+0.03 [+0.005, +0.06]); after the α = 0 control only (5, ·, 50) stays positive. Legacy numbers (legacy split, not a JKO step of F — `ERRATUM.md` §12): fast 2.04 bits at (10, 1, 0), rigorous 1.28–1.46 bits; both below equilibrium everywhere | `results/excess_over_gibbs_2026_10_25/`, `results/jko_entropic_sweep_2026_10_25/`, `paper/hyperparam_sweep_dense.json`, `paper/rigorous_specialisation_wide_2026_10_24_reg/summary.json` |
| Streaming surrogate latency | p50 = 0.04 ms, p99 = 0.06 ms (v0.1, `state_dim=16`); p50 = 0.37 ms (v0.2-d128, `state_dim=128`) | `bench/T3_latency.jsonl` |
| Continual-learning retention (honest trade-off) | Task 1 with cons. 0.90 ± 0.07 / without 0.29 ± 0.07; Task 3 with 0.004 / without 0.81 | `paper/cl_benchmark.json` |
| Real-LLM continual learning, 500-sample regime (Qwen3-4B 4-bit LoRA, 5 seeds) | SST-2: forgetting 0.085 ± 0.030; CoLA: apparent positive transfer; BoolQ: 0.565 ± 0.023 (data-limited) | `bench/cl_llm/runs/e2_5seeds_summary.json` |
| Real-LLM continual learning, 5000-sample regime (5 seeds) | SST-2: forgetting **0.361 ± 0.010** (4.2× higher); CoLA: forgetting 0.129 ± 0.021; positive-transfer artefact disappears | `bench/cl_llm/runs/e3_5k_5seeds_summary.json` |
| ε-dependence of one entropic prox step (ε as Sinkhorn regularisation only, prox step h = 0.05) | KL(ρ_ε ‖ ρ_0.001) = 3.5e-4 nats at ε = 0.01, 0.011 at 0.05, 0.016 at 0.1. The former "0 → 8.3 bits" was an artefact of the defective prox; the 2026-10-24 intermediate values (0.23 / 0.67 nats) conflated ε with the prox step (`ERRATUM.md` §9) | `paper/erratum_2026_10_24_reg/epsilon_sweep_reg.json` |
| Entropic JKO is the real Peyré bias, not a grid artefact (pre-registered OU test) | Grid-invariant 321→4001 pts (0 % change); no-entropy null Var* ≈ 2.8e-15; stationary floor Var* = (ε/τ)·g(τ) with g(τ)→0.5 as τ→0 (textbook ε/(2τ)) | `experiments/results/e1b_peyre_bias.json` |
| Propagation-of-chaos preconditions (E2) | Convex on simplex tangent (worst eig 2.831), L ≈ 18.5, Bakry–Émery LSI ≥ 2.831 — but only because KL entropy dominates the indefinite coupling J_sym (min eig −0.288); **not uniform** | `experiments/results/e2_poc_preconditions.json` |
| MLX vs NumPy W2 prox (corrected operator, per call) | MLX **slower** at the paper grid sizes (NumPy/MLX = 0.10 at n = 16, 0.15 at n = 64) and faster only from n ≈ 1024 (1.32 at 1024, 4.26 at 2048; M1 Max). The former "5.13× faster than POT" claim is **withdrawn** — see `ERRATUM.md` | `PERFORMANCE.md`, `tests/track2_paper/test_mlx_prox_w2.py` |

Results that did **not** hold are reported honestly in the paper:
the default flat-potential setup does not induce specialization
(entropy ≈ maximum), and naive prior-averaged consolidation
over-stabilizes on the first task at the expense of later ones.

Following an external review (`REVIEW.md`), the `track2_paper` JKO scheme
is described as an **entropic-regularized** JKO, not an exact-Wasserstein
"rigorous proxy": the entropic Sinkhorn bias is a real model bias (the
stationary floor ε/(2τ) of the entropic JKO scheme (Peyré 2015;
Baradat, Hraivoronska & Santambrogio 2025), grid-invariant — `EXPERIMENT-BACKLOG.md`
E1/E1b), and the propagation-of-chaos preconditions hold only in the
entropy-dominated regime, with no uniform log-Sobolev constant (E2).

## Integration with micro-kiki

This repository ships three artifacts to bridge to an existing
micro-kiki deployment:

1. `kiki_flow_core/track3_deploy/weights/v0.2-d128.safetensors` — trained
   surrogate weights matched to a 32-stack × 4-ortho = 128-dim state.
2. Two unified-diff patches and a design document in
   `docs/superpowers/patches/` that resolve three of the four known
   integration dependencies.
3. A working `KikiFlowBridge` reference implementation (in
   `patches/02-micro-kiki-dep3-runner-factory.md`) that owns its own
   tokenizer and returns None on any failure, guaranteeing zero impact
   on the nominal routing pathway when `KIKI_FLOW_ENABLED=0`.

See [`docs/superpowers/integration-notes.md`](docs/superpowers/integration-notes.md)
for the full status of each dependency.

## Repository layout

```
kiki-flow-research/
├── kiki_flow_core/                 # 45 source files, strict mypy + ruff
│   ├── state.py                    # FlowState Pydantic v2 + invariants
│   ├── master_equation.py          # FreeEnergy ABC + JKOStep
│   ├── wasserstein_ops.py          # prox_w2, w2_distance, Sinkhorn
│   ├── species/                    # CanonicalSpecies, MixedCanonicalSpecies
│   ├── modules/                    # AdvectionDiffusion, scheduler, phono loop
│   ├── hooks/                      # Aeon, MoE-LoRA, Routing adapters
│   ├── telemetry/                  # StructuredLogger, Metrics
│   ├── track1_perf/                # Eulerian grid + offline consolidator
│   ├── track2_paper/               # Particles + JKO + figures
│   └── track3_deploy/              # Pure-numpy surrogate + streaming runner
├── tests/                          # pytest suite, golden NPZ regressions; count and coverage as measured by CI (see above)
├── scripts/                        # Reproducibility scripts
│   └── cl_llm_bench/               # Real-LLM CL: stub / preflight / real modes
│       └── kxkm_trainer/           # Standalone 4-bit QLoRA trainer (runs on kxkm-ai)
├── paper/                          # main.tex, main.pdf, figures, JSON stats
├── paper_rigorous/                 # Historical 84-min rigorous run (defective prox, see ERRATUM.md; kept for provenance)
├── paper_rigorous_fixedprox/       # Corrected rigorous reruns, 12 min 35 s to 12 min 45 s on M1 Max (ERRATUM.md §7, §9, §10)
├── paper_rigorous_fixedeps/
├── paper_rigorous_maxiter/
├── bench/                          # SLO latency records
│   └── cl_llm/runs/                # Per-seed real-CL manifests + summary
└── docs/superpowers/               # Specs, plans, integration, patches
    └── runbooks/real-cl-bench.md   # Real-mode CL manual runbook
```

## Program context

Young-flow's Wasserstein engine is the upstream species model for the
dream-of-kiki consolidation framework. As of 2026-05-19,
[`dream-of-kiki/docs/specs/2026-05-20-biophysical-stratification.md`](https://github.com/hypneum-lab/dream-of-kiki/blob/main/docs/specs/2026-05-20-biophysical-stratification.md)
§3 routes Młynarski & Hermundstad 2025 *(Convergence of efficient and
predictive coding in multimodal sensory processing, bioRxiv 2025)* as
the normative scaffolding for this engine.

## AI assistance (transparency)

This repository is developed by a one-person lab with AI assistants
used as tools, never as authors:

- **Claude Code (Anthropic)** assists with code, documentation,
  operations and literature search in interactive sessions driven by
  the PI.
- **Codex CLI (OpenAI, `gpt-6.1-sol` / `gpt-6-astra`)** runs
  independent review passes before a release, a pre-registration or a
  submission; the briefs, reports and verdict tables are published next
  to the reviewed material whenever a pass is run.

Every scientific claim, verdict and pre-registration decision is taken
by the PI; results are reproduced by the test suite and CI, and review
reports are evidence, not endorsements. No model is listed as a
co-author of a publication or as a co-author of a commit.

## Citation

Draft pre-print, single author (Clément Saillant). The latest tagged
paper version is `paper-v0.13-draft` (2026-10-25, erratum update:
solver ε and `max_iter` fixes; Zenodo
[10.5281/zenodo.23636477](https://doi.org/10.5281/zenodo.23636477)),
preceded by `paper-v0.12-draft` (2026-10-24, corrected-prox reruns;
Zenodo [10.5281/zenodo.23619954](https://doi.org/10.5281/zenodo.23619954)).
The concept DOI
[10.5281/zenodo.19666755](https://doi.org/10.5281/zenodo.19666755)
resolves to the latest version. Every tag up to `paper-v0.11-draft`,
and the Zenodo record 10.5281/zenodo.19666756 (`paper-v0.8-draft`),
carry claims withdrawn by `ERRATUM.md`. The package version in
`pyproject.toml` (`0.0.1`) is independent of the paper tags. The
packaged `paper/kiki-flow-arxiv-v0.5.zip` / `paper/arxiv_bundle/` is
an older v0.5 snapshot, **superseded** since the 2026-10-24 erratum: it
still carries the withdrawn fast-vs-rigorous agreement, the old Fig. 1
and the old ε-sweep. Use `paper/main.pdf`; the bundle must be
regenerated from `paper/main.tex` before any submission. Not submitted
to arXiv.

```bibtex
@unpublished{kiki-flow-2026,
  author = {Saillant, Clément},
  title  = {Hybrid Cognitive-Fluid Dynamics for Language Production:
            A Wasserstein Gradient Flow Perspective},
  year   = 2026,
  doi    = {10.5281/zenodo.23636477},
  url    = {https://github.com/hypneum-lab/kiki-flow-research},
  note   = {Draft; code, data, and figures released under MIT at tag paper-v0.13-draft; see ERRATUM.md}
}
```

## License

MIT. See [`LICENSE`](./LICENSE).

## Acknowledgments

Built on top of the excellent open-source work of
[Apple MLX](https://github.com/ml-explore/mlx),
[POT (Python Optimal Transport)](https://github.com/PythonOT/POT),
[JAX](https://github.com/google/jax),
[scikit-learn](https://github.com/scikit-learn/scikit-learn), and
[sentence-transformers](https://github.com/UKPLab/sentence-transformers).
