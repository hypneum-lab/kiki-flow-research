# Changelog

All notable releases of `kiki-flow-research`. Dates in YYYY-MM-DD.

## [paper-v0.4-draft] — 2026-04-18

First releasable paper with measured data on every figure.

### Added
- `scripts/cl_benchmark.py` — 3-task distributional continual-learning
  proxy, 5 seeds, produces the real fig5.
- `scripts/epsilon_sweep.py` — measured Sinkhorn entropic bias across
  5 epsilon values on a smooth peaked prox problem.
- `scripts/hyperparam_sweep.py` — 27-configuration sweep over
  `(alpha, beta, gamma)` surfacing the best specialization point.
- `scripts/dump_hybrid_pairs.py` — Mode-A trajectory generator at
  `state_dim=128` for the T3 surrogate retrain.
- `kiki_flow_core/track3_deploy/weights/v0.2-d128.safetensors` —
  surrogate matched to a 32-stack × 4-ortho micro-kiki state.
- `kiki_flow_core/track2_paper/mlx_particle_simulator.py` and
  `mlx_wasserstein.py` — Metal-backed alternatives to the numpy/POT
  defaults (simulator gives 2.6× speedup; Sinkhorn is scaffolded).
- `save_trajectories` and `make_all_figures` flags on `run_paper`.
- `_aggregate_rho_stats` reporting peak amplitude, Shannon entropy,
  and peak-index statistics per species across seeds.
- Five figure generators invoked end-to-end (fig1 phase portraits,
  fig2 F-decay, fig3 Turing profiles, fig4 KL vs epsilon, fig5 CL gap).
- Shared matplotlib `rcParams` for consistent serif 11 pt paper figures.
- `docs/superpowers/patches/` — two unified-diff patches and a
  reference implementation for micro-kiki integration.
- `paper/main.tex` + compiled `main.pdf` (~480 KB) with four related-
  work paragraphs, honest mixed-sign findings, reproducibility
  statement, acknowledgments.
- `bench/T3_latency.jsonl` — SLO latency records for both v0.1 and
  v0.2-d128 surrogate variants.
- Public `README.md`, `LICENSE` (MIT), `CONTRIBUTING.md`.

### Changed
- Abstract rewritten twice for honesty: first to remove unverified
  overclaims, then again to report the measured specialization gap.
- Related-work section expanded from four 1-sentence citations to
  four paragraphs covering psycholinguistics, Wasserstein flows,
  continual learning, and neuromorphic mean-field inference.
- Visibility flipped to public on 2026-04-18.

### Known limitations
- fig5 is a distributional proxy, not a full LLM continual-learning
  benchmark (issue #1).
- Hyperparameter sweep is coarse; EWC-style refinement pending
  (issue #2).
- Rigorous JKO path is CPU-bound via POT (issue #3).
- End-to-end micro-kiki integration still requires a small
  upstream refactor (issue #4).

## [paper-v0.3-draft] — 2026-04-17

- Measured Sinkhorn epsilon sweep (replaces synthetic fig4).
- 27-config hyperparameter sweep surfacing the specialization regime.
- Abstract updated with quantitative specialization finding.

## [paper-v0.2-draft] — 2026-04-17

- Five figures (two real, two synthetic placeholders, one panel) +
  aggregate rho statistics (entropy, peak, peak-index).
- Related-work section expanded.
- Trajectory persistence in `paper_run.py`.
- Honest abstract restatement + reproducibility statement +
  acknowledgments paragraph.

## [paper-v0.1-draft] — 2026-04-17

- First compiled `main.pdf` with five phase portraits (one per seed).
- Scaffolded Experiments, Discussion, Conclusion sections.

## [track3-v0.1] — 2026-04-17

- `track3_deploy/` — pure-NumPy neural surrogate (`NeuralSurrogate`),
  MiniLM query encoder (`QueryEncoder`) with stub fallback, streaming
  runner (`StreamingRunner`), JAX-based training (`SurrogateTrainer`),
  latency microbench with p50 < 25 ms / p99 < 100 ms CI gates.
- Public API exposed at `kiki_flow_core.*`.

## [track2-v0.1] — 2026-04-17

- `track2_paper/` — `ParticleSimulator`, `FullJKOSolver`,
  `T2FreeEnergy` (4-term functional), `MultiscaleLoop`, 5 figure
  generators, `paper_run` driver.
- PCA projection helper for Aeon embeddings.

## [track1-v0.1] — 2026-04-17

- `track1_perf/` — `T1FreeEnergy` (phenomenological 3-term),
  `EulerianGridSolver`, `DriftTrigger`, `offline_consolidator` CLI,
  checkpoint I/O via safetensors.

## Earlier (unreleased) commits — 2026-04-17

- Bootstrap: `pyproject.toml`, pre-commit hooks, CI workflow,
  pytest + coverage + ruff + mypy strict.
- `kiki_flow_core/` shared core: `FlowState` (Pydantic v2),
  `JKOStep`, `wasserstein_ops` (prox_w2, w2_distance, Sinkhorn),
  `OrthoSpecies` (Levelt-Baddeley coupling from YAML),
  `HybridSpecies` (4 × N projection), `AdvectionDiffusion`,
  `ScaffoldingScheduler`, `PhonologicalLoop`, `hooks` (Aeon, MoE,
  Routing with circuit breaker), `telemetry` (JSON logger +
  Prometheus metrics), end-to-end smoke + double-run determinism.
- 56 core tests passing, 95 % coverage.
