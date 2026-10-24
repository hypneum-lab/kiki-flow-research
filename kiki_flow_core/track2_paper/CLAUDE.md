# track2_paper — paper-faithful numerics

T2 subpackage. Source of every number, curve, and phase portrait that
appears in `paper/main.tex` (and the rigorous appendix in
`paper_rigorous/`). Discipline: equations match the paper, figures
regenerate bit-stably, claims trace to JSONs.

## Module map

| File | Role |
|---|---|
| `paper_run.py` | `run_paper(...)` — top-level driver, 5 seeds, emits trajectories + figures + `stats.json` |
| `paper_f.py` | `T2FreeEnergy` — V + KL prior + Levelt-Baddeley reaction (J_sym/J_asym split) + Turing |
| `paper_f_separable.py` | Separable variant of T2FreeEnergy used in the analytical-gradient ablation |
| `full_jko_solver.py` | `FullJKOSolver` (POT) and `MLXFullJKOSolver` (rigorous-path Sinkhorn) |
| `mlx_wasserstein.py` | Native MLX log-domain Sinkhorn + `mlx_prox_w2` (injected via `prox_fn`) |
| `mlx_particle_simulator.py` / `particle_simulator.py` | N-particle Langevin samplers (MLX vs NumPy) |
| `multiscale_loop.py` | Fast/slow nested loop — particle Langevin (fast) drives JKO (slow) |
| `pca_projection.py` | 1D projection of N-particle clouds onto the histogram support |
| `figures/` | One generator per published figure; `Agg` backend; emits PDF + PNG |

## Coupling with `paper/` and `paper_rigorous/`

| Direction | Contract |
|---|---|
| `paper_run.run_paper(out_dir=Path("paper"))` | writes `paper/figures/fig{1..6}*` + `paper/stats.json` + `paper/trajectories/*.safetensors` |
| `paper_run.run_paper(use_w2_prox=True, n_slow=100, out_dir=Path("paper_rigorous"))` | writes the rigorous artifacts (committed ones: POT backend, pre-`ffee5d9` prox; see `ERRATUM.md`) |
| Figure naming | `fig{N}_<slug>` — match the slug used by `\includegraphics{figures/figN_name}` exactly; LaTeX picks PDF |
| JSON schema | additive only; if you change a key, update `paper/main.tex` in the same commit (see `scripts/CLAUDE.md`) |

`scripts/*.py` orchestrate the full reproducibility runs and call into
this subpackage; never duplicate driver logic in `scripts/`. Generators
under `figures/` are the single PDF-emission path — the scripts do not
write PDFs directly.

## Numerical discipline

- Every functional in `paper_f*.py` clips with `1e-12` before `log` and
  routes coupling through `J_sym` (the `J_asym` contraction with
  symmetric `<rho_i, rho_k>` vanishes identically — derivation lives in
  `paper_f.py:T2FreeEnergy.value`).
- 4-species canonical names `phono / lex / syntax / sem` are the only
  accepted ordering; figures and JSON keys depend on it.
- Particle -> histogram conversion in `multiscale_loop._particles_to_flow_state`
  falls back to uniform `1.0 / n_bins` for an empty species mask — never
  to zeros (would break `assert_invariants`).
- Rigorous path uses `epsilon=0.01`, `max_iter=1000`, `n_inner=50`,
  `apply_w2_prox=True`. Don't loosen these to "make MLX faster" — the
  appendix claim is anchored to this exact tuple.
- `Sinkhorn did not converge` warnings on the rigorous path are expected
  and recorded in `paper_rigorous/run.log`; do not silence them
  globally (see root `CLAUDE.md`).

## Anti-patterns (T2-specific)

- Modifying an equation in `paper_f.py` (or any solver constant in
  `full_jko_solver.py`) without bumping the FC axis and re-running every
  script that produces a `paper/*.json`. The paper claims are versioned.
- Writing a PDF straight from a script — figures live exclusively in
  `figures/` generators, called by `paper_run.run_paper` or
  `scripts/*.py`. Bypassing them produces a figure with no manifest.
- Adding a new figure under `figures/` without wiring it into
  `paper_run` (or a `scripts/` entry) and without an `\includegraphics`
  line in `main.tex`. Orphan generators rot.
- Using the non-MLX `ParticleSimulator` for the rigorous run, or vice
  versa, without updating the `backend` field in
  `paper_rigorous/stats.json`. Backend is part of the claim.
- Importing from `track1_perf/` or `track3_deploy/` (root rule, repeated
  here because new figure code is the most common offender).
