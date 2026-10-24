# Widened specialisation sweep, rigorous path rerun with ε = 0.01

This directory supersedes the rigorous half of
`paper/rigorous_specialisation_wide_2026_10_24/`. That run used
`FullJKOSolver` before the ε fix (ERRATUM.md §9), so the prox ran with
Sinkhorn reg = h = 0.05 instead of the configured ε = 0.01.

- Driver: `experiments/erratum_reg_reruns_2026_10_24.py spec` at
  `0d5683f` (fix `016d71a`), macM1 CPU, `nice -n 19`, one process.
  1380 rigorous runs, same 138 points and 10 seeds as the widened sweep.
- The fast-path records are copied from the widened sweep: the fast
  path has no prox, and its summaries are identical.
- Files: `sweep_wide_records.json` and `summary.json`, same schema as
  the widened sweep. Read that sweep's `ANALYSIS.md` for the method.

| Rigorous path (10 seeds) | reg = 0.05 (old) | ε = 0.01 (this) |
|---|---|---|
| best point | (15, 0.2, 0): gap 1.44 ± 0.05 | (15, 0.2, 0): gap 1.46 ± 0.05 |
| (10, 1, 0) | 1.24 ± 0.06 | 1.28 ± 0.07 |
| best of the 72-point grid, seeds 0–4 | (8, 0.5, 5): 1.35 ± 0.06 | (8, 0.5, 5): 1.39 ± 0.06 |
| gap at α = 30, β ≥ 0.5 | 1.20–1.23 | 1.23–1.26 |
| S (vs α = 0 control) at α ≥ 8, β ≥ 0.2 | 0.79–1.27 | 0.78–1.26 |
| α = 1: S | ≤ 0.08 | ≤ 0.07 (0.003 for β ≥ 0.3) |
| α > 0 points with S's 95 % CI ≤ 0 | 0 / 116 | 1 / 116 (α = 1, β = 0.3) |
| α > 0 points with rigorous H > fast H (95 % CI) | 115 / 116 | 102 / 116 |
| phono/lex less specialised than syntax/sem (α ≥ 3) | 58 / 100 | 55 / 100 |
| max \|H(γ) − H(γ = 0)\| | 0.054 | 0.046 |
| β slope of H per doubling, α = 3–20 | +0.04 to +0.07 | +0.04 to +0.07 |

The largest change in a point's mean entropy is 0.061 bits (mean
change −0.036). The findings of the widened sweep all hold:
- specialisation on the rigorous path is smaller than on the fast path
  and saturates;
- the β trend has opposite signs on the two paths;
- phono/lex de-specialise at large α;
- γ is negligible.
