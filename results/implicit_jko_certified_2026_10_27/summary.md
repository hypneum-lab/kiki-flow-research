# Certified implicit JKO step — Codex B1 cases (2026-10-27, rerun 2026-10-28)

One certified step per case on electron-server (nice 19, one process each): grid 16 on [-2, 2],
T2 energy with declared Turing smoothing delta = 1e-5, Dirichlet input of seed 500,
h = 0.05, reg = h^2, kkt_tol = 1e-6, defaults otherwise. Produced at 18:41Z on 2026-10-27 by
the working tree that became the commits of this PR (probe equivalent to
`experiments/implicit_jko_certified_validation_2026_10_27.py b1`); raw records in
`b1_probe.jsonl`. Superseded by the rerun below.

| case (alpha, gamma) | status | KKT residual (TV) | outer iterations | max anchor weight L | prox calls | max inner iterations | wall time | Q_h explicit output → certified |
|---|---|---|---|---|---|---|---|---|
| (8.0,5.0) | converged | 9.3e-07 | 41 | 64 | 620 | 39871 | 407 s | -1.08940 → -1.11587 |
| (5.0,30.0) | converged | 7.9e-07 | 65 | 16 | 800 | 12848 | 284 s | 1.07766 → 0.81783 |
| (10.0,0.0) | converged | 9.9e-07 | 6 | 0 | 32 | 1354 | 4 s | -3.33280 → -3.34252 |

Codex measured on PR #50 for the same inputs: final TV residual 0.0052 at (8, 5) and
0.042 at (5, 30), accepted silently; (10, 0) converged. The explicit step costs about 0.6 s.

## Rerun with the two-residual stopping rule (2026-10-28)

The first run above stopped on the TV residual only, which the review of PR #51 showed
can accept false stationary points in low-mass bins. Rerun at 17:08Z on 2026-10-28 by
`experiments/implicit_jko_certified_validation_2026_10_27.py b1` (solver
`certified-implicit-2`, `kkt_tol` = 1e-6, `log_tol` = 1e-5, 3 workers, nice 19, shared
machine); raw records in `b1_rerun_20261028T170811Z.jsonl` (`b1_probe.jsonl` is kept as
committed).

| case (alpha, gamma) | status | TV residual | log residual | q_min | outer iterations | wall time | Q_h explicit → certified |
|---|---|---|---|---|---|---|---|
| (8.0,5.0) | converged | 9.3e-07 | 7.8e-06 | 1.4e-02 | 41 | 439 s | -1.08940 → -1.11587 |
| (5.0,30.0) | converged | 7.9e-07 | 9.0e-06 | 2.2e-02 | 65 | 301 s | 1.07766 → 0.81783 |
| (10.0,0.0) | converged | 9.9e-07 | 5.8e-06 | 4.0e-03 | 6 | 6 s | -3.33280 → -3.34252 |

Same iterates as the first run (no low-mass bin on these inputs). Each case reaches a
stationary point of the smoothed problem (energy F_delta); there is no global guarantee.
