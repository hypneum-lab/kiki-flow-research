# Certified implicit JKO step — Codex B1 cases (2026-10-27)

One certified step per case on electron-server (nice 19, one process each): grid 16 on [-2, 2],
T2 energy with declared Turing smoothing delta = 1e-5, Dirichlet input of seed 500,
h = 0.05, reg = h^2, kkt_tol = 1e-6, defaults otherwise. Produced at 18:41Z on 2026-10-27 by
the working tree that became the commits of this PR (probe equivalent to
`experiments/implicit_jko_certified_validation_2026_10_27.py b1`); raw records in
`b1_probe.jsonl`.

| case (alpha, gamma) | status | KKT residual (TV) | outer iterations | max anchor weight L | prox calls | max inner iterations | wall time | Q_h explicit output → certified |
|---|---|---|---|---|---|---|---|---|
| (8.0,5.0) | converged | 9.3e-07 | 41 | 64 | 620 | 39871 | 407 s | -1.08940 → -1.11587 |
| (5.0,30.0) | converged | 7.9e-07 | 65 | 16 | 800 | 12848 | 284 s | 1.07766 → 0.81783 |
| (10.0,0.0) | converged | 9.9e-07 | 6 | 0 | 32 | 1354 | 4 s | -3.33280 → -3.34252 |

Codex measured on PR #50 for the same inputs: final TV residual 0.0052 at (8, 5) and
0.042 at (5, 30), accepted silently; (10, 0) converged. The explicit step costs about 0.6 s.
