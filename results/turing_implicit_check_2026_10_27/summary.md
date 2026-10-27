# Implicit Turing treatment: dissipation and gamma <= 5 checks (2026-10-27)

96 configurations x 10 slow steps on the grid (alpha in [2.0, 5.0, 10.0], h in [0.05, 0.025], initial states Dirichlet seeds [500, 501]), reg = h^2, Sinkhorn early stop 1e-12. Steps with F increasing by more than 1e-12:

| gamma | steps | implicit: F increases | explicit: F increases | max halvings | implicit s/step |
|---|---|---|---|---|---|
| 0 | 120 | 0 | 0 | 0 | 15.0 |
| 5 | 120 | 0 | 0 | 0 | 18.1 |
| 10 | 120 | 0 | 0 | 0 | 18.6 |
| 20 | 120 | 0 | 5 | 0 | 18.7 |
| 30 | 120 | 0 | 11 | 0 | 19.0 |
| 50 | 120 | 0 | 22 | 3 | 21.7 |
| 70 | 120 | 0 | 48 | 0 | 19.2 |
| 100 | 120 | 0 | 76 | 0 | 19.0 |

Dissipation criterion (0 increases, all configurations): PASS.

gamma <= 5, per-step max TV implicit vs explicit (same input): 5.18e-02 (tolerance 0.001) — FAIL.

| key point (alpha, gamma) | gap implicit | gap explicit (PR #46) | diff mean ± sd | within ±0.05 |
|---|---|---|---|---|
| (10, 0) | 2.708 | 2.728 | -0.020 ± 0.015 | yes |
| (8, 5) | 2.393 | 2.434 | -0.041 ± 0.032 | yes |
