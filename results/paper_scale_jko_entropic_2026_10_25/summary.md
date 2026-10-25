# Paper-scale run under the entropic JKO default (2026-10-25)

Grid 64, 10 000 particles, 1000 fast x 100 slow steps, h = 0.05, seeds 0-4, MLX simulator on the CPU, `FullJKOSolver` default (entropic JKO step of F, reg = h^2), `MultiscaleLoop(reinject=True)`. Final entropy per species (bits, mean ± sd over seeds) next to the legacy rigorous run `paper_rigorous_maxiter/` (legacy split, epsilon 0.01, no reinjection; seed-mean only).

| species | entropic default | legacy rigorous |
|---|---|---|
| phono | 5.9922 ± 0.0046 | 5.8917 |
| lex | 5.9919 ± 0.0038 | 5.9088 |
| syntax | 5.9907 ± 0.0052 | 5.9032 |
| sem | 5.9860 ± 0.0077 | 5.9034 |

Gap log2(64) − mean H: 0.0098 bits (legacy 0.0982). Runtime per seed: 1744 s, 1742 s, 1757 s, 1737 s, 1755 s.
