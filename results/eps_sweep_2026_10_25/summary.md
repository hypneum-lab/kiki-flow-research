# Sinkhorn regularisation sweep, rigorous specialisation gap (2026-10-25)

Driver `experiments/eps_sweep_2026_10_25.py` at `7b3f7ee2`, host `electron-server` (x86_64), Python 3.14.4, MLX 0.31.1 on `Device(cpu, 0)`, 2026-10-24T22:43:30Z.

Problem of `experiments/rigorous_specialisation_2026_10_24.py` (grid 16, 500 particles, 200 fast x 30 slow steps, h = 0.05). epsilon is the Sinkhorn regularisation of the rigorous prox (`FullJKOSolver.epsilon`, forwarded as `reg`); the proximal pull stays at h. Gap = log2(16) - mean H over the four species, in bits; S = H(alpha = 0, same beta, gamma, seed) - H; KL in nats, summed over species at the final slow step, on this reduced setup. Mean ± sd (ddof = 1) over seeds.

| eps | (alpha, beta, gamma) | n_inner | rigorous gap | S vs alpha=0 | fast gap | fast - rigorous gap | KL(rig‖fast) | seeds |
|---|---|---|---|---|---|---|---|---|
| 0.01 | (10, 1, 0) | 10 | 1.276 ± 0.031 | 1.057 ± 0.043 | 2.032 ± 0.027 | 0.757 ± 0.032 | 22.148 ± 2.884 | 5 |
| 0.01 | (8, 0.5, 5) | 5 | 1.388 ± 0.055 | 1.173 ± 0.062 | 1.802 ± 0.029 | 0.414 ± 0.046 | 16.446 ± 1.557 | 5 |
| 0.02 | (10, 1, 0) | 10 | 1.266 ± 0.028 | 1.071 ± 0.038 | 2.032 ± 0.027 | 0.766 ± 0.029 | 22.181 ± 2.858 | 5 |
| 0.02 | (8, 0.5, 5) | 5 | 1.380 ± 0.059 | 1.188 ± 0.064 | 1.802 ± 0.029 | 0.422 ± 0.048 | 16.481 ± 1.563 | 5 |
| 0.05 | (10, 1, 0) | 10 | 1.234 ± 0.028 | 1.064 ± 0.034 | 2.032 ± 0.027 | 0.798 ± 0.027 | 22.390 ± 2.733 | 5 |
| 0.05 | (8, 0.5, 5) | 5 | 1.349 ± 0.059 | 1.182 ± 0.061 | 1.802 ± 0.029 | 0.453 ± 0.050 | 16.766 ± 1.557 | 5 |

At (10, 1, 0) the rigorous gap spans 0.042 bits across epsilon (seed sd 0.028-0.031 bits).

At (8, 0.5, 5) the rigorous gap spans 0.039 bits across epsilon (seed sd 0.055-0.059 bits).

## Committed macM1 references at the same points

| source | (alpha, beta, gamma) | rigorous gap | fast gap |
|---|---|---|---|
| reg_0.05_prefix | (8.0,0.5,5.0) | 1.349 ± 0.059 | 1.802 ± 0.029 |
| reg_0.05_prefix | (10.0,1.0,0.0) | 1.234 ± 0.028 | 2.032 ± 0.027 |
| eps_0.01_fixed | (10.0,1.0,0.0) | 1.276 ± 0.031 | — |
| eps_0.01_fixed | (8.0,0.5,5.0) | 1.388 ± 0.055 | — |

- `reg_0.05_prefix`: paper/rigorous_specialisation_2026_10_24/summary.json (before the fix: reg = h = 0.05), seeds 0-4
- `eps_0.01_fixed`: paper/rigorous_specialisation_wide_2026_10_24_reg/sweep_wide_records.json (fix forwarded, epsilon = 0.01), seeds [0, 1, 2, 3, 4]
