# Specialisation sweep on the rigorous path (2026-10-24)

Question (ERRATUM.md §7–8): does the specialisation of the dense sweep
(`paper/hyperparam_sweep_dense.json`, fast path, 2.04-bit gap at
(α, β, γ) = (10, 1, 0)) hold on the rigorous path?

## Setup

- Driver: `experiments/rigorous_specialisation_2026_10_24.py` at
  `c55c784`. Same 72-point grid, potentials, β → `n_inner` map, grid 16,
  500 particles, 200 fast / 30 slow steps as `scripts/hyperparam_sweep_dense.py`.
- Rigorous = `FullJKOSolver`: the same projected-gradient substeps, then
  the corrected entropic W2 prox (`prox_w2`, Dykstra, post-`ffee5d9`).
  As in the §7 reruns, `JKOStep` calls the prox with ε = h = 0.05 and
  100 iterations (`FullJKOSolver.epsilon` is stored but not used).
- Fast = the historical scheme, rerun on the same host as a control.
- Seed 0 is the seed of the historical sweep. Seeds 1–4 were added to
  measure the seed-to-seed spread (the historical sweep has one seed).
- Host macM1 (Apple M1 Max), Python 3.14.4, MLX on the CPU device,
  `nice -n 19`, one process. 720 runs (2 paths × 72 × 5 seeds) in
  20 min: 1.7 s per rigorous run, 1.65 s per fast run.
- Metric (as in the fast sweep): H = mean over the 4 species of the final
  entropy, in bits; gap = log2(16) − H = 4 − H.

Check: the fast rerun at seed 0 reproduces `paper/hyperparam_sweep_dense.json`
exactly (max |ΔH| = 0.0 over the 72 points).

## Result

| | fast | rigorous |
|---|---|---|
| Best point (seed 0) | (10, 1, 0): H 1.961, gap **2.04** bits (51 %) | (8, 0.5, 5): H 2.577, gap **1.42** bits (36 %) |
| Best point (mean of 5 seeds) | (10, 1, 0): H 1.968 ± 0.027, gap 2.03 | (8, 0.5, 5): H 2.651 ± 0.059, gap 1.35 (34 %) |
| At (10, 1, 0), 5 seeds | H 1.968 ± 0.027 (gap 2.03) | H 2.766 ± 0.028 (gap 1.23) |
| α = 0 control (12 points) | H 3.91–3.98 (gap ≤ 0.09) | H 3.83–3.85 (gap 0.15–0.17) |
| Effect of γ (max |H(γ=5) − H(γ=0)|) | 0.015 bits | 0.028 bits |

Uncertainty is the sample SD over seeds 0–4 (n = 5). Full data:
`summary.json` (every point, both paths, mean/SD/min/max and the paired
rigorous − fast difference), `sweep_dense_rigorous.json`,
`sweep_dense_fast_rerun.json`.

Per-species final entropy at (10, 1, 0), 5 seeds (bits):

| species | fast | rigorous |
|---|---|---|
| phono | 1.71 ± 0.07 | 3.03 ± 0.08 |
| lex | 1.67 ± 0.04 | 3.04 ± 0.15 |
| syntax | 2.15 ± 0.07 | 2.64 ± 0.07 |
| sem | 2.34 ± 0.06 | 2.35 ± 0.11 |

γ = 0 slice (H in bits; seed-0 values, then mean ± SD over seeds 0–4):

| α | β | fast s0 | rig. s0 | fast | rigorous | rig. − fast |
|---|---|---|---|---|---|---|
| 0 | 0 | 3.930 | 3.838 | 3.912 ± 0.053 | 3.850 ± 0.021 | -0.062 ± 0.070 |
| 0 | 0.5 | 3.975 | 3.818 | 3.978 ± 0.006 | 3.832 ± 0.022 | -0.146 ± 0.017 |
| 0 | 1 | 3.971 | 3.816 | 3.974 ± 0.006 | 3.830 ± 0.021 | -0.145 ± 0.016 |
| 0 | 2 | 3.969 | 3.815 | 3.972 ± 0.006 | 3.829 ± 0.021 | -0.143 ± 0.016 |
| 1 | 0 | 3.680 | 3.793 | 3.647 ± 0.056 | 3.767 ± 0.033 | +0.120 ± 0.082 |
| 1 | 0.5 | 3.780 | 3.806 | 3.767 ± 0.009 | 3.822 ± 0.022 | +0.055 ± 0.024 |
| 1 | 1 | 3.793 | 3.805 | 3.777 ± 0.011 | 3.820 ± 0.021 | +0.042 ± 0.024 |
| 1 | 2 | 3.798 | 3.804 | 3.782 ± 0.012 | 3.819 ± 0.021 | +0.037 ± 0.024 |
| 3 | 0 | 3.174 | 3.288 | 3.150 ± 0.037 | 3.301 ± 0.041 | +0.151 ± 0.033 |
| 3 | 0.5 | 3.093 | 3.368 | 3.185 ± 0.063 | 3.480 ± 0.100 | +0.295 ± 0.054 |
| 3 | 1 | 3.077 | 3.336 | 3.105 ± 0.032 | 3.448 ± 0.075 | +0.343 ± 0.052 |
| 3 | 2 | 3.130 | 3.566 | 3.096 ± 0.035 | 3.546 ± 0.057 | +0.450 ± 0.050 |
| 5 | 0 | 3.020 | 3.030 | 2.987 ± 0.030 | 3.093 ± 0.053 | +0.105 ± 0.063 |
| 5 | 0.5 | 2.649 | 2.916 | 2.657 ± 0.057 | 2.986 ± 0.112 | +0.329 ± 0.107 |
| 5 | 1 | 2.602 | 3.034 | 2.617 ± 0.018 | 3.147 ± 0.102 | +0.530 ± 0.086 |
| 5 | 2 | 2.573 | 3.248 | 2.597 ± 0.035 | 3.220 ± 0.089 | +0.624 ± 0.086 |
| 8 | 0 | 2.939 | 3.052 | 2.884 ± 0.041 | 3.054 ± 0.028 | +0.170 ± 0.047 |
| 8 | 0.5 | 2.163 | 2.579 | 2.198 ± 0.034 | 2.652 ± 0.059 | +0.454 ± 0.049 |
| 8 | 1 | 2.138 | 2.872 | 2.149 ± 0.024 | 2.834 ± 0.064 | +0.685 ± 0.071 |
| 8 | 2 | 2.114 | 2.897 | 2.131 ± 0.020 | 2.948 ± 0.036 | +0.817 ± 0.038 |
| 10 | 0 | 2.899 | 2.974 | 2.845 ± 0.041 | 2.991 ± 0.035 | +0.146 ± 0.043 |
| 10 | 0.5 | 2.073 | 2.770 | 2.052 ± 0.015 | 2.770 ± 0.054 | +0.718 ± 0.055 |
| 10 | 1 | 1.961 | 2.748 | 1.968 ± 0.027 | 2.766 ± 0.028 | +0.798 ± 0.027 |
| 10 | 2 | 1.966 | 2.824 | 1.974 ± 0.018 | 2.856 ± 0.033 | +0.882 ± 0.033 |

## Reading

- **Specialisation holds on the rigorous path, but it is smaller.** The
  gap grows with α on both paths. On the rigorous path it reaches
  1.35 ± 0.06 bits at its best point (1.42 bits at seed 0), against
  0.15–0.17 bits with no potential. The 2.04-bit figure is specific to
  the fast path. At the paper's operating point (10, 1, 0) the rigorous
  gap is 1.23 ± 0.03 bits, 0.80 ± 0.03 bits less than the fast path.
- **The landscape differs.** On the fast path, a larger β helps up to
  β = 1. On the rigorous path, a larger β reduces specialisation at
  α ≥ 3: the prox pulls each update back towards the slow-step
  reference, and more inner steps move ρ further from it. The rigorous
  best point is (8, 0.5), not (10, 1). For every α > 0 point, the
  rigorous entropy is higher than the fast entropy.
- **The species pattern differs.** On the fast path, phono and lex (the
  linear potentials) specialise most. On the rigorous path they
  specialise least; syntax and sem (the |x| potentials) carry most of
  the gap. Sem is the same on both paths.
- γ has a negligible effect on both paths (≤ 0.03 bits), as in the fast
  sweep.
- With no potential, the rigorous path ends about 0.13 bits lower than
  the fast path. This is the same anchoring effect as in ERRATUM.md §7
  (5.90 vs 6.00 bits at grid 64).
