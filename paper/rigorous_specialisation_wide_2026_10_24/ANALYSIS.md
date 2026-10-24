# Widened specialisation sweep, fast vs rigorous (2026-10-24)

Follow-up to `paper/rigorous_specialisation_2026_10_24/` (72 points,
5 seeds). Questions: where does specialisation hold or fail on the rigorous
path, what happens per species (phono/lex), and is the fast-path β trend
an artefact?

## Setup

- Driver `experiments/rigorous_specialisation_wide_2026_10_24.py` at
  `f5682fe`. It calls the `run_one` of the first sweep (same problem: grid
  16, 500 particles, 200 fast / 30 slow steps, asymmetric potentials,
  `n_inner = max(1, int(10 β))`). Rigorous = `FullJKOSolver` with the
  corrected Dykstra prox (ε = h = 0.05, 100 iterations).
- 138 points, 10 seeds (0–9), both paths, 2760 records:
  - core: the original 72 points. Seeds 0–4 are reused from the first
    sweep, seeds 5–9 are new.
  - β: {0.2, 0.3, 4, 8} at α ∈ {0, 1, 3, 5, 8, 10}, γ = 0.
  - α: {15, 20, 30} × β ∈ {0, 0.2, 0.5, 1, 2, 4}, γ = 0.
  - γ: {10, 20, 50} × α ∈ {0, 5, 8, 10} × β ∈ {0.5, 1}.

  β only enters through `n_inner`, so β < 0.2 is the same run as β = 0
  and is not swept. Every point has an α = 0 control at the same (β, γ).
- Host macM1 (M1 Max), Python 3.14.4, MLX on the CPU device,
  `nice -n 19`, one process, under a watchdog (pause on memory-pressure
  warning or when N10-C checkpoints stall). 2040 new runs, 58.5 min of
  compute.
- Metrics (bits): H = mean final entropy over the 4 species; gap =
  4 − H; S = H(α = 0, same β, γ, seed) − H, paired by seed. The
  specialisation S removes the path's own entropy offset at α = 0 (the
  rigorous path ends about 0.14 bits lower there). Uncertainty: ± SD over
  10 seeds. `summary.json` also gives 95 % t intervals.

Files: `sweep_wide_records.json` (every run), `summary.json` (every
point: H, gap, S and per-species H for both paths, paired rigorous − fast,
β trends, best points).

## Main results

| | fast | rigorous |
|---|---|---|
| best point | (30, 4, 0): gap **2.94 ± 0.03** | (15, 0.2, 0): gap **1.44 ± 0.05** |
| best on the original 72 points | (10, 1, 0): gap 2.03 ± 0.03 | (8, 0.5, 0/5): gap 1.32–1.33 |
| gap at α = 30, β ≥ 0.5 | 2.84–2.94 (still growing) | 1.20–1.23 (saturated) |
| S at α ≥ 8, any β ≥ 0.2 | 1.6–2.9 | 0.79–1.27 |
| points with α > 0 and S's 95 % CI ≤ 0 | 0 / 116 | 0 / 116 |
| points with α > 0 and rigorous H > fast H (95 % CI) | — | 115 / 116 |
| max \|H(γ) − H(γ = 0)\|, γ ≤ 50 | 0.047 | 0.054 |

1. **Specialisation holds on the rigorous path everywhere α > 0, but it
   is weaker and it saturates.** S is positive at all 116 points with
   α > 0. It is small for α = 1 (≤ 0.08 bits against ≤ 0.26 for fast),
   0.26–0.88 for α = 3–5, and plateaus at about 0.8–1.3 bits for α ≥ 8.
   The fast-path gap keeps growing with α up to 2.94 bits at α = 30.
   So "(10, 1, 0) is the best point" is an edge effect of the 72-point
   grid, not a fast-path optimum. The rigorous optimum (15, 0.2) is
   inside the widened grid.
2. **The β trend has opposite signs on the two paths.** At α ≥ 3 the fast
   path gains from more inner steps: H falls with log β and levels off
   from β ≈ 2. The rigorous path loses: H rises with β (slope
   +0.04 to +0.07 bits per doubling at α = 3–20) and, for α ≤ 10, is
   still rising at β = 8. Its best β is the smallest one swept (0.2–0.5). β only changes
   the number of projected-gradient substeps inside one step of fixed
   length h, so it is a numerical parameter, not a physical one. That the
   sign flips with the scheme means the fast-path "larger β helps"
   reading is a property of the fast scheme, not of the flow.
3. **Phono/lex behave differently.** On the fast path, phono and lex
   (linear potentials) are the most specialised species at every α ≥ 3
   (97 of 100 points). On the rigorous path they specialise up to α ≈ 5
   (H ≈ 2.65–2.95), then **de-specialise** as α grows (H ≈ 3.05–3.25 at
   α = 30). Syntax and sem (|x| potentials) then carry the whole gap.
   They are less specialised than syntax/sem at 58 of the 100 points with
   α ≥ 3. A plausible mechanism, not tested here: the prox anchors each
   update to the slow-step reference, which limits how far mass can be
   transported towards the boundary per step under a steep linear
   potential.
4. **γ (Turing coupling) stays negligible** up to γ = 50 (turing
   strength 5) on both paths: ≤ 0.05 bits.

## γ = 0 slice

Gap and S in bits, mean ± SD over 10 seeds:

| α | β | gap fast | gap rigorous | S fast | S rigorous | rig. − fast (H) |
|---|---|---|---|---|---|---|
| 1 | 0.2 | 0.27 ± 0.03 | 0.19 ± 0.03 | 0.25 ± 0.03 | 0.03 ± 0.02 | +0.08 ± 0.04 |
| 1 | 0.5 | 0.23 ± 0.01 | 0.18 ± 0.03 | 0.21 ± 0.01 | 0.01 ± 0.00 | +0.05 ± 0.03 |
| 1 | 1 | 0.22 ± 0.01 | 0.18 ± 0.03 | 0.20 ± 0.01 | 0.01 ± 0.00 | +0.04 ± 0.03 |
| 1 | 2 | 0.22 ± 0.01 | 0.18 ± 0.03 | 0.19 ± 0.01 | 0.01 ± 0.00 | +0.04 ± 0.03 |
| 1 | 4 | 0.22 ± 0.01 | 0.18 ± 0.03 | 0.19 ± 0.01 | 0.01 ± 0.00 | +0.03 ± 0.03 |
| 1 | 8 | 0.22 ± 0.01 | 0.18 ± 0.03 | 0.19 ± 0.01 | 0.01 ± 0.00 | +0.03 ± 0.03 |
| 3 | 0.2 | 0.79 ± 0.04 | 0.70 ± 0.08 | 0.77 ± 0.04 | 0.54 ± 0.07 | +0.09 ± 0.07 |
| 3 | 0.5 | 0.81 ± 0.06 | 0.52 ± 0.08 | 0.79 ± 0.06 | 0.35 ± 0.08 | +0.30 ± 0.06 |
| 3 | 1 | 0.89 ± 0.03 | 0.52 ± 0.09 | 0.86 ± 0.03 | 0.34 ± 0.09 | +0.37 ± 0.08 |
| 3 | 2 | 0.91 ± 0.03 | 0.51 ± 0.09 | 0.89 ± 0.03 | 0.34 ± 0.09 | +0.40 ± 0.08 |
| 3 | 4 | 0.92 ± 0.03 | 0.47 ± 0.04 | 0.89 ± 0.03 | 0.30 ± 0.03 | +0.45 ± 0.05 |
| 3 | 8 | 0.92 ± 0.03 | 0.43 ± 0.04 | 0.89 ± 0.03 | 0.26 ± 0.03 | +0.49 ± 0.05 |
| 5 | 0.2 | 1.23 ± 0.03 | 0.95 ± 0.08 | 1.21 ± 0.03 | 0.79 ± 0.09 | +0.28 ± 0.06 |
| 5 | 0.5 | 1.33 ± 0.05 | 1.01 ± 0.09 | 1.31 ± 0.05 | 0.84 ± 0.08 | +0.32 ± 0.09 |
| 5 | 1 | 1.39 ± 0.02 | 0.86 ± 0.08 | 1.36 ± 0.02 | 0.69 ± 0.08 | +0.53 ± 0.07 |
| 5 | 2 | 1.41 ± 0.03 | 0.77 ± 0.08 | 1.38 ± 0.04 | 0.60 ± 0.09 | +0.64 ± 0.08 |
| 5 | 4 | 1.42 ± 0.03 | 0.71 ± 0.05 | 1.39 ± 0.03 | 0.54 ± 0.06 | +0.71 ± 0.06 |
| 5 | 8 | 1.42 ± 0.03 | 0.66 ± 0.03 | 1.40 ± 0.03 | 0.49 ± 0.03 | +0.76 ± 0.05 |
| 8 | 0.2 | 1.66 ± 0.02 | 1.20 ± 0.06 | 1.64 ± 0.02 | 1.04 ± 0.05 | +0.46 ± 0.07 |
| 8 | 0.5 | 1.80 ± 0.03 | 1.32 ± 0.08 | 1.77 ± 0.03 | 1.15 ± 0.08 | +0.47 ± 0.07 |
| 8 | 1 | 1.85 ± 0.02 | 1.19 ± 0.06 | 1.82 ± 0.02 | 1.02 ± 0.06 | +0.66 ± 0.07 |
| 8 | 2 | 1.87 ± 0.02 | 1.05 ± 0.04 | 1.84 ± 0.02 | 0.88 ± 0.03 | +0.81 ± 0.04 |
| 8 | 4 | 1.86 ± 0.01 | 1.00 ± 0.07 | 1.84 ± 0.02 | 0.82 ± 0.05 | +0.87 ± 0.08 |
| 8 | 8 | 1.86 ± 0.02 | 0.97 ± 0.05 | 1.83 ± 0.02 | 0.79 ± 0.04 | +0.90 ± 0.06 |
| 10 | 0.2 | 1.84 ± 0.02 | 1.31 ± 0.07 | 1.81 ± 0.02 | 1.14 ± 0.05 | +0.53 ± 0.07 |
| 10 | 0.5 | 1.95 ± 0.02 | 1.23 ± 0.05 | 1.93 ± 0.02 | 1.06 ± 0.03 | +0.72 ± 0.05 |
| 10 | 1 | 2.03 ± 0.03 | 1.24 ± 0.06 | 2.01 ± 0.03 | 1.07 ± 0.05 | +0.79 ± 0.05 |
| 10 | 2 | 2.02 ± 0.02 | 1.14 ± 0.07 | 2.00 ± 0.02 | 0.96 ± 0.06 | +0.89 ± 0.06 |
| 10 | 4 | 2.03 ± 0.02 | 1.10 ± 0.05 | 2.00 ± 0.02 | 0.92 ± 0.04 | +0.94 ± 0.05 |
| 10 | 8 | 2.04 ± 0.02 | 0.99 ± 0.03 | 2.01 ± 0.02 | 0.82 ± 0.03 | +1.05 ± 0.04 |
| 15 | 0.2 | 2.06 ± 0.01 | 1.44 ± 0.05 | 2.03 ± 0.01 | 1.27 ± 0.03 | +0.62 ± 0.05 |
| 15 | 0.5 | 2.33 ± 0.02 | 1.22 ± 0.04 | 2.30 ± 0.02 | 1.05 ± 0.02 | +1.11 ± 0.03 |
| 15 | 1 | 2.35 ± 0.03 | 1.25 ± 0.04 | 2.33 ± 0.03 | 1.08 ± 0.04 | +1.10 ± 0.04 |
| 15 | 2 | 2.38 ± 0.02 | 1.20 ± 0.05 | 2.35 ± 0.02 | 1.03 ± 0.05 | +1.17 ± 0.05 |
| 15 | 4 | 2.39 ± 0.02 | 1.17 ± 0.07 | 2.36 ± 0.02 | 0.99 ± 0.06 | +1.22 ± 0.06 |
| 20 | 0.2 | 2.16 ± 0.01 | 1.38 ± 0.05 | 2.14 ± 0.02 | 1.22 ± 0.04 | +0.78 ± 0.06 |
| 20 | 0.5 | 2.48 ± 0.01 | 1.23 ± 0.05 | 2.46 ± 0.02 | 1.06 ± 0.05 | +1.25 ± 0.04 |
| 20 | 1 | 2.56 ± 0.02 | 1.20 ± 0.07 | 2.53 ± 0.02 | 1.03 ± 0.06 | +1.35 ± 0.06 |
| 20 | 2 | 2.58 ± 0.03 | 1.18 ± 0.06 | 2.55 ± 0.03 | 1.01 ± 0.05 | +1.40 ± 0.06 |
| 20 | 4 | 2.58 ± 0.04 | 1.18 ± 0.06 | 2.55 ± 0.04 | 1.01 ± 0.05 | +1.40 ± 0.08 |
| 30 | 0.2 | 2.34 ± 0.02 | 1.29 ± 0.04 | 2.32 ± 0.02 | 1.13 ± 0.03 | +1.04 ± 0.04 |
| 30 | 0.5 | 2.84 ± 0.03 | 1.22 ± 0.05 | 2.82 ± 0.03 | 1.05 ± 0.04 | +1.62 ± 0.03 |
| 30 | 1 | 2.91 ± 0.02 | 1.20 ± 0.04 | 2.89 ± 0.03 | 1.03 ± 0.04 | +1.71 ± 0.03 |
| 30 | 2 | 2.94 ± 0.03 | 1.23 ± 0.05 | 2.91 ± 0.03 | 1.06 ± 0.06 | +1.71 ± 0.04 |
| 30 | 4 | 2.94 ± 0.03 | 1.23 ± 0.06 | 2.91 ± 0.03 | 1.06 ± 0.07 | +1.71 ± 0.04 |
## Per-species final entropy at β = 0.5, γ = 0 (bits, 10-seed mean)

| α | path | phono | lex | syntax | sem |
|---|---|---|---|---|---|
| 0 | fast | 3.98 | 3.98 | 3.98 | 3.98 |
| 0 | rigorous | 3.83 | 3.83 | 3.82 | 3.84 |
| 3 | fast | 2.95 | 2.98 | 3.34 | 3.48 |
| 3 | rigorous | 3.22 | 3.31 | 3.69 | 3.71 |
| 5 | fast | 2.33 | 2.35 | 2.92 | 3.07 |
| 5 | rigorous | 2.65 | 2.70 | 3.34 | 3.26 |
| 8 | fast | 1.89 | 1.87 | 2.62 | 2.43 |
| 8 | rigorous | 2.69 | 2.70 | 3.12 | 2.19 |
| 10 | fast | 1.77 | 1.74 | 2.18 | 2.51 |
| 10 | rigorous | 2.96 | 2.99 | 2.66 | 2.47 |
| 15 | fast | 1.39 | 1.40 | 2.02 | 1.89 |
| 15 | rigorous | 3.05 | 3.11 | 2.49 | 2.49 |
| 20 | fast | 1.24 | 1.25 | 1.74 | 1.86 |
| 20 | rigorous | 3.09 | 3.15 | 2.20 | 2.63 |
| 30 | fast | 0.89 | 0.89 | 1.32 | 1.55 |
| 30 | rigorous | 3.15 | 3.24 | 2.16 | 2.59 |
