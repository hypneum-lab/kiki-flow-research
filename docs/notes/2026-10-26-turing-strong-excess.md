# Strong-Turing excess over Gibbs: result of the L1 preregistration

Date: 2026-10-26. Preregistration:
`docs/prereg/2026-10-26-turing-strong-excess.md`, frozen by the annotated
tag `prereg-turing-strong-20261026` on commit `681c966` (tagger date
2026-10-26T22:02:58Z, pushed before the run). Results:
`results/turing_strong_excess_2026_10_26/` (`partial.jsonl`, 200 records;
`excess.json`; `summary.md`).

## Execution

- **Host and run:** electron-server, `nice 19`, 6 workers. The checkout
  was the tagged commit (`git describe` → `prereg-turing-strong-20261026`).
- **Timing:** started 2026-10-26T22:03:08Z, 10 s after the tag. The 200
  runs ended at about 22:16Z (≈ 23 s per run). The analysis (same
  commit, `analyse`) ran at 22:19Z.
- **Deviations:** none. All 200 preregistered runs completed. Nothing was
  rerun, no other seed was used, and the analysis code is the one frozen
  by the tag.

## Result (preregistered primary statistic)

The primary statistic is d = [gap − gap_Gibbs] − gap(α = 0, same γ,
seed), in bits. Each cell reports the mean over seeds 100–109, the
95 % bootstrap CI, and the Holm-adjusted one-sided p over 15 cells.

| α \ γ | 20 | 30 | 50 | 70 | 100 |
|---|---|---|---|---|---|
| 2 | −0.891 [−1.002, −0.779]; 1 | −0.749 [−0.927, −0.546]; 1 | +0.324 [−0.118, +0.656]; 0.89 | −0.292 [−0.666, +0.043]; 1 | −1.133 [−1.333, −0.915]; 1 |
| 5 | +0.075 [−0.043, +0.204]; 1 | **+0.248 [+0.102, +0.390]; 0.0064** | +0.267 [+0.061, +0.419]; 0.11 | −0.726 [−0.992, −0.485]; 1 | −1.998 [−2.148, −1.849]; 1 |
| 10 | −0.121 [−0.197, −0.047]; 1 | −0.039 [−0.097, +0.022]; 1 | −0.143 [−0.313, −0.010]; 1 | −1.137 [−1.399, −0.896]; 1 | −2.399 [−2.529, −2.274]; 1 |

**Verdict under the preregistered rule: "supported", restricted to one
cell, (α, γ) = (5, 30).** In that cell the excess is +0.248 bits, the
Holm p is 0.0064, and the one-sided lower bound at its Holm level is
+0.045. The other 14 cells are "inconclusive".

The original observation, (5, 50) on seeds 0–9 (+0.223
[+0.132, +0.308]), is **not confirmed after Holm**. On the new seeds it
gives +0.267 [+0.061, +0.419], with an unadjusted p of 0.0078 but a Holm
p of 0.11.

## Descriptive (not part of the decision)

- **Excess without the α = 0 control:** positive in 13 of 15 cells,
  every cell except α = 2 with γ ∈ {20, 30}. The final states are more
  concentrated than the Gibbs state of the potentials alone throughout
  the strong-Turing regime.
- **The α = 0 control is not a pure finite-particle bias at large γ.**
  Its gap grows with γ: 0.22, 0.21, 0.31, 1.32 and 2.59 bits at γ = 20,
  30, 50, 70 and 100. The Turing term alone structures the species even
  with no potential. The preregistration anticipated this limit (§5:
  the control removes the bias "only to the extent that the bias does
  not depend on α"). For γ ≥ 70 the primary statistic subtracts a
  Turing-driven structure, not a bias. Its strongly negative values
  there do not mean "less specialised than equilibrium".
- **Full-energy reference:** the damped fixed point did not converge in
  any cell (300 iterations, residual 0.01–0.43). The column is reported
  in `summary.md` and is not interpretable.

## Reading

Within the preregistered frame, one cell shows a robust positive excess
over the Gibbs equilibrium of the potentials after removing the α = 0
gap. Three things make it fragile:

- It is isolated. Its neighbours (5, 20) and (5, 50) are positive but
  inconclusive.
- It depends on the control, which itself becomes Turing-dominated as
  γ grows.
- Per §5 of the preregistration, it does not establish that
  specialisation emerges beyond the equilibrium of the full energy F.
  The Turing term is part of F, and the reference that would settle
  this (the minimiser of the full energy) is not available in this
  regime.

The defensible statement is: "under a strong Turing term (γ = 30,
α = 5), the final state is more specialised than the equilibrium of the
potentials, beyond what the same Turing term produces without
potentials".

## What would settle it (not run)

- A converged minimiser of the full T2 energy for γ ≥ 20. For example:
  run the entropic JKO step to stationarity with large h, or use a
  proximal method on the non-smooth Turing term.
- With that minimiser, test the excess over the equilibrium of F
  itself, under a new preregistration.
