# Implicit treatment of the Turing term in the entropic JKO step (2026-10-27)

**Status.** The fix is behind an option and the default is unchanged
(`nonseparable="explicit"`). The decision to switch the default is left to
the PI, and the PR proposes it.

**Origin.** PI decision of 2026-10-27 06:48. ERRATUM §14 found that the
default slow step does not make F decrease for γ ≥ 10.

**Files.**

- Code: `kiki_flow_core/master_equation.py` (`JKOStep(nonseparable=...)`)
  and `kiki_flow_core/track2_paper/full_jko_solver.py`, which forwards
  `nonseparable` and `prox_tol`.
- Check: `experiments/turing_implicit_check_2026_10_27.py`.
- Results: `results/turing_implicit_check_2026_10_27/`, computed on
  electron-server with `nice 19` and 6 workers, between 11:52Z and 13:06Z.
- Tests: `tests/test_jko_implicit_nonseparable.py`.

## What changes

The terms of F that have no closed-form prox are the J_sym coupling and
the Turing term Σ|∇ρ_i ∇ρ_k|. The explicit mode linearises them once, at
rho_tau. The implicit mode (`nonseparable="implicit"`) works in two
stages:

1. **Implicit (backward-Euler) treatment.** The point at which these terms
   are linearised is iterated until it equals the step's own output. This
   is a damped fixed point over all species jointly (damping 0.5, up to 30
   iterations, tolerance 1e-9 in TV).
2. **Safeguard.** If F(output) > F(input), the step is split into two
   implicit half-steps, recursively, up to 8 levels. The Sinkhorn
   regularisation is scaled by (h_sub / h)² so that reg / h² stays
   constant.

## Check 1: F never increases (criterion: 0)

**Setup.**

- Grid: γ ∈ {0, 5, 10, 20, 30, 50, 70, 100} × α ∈ {2, 5, 10} ×
  h ∈ {0.05, 0.025}.
- Starting states: two per cell, drawn per species from Dirichlet(1) with
  seeds 500 and 501.
- 10 slow steps per state on the grid, reg = h², Sinkhorn early stop at
  1e-12.
- The explicit mode runs on the same states as a comparison.

**Results.** Each cell gives the number of steps with F increasing by more
than 1e-12, implicit / explicit (20 steps per cell):

| γ | α=2, h=0.05 | α=2, h=0.025 | α=5, h=0.05 | α=5, h=0.025 | α=10, h=0.05 | α=10, h=0.025 |
|---|---|---|---|---|---|---|
| 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 5 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 10 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 20 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 5 | 0 / 0 |
| 30 | 0 / 0 | 0 / 0 | 0 / 5 | 0 / 0 | 0 / 5 | 0 / 1 |
| 50 | 0 / 6 | 0 / 0 | 0 / 6 | 0 / 0 | 0 / 6 | 0 / 4 |
| 70 | 0 / 17 | 0 / 5 | 0 / 8 | 0 / 6 | 0 / 8 | 0 / 4 |
| 100 | 0 / 17 | 0 / 18 | 0 / 13 | 0 / 11 | 0 / 9 | 0 / 8 |

**Implicit mode: 0 increases over the 960 steps, so the check passes.** The
largest change of F in a single step is −2.1e-14. The safeguard split a
step at most 3 levels deep (at γ = 50).

The explicit mode increases F in 37 of the 96 configurations. From these
random starting states it does not increase F at γ = 10. From the Gibbs
state, the 2026-10-27 dissipation check found 16 increases out of 30 at
γ = 10.

**Cost.** 15 to 22 s per slow step in implicit mode (grid 16, 4 species),
against about 0.6 s in explicit mode, i.e. 25 to 35 times slower.

## Check 2: are the γ ≤ 5 results unchanged?

**(a) Step level.** For γ ∈ {0, 1, 5} and α ∈ {2, 5, 10}, both modes were
applied to the same input, over 10 consecutive steps. The maximum TV
between the two outputs is 0.023–0.052. That maximum always occurs at the
first step from the random start; after 10 steps the TV is 3e-5 to 3e-3.
**The tolerance of 1e-3 set in the driver is not met at the first step.**
The two modes linearise the coupling at different points, and the
difference is largest when the step moves far from rho_tau. Both modes
tend toward the same states.

**(b) The paper's key points.** Full multiscale runs (500 particles,
200 fast × 30 slow steps, reinjection, seeds 0–4) in implicit mode,
compared with the explicit runs of PR #46 on the same seeds:

| (α, γ) | implicit gap | explicit gap (#46) | paired difference (mean ± sd) | within ±0.05 bit |
|---|---|---|---|---|
| (10, 0) | 2.708 | 2.728 | −0.020 ± 0.015 | yes |
| (8, 5) | 2.393 | 2.434 | −0.041 ± 0.032 | yes |

Both points are within the declared tolerance. The shift is nonetheless
systematic: the implicit mode gives slightly smaller gaps, by 0.02–0.04
bits. The excess over Gibbs at these points stays clearly negative
(−0.39 and −0.49 bits under the explicit step), so the conclusion "no
specialisation beyond equilibrium" is not affected.

## Proposal (not applied): switch the default

The fix passes the dissipation criterion everywhere it was tested and
keeps the paper's key points within ±0.05 bit. We therefore propose making
`nonseparable="implicit"` the default of the entropic scheme, with three
caveats:

- **Cost.** Each slow step is 25 to 35 times slower. The 330-run sweep of
  #46 would take on the order of 15 to 20 h on electron-server instead of
  42 min. The paper-scale run (grid 64) would be correspondingly longer.
- **Shift.** The γ ≤ 5 gaps move by −0.02 to −0.04 bits. Results would
  have to be quoted with their scheme, as for §12–14 of the ERRATUM.
- **Validity of earlier results.** Values at γ ≥ 10 remain not
  interpretable until they are recomputed with the implicit mode
  (ERRATUM §14).

No new preregistration and no confirmatory run were made (PI instruction).
