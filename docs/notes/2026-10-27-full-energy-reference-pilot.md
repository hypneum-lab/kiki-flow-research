# Full-energy reference for the strong-Turing grid: pilot (2026-10-27)

**Outcome: the reference does not converge in any of the 15 cells, so no
preregistration was frozen, no tag was pushed and no run was made. The
pilot also shows that, for gamma ≥ 10, the default slow step is not
dissipative. The positive cell of the 2026-10-26 preregistration, (5, 30),
is therefore not interpretable.**

Decision: PI, QCM of 2026-10-26 22:25 (issue #31). The plan was to:

1. compute a reference for the full T2 energy F (J_sym coupling plus the
   Turing term) that converges for γ ≥ 20;
2. check its convergence on the 15 cells of
   `docs/prereg/2026-10-26-turing-strong-excess.md` with two distinct pilot
   seeds;
3. only then freeze a preregistration (primary statistic gap − gap_Fmin,
   seeds 200–209, Holm over the cells, cells with a non-converged reference
   excluded).

Code: `experiments/full_energy_excess_2026_10_26.py` (the `pilot`
subcommand) and `experiments/turing_dissipation_check_2026_10_27.py`.
Artefacts: `results/full_energy_excess_2026_10_26/`. Everything ran on
electron-server with `nice 19`.

## 1. Entropic JKO step iterated to stationarity (the requested method)

Probe: `pilot_jko_probe.log`. At (5, 30), starting from the Gibbs state of
the potentials (F = −17.13), the step was run with h = 0.5 and reg = 1e-3
or 1e-2, the Sinkhorn iteration capped at 5000 or 500. In every
configuration:

- **F increases**, to between −10 and −4;
- the iterates oscillate with period 2;
- each step moves W2 by about one cell (0.27).

Two causes:

- The step linearises the non-smooth Turing term,
  Σ_{i<k} Σ_x |∇ρ_i ∇ρ_k|, at ρ_τ. This treatment is explicit, so the
  scheme is not dissipative.
- On the 16-point grid, a small Sinkhorn reg pins mass to the grid (the
  neighbour kernel weight is e^(−71) at reg = 1e-3). A large reg instead
  biases the fixed point by O(reg / 2h).

This method cannot provide the reference.

## 2. Monotone mirror descent on F (substitute)

The minimiser of F does not depend on the metric. We therefore used mirror
descent (exponentiated gradient, KL geometry), which only accepts steps
that decrease F.

- **Finite-difference gradient** (`pilot_mirror_fd_probe.log`): the descent
  stalls before stationarity at γ = 30 (ΔF ≈ 2e-4, W2 ≈ 5e-3), and the
  initialisations disagree (gap 1.92–2.00).
- **Smoothed Turing term with exact gradient**: |u| → √(u² + δ²) − δ,
  continuation δ = 1e-2 → 1e-6, warm-started stage by stage
  (`pilot_smoothed_probe.log`). The analytic gradient matches finite
  differences to 2e-5, and F_δ → F. At (5, 10) and (5, 30), all three
  initialisations reach the same F to 1e-8, and the gap is stable in δ to
  1e-4.

## 3. Pilot on the 15 cells plus 3 validation cells

File: `pilot.json`.

- **Two pilot seeds:** 1001 and 1002, each run with 3 initialisations
  (Gibbs, uniform, Dirichlet).
- **Convergence criterion, per δ stage:** five consecutive accepted steps
  with ΔF < 1e-10 and W2 < 1e-6, or a stalled line search with a
  first-order stationarity residual below 1e-6.
- **Further conditions:** every stage must converge, the gap must be stable
  to within 1e-3 between the last two δ, and each stage is capped at 20 000
  steps.

Results:

| cells | converged (seed 1001 / seed 1002) | final-stage residual | gap across initialisations |
|---|---|---|---|
| validation (10, 0), (8, 5) | 3/3 and 3/3 | ≤ 1e-7 | identical (F range ≤ 2e-12) |
| validation (5, 10) | 0/3 and 1/3 | 3.9e-6 | identical |
| the 15 test cells (γ ≥ 20) | **0/3 and 0/3 everywhere** | 1e-5 to 4e-2 (cap of 20 000 steps reached) | equal to ~1e-4 in 9 cells; distinct local minima in 6 cells (F range up to 2e-2, gap difference up to 0.008 bits) |

At the last δ stage the problem is stiff (curvature ~ 1/δ), and a
first-order method does not reach stationarity. The approximate minimum
nonetheless pins the gap to about 0.01 bit:

- (5, 30): 2.039;
- (5, 20): 2.054;
- (10, 30): 2.758–2.765;
- (2, 20): 1.028.

By the rule decided in advance (a cell whose reference does not converge
is excluded), **all 15 cells would be excluded**. The test would be empty,
so it was not frozen and not run.

## 4. Finding: the default slow step is not dissipative for γ ≥ 10

File: `dissipation_check.log`. The default `FullJKOSolver` (h = 0.05,
reg = h²) was iterated on the grid from the Gibbs state at α = 5, 30 steps
per cell:

| γ | F start → end | steps with F increasing |
|---|---|---|
| 0 | −19.658 → −19.705 | 0/30 |
| 5 | −19.237 → −19.585 | 0/30 |
| 10 | −18.816 → −16.883 (max −15.39) | 16/30 |
| 20 | −17.974 → −14.054 (max −10.95) | 15/30 |
| 30 | −17.132 → −10.938 (max −7.63) | 15/30 |

File: `final_state_energy.log`. F at the final states of the 2026-10-26
preregistered runs (seeds 100–109), compared with the approximate minimum
and with the Gibbs state of the potentials:

| cell | F of the final states, mean [min, max] | F_min (approx.) | F(Gibbs of the potentials) |
|---|---|---|---|
| (5, 20) | −13.07 [−17.02, −8.49] | −19.44 | −17.97 |
| (5, 30) | −10.66 [−17.30, −4.57] | −19.34 | −17.13 |
| (5, 50) | −7.10 [−13.00, −2.77] | −19.20 | −15.45 |
| (10, 30) | −36.97 [−42.79, −29.58] | −46.54 | −41.68 |
| (5, 100) | +6.13 [−1.16, +13.93] | −19.03 | −11.24 |

Consequences:

- **The "excess > 0" verdict at (5, 30) in
  `docs/notes/2026-10-26-turing-strong-excess.md` is not interpretable as
  specialisation.** The final states lie far above the minimum of F, and
  even above the Gibbs state of the potentials. The extra concentration
  comes from high-energy states that a non-dissipative step leaves behind,
  not from relaxation toward an equilibrium.
- The same applies to every γ ≥ 10 value of PR #46 and of
  `results/excess_over_gibbs_2026_10_25/` (γ ∈ {10, 20, 50}). This
  includes the positive excesses at γ ≥ 20 that motivated the 2026-10-26
  test.
- The step is dissipative for γ ≤ 5. The paper's operating points,
  (10, 1, 0) and (8, 0.5, 5), and the conclusion "no specialisation beyond
  equilibrium" (ERRATUM §13) are not affected.

## 5. Proposals (decisions left to the PI)

1. **ERRATUM (to be added, status PROPOSED):**
   - the default step is not dissipative for γ ≥ 10;
   - results at γ ≥ 10 under the entropic step (PR #46, the excess
     analysis, PR #48) are not interpretable;
   - the "supported" verdict of the 2026-10-26 preregistration is
     withdrawn as an artefact.

   Not written in this PR, pending validation.
2. **Fix the step before any new test:** treat the Turing term implicitly
   inside the prox, using the smoothed |·| with its exact gradient and a
   few inner fixed-point iterations (or a proximal-gradient split). Check
   with `turing_dissipation_check_2026_10_27.py step` that F decreases at
   every step for γ up to 100.
3. **Reference:** replace the first-order mirror descent with a
   second-order method on the smoothed problem (Newton or L-BFGS in the
   softmax parameterisation) so that the δ = 1e-6 stage reaches the
   residual. Alternatively, accept the approximate minimum with an explicit
   ±0.01-bit uncertainty, which would be a change of criterion, to be
   decided before any preregistration.
4. Only then, a new preregistration on the strong-Turing grid.
