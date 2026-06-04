# Experiment backlog — kiki-flow-research

From the external review (REVIEW.md). Run in a fresh session with the repo env
(`uv sync --all-extras`). Each experiment registers an R1 `run_id` and bumps EC if it
changes a claim.

## Status summary (2026-06-04)

| Exp | Status | Verdict |
|---|---|---|
| E1  | **done** | After fixing a fake `prox_w2` (see below), `Var(ρ_K)` decreases **monotonically** toward the exact OU reference `T(x)=x/(1+τ)`; the invariant parasitic floor (~10.48) is gone. Caveat: low-reg runs needed a grid refinement (Gibbs-kernel underflow). |
| E1b | **done** | Pre-registered OU test: the parasitic diffusion is the **real entropic Peyré model bias, not a grid artefact** (grid-invariant 321→4001: 0%; no-entropy null ≈ 2.8e-15). Prereg slope band [0.45,0.55] honestly **REFUTED at τ=1** (slope 0.60), then reconciled by τ→0 limit recovering the textbook `0.5 = ε/(2τ)`. No HARKing. |
| E2  | **done** | PoC preconditions only **conditionally** met (entropy-dominated regime). Convexity on the simplex tangent holds only because the strongly-convex KL entropy dominates the **indefinite** symmetric coupling `J_sym` (min eig −0.288); no uniform log-Sobolev constant. The "rigorous proxy" label must stay qualified. |

**Root-cause fix enabling E1/E1b** (commit `ffee5d9`): the original `prox_w2`
gradient term `transport @ cost.diagonal()` was identically zero (the sqeuclidean
self-distance diagonal is null), so the JKO step degenerated into pure anti-KL
repulsion and E1's variance was a constant parasitic floor (~10.48), invariant in ε.
Replaced with the **true entropic W2 proximal operator** (Peyré: Dykstra scaling on
the Gibbs kernel), validated against finite differences and a Gaussian-variance
contraction test.

## E1 — ε→0 sweep (quantify the spurious (α/2)Δρ diffusion)  [light, no large model]

> **DONE** — see status table above. Artifacts: `experiments/e1_epsilon_sweep_var.py`,
> `experiments/results/e1_epsilon_sweep_var.json`. At production ε=0.01 (τ_total=1)
> the spurious floor `ε/(2τ)` is non-negligible vs the density scale, so the
> "exact-Wasserstein rigorous proxy" label is qualified (now read as
> entropic-regularized JKO). `Var(ρ_K)` is monotone decreasing in ε toward the
> exact reference. Grid refinement was required at low reg (kernel underflow) — this
> resolvability issue, not the bias value, was the source of the E1 grid ambiguity
> later closed by E1b.

**Why**: `track2_paper` uses POT entropic Sinkhorn at fixed ε=0.01
(`kiki_flow_core/wasserstein_ops.py`, `track2_paper/full_jko_solver.py`: ε=0.01, h=0.05).
Theory (arXiv 2502.12666, 2604.08283): with ε=α·τ the entropic-JKO limiting PDE gains
`(α/2)·Δρ`; the scheme is NOT the exact-W2 gradient flow unless ε/τ → 0.

**Protocol**:
1. Canonical 1D potential energy `F[ρ] = ∫ V ρ`, `V(x)=x²/2` (OU contraction; exact-W2 JKO
   map is analytic: `T(x)=x/(1+τ)`, pure transport, NO diffusion).
2. Run `FullJKOSolver` at fixed τ for ε ∈ {1e-2, 5e-3, 2e-3, 1e-3, 5e-4}, K steps.
3. Measure `Var(ρ_K)` and compare to the analytic exact reference `Var0·(1+τ)^(-2K)`.
4. **Acceptance**: `Var_entropic − Var_exact` scales ≈ linearly in ε and → 0 as ε→0;
   the stationary spurious floor ≈ `ε/(2τ)`.
5. **Deliverable number**: report the floor at the production ε=0.01 and the actual τ —
   if `ε/(2τ)` is non-negligible vs the density scale, the "rigorous proxy" label must be
   qualified (debiased Sinkhorn divergence partially cancels it; quantify residual).

**Decision**: if the discrepancy is non-negligible → either drive ε→0 with τ (and cite
2604.08283's convexity + joint-vanishing rate), or restate track2 as "entropic-regularized
JKO" not "exact-Wasserstein rigorous proxy".

## E1b — pre-registered OU test: grid artefact vs real Peyré bias  [light]

> **DONE** — see status table above. Artifacts:
> `experiments/e1b_peyre_bias_preregistered.py`,
> `experiments/results/e1b_peyre_bias.json`. Thresholds were fixed before the run
> (no HARKing). Findings:
> - **Claim 1.1 (slope = ε/(2τ))**: pre-spec band [0.45,0.55] **REFUTED** at τ=1 —
>   measured slope 0.60 (R²=1.00). Kept as-is; reconciled as a finite-step fixed
>   point `Var* = (ε/τ)·g(τ)` with `g(τ): 0.60 → 0.556 → 0.512 → 0.502 → 0.5` as
>   τ→0, recovering the textbook Peyré 2015 first-order slope 0.5.
> - **Claim 1.3 (model vs grid)**: grid-invariant (321 vs 1281 → 0.0% change,
>   confirmed to 4001 pts) → **REAL Peyré model bias, NOT a discretisation artefact**.
>   The E1 grid issue affected low-reg *resolvability*, not the bias value.
> - **Null control (exact-W2, no entropy)**: `Var* ≈ 2.8e-15`, slope ≈ 0 → all the
>   diffusion is entropic; pure transport collapses to a Dirac as expected.
> Validates the `prox_w2` fix and closes the grid-vs-theory ambiguity left open by E1.

## E2 — N-particle Langevin propagation-of-chaos preconditions  [analysis]

> **DONE** — see status table above. Artifacts: `experiments/e2_poc_preconditions.py`,
> `experiments/results/e2_poc_preconditions.json`. Verdict: PoC preconditions only
> **conditionally** met. At grid=8 with a uniform prior the total free energy is convex
> on the simplex tangent (worst tangent Hessian eigenvalue 2.831 > 0) and
> finite-Lipschitz (L ≈ 18.5), giving a Bakry–Émery LSI const ≥ 2.831 — **but** this
> convexity is supplied entirely by the strongly-convex KL entropy term; the symmetric
> coupling `J_sym` is indefinite (min eig −0.288 < 0). Convexity is therefore
> conditional on entropy strength vs ‖J_sym‖, grid resolution and prior sharpness, and
> is **not uniform**, so no uniform log-Sobolev constant is established. The
> "rigorous proxy" / propagation-of-chaos label is supported only in the
> entropy-dominated regime and is qualified as such.

**Why**: `track2_paper/mlx_particle_simulator.py` claims a mean-field Langevin × JKO
proxy. Propagation-of-chaos (N-particle → mean-field) needs convexity, Lipschitz gradient,
uniform log-Sobolev — none established for the non-convex multi-species free energy.

**Protocol**: numerically estimate (a) the free-energy Hessian eigenvalues (convexity?),
(b) the gradient Lipschitz constant on the simplex, (c) a log-Sobolev constant proxy.
**Acceptance**: either exhibit the constants (→ proxy justified) or document their absence
(→ "proxy" is heuristic; restate).
