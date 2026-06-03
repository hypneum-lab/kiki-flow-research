# Experiment backlog — kiki-flow-research

From the external review (REVIEW.md). Run in a fresh session with the repo env
(`uv sync --all-extras`). Each experiment registers an R1 `run_id` and bumps EC if it
changes a claim.

## E1 — ε→0 sweep (quantify the spurious (α/2)Δρ diffusion)  [light, no large model]

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

## E2 — N-particle Langevin propagation-of-chaos preconditions  [analysis]

**Why**: `track2_paper/mlx_particle_simulator.py` claims a mean-field Langevin × JKO
proxy. Propagation-of-chaos (N-particle → mean-field) needs convexity, Lipschitz gradient,
uniform log-Sobolev — none established for the non-convex multi-species free energy.

**Protocol**: numerically estimate (a) the free-energy Hessian eigenvalues (convexity?),
(b) the gradient Lipschitz constant on the simplex, (c) a log-Sobolev constant proxy.
**Acceptance**: either exhibit the constants (→ proxy justified) or document their absence
(→ "proxy" is heuristic; restate).
