# External reviewer critique — kiki-flow-research

> Status: external review (deep-research harness, findings verified 3-0), 2026-06-03.
> Resolution pass merged 2026-06-04 — see "Resolution" block below.

## Resolution (2026-06-04, merged)

The critique was investigated, not deflected. Outcome (see `EXPERIMENT-BACKLOG.md`
for full verdicts and artifacts):

- **The discrepancy is real and now quantified, not hand-waved.** Investigating it
  first surfaced a bug: `prox_w2` was a *fake* proximal operator (its gradient term
  `transport @ cost.diagonal()` was identically zero), so the prior ε-sweep measured a
  parasitic floor, not the entropic bias. Fixed to the true entropic W2 prox (Peyré
  Dykstra scaling on the Gibbs kernel; commit `ffee5d9`).
- **Open question 4 (numerical magnitude / would ε→0 change the conclusion):** the
  pre-registered E1b OU test shows the parasitic diffusion is the **real entropic
  Peyré model bias, grid-invariant (321→4001: 0%)**, not a discretisation artefact;
  no-entropy null ≈ 2.8e-15. The stationary floor is `Var* = (ε/τ)·g(τ)` with
  `g(τ)→0.5` as τ→0, i.e. the textbook `ε/(2τ)` law. At production ε=0.01 this floor
  is non-negligible vs the density scale. **The reviewer is correct**: the
  "exact-Wasserstein rigorous proxy" framing is overstated and is now read as an
  entropic-regularized JKO with a quantified, qualified bias.
- **Open question 1 (PoC / mean-field faithfulness):** E2 confirms PoC preconditions
  are met only **conditionally** (entropy-dominated regime). `J_sym` is indefinite
  (min eig −0.288); convexity comes solely from the strongly-convex KL term and is not
  uniform → **no uniform log-Sobolev constant**. The "rigorous proxy" label stays
  qualified.
- Open questions 2 (psycholinguistic empirical content) and 3 (consolidation
  baseline) remain open.

## Verdict
The Optimal-Transport framing is **substantive, not decorative** — but the
**"rigorous proxy" label is overstated**: the implemented scheme provably converges to a
different PDE than the advertised exact-Wasserstein gradient flow.

## Core critique (verified 3-0)
`track2_paper` uses POT **entropic Sinkhorn at fixed ε=0.01** (paper/main.tex:294-295,
kiki_flow_core/wasserstein_ops.py). Established OT theory shows **entropic-JKO does not
converge to the same PDE as vanilla exact-Wasserstein JKO**: with ε=α·τ the limiting PDE
gains a linear diffusion term **(α/2)·Δρ** — the flow is heat-diffused unless ε/τ → 0
(arXiv 2502.12666, 2604.08283). A convergence rate is provable only **under convexity +
joint vanishing of α and τ** (Baradat & Cherf 2026); the non-convex multi-species free
energy meets neither condition and the paper does not cite the result.

- **Partial mitigation acknowledged**: ε-sweep (PERFORMANCE.md:42-56) + debiased Sinkhorn
  divergence (Feydy 2019, main.tex:211-212) — but no formal bound.
- **Honest nuance**: arXiv 2604.08283 is a non-peer-reviewed April-2026 preprint; the code
  mitigations likely make the practical bias smaller than worst-case → the critique is an
  **uncharacterized discrepancy**, not a demonstrated large error.

## Open questions
1. Is the N-particle Langevin × JKO "rigorous proxy" a faithful mean-field discretization?
   Propagation-of-chaos needs convexity, Lipschitz derivative, uniform log-Sobolev — none
   established for the non-convex free energy.
2. Does the psycholinguistic mapping (phono/lex/syntax/sem as Wasserstein densities on a
   simplex) have empirical content, or is it analogy? WEAVER++ is staged feedforward with
   no co-equal syntax density — in tension with four simultaneously-evolving species.
3. Is "consolidation" operationally defined and measured against a real baseline? Is
   track3's sub-ms advisory routing benchmarked against a genuine comparator?
4. What is the numerical magnitude of the entropic-diffusion discrepancy at ε=0.01 in the
   reported trajectories — would sweeping ε→0 with τ change the conclusions?
