# External reviewer critique — kiki-flow-research

> Status: external review (deep-research harness, findings verified 3-0), 2026-06-03.
> Leads to confirm.

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
