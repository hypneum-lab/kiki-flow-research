# Is `JKOStep` a JKO step? Identification of the step of `master_equation.py`

Date: 2026-10-25. Scope: NH-60 prerequisite (issue #31, PI decision of
2026-10-25) raised by the cross review of 2026-10-25 (annex 12 of the
hypotheses review, `hypneum-lab/.github`, point B5): before any
coarsening sweep, establish whether the step of
`kiki_flow_core/master_equation.py:91-116` (as run by
`track2_paper/full_jko_solver.py`) is the JKO step
`argmin_rho F(rho) + W2^2(rho, rho_tau) / (2h)`.

Companion code: `experiments/jko_step_check_2026_10_25.py`, results in
`results/jko_step_check_2026_10_25/` (`results.json`, `summary.md`,
`fig_jko_step_check.{png,pdf}`), tests in
`tests/test_jko_step_check_2026_10_25.py`. Code at `771dc69`.

## 0. Conclusion first

**Not identified.** The step is the composition of two maps, neither of
which is a JKO step of F, and whose composition is not one either, at
any order in h:

1. an explicit Euler integration, over time h in `n_inner` sub-steps, of
   the *Euclidean* (ℓ², reaction-type) gradient flow of F restricted to
   the simplex — a pointwise relaxation with no transport;
2. an entropic JKO step in the sense of Peyré (2015) — correct as such —
   but for the functional `KL(. || d)`, where `d` is the output of (1),
   not for F. It equals the entropic JKO step of F only when F is
   itself `KL(. || pi)` with unit temperature and `d = pi`, i.e. never
   along a trajectory.

Numerically (section 3), on a 48-point 1-D support with three energies
whose exact JKO minimiser is computed to a certified gap ≤ 3e-8, the W2
distance between the rigorous step (`FullJKOSolver`) and the exact JKO
minimiser is 60 % (quadratic potential + entropy), 64 % (double well +
entropy) and 14 % (entropy alone) of the JKO displacement at the paper's
h = 0.05, and it **grows** as h decreases (94 %, 125 %, 49 % at
h = 0.00625; against the un-pinned Fokker–Planck flow: 94 %, 114 %,
49 %). The fast path (no prox) is farther from the JKO minimiser than
rho_tau itself for h ≤ 0.1 and violates the dissipation inequality in
50/63, 22/63 and 38/63 configurations; the rigorous path violates it at
the smallest h (53/252, 62/252, 9/252). Hence no exponent prediction for
NH-60 can rest on this step, and the sweep is not launched. Section 5
gives the minimal modification that turns the step into an entropic JKO
step of F at *lower* cost than the present code; its error is the
entropic bias (section 3.2.1, `Peyré` column), already measured.

## 1. What the step computes

### 1.1 Notation

Support `x ∈ R^N` (`support`, N points), cost `C_ij = |x_i − x_j|^2`,
simplex Δ_N, current iterate `rho_tau ∈ Δ_N`, free energy
`F : Δ_N → R`, JKO time step `h` (`JKOStep.h`), inner count `n_inner`
(default 50), sub-step `s = h / n_inner` (line 93), clip floor 1e-12.
Prox parameters: pull `epsilon = h` (line 110), Sinkhorn regularisation
`γ = reg` (default `None` → `γ = h`, floored at 1e-3,
`wasserstein_ops.py:79`), Dykstra count `M = prox_n_iter` (100 for a
plain `JKOStep`, `max_iter = 1000` through `FullJKOSolver` since
ERRATUM §10). `FullJKOSolver.epsilon` is forwarded as `γ` since ERRATUM
§9; it is *not* the proximal pull. Numerical gradient `ĝ = grad_rho`:
the base class perturbs one coordinate by `eps = 1e-4`, renormalises and
differences F, so `ĝ_i = ∂_i F − ⟨∇F, rho⟩ + O(eps)` (the derivative
along the tangent `e_i − rho`); `T2FreeEnergy` does not override it.

### 1.2 Phase A — projected Euclidean descent (lines 98–104)

```
rho^(0) = rho_tau
for m = 1 .. n_inner:
    rho^(m) = Π( rho^(m-1) − s · ĝ(rho^(m-1)) ),   Π(v) = max(v, 1e-12) / Σ max(v, 1e-12)
d := rho^(n_inner)
```

With `Π(rho − s ĝ) = rho − s ĝ + s rho ⟨1, ĝ⟩ + O(s^2)`, Phase A is the
explicit Euler scheme, over total time h, of the ODE

    d rho_i / dt = −ĝ_i(rho) + rho_i ⟨1, ĝ(rho)⟩,

a gradient descent of F in the Euclidean metric of R^N retracted onto
the simplex (radial retraction, not the orthogonal projection). Its
defining property: `d rho_i / dt` depends on the local value `∂_i F`
and on global scalars only. Mass changes *in place*; there is no flux
between neighbouring cells. In the continuum this is the reaction-type
flow `∂_t rho = −(δF/δrho − λ(t)) + μ(t) rho` (Euclidean / "L²"
gradient flow), whereas the Wasserstein gradient flow of F is the
continuity equation `∂_t rho = ∇·(rho ∇ δF/δrho)` (mobility rho), of
which the JKO step is the implicit Euler scheme in the W2 geometry
(Jordan–Kinderlehrer–Otto 1998). For `F = ∫ V rho + θ ∫ rho log rho`
the two flows are respectively

    ∂_t rho = −V − θ log rho + λ(t)          (Phase A, pointwise relaxation to the Gibbs state),
    ∂_t rho = ∇·(rho ∇V) + θ Δ rho           (W2 flow: Fokker–Planck).

The first fills depleted regions at rate `−θ log rho` (≈ 27 θ at the
clip floor) without moving anything across them; the second must
transport mass through them. On a double well this is the difference
between teleporting mass between wells and a Kramers barrier crossing.

### 1.3 Phase B — entropic prox (lines 105–114, `wasserstein_ops.prox_w2`)

With `apply_w2_prox=True` (the rigorous path),

    q = argmin_{q ∈ Δ_N}  W_γ(q, rho_tau) + 2h · KL(q || d),

where `W_γ(q, r) = min_{π ∈ Π(q, r)} ⟨C, π⟩ + γ KL(π || 1)` is the
entropic OT cost (`OT_reg` in the docstring). It is solved by the
scaling iteration on `π = diag(a) K diag(b)`, `K = exp(−C/γ)`:

    a ← d^(1−λ) · (K b)^(λ−1),      λ = γ / (γ + 2h)        (KL-prox of (2h/γ)·KL(.||d))
    b ← rho_tau / (Kᵀ a)                                     (second marginal = rho_tau)

for M iterations, output `q = a ⊙ (K b)` renormalised. `rho_tau` enters
only as the fixed second marginal (the W2 anchor); `d` enters only as
the KL target. The test
`test_generalised_sinkhorn_reproduces_prox_w2_for_kl_to_target`
checks, to 1e-9, that this loop is the generalised Sinkhorn iteration of
Peyré (2015; convergence of the two-block scaling scheme for general
convex marginal functionals: Chizat, Peyré, Schmitzer, Vialard 2018)
applied to the functional `G_d(q) = KL(q || d)`.

### 1.4 The composite map

Writing `E_h` for Phase A and `J^γ_h[G]` for the entropic JKO step of a
functional G with step h and regularisation γ,

    S_h(rho_tau) = J^γ_h[ KL(. || E_h(rho_tau)) ](rho_tau)
                 = argmin_q  KL(q || E_h(rho_tau)) + W_γ(q, rho_tau) / (2h).

The fast path (`apply_w2_prox=False`) is `S_h = E_h`. The targets are

    J_h[F](rho_tau)   = argmin_rho F(rho) + W2^2(rho, rho_tau) / (2h)     (exact JKO step),
    J^γ_h[F](rho_tau) = argmin_rho F(rho) + W_γ(rho, rho_tau) / (2h)      (entropic JKO step, Peyré 2015).

### 1.5 Where the step is called

`track2_paper/multiscale_loop.py:96-106`: at each slow step the
particles are advanced by `n_fast` Langevin steps with zero drift
(`potential_fn=_zero_potential` / `None`), histogrammed into a
`FlowState`, passed once through `jko.step`, and the result is appended
to `trajectory` — the particles are **not** updated from it. The slow
dynamics recorded by the paper is therefore `S_h` applied once to the
histogram of a free diffusion, not an iteration of `S_h` and a fortiori
not a JKO sequence. Any coarsening law read on `trajectory` would be
the law of the free Langevin histogram post-processed by one step.

## 2. Comparison with the JKO step

### 2.1 Phase B is a JKO step of the wrong functional

Let `F = ⟨V, .⟩ + θ Σ rho log rho`, the class of all three test energies
and of the first two terms of `T2FreeEnergy` (`V = potentials`, θ = 1,
prior uniform). Then `F = θ · KL(. || π) + const` with the Gibbs state
`π ∝ exp(−V/θ)`, and

    S_h(rho_tau) = argmin_q KL(q || d) + W_γ(q, rho_tau)/(2h)
                 = argmin_q (1/θ) [ θ KL(q || d) + W_γ(q, rho_tau) / (2h/θ) ]
                 = J^γ_{h/θ}[ θ KL(. || d) ](rho_tau).

So Phase B is an entropic JKO step of an energy of the same family as F,
with two substitutions: the Gibbs state `π` is replaced by `d = E_h(rho_tau)`
(the Phase A output, which equals `π` only in the limit h → ∞ of Phase A)
and the time step h by `h/θ` (a factor 10 for `quad`, 20 for `dw`,
1 for `ent` and for `T2FreeEnergy`). `S_h = J^γ_h[F]` iff `d = π` and
θ = 1. For energies that are not a KL divergence (the `J_sym` coupling
and the Turing term of `T2FreeEnergy`, the double-well-in-rho energy a
Cahn–Hilliard model needs) no `d` makes `KL(. || d) ∝ F`: the
identification fails structurally, not only quantitatively.

### 2.2 Small-h expansion (exact OT, γ → 0)

`E_h(rho_tau) = rho_tau − h ĝ + O(h^2)`, so `log(rho_tau / d) = h ĝ/rho_tau + O(h^2)`.
A JKO step of G moves by `−h ∇_W G(rho_tau) + O(h^2)` with
`∇_W G = −∇·(rho ∇ δG/δrho)`; for `G = KL(. || d)`,
`δG/δq(rho_tau) = log(rho_tau/d) + 1 = O(h)`, hence

    S_h(rho_tau) − rho_tau = h^2 · ∇·( rho_tau ∇(ĝ / rho_tau) ) + O(h^3),

against `J_h[F](rho_tau) − rho_tau = h ∇·(rho_tau ∇ δF/δrho) + O(h^2)`.
The composite step is not a consistent discretisation of the W2 flow of
F: `(S_h(rho) − rho)/h → 0`. Its "relative error" `W2(S_h, J_h) / W2(rho_tau, J_h)`
tends to 1 as h → 0, which is what section 3.3 shows (and the fast path,
`E_h`, moves by O(h) in the wrong direction, relative error > 1).

### 2.3 Fixed γ, small h: the entropic blur

At fixed `γ = reg` and h → 0, `λ → 1`, `a → 1`, and `q → K(rho_tau / K1)`,
the Gibbs-kernel blur of `rho_tau` (width √γ): the step tends to a
non-trivial smoothing instead of the identity. This is the entropic
bias of Peyré's scheme, which also affects `J^γ_h[F]`; it is measured
by `W2(J^γ_h[F], J_h[F])` in the tables of section 3 (1.3–12 × smaller
than the error of `S_h` at γ = 0.01, and controlled by γ), and is
orthogonal to the
identification problem: with `γ = 0.01`, `h = 0.05` (paper values),
`λ = 1/11`, the KL pull toward `d` dominates and `S_h ≈ d^(10/11) · (blur)^(1/11)`
before renormalisation — the output is essentially the Phase A output,
smoothed, unless `d` is far from `rho_tau` in W2, in which case the
anchor pulls part of the teleported mass back (section 3.2, `dw`).

### 2.4 Role of `reg` and of ERRATUM §9–11

`reg` is the Sinkhorn regularisation of Phase B only. ERRATUM §9 (ε
stored but not forwarded, so `γ = h = 0.05` instead of 0.01), §10
(`max_iter` not forwarded) and §11 (ε-sweep of the specialisation gap,
0.04 bit) concern the accuracy with which Phase B solves *its* problem.
They do not touch which problem it solves; the present note does. The
PI's reading in ERRATUM §8–11 that the rigorous path is "the JKO step
with the W2 prox" should be amended to "an entropic JKO step of
`KL(. || d)` anchored on `rho_tau`, after a Euclidean descent of F".

### 2.5 Dissipation inequality

`J_h[F]` satisfies `F(rho_1) + W2^2(rho_1, rho_0)/(2h) ≤ F(rho_0)` by
construction (`rho_0` is feasible) and minimises the left-hand side. Any
other output has a larger value: the "defect"
`F(rho_1) + W2^2(rho_1, rho_0)/(2h) − F(rho_0)` of `S_h` is necessarily
≥ that of `J_h`, and nothing guarantees it is ≤ 0: Phase A lowers F by
O(h) but can move mass by O(1) in W2 (teleportation), so its defect is
O(1) > 0 at small h. Section 3 measures it.

## 3. Numerical check

### 3.1 Setup

`experiments/jko_step_check_2026_10_25.py`, run on electron-server
(Xeon X5680, no AVX, Python 3.14.4, NumPy 2.4.4, SciPy 1.17.1, POT
0.9.6, cvxpy 1.9.3 + Clarabel 0.11.1 in the scratch venv only, `nice 19`,
`OMP_NUM_THREADS=4`), deterministic. Support N = 48 on [−1, 1]
(Δx = 0.0426), cost |x − y|². Energies (all `F = ⟨V, rho⟩ + θ Σ rho log rho`,
convex, so the JKO minimiser is unique):

| name | V | θ | rho_tau |
|---|---|---|---|
| `quad` | 4x² | 0.1 | N(−0.5, 0.12) |
| `dw` | (x² − ¼)² / (1/16): wells ±0.5, barrier 1 (20 θ) | 0.05 | 0.7 N(−0.5, 0.1) + 0.3 N(0.5, 0.1) |
| `ent` | 0 (KL to the uniform prior: the leading term of `T2FreeEnergy`) | 1 | same bimodal |

Step under test: `FullJKOSolver(h, epsilon=reg, max_iter=1000, n_inner)`
(`rig`) and `JKOStep(apply_w2_prox=False)` (`fast`), with
h ∈ {0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625}, n_inner ∈ {1, 10, 50},
reg ∈ {0.005, 0.01, 0.05, default = h}, gradient convention ∈
{base-class numerical (as `T2FreeEnergy`), analytic tangential, raw}.

References from the same `rho_tau`:

- exact JKO minimiser `J_h[F]`: Kelley cutting planes on the convex
  piecewise-affine `W2^2(., rho_tau)` (each cut = LP dual potential of
  `ot.emd`, SLSQP inner solves on the simplex), stopped at a duality gap
  ≤ 1e-9 (reported; the inner tolerance limits the certificate to ≈1e-8),
  cross-checked against a conic solve of the plan formulation (cvxpy /
  Clarabel): max TV between the two solvers 3.5e-4;
- entropic JKO minimiser of F, `J^γ_h[F]` (Peyré 2015), generalised
  Sinkhorn with the closed-form KL-prox of F, γ = reg;
- discrete Fokker–Planck flow `exp(h L) rho_tau`, L the
  Scharfetter–Gummel generator of the W2 flow of F on the grid (mass
  conserving, Gibbs state stationary; a gradient flow of the discrete F
  for a discrete transport metric, Maas 2011) — a reference that is not
  quantised by the grid (section 3.3).

Metrics: TV and exact W2 (`ot.emd2`) between the step output and each
reference; `rel.` = W2(step, J_h) / W2(rho_tau, J_h); dissipation defect;
mass on x > 0 (right well). Trajectories: 10–20 steps at
h ∈ {0.1, 0.05, 0.025} to t = 0.5 with the paper's `n_inner = 50`,
`reg = 0.01`, against the exact JKO sequence and the FP flow.

### 3.2 Results

Full tables: `results/jko_step_check_2026_10_25/summary.md`; figure:
`fig_jko_step_check.png` (left: relative error vs h; middle: dissipation
defect; right: densities after one step at h = 0.05). Run: `771dc69`,
2026-10-25T12:25Z, 1844 s.

#### 3.2.1 One step: distance to the exact JKO minimiser and to the FP flow

Relative W2 error `W2(step, J_h) / W2(rho_tau, J_h)` (`rig` =
`FullJKOSolver`, n_inner 50, reg 0.01, numerical gradient; `fast` = no
prox; `Peyré` = entropic JKO of F at γ = 0.01), then the same distances
relative to the FP flow (`/ W2(rho_tau, flow)`), the exact JKO's own
distance to the flow being the discretisation scale of the grid:

| energy | h | rig / JKO | fast / JKO | Peyré / JKO | rig / flow | fast / flow | Peyré / flow | JKO / flow |
|---|---|---|---|---|---|---|---|---|
| quad | 0.4 | 0.26 | 0.32 | 0.022 | 0.07 | 0.08 | 0.21 | 0.21 |
| quad | 0.2 | 0.40 | 0.60 | 0.051 | 0.18 | 0.26 | 0.22 | 0.23 |
| quad | 0.1 | 0.48 | 1.02 | 0.090 | 0.40 | 0.66 | 0.19 | 0.22 |
| quad | **0.05** | **0.60** | 1.79 | 0.187 | 0.56 | 1.33 | 0.18 | 0.22 |
| quad | 0.025 | 0.70 | 2.51 | 0.246 | 0.69 | 2.36 | 0.24 | 0.14 |
| quad | 0.0125 | 0.89 | 4.28 | 0.547 | 0.82 | 3.73 | 0.47 | 0.27 (pinned) |
| quad | 0.00625 | 0.94 | 3.18 | 0.730 | 0.94 | 4.41 | 0.76 | 0.88 (pinned) |
| dw | 0.4 | 0.40 | 3.03 | 0.170 | 0.41 | 2.93 | 0.22 | 0.15 |
| dw | 0.2 | 0.48 | 2.11 | 0.246 | 0.49 | 2.00 | 0.31 | 0.20 |
| dw | 0.1 | 0.54 | 1.56 | 0.330 | 0.58 | 1.45 | 0.41 | 0.28 |
| dw | **0.05** | **0.64** | 0.85 | 0.498 | 0.66 | 0.78 | 0.52 | 0.37 |
| dw | 0.025 | 0.75 | 0.71 | 0.700 | 0.74 | 0.69 | 0.67 | 0.31 |
| dw | 0.0125 | 0.86 | 1.15 | 0.911 | 0.89 | 1.08 | 0.90 | 0.54 (pinned) |
| dw | 0.00625 | 1.25 | 1.63 | 1.472 | 1.14 | 1.23 | 1.26 | 0.78 (pinned) |
| ent | 0.4 | 0.04 | 0.60 | 0.038 | 0.17 | 0.37 | 0.17 | 0.18 |
| ent | 0.2 | 0.02 | 0.89 | 0.020 | 0.16 | 0.69 | 0.16 | 0.16 |
| ent | 0.1 | 0.06 | 1.26 | 0.062 | 0.20 | 0.96 | 0.19 | 0.20 |
| ent | **0.05** | **0.14** | 1.55 | 0.080 | 0.28 | 1.15 | 0.22 | 0.23 |
| ent | 0.025 | 0.27 | 1.90 | 0.148 | 0.38 | 1.41 | 0.21 | 0.24 |
| ent | 0.0125 | 0.40 | 2.42 | 0.247 | 0.45 | 1.92 | 0.15 | 0.25 |
| ent | 0.00625 | 0.49 | 3.22 | 0.465 | 0.49 | 2.74 | 0.29 | 0.29 |

Reading:

- The error of the rigorous step **grows as h decreases**, toward 1
  (quad 0.26 → 0.94, dw 0.40 → 1.25, ent 0.04 → 0.49): the step freezes
  relative to the flow, as section 2.2 predicts; there is no order k at
  which it would be a JKO step. The same holds against the FP flow, so it
  is not an artefact of the grid pinning of the exact reference.
- The fast path is farther from the JKO minimiser than `rho_tau` itself
  (ratio > 1) for h ≤ 0.1 (quad, ent) and at both ends of the h range
  (dw): the Euclidean step moves mass in the wrong direction.
- The exact JKO step and the FP flow differ by 14–23 % of the
  displacement for h ≥ 0.025 (two first-order schemes with different
  spatial discretisations on 48 points); the entropic JKO step of F
  (`Peyré`) sits at the same distance from the flow (quad, ent), i.e. it
  is as good a JKO step as the exact one at the grid's resolution, and
  its distance to the exact minimiser (2–25 % for quad and ent at
  h ≥ 0.025) is the entropic bias of section 2.3, which grows as h
  decreases at fixed γ and is 1.3–12 × below the rigorous step's error
  (quad 3.2 ×, dw 1.3 ×, ent 1.8 × at h = 0.05; 12 × for quad at
  h = 0.4). For `dw` the bias at γ = 0.01 is large (0.17–0.70): the blur
  width √γ = 0.1 equals the well width; γ must be reduced for such
  energies.
- **The identity of section 2.1 is observed.** For `ent` (θ = 1) at
  h = 0.4, where Phase A has converged to the uniform state π,
  `W2(rig, Peyré) = 1.1e-6` (TV 2.5e-11): the rigorous step *is* the
  entropic JKO step of F exactly when d = π and θ = 1. The agreement
  degrades as h decreases and d moves away from π (4.5e-4 at h = 0.2,
  7.2e-3 at 0.1, 2.3e-2 at 0.05). For `quad` (θ = 0.1) `W2(rig, Peyré)`
  stays at 0.08–0.12 for every h: the step h/θ = 10 h and the target
  d ≠ π keep the two apart.

#### 3.2.2 n_inner, reg, gradient convention (h = 0.05)

| energy | fast | rig, n_inner 1 / 10 / 50 (reg 0.01) | rig, reg 0.005 / 0.01 / 0.05 / h (n_inner 50) | rig, gradient numerical / tangential / raw |
|---|---|---|---|---|
| quad | 1.79 | 0.70 / 0.61 / 0.60 | 0.60 / 0.60 / 0.67 / 0.67 | 0.60 / 0.61 / 0.63 |
| dw | 0.85 | 0.65 / 0.64 / 0.64 | 0.63 / 0.64 / 0.72 / 0.72 | 0.64 / 0.70 / 0.69 |
| ent | 1.55 | 0.61 / 0.11 / 0.14 | 0.15 / 0.14 / 0.12 / 0.12 | 0.14 / 0.14 / 0.08 |

None of the knobs changes the picture; `n_inner = 1` is worse for `ent`
(one Euclidean step of size h with a gradient of ≈ 27 at clipped cells
overshoots and clips).

#### 3.2.3 Dissipation inequality

Defect `F(rho_1) + W2^2(rho_1, rho_tau)/(2h) − F(rho_tau)` (≤ 0
required; the exact JKO minimiser attains the minimum):

| energy | h | JKO | rig | fast | Peyré |
|---|---|---|---|---|---|
| quad | 0.4 / 0.05 / 0.0125 / 0.00625 | −0.76 / −0.28 / −0.091 / −0.018 | −0.71 / −0.17 / −0.018 / **+0.015** | −0.68 / **+0.53** / **+1.41** / **+1.68** | ≤ 0 |
| dw | 0.4 / 0.05 / 0.0125 / 0.00625 | −0.089 / −0.064 / −0.028 / −0.014 | −0.083 / −0.046 / −0.011 / **+0.016** | −0.053 / −0.044 / **+1e-4** / **+0.028** | ≤ 0 |
| ent | 0.4 / 0.05 / 0.0125 / 0.00625 | −0.89 / −0.63 / −0.32 / −0.19 | −0.89 / −0.62 / −0.28 / −0.16 | −0.85 / **+0.006** / **+1.22** / **+1.94** | ≤ 0 |

Over all 252 rigorous configurations (7 h × 3 n_inner × 4 reg × 3
gradient conventions) the inequality fails 53 times for `quad`
(h ∈ {0.2, 0.1, 0.0125, 0.00625}), 62 times for `dw` and 9 times for
`ent` (h = 0.00625); over the 63 fast configurations, 50 / 22 / 38
times. The rigorous step dissipates less than the JKO minimiser at every
h (necessarily, section 2.5): e.g. `quad` at h = 0.05, −0.17 against
−0.28.

#### 3.2.4 Mass across the barrier (teleportation)

`quad` (`rho_tau` entirely on x < 0): mass on x > 0 after one step at
h = 0.05 — JKO 0.0009, FP flow 0.0022, rig **0.042**, fast **0.27**; at
h = 0.4 — JKO 0.15, flow 0.42, rig 0.43, fast 0.50. `dw`: JKO, flow and
rig keep 0.3000 (the W2 anchor removes what Phase A teleported; the
Kramers crossing over a 20 θ barrier is invisible at these times), while
fast moves 0.3019 (h = 0.05) → 0.3496 (h = 0.4): 5 % of the mass across
the barrier in one step.

#### 3.2.5 Trajectories to t = 0.5 (n_inner 50, reg 0.01; exact JKO sequence restarted from its own iterate)

| energy | h (steps) | W2(rig, JKO seq.) | W2(rig, flow) | W2(JKO seq., flow) | W2(fast, flow) | F at t = 0.5: rig / JKO / flow-like fast |
|---|---|---|---|---|---|---|
| quad | 0.1 (5) / 0.05 (10) / 0.025 (20) | 0.046 / 0.064 / 0.089 | 0.054 / 0.085 / 0.127 | 0.027 / 0.043 / 0.098 | 0.028 / 0.028 / 0.028 | −0.169 / −0.186 / −0.185; −0.143 / −0.177; −0.094 / −0.143 |
| dw | idem | 0.022 / 0.024 / 0.028 | 0.025 / 0.028 / 0.037 | 0.012 / 0.018 / 0.026 | 0.207 / 0.207 / 0.207 | −0.066 / −0.073; −0.061 / −0.072; −0.047 / −0.066 |
| ent | idem | 0.012 / 0.056 / 0.098 | 0.021 / 0.021 / 0.032 | 0.023 / 0.067 / 0.124 | 0.080 / 0.080 / 0.080 | −3.862 / −3.862; −3.862 / −3.847; −3.858 / −3.818 |

(`quad` flow displacement 0.49, `dw` 0.063, `ent` 0.28.) At t = 0.5 the
`quad` flow is within 2 % of the Gibbs state (relaxation time 1/8), so
the final-time distances are dominated by the common equilibrium (fast
is as close to the flow as the exact sequence); the one-step numbers are
the discriminating ones. Still, the composite trajectory relaxes more
slowly than the flow and more slowly for smaller h (its F at t = 0.5 is
−0.094 at h = 0.025 against −0.185 for the flow): the freezing of
section 2.2 accumulates. For `ent` the exact grid-JKO sequence itself
drifts from the flow as h decreases (0.023 → 0.124: lattice pinning
accumulates over 20 steps of ≈ one cell each) while the entropic
composite does not (0.021–0.032) — entropic regularisation cures
pinning — and the composite tracks the heat flow to 8–12 % of the
displacement at θ = 1, for the reason of section 2.1, not in general
(`quad`, `dw`: 1.4–2 × the exact sequence's distance to the flow).

#### 3.2.6 Solver diagnostics

Cutting planes: 4–63 cuts and 0.3–36 s per exact reference, |gap| ≤
1.7e-12 except `dw` at h = 0.05 (−2.8e-8, inner tolerance); agreement
with cvxpy/Clarabel ≤ 3.5e-4 in TV (7 h values × 3 energies). Generalised
Sinkhorn for the Peyré reference converged (log-scaling change < 1e-13)
in 13–2533 iterations. The step under test ran the production code path
(`FullJKOSolver.step` → `JKOStep.step` → `prox_w2`, 1000 Dykstra
iterations) with the forwarded `reg` and `max_iter` pinned by the test
`test_kiki_step_forwards_h_reg_and_dykstra_count`.

### 3.3 Caveats

- **Grid pinning of the exact reference.** `W2^2(., rho_tau)` is
  piecewise affine on a fixed grid, so for small h the exact JKO
  minimiser either moves mass by whole cells or not at all:
  `W2(rho_tau, J_h)` stalls at ≈ Δx = 0.0426 for h ≤ 0.0125 (`quad`).
  The small-h statements rest on the FP-flow reference, which has no
  such quantisation; the exact reference is used where the displacement
  is ≥ 2 cells (h ≥ 0.025).
- **Certificate.** The cutting-plane gap is computed with the model
  value at the SLSQP point, an upper bound of the model minimum; the
  certificate is therefore "gap ≤ 1e-9 up to the inner tolerance"
  (worst observed −2.8e-8), consistent with the ≤ 3.5e-4 TV agreement
  with Clarabel. Irrelevant at the 0.1–1 relative-error scale of the
  finding.
- **Energies.** Potential + entropy only (what makes the exact reference
  computable). The structural argument of section 2.1 extends the
  negative conclusion to non-KL energies; the quantitative one does not
  need to.
- **Gradient convention.** The base-class finite difference with
  `eps = 1e-4` underestimates the entropy gradient at clipped cells
  (`log 1e-4` instead of `log 1e-12`); the analytic conventions give the
  same qualitative picture (table "gradient convention" in `summary.md`).

## 4. Conclusion and hypotheses

Under the hypotheses (i) 1-D support, cost |x − y|², (ii) F convex of
the form potential + θ entropy, (iii) the code's clipping and
renormalisation conventions, (iv) h ≥ 0.025 for the exact reference and
any h for the FP-flow reference:

- `JKOStep.step` (fast path) is an explicit Euler scheme of the
  Euclidean gradient flow of F on the simplex: **not a JKO step, not at
  any order** (relative error > 1 for h ≤ 0.1, dissipation inequality
  violated in 50/63, 22/63, 38/63 configurations, mass teleported across
  the domain).
- `FullJKOSolver.step` (rigorous path) is Peyré's entropic JKO step of
  `KL(. || d)` anchored on `rho_tau`, `d` the fast-path output: **not a
  JKO step of F, not identified at any O(h^k)**; relative W2 error 0.60
  (`quad`), 0.64 (`dw`), 0.14 (`ent`) at h = 0.05, increasing toward 1
  as h decreases and only shrinking for large h (0.26, 0.40, 0.04 at
  h = 0.4); it dissipates less than the JKO minimiser at every h and
  violates the inequality at h = 0.00625 (and in 21–25 % of the `quad`
  / `dw` configurations); over 10–20 steps it relaxes more slowly than
  the flow, the more so the smaller h (section 3.2.5).
- Phase B *is* a correct entropic JKO step — of the wrong functional,
  and it coincides with the entropic JKO step of F exactly when the
  conditions of section 2.1 hold (`ent`, h = 0.4: 1e-6). This is the
  lever for the fix.

## 5. Minimal modification and its cost

Replace Phases A + B by the generalised Sinkhorn iteration of Phase B
with the KL-prox of `(2h/γ) F` in place of the KL-prox of
`(2h/γ) KL(. || d)`:

```
a ← prox^KL_{(2h/γ) F}(K b) / (K b)
b ← rho_tau / (Kᵀ a)
```

- For `F = ⟨V, rho⟩ + KL(rho || prior)` (the separable part of
  `T2FreeEnergy`, θ = 1): closed form
  `prox(p) = (p · prior^τ · e^{−τV})^{1/(1+τ)}`, `τ = 2h/γ`
  (`jko_entropic` + `PotentialEntropyF.kl_prox` in the experiment
  script, general θ).
- For the `J_sym` coupling and the Turing term: linearise at `rho_tau`
  (freeze them into an effective potential `V_eff = V + ∂(coupling)/∂rho |_{rho_tau}`),
  i.e. the semi-implicit entropic JKO scheme, first order in h like the
  JKO step itself; or a few inner fixed-point iterations on the prox.
- Code: a `kl_prox(p, tau)` method on `FreeEnergy` (default: the
  linearised-potential prox built from `grad_rho`), `prox_w2` takes the
  prox as argument, `JKOStep.step` drops Phase A and calls the prox with
  `reference = rho_tau`. About 40 lines in `master_equation.py` and
  `wasserstein_ops.py`; `apply_w2_prox=False` keeps the fast path
  unchanged for comparison.
- Cost: identical per Dykstra iteration (two N × N matvecs) and *no*
  `n_inner × N` finite-difference evaluations of F any more: cheaper
  than today's rigorous path (the paper's "55 × slower" rigorous path
  spends most of its time in Phase A's 50 × N `model_copy` + `value`
  calls, not in the prox).
- What it buys: `W2(J^γ_h[F], J_h[F])` is the `Peyré` column of the
  tables (section 3.2.1): the error becomes the entropic bias,
  O(γ log 1/γ), 1.3–12 × smaller than the current error at γ = 0.01
  (and as close to the FP flow as the exact JKO step for `quad`, `ent`),
  tunable by γ (log-domain Sinkhorn below 5e-3, or Sinkhorn-divergence
  debiasing); for narrow structures (`dw`, well width 0.1 = √γ) γ must
  be lowered, which the `dw` row shows is not optional.
- Validation: the same experiment script, with `kiki_step` pointed at the
  new solver, must show `rel.` → 0 with γ and the dissipation defect ≤ 0.
- Reinjection: `MultiscaleLoop.run` must feed the JKO output back
  (resample the particles from `state.rho`, or drop the particle layer
  and iterate the step on the grid) before any slow-time law is read.

## 6. Consequences for NH-60 (issue #31)

Not launched: the exponent prediction `L(t) ∝ t^{1/3}` is a statement
about the conserved (H^{−1} / Wasserstein, mobility-limited) gradient flow
of a Cahn–Hilliard energy, and the current step integrates neither that
energy nor that flow. Prerequisites before an L1 preregistration, in
order:

1. the step of section 5 (entropic JKO of F, reinjected), validated by
   the experiment of section 3 (`rel.` → 0 with γ, defect ≤ 0);
2. an energy with a double well *in the density* plus an interfacial
   term, `F[rho] = ∫ W(rho) + (κ/2) ∫ |∇rho|^2`, and an explicit
   mobility (W2 = mobility rho, degenerate Cahn–Hilliard; constant
   mobility for the classical t^{1/3}) — a double-well *potential* V(x)
   on a Fokker–Planck flow (the `dw` case here) has no coarsening: it is
   a linear PDE converging to the Gibbs state;
3. the observable (structure factor peak or interface count) and the
   null (no power law / exponent outside the agreed window [0.28, 0.38]
   vs [0.25, 0.45], to be harmonised) — unchanged from the catalogue.

Cost estimate for such a pilot once 1–2 are in place, on electron-server:
1-D grid N = 1024, Gaussian Gibbs kernel applied by FFT (O(N log N)) or
dense matvec (2 × 10^6 flops), ≈ 200 Sinkhorn iterations per step at
γ ≈ 0.01: ≈ 0.1–0.5 s per step, 10^4 steps ≈ 0.3–1.5 h per seed, 5 seeds
in parallel on 8 cores ≈ 1–2 h wall; N = 4096 with FFT ≈ 4 × that. Within
the 6 h budget — but only after the two prerequisites.

## References

- Jordan, Kinderlehrer, Otto, *The variational formulation of the
  Fokker–Planck equation*, SIAM J. Math. Anal. 29 (1998).
- Peyré, *Entropic approximation of Wasserstein gradient flows*, SIAM J.
  Imaging Sci. 8 (2015).
- Chizat, Peyré, Schmitzer, Vialard, *Scaling algorithms for unbalanced
  optimal transport problems*, Math. Comp. 87 (2018).
- Maas, *Gradient flows of the entropy for finite Markov chains*, J.
  Funct. Anal. 261 (2011).
- Scharfetter, Gummel, *Large-signal analysis of a silicon Read diode
  oscillator*, IEEE Trans. Electron Devices 16 (1969).
- Santambrogio, *Optimal Transport for Applied Mathematicians*,
  Birkhäuser (2015), ch. 8.
- Bray, *Theory of phase-ordering kinetics*, Adv. Phys. 43 (1994) (NH-60
  context).
- `ERRATUM.md` §9–11 of this repository.
