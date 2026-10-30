# Preregistration (L1): Lie and Strang splitting of the entropic JKO step of T2

- **Date:** 2026-10-30. Written before any confirmatory record exists.
- **Tag:** `prereg-jko-splitting-20261030` (annotated, pushed before the
  confirmatory run).
- **Level:** L1 (internal preregistration, frozen by a pushed tag; not on OSF).
- **Decision:** PI, 2026-10-30 14:08 (write, code and time a pilot; the
  confirmatory run waits for a separate GO of the PI).
- **Schemes:** `kiki_flow_core/track2_paper/jko_splitting.py`.
- **Driver:** `experiments/jko_splitting_2026_10_30.py` (`pilot`, `confirm`,
  `analyse`), frozen at the tagged commit, including the analysis code.
- **Tests:** `tests/track2_paper/test_jko_splitting.py`,
  `tests/test_jko_splitting_driver.py`.
- **Pilot note:** `docs/notes/2026-10-30-jko-splitting-pilot.md` (timing only).

## 1. Background

The certified implicit JKO step (#51, merged in `31feab6`) solves
Q_h(q) = F(q) + Σ_i OT_reg(q_i, ρ_τ,i)/(2h) for the smoothed T2 energy and
certifies a stationary point, or fails explicitly. The M5 `certify` check
(reading rule `prereg-m5-certify-20261030`, PR #52) shows that it fails to
certify in most configurations at γ ≥ 70, and also at (α, γ, h, seed) =
(10, 10, 0.05, 500) on its second step.

The revue d'hypothèses of 2026-10-24 (hypneum-lab/.github,
`docs/research/2026-10-24-revue-hypotheses/`, NH-20, R11 H7) proposed reading
operator ordering as a splitting, with the textbook expectation "Lie O(h),
Strang O(h²)". Its independent critique (`synthese/12-critique-astra.md`,
§ splitting) objects that these global orders assume exact (or second-order)
sub-flows: palindromic composition of first-order steps (implicit Euler, JKO)
is not second order in general. This preregistration tests both readings on
the T2 step, and asks the practical question that motivates it.

## 2. Questions

Q1 (substitution). On the configurations where the certified implicit step
works (the M5 criterion stratum, γ ≤ 30), is a splitting step (Lie or Strang)
an acceptable substitute: certified, as accurate as the reference's own time
discretisation, and dissipative?

Q2 (coverage). On the stress stratum (γ ≥ 50), where the implicit step mostly
fails, are the splitting steps certified and dissipative where it is not?
(Accuracy is not decidable there; see §3.)

Q3 (order). Is the observed convergence order of the Strang step 2 (the
textbook expectation of NH-20) or 1 (the prediction when sub-steps are
first-order JKO steps)? Lie is expected at order 1 under both readings.

## 3. Schemes and references

Energy. F = F_A + F_B, the smoothed T2 energy (δ = 1e-5) of the M5 problems:

- F_A (separable) = Σ_i ⟨ρ_i, V_i⟩ + KL(ρ_i ‖ prior_i), exact entropy. Its
  KL-prox is in closed form, so its entropic JKO step is exact: one certified
  generalised Sinkhorn solve per species (the certified loop stops at its first
  check, with a KKT residual at the inner-solver level).
- F_B (non-separable) = Σ_{i,k} J_sym[i,k] ⟨ρ_i, ρ_k⟩ + smoothed Turing term,
  with exact gradients. Its entropic JKO step is solved by the certified
  implicit solver of `JKOStep` applied to F_B alone (no entropy in F_B).

The parts sum to F (test, 1e-12), and the composition of their linearised
proxes is the linearised prox of F (test, 1e-12).

Sub-step. A sub-step of length s is `JKOStep(part, s, support,
scheme="jko_entropic", nonseparable="implicit")` with the defaults of
`31feab6`: Sinkhorn reg = s², `kkt_tol` 1e-6, `log_tol` 1e-5, `inner_log_tol`
1e-9, `max_outer` 500, `inner_n_iter` 200 000. A step is **certified** when
every sub-step is.

Schemes (composition, rightmost first):

- **S** (`implicit`): the certified implicit step of F, `31feab6` defaults;
- **Lie**: A_h ∘ B_h (B over h, then A over h);
- **Strang**: A_{h/2} ∘ B_h ∘ A_{h/2}.

No sub-step is merged across steps (A_{h/2} ∘ A_{h/2} ≠ A_h for JKO steps).

References for a step of length h from ρ0:

- **Primary reference:** S_h(ρ0), when certified. It is the object the
  splitting approximates: a stationary point of Q_h.
- **Small-h reference:** R_h(ρ0) = S_{h/2}(S_{h/2}(ρ0)), when both steps are
  certified. Used descriptively only. Reason: on the 16-point grid the
  discrete W2 cost is piecewise affine, so the JKO step pins as h decreases
  (mass moves only where neighbouring first variations differ by more than
  about Δx²/(2h); `docs/notes/2026-10-25-jko-step-identification.md`, grid
  pinning). A reference built from smaller steps therefore converges to
  freezing, not to a gradient flow, and cannot arbitrate accuracy. h/2 is the
  smallest refinement, and the one already computed for the margin below.

No reference is defined where neither exists; such configurations count only
for certification, dissipation and cost.

## 4. Design (fixed in advance)

- **Problems and inputs:** those of M5 `certify`, imported from its driver:
  grid 16 on [−2, 2], potentials α(x + 1.5), α(x − 1.5), α|x|, −α|x|, uniform
  prior, Turing strength 0.1 γ, δ = 1e-5, inputs Dirichlet(1) per species from
  `default_rng(seed)`.
- **Grid:** γ ∈ {0, 5, 10, 20, 30, 50, 70, 100} × α ∈ {2, 5, 10} × seeds
  {500, 501}: 48 cells. The M5 configurations are these cells × h ∈ {0.05,
  0.025} (96 configurations).
- **Step-size ladder:** h ∈ {0.1, 0.05, 0.025, 0.0125}.
- **Trajectories per cell (one job each):**

  | scheme | h = 0.1 | 0.05 | 0.025 | 0.0125 |
  |---|---|---|---|---|
  | S | 1 step | 2 | 2 | 2 |
  | Lie, Strang | 1 | 3 | 3 | 2 |

  576 jobs. A trajectory stops at its first uncertified step.
- **Wall-clock cap:** 1 800 s per step (all sub-steps), `signal.alarm`; a
  capped step is `status = "timeout"`, i.e. not certified. Same cap for all
  schemes. (M5 recorded single failing steps of up to 6 h at γ = 100; the cap
  bounds the cost and is a budget, not a property of the solvers.)
- **Recorded per step:** status and certificates of every sub-step, wall time,
  F before and after, output densities, and at h ∈ {0.05, 0.025} certified
  brackets of Q_h(out; in) and Q_h(in; in) for the full F
  (`JKOStep.objective_bounds`, gap ≤ 1e-10 per OT_reg).
- **Seeds:** confirmatory {500, 501} (= M5). Pilot and bring-up: 999 only.
- **Host:** KXKM-AI (`/ai/saisail/jko-splitting/`), CPU only
  (`CUDA_VISIBLE_DEVICES=""`), `nice 19`, at most 4 workers, environment from
  `uv.lock` with uv 0.13.5. Every record carries the code identity (commit,
  dirty flag, diff hash) and settings; `confirm` resumes only from records with
  the same key and refuses to start without `--go <reference of the PI's GO>`.

## 5. Metrics

For a configuration (cell, h ∈ {0.05, 0.025}) and X ∈ {Lie, Strang}:

- **error** e_X(h) = max_i TV(X_h(ρ0)_i, S_h(ρ0)_i) (primary reference);
- **margin** s_S(h) = max_i TV(S_h(ρ0)_i, R_h(ρ0)_i): the difference between
  one step of h and two steps of h/2 of the reference scheme itself;
- **error vs the small-h reference** e'_X(h) = max_i TV(X_h(ρ0)_i, R_h(ρ0)_i)
  (descriptive);
- **dissipation:** on each of the 3 steps, the discrete JKO inequality for the
  full F, Q_h(out; in) ≤ Q_h(in; in) (satisfied by any global minimiser of Q_h,
  since q = in is feasible); a **violation** is lower(Q_h(out; in)) >
  upper(Q_h(in; in)), or a bracket that cannot be certified. F_out − F_in is
  reported descriptively (F can rise at the minimiser of the non-debiased
  entropic objective, ERRATUM §16 B2);
- **cost:** wall time per step, and the ratio X/S on step 1 where both are
  certified (descriptive).

Order, per primary cell (γ ≤ 30; 30 cells) and Y ∈ {S, Lie, Strang}:

- **self-convergence defect** s_Y(h) = max_i TV(Y_h(ρ0)_i, Y_{h/2}(Y_{h/2}(ρ0))_i)
  for h ∈ {0.1, 0.05, 0.025}. For a one-step method of global order p,
  s_Y(h) ≈ (1 − 2^−p) · local error = O(h^{p+1}). It needs no external
  reference.
- **slope** p̂_Y = least-squares slope of log s_Y against log h over the three
  h, computed only if all three s_Y ≥ 1e-5 and all steps involved are
  certified.

## 6. Constants and their justification

- **Accuracy margin s_S(h).** Splitting and implicit steps are both first-order
  discretisations of the same flow over time h. s_S(h) is the change in the
  reference when the same time h is covered by two half steps instead of one:
  an empirical measure of the reference's own time-discretisation ambiguity.
  A splitting step that differs from S_h by less than that is not
  distinguishable from it at the resolution of the time step. No free
  constant.
- **Order threshold 2.5.** The two readings predict s_Strang = O(h²) (order 1)
  and O(h³) (order 2). On the log-log scale 2.5 is equidistant from both
  predictions; a slope is attributed to the nearer one.
- **Control gate p̂_S ∈ [1.5, 2.5].** S is a first-order scheme, so its slope
  is predicted at 2. A cell where S does not show it (within half the gap
  between consecutive orders, the same criterion as above) is not in the
  asymptotic regime (pinning, pre-asymptotic saturation of the first step) and
  is excluded from the order verdict.
- **Resolution floor 1e-5.** Each certified sub-step is known to a KKT
  residual below 1e-6 in TV per species; a self-convergence defect combines at
  most 2 steps of up to 3 sub-steps (≤ 6e-6). Differences below 1e-5 are not
  resolved.
- **Fractions 1/2.** A verdict computed on fewer than half of the cells it is
  meant to describe would describe a minority of the grid; it is then declared
  not decidable (accuracy) or not identifiable (order).

There is no equivalence test and no tolerance band: like the M5 reading rule,
the verdicts are deterministic on a fixed grid (two seeds are not a sample to
bootstrap).

## 7. Decision rule (fixed in advance)

**Q1, per X ∈ {Lie, Strang}, primary stratum (γ ≤ 30, 60 configurations):**

- C1: X certified on its 3 steps in every configuration;
- C2: e_X(h) ≤ s_S(h) in every configuration where both are defined;
- C3: no dissipation violation on any of the 3 steps of any configuration.

Verdict: **substitutable** if C1, C2 and C3 hold; **not substitutable**
otherwise, reported with the failing configurations; **undecided** if the
margin is defined in fewer than 30 of the 60 configurations (then C1 and C3
are still reported).

**Q2, stress stratum (γ ≥ 50, 36 configurations), descriptive with one count:**
a configuration is **rescued** by X when S_h is not certified on step 1 while
X is certified on 3 steps with no dissipation violation. Reported: rescued
configurations, certification rates of S, Lie and Strang by γ, α and h, and
e'_X where R_h exists. Rescue means certified and dissipative, not accurate.

**Q3, per X, primary cells (30):** a cell is **gated** when p̂_S exists and lies
in [1.5, 2.5]. If fewer than 15 cells are gated, or X has no slope in a gated
cell, the verdict is **not identifiable**. Otherwise each gated cell with a
slope is classed **second-order-like** (p̂_X ≥ 2.5) or **first-order-like**
(p̂_X < 2.5); the verdict is the majority class (**tied** on equality). The
median slope and the per-cell slopes are reported.

The analysis code that implements this section is `analyse` in the driver,
tested on synthetic records (`tests/test_jko_splitting_driver.py`).

## 8. What counts as a null

- Q1: "not substitutable" or "undecided" for both schemes: no splitting step
  can replace the certified implicit step on the primary stratum. It says
  nothing about other energies, grids or step sizes.
- Q2: no rescued configuration: the splitting does not extend the certified
  domain at γ ≥ 50 under the same cap.
- Q3: Strang "first-order-like", "tied" or "not identifiable": no evidence of
  second order. As the astra critique notes, this refutes neither DR-2 nor the
  splitting reading of NH-20 in general: it says that composing first-order
  JKO sub-steps on this grid does not produce it. "Not identifiable" is the
  expected outcome if pinning dominates the ladder.

## 9. Predictions stated in advance, and what was seen before

Theory (linear analysis, implicit-Euler sub-steps; reproduced in
`test_strang_of_implicit_euler_sub_steps_is_first_order`):

- Lie and Strang with first-order sub-steps are both of global order 1;
  Strang reaches order 2 only with exact or second-order sub-flows
  (`test_order_with_exact_sub_flows_is_one_for_lie_and_two_for_strang`).
- Against S_h, the Strang step carries a defect absent from Lie even when F_A
  and F_B commute: A_{h/2} ∘ A_{h/2} ≠ A_h for JKO steps. Strang need not be
  closer to S_h than Lie.
- On this grid, pinning should amplify that defect: two half steps of A each
  face twice the transport threshold of one full step, so they move less.

Seen before writing this document (exploratory, to be disclosed, not
confirmatory). During code bring-up, one run on seed 999 (not a confirmatory
seed) at (α, γ) = (5, 0) and (5, 5), h ∈ {0.1, 0.05, 0.025}, computed
TV(X_h, S_h) and the displacement TV(S_h, ρ0):

| (α, γ) | h | TV(S_h, ρ0) | TV(Lie, S_h) | TV(Strang, S_h) |
|---|---|---|---|---|
| (5, 0) | 0.1 | 0.445 | 0.040 | 0.080 |
| (5, 0) | 0.05 | 0.435 | 0.010 | 0.166 |
| (5, 0) | 0.025 | 0.272 | 0.008 | 0.241 |
| (5, 5) | 0.1 | 0.444 | 0.055 | 0.081 |
| (5, 5) | 0.05 | 0.432 | 0.015 | 0.161 |
| (5, 5) | 0.025 | 0.278 | 0.008 | 0.244 |

These numbers agree with the second and third predictions; the margin s_S, the
order slopes and the stress stratum were not computed. A separate unit test on
a 6-point grid with support [−0.5, 0.5] and h = 0.2 found the B sub-step
uncertified (`max_outer`), so B-step failures are possible. The Strang-vs-S
part of Q1 is therefore a confirmation on new seeds of an exploratory
observation, not a blind test; Q2, Q3 and C1–C3 for Lie are blind.

The pilot (§10) records statuses and wall times only, on seed 999; it computes
no error.

## 10. Pilot and budget

See `docs/notes/2026-10-30-jko-splitting-pilot.md`. The pilot ran after this
decision rule was committed (§7 unchanged since); only this section and the
pilot note were added afterwards.

BUDGET-PLACEHOLDER

## 11. Limits

- One grid (16 points on [−2, 2]), one smoothing δ, one input family, two
  seeds. The grid is coarse enough for pinning to dominate small steps; the
  order question may not be answerable on it (declared null).
- The reference S_h is a stationary point of a non-convex Q_h, not a global
  minimiser; e_X measures the distance to that point. ΔQ_h is not used in the
  decision.
- The sub-step regularisation is s² (the `JKOStep` default for a step of
  length s). Another rule (for example h² for every sub-step) gives other
  schemes; not tested.
- The B sub-step has no entropy; its convexity comes only from OT_reg/(2h).
  Whether it certifies more easily than S is an empirical question (Q2).
- The wall-clock cap makes "not certified" partly a budget statement at
  γ ≥ 50, for all schemes alike.
- The margin s_S uses S at h/2, which pins more than S at h; the margin may be
  inflated where pinning is strong. It is reported per configuration.
- TV is the only distance in the decision; Q_h values and W2 are not.
- Platform: records are pooled from one host (KXKM-AI); trajectories on this
  path are not bit-portable across machines (ERRATUM §16).

## 12. Out of scope

Nothing in this study recomputes, replaces or amends a result of the paper.
Using a splitting step in any other driver needs a separate PI decision.
