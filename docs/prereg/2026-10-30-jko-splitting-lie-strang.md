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
- **History:** first version committed at `cdb9f33`, before the pilot; revised
  after an independent review (Annex A) before the tag. The pilot had run to
  completion when the revision was written; Annex A lists what of it was seen.

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
is not second order in general.

This study is related to NH-20, not a direct test of it: R11 H7 splits a
Wasserstein transport step from a Fisher–Rao reaction step, whereas here both
sub-steps are entropic transport (JKO) steps of species of fixed mass, split
by energy terms. Its first purpose is practical: an alternative slow step
where the implicit step does not certify.

## 2. Questions

Q1 (substitution). On the M5 criterion stratum (γ ≤ 30, a parameter stratum,
not the set where S succeeds), is a splitting step (Lie or Strang) an
acceptable substitute for the certified implicit step S: certified on 3 steps,
close to S on the first step (within S's own refinement defect, §5), and with
no detected violation of the discrete dissipation inequality?

Q2 (coverage). On the stress stratum (γ ≥ 50), are the splitting steps
certified, with no detected dissipation violation, where S is not? Accuracy is
not decidable there (§3).

Q3 (local exponent). On this step-size ladder, does the self-convergence
defect of the Strang step scale like h² (the profile of a first-order method,
predicted when sub-steps are first-order JKO steps) or like h³ (the profile of
a second-order method, the textbook expectation of NH-20)? Lie is expected at
h² under both readings. This is a diagnostic of a local exponent on a finite
ladder of a fixed grid, not a proof of a global order.

## 3. Schemes and references

Energy. F = F_A + F_B, the smoothed T2 energy (δ = 1e-5) of the M5 problems:

- F_A (separable) = Σ_i ⟨ρ_i, V_i⟩ + KL(ρ_i ‖ prior_i), exact entropy. Its
  KL-prox is in closed form, so its entropic JKO step needs no linearisation:
  the certified loop stops at its first check (two certified generalised
  Sinkhorn solves per species, the second warm-started). It is exact up to the
  solver tolerances.
- F_B (non-separable) = Σ_{i,k} J_sym[i,k] ⟨ρ_i, ρ_k⟩ + smoothed Turing term,
  with exact gradients. Its entropic JKO step is solved by the certified
  implicit solver of `JKOStep` applied to F_B alone (no entropy in F_B). The
  transport term adds positive curvature; it does not make the B problem
  convex against non-convex couplings.

The parts sum to F (test, 1e-12), and the composition of their linearised
proxes is the linearised prox of F (test, 1e-12; this says nothing about the
composed JKO steps).

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

- **Primary reference:** S_h(ρ0), when certified: the output of the certified
  procedure (its initialisation and tolerances), a stationary point of Q_h, not
  a canonical global minimiser.
- **Small-h reference:** R_h(ρ0) = S_{h/2}(S_{h/2}(ρ0)), when both steps are
  certified. It enters the decision only through the margin s_S (§5) and is
  otherwise descriptive. Reason: on a fixed grid the unregularised W2² cost is
  polyhedral (moving mass t between neighbours costs Δx²|t|), so the JKO step
  pins as h decreases (mass moves only where neighbouring first variations
  differ by more than about Δx²/(2h);
  `docs/notes/2026-10-25-jko-step-identification.md`, grid pinning). The
  entropic cost with reg = h² smooths this threshold but becomes extremely stiff
  as h decreases (Δx²/reg ≈ 7, 28, 114, 455 on the ladder; ≈ 1 820 for the A
  half-steps at h = 0.0125). A reference built from smaller steps can drift
  towards freezing rather than towards a gradient flow; it cannot arbitrate
  accuracy, and the h → 0 limit on 16 sites is not the continuous flow of
  interest.

## 4. Design (fixed in advance)

- **Problems and inputs:** those of M5 `certify`, imported from its driver:
  grid 16 on [−2, 2], potentials α(x + 1.5), α(x − 1.5), α|x|, −α|x|, uniform
  prior, Turing strength 0.1 γ, δ = 1e-5, inputs Dirichlet(1) per species from
  `default_rng(seed)`.
- **Grid:** γ ∈ {0, 5, 10, 20, 30, 50, 70, 100} × α ∈ {2, 5, 10} × seeds
  {500, 501}: 48 cells. The M5 configurations are these cells × h ∈ {0.05,
  0.025} (96 configurations; 60 in the primary stratum γ ≤ 30).
- **Step-size ladder:** h ∈ {0.1, 0.05, 0.025, 0.0125}.
- **Trajectories per cell (one job each):**

  | scheme | h = 0.1 | 0.05 | 0.025 | 0.0125 |
  |---|---|---|---|---|
  | S | 1 step | 2 | 2 | 2 |
  | Lie, Strang | 1 | 3 | 3 | 2 |

  576 jobs. A trajectory stops at its first uncertified step.
- **Caps (budgets, not properties of the solvers):** 1 800 s per step solve
  (all sub-steps) and 600 s for the Q_h brackets of a step, by `signal.alarm`
  (cooperative: a long native call can delay the interruption). A capped solve
  is `status = "timeout"` (not certified; a splitting step keeps the
  diagnostics of its finished sub-steps); capped brackets are a missing
  measurement. An unexpected exception inside a step is recorded as
  `status = "step_error"` (not certified), keeping the earlier steps of the
  trajectory. Same caps for all schemes. (M5 recorded single failing steps of
  up to 6 h at γ = 100.)
- **Recorded per step:** status and certificates of every sub-step, solver and
  bracket wall times, F before and after, output densities, and at h ∈ {0.05,
  0.025} certified brackets of Q_h(out; in) and Q_h(in; in) for the full F
  (`JKOStep.objective_bounds`, gap ≤ 1e-10 per OT_reg). Each record carries
  the code identity (commit, dirty flag, diff hash) and settings (`RUN_KEY`),
  and the host, library versions, thread settings, `CUDA_VISIBLE_DEVICES` and
  niceness.
- **Attempts.** Each job writes one record: a trajectory, a `job_error` of the
  job itself (kind `code`), or a `job_error` of kind `infrastructure` when the
  worker pool breaks. The observation of a job is its **first record that is
  not of kind infrastructure**; a `job_error` of kind code counts as not
  certified and is never re-run by `confirm`. Infrastructure failures are
  re-run only with `--retry-infrastructure`, and their records stay in the
  file. Records of another `RUN_KEY` are ignored. Every extra record of a job
  is listed in the summary.
- **Completeness.** Verdicts are final only when each of the 576 jobs has an
  observation; otherwise `analyse` writes them as `provisional_*` with the list
  of missing jobs. The summary records the records file, its SHA-256 and the
  `RUN_KEY`.
- **Seeds:** confirmatory {500, 501} (= M5). Pilot and bring-up: 999 only.
- **Host:** KXKM-AI (`/ai/saisail/jko-splitting/`), CPU only
  (`CUDA_VISIBLE_DEVICES=""`, one BLAS thread per worker), `nice 19`, at most
  4 workers, environment from `uv.lock` with uv 0.13.5 (SHA-256 in the pilot
  note). The launcher script is archived with the records. `confirm` refuses
  to start without `--go <reference of the PI's GO>`. Because of the caps, the
  load of the host can turn into certification outcomes; the host is shared
  (§11).

## 5. Metrics

For a configuration (cell, h ∈ {0.05, 0.025}) and X ∈ {Lie, Strang}:

- **displacements** d_S = max_i TV(S_h(ρ0)_i, ρ0_i), and likewise for R_h and
  X_h (reported);
- **error** e_X(h) = max_i TV(X_h(ρ0)_i, S_h(ρ0)_i), first step only;
- **margin** s_S(h) = max_i TV(S_h(ρ0)_i, R_h(ρ0)_i): the change of the
  reference when the same time h is covered by two half steps. It is
  **usable** when it exists and s_S < ½ d_S; otherwise it is **degenerate**
  (the reference changes by more than half its own displacement under
  refinement, e.g. R_h ≈ ρ0 when the half steps freeze) and the configuration
  has no usable margin. Usability depends on S and R only, never on X;
- **error vs the small-h reference** e'_X(h) = max_i TV(X_h(ρ0)_i, R_h(ρ0)_i)
  (descriptive);
- **dissipation**, on each certified step: the discrete JKO inequality for the
  full F, Q_h(out; in) ≤ Q_h(in; in) (satisfied by any global minimiser of Q_h,
  since q = in is feasible), in four states: **proven** (upper(out) ≤
  lower(in)), **violated** (lower(out) > upper(in)), **indeterminate**
  (overlapping brackets: the comparison is not resolved at the certified
  precision), **missing** (brackets not certified, timed out, absent,
  non-finite or reversed). F_out − F_in is descriptive (F can rise at the
  minimiser of the non-debiased entropic objective, ERRATUM §16 B2);
- **cost:** solver and bracket wall time per step; the ratio X/S of the solver
  wall time on step 1 where both first steps are certified (descriptive);
- **certification rates** by γ, α and h: S on step 1, X on step 1 and on 3
  steps.

Local exponent, per primary cell (γ ≤ 30; 30 cells) and Y ∈ {S, Lie, Strang}:

- **self-convergence defect** s_Y(h) = max_i TV(Y_h(ρ0)_i,
  Y_{h/2}(Y_{h/2}(ρ0))_i) for h ∈ {0.1, 0.05, 0.025}. If
  Y_h = φ_h + c h^{p+1} + O(h^{p+2}) for a smooth flow φ, then s_Y(h) ≈
  (1 − 2^−p) |c| h^{p+1}: exponent 2 for a first-order method, 3 for a
  second-order one. These hypotheses are not established on this grid (§3);
  the exponent is read as a profile on the ladder, not as a global order.
- **adjacent exponents** a1 = log2(s_Y(0.1)/s_Y(0.05)), a2 =
  log2(s_Y(0.05)/s_Y(0.025)), computed only if the three s_Y ≥ 1e-5 and every
  step involved is certified; the least-squares slope over the three points is
  reported (it equals (a1 + a2)/2 and ignores curvature, so it does not
  classify).
- **class** of Y in the cell: **first** if a1 and a2 both lie in [1.5, 2.5);
  **second** if both lie in [2.5, 3.5]; **neither** otherwise (inconsistent or
  outside both profiles); **unresolved** if the exponents cannot be computed.

## 6. Constants (conventions, stated as such)

- **Margin s_S, factor 1, usability ½.** C2 is a proximity criterion to S, not
  an accuracy-to-the-flow criterion: in a smooth regime, with S_h = φ_h + c_S h²
  and X_h = φ_h + c_X h², C2 compares |c_X − c_S| with |c_S|/2, so a scheme
  whose error to the flow is larger than S's can pass. The margin is the size
  of a change of the reference that its own refinement produces; it has no free
  constant fitted to data. The ½ usability bound excludes margins that are not
  small against the step's own motion (frozen half steps, counter-example
  R_h = X_h = ρ0); it is a convention.
- **Exponent bands.** The two readings predict exponents 2 and 3; a band of
  half the gap around each, [1.5, 2.5) and [2.5, 3.5], attributes an exponent
  to the nearer prediction. Requiring both adjacent exponents in the same band
  rejects non-monotone or curved profiles. The same band [1.5, 2.5) gates S,
  whose prediction is 2. These are conventions, not a statistical test: a gated
  cell is one where S shows a first-order profile on the ladder, not a proof of
  an asymptotic regime.
- **Resolution floor 1e-5 (heuristic).** The certificate of a sub-step bounds
  the fixed-point residual TV(T_0(q), q) < 1e-6, not the distance to the exact
  solution, and no contraction or conditioning bound is available here
  (non-convex B). A self-convergence defect compares one step with two half
  steps: 3 sub-steps for S, 6 for Lie, 9 for Strang. 1e-5 is a heuristic floor
  under which defects are declared unresolved; it is not a certified bound.
  C2 comparisons within 1e-5 of equality are listed as near ties (they keep
  their pass or fail).
- **Fractions ½.** A verdict that would describe fewer than half of the cells
  it is meant to describe is declared not decidable (Q1) or not identifiable
  (Q3); a coverage convention, not a calibration.

There is no equivalence test and no tolerance band: like the M5 reading rule,
the verdicts are deterministic on a fixed grid (two seeds are not a sample to
bootstrap).

## 7. Decision rule (fixed in advance)

**Q1, per X ∈ {Lie, Strang}, primary stratum (γ ≤ 30, 60 configurations):**

- C1: X certified on its 3 steps in every configuration;
- C2: in every configuration with a usable margin, e_X(h) ≤ s_S(h); a
  configuration with a usable margin where X has no certified first step fails
  C2 ("not evaluable", also a C1 failure);
- C3: on every certified step of X in every configuration, the dissipation
  state is proven or indeterminate (violated or missing fails C3).

Verdict, in this order: **not substitutable** if C1, C2 or C3 fails anywhere
(reported with the failing configurations; a certain failure decides first);
otherwise **undecided** if fewer than 30 of the 60 configurations have a usable
margin; otherwise **substitutable**. "Substitutable" means: certified, close
to S on the first step within S's refinement defect, and no detected
dissipation violation (indeterminate comparisons are counted and reported); it
does not mean more accurate than S, nor valid over long trajectories.

**Q2, stress stratum (γ ≥ 50, 36 configurations), descriptive with one count:**
a configuration is **rescued** by X when S has an observation whose step 1 is
not certified (an S `job_error` or a missing S does not count), while X is
certified on 3 steps with every dissipation state proven or indeterminate.
Reported: rescued configurations (and those among them with an indeterminate
state), certification rates by γ, α and h, and e'_X where R_h exists. Rescue
means certified with no detected violation, not accurate.

**Q3, per X, primary cells (30):** a cell is **gated** when S's class is
first. If fewer than 15 cells are gated, the verdict is **not identifiable**.
Otherwise X's classes are counted over all gated cells (unresolved and neither
included); the verdict is **first-order-like** or **second-order-like** if that
class holds a strict majority of the gated cells, and **not identifiable**
otherwise. The class counts, the per-cell exponents and slopes and their median
are reported in every case.

The analysis code that implements this section is `analyse` / `analyse_file`
in the driver, tested on synthetic records (`tests/test_jko_splitting_driver.py`,
including missing slopes, inconsistent exponents, margins with failing X,
degenerate margins, the four dissipation states, job errors, duplicates and an
incomplete record set).

## 8. What counts as a null

- Q1: "not substitutable" or "undecided" for both schemes: substitution of S by
  a splitting step is **not established** on the primary stratum. A failure
  found (C1, C2, C3) and an unavailable comparison (undecided) are reported
  separately. Nothing follows for other energies, grids or step sizes.
- Q2: no rescued configuration: the splitting does not extend the certified
  domain at γ ≥ 50 under the same caps.
- Q3: Strang "first-order-like" or "not identifiable": no evidence of a
  second-order profile on this ladder. "Not identifiable" means the set-up
  cannot establish the exponent, not that the composition lacks it;
  "first-order-like" does not prove a global order 1. As the astra critique
  notes, neither refutes DR-2 nor the splitting reading of NH-20 in general.
  "Not identifiable" is a plausible outcome if pinning dominates the ladder.

## 9. Predictions stated in advance, and what was seen before

Theory (linear model with implicit-Euler sub-steps E_A(h) = (I − hA)^−1;
reproduced in `test_strang_of_implicit_euler_sub_steps_is_first_order`):

- Strang = I + h(A+B) + h²[3A²/4 + (AB+BA)/2 + B²] + O(h³) against the exact
  flow's h²(A+B)²/2: local defect h²(A²/4 + B²/2), generically non-zero, so
  with these sub-steps order 1 is generic. Exact sub-flows, or sub-integrators
  of order at least 2 (or suitably adjoint pairs), give the classical Strang
  guarantee under regularity and stability
  (`test_order_with_exact_sub_flows_is_one_for_lie_and_two_for_strang`).
- Against S_h, Strang carries a defect of its own: E_A(h/2)² − E_A(h) =
  −h²A²/4 + O(h³), present even with B = 0 (where Lie equals S). Lie also
  differs from S in general (Lie − S = −h²BA + O(h³)). Strang need not be
  closer to S than Lie.
- Hypothesis (not a theorem): pinning may amplify Strang's defect in an
  intermediate range of h, since each A half step faces about twice the
  transport threshold of a full step; the entropic smoothing, the change of
  state between sub-steps and B in between can weaken this, and at very small
  h all steps may freeze. The entropic step of a zero energy is not the
  identity (its plan is ∝ r_j exp(−C_ij/s²)), so the comparison mixes the energy
  split, first-order integration and the regularisation rule.

Seen before writing this document (exploratory, disclosed, not
confirmatory). During code bring-up, one run on seed 999 (not a confirmatory
seed) at (α, γ) = (5, 0) and (5, 5), h ∈ {0.1, 0.05, 0.025}, computed
TV(X_h, S_h) for Lie and Strang and the displacement TV(S_h, ρ0):

| (α, γ) | h | TV(S_h, ρ0) | TV(Lie, S_h) | TV(Strang, S_h) |
|---|---|---|---|---|
| (5, 0) | 0.1 | 0.445 | 0.040 | 0.080 |
| (5, 0) | 0.05 | 0.435 | 0.010 | 0.166 |
| (5, 0) | 0.025 | 0.272 | 0.008 | 0.241 |
| (5, 5) | 0.1 | 0.444 | 0.055 | 0.081 |
| (5, 5) | 0.05 | 0.432 | 0.015 | 0.161 |
| (5, 5) | 0.025 | 0.278 | 0.008 | 0.244 |

Not computed then: the margin s_S, the dissipation brackets, trajectories of
more than one step, the exponents and the stress stratum. A unit test on a
6-point grid with support [−0.5, 0.5] and h = 0.2 found the B sub-step
uncertified (`max_outer`), so B-step failures are possible. Consequently, the
first-step errors of both schemes (one input of C2) were seen on a seed of the
same problem family; C2 is a confirmation on new seeds, not a blind test, for
both Lie and Strang. C1, C3 and Q3 were not seen on any step; the pilot (§10,
Annex A) adds certification statuses of first steps on seed 999.

## 10. Pilot and budget

See `docs/notes/2026-10-30-jko-splitting-pilot.md`. The decision rule of §7
was first committed (`cdb9f33`) before the pilot started; it was then revised
in response to the review of Annex A, while the pilot was running and after it
ended, with the exposure listed there.

Pilot (timing only, seed 999, 5 cells × 3 schemes × 4 h, one step each, cap
240 s, KXKM-AI, 2026-10-30 13:39–14:04Z): 60/60 jobs in 24.5 min. Estimated
cost of the confirmatory run with the preregistered caps: 18 to 150 CPU-h,
i.e. about 4.5 h to 38 h of wall time on 4 workers, central estimate about
23 h (90 CPU-h); with a 600 s step cap it would be 4.5 h to 14 h (central
9 h). The cap of 1 800 s is kept; changing it is a PI decision to be taken
before the tag, or an amendment.

The confirmatory run starts only on the PI's GO, after the end of E1-small and
of the NH-81 measurement on the host (the pilot, at nice 19 and CPU-only,
slowed the overlapping E1-small cells from about 177 s to up to 345 s), and
records the GO reference in every record.

## 11. Limits

- One grid (16 points on [−2, 2]), one smoothing δ, one input family, two
  seeds. The grid is coarse enough for pinning to dominate small steps; the
  exponent question may not be answerable on it (declared null).
- The reference S_h is a stationary point of a non-convex Q_h, not a global
  minimiser; e_X measures the distance to that point, and branch changes of
  stationary points between h are possible and not controlled. The X versus S
  objective gap is not used as a metric; the output-versus-input comparison of
  the full Q_h enters C3.
- C2 tests the first step only; C1 and C3 cover 3 steps; nothing tests
  accuracy over a trajectory, and 3 steps cover different times at different h.
- The sub-step regularisation is s² (the `JKOStep` default for a step of
  length s). Another rule (for example h² for every sub-step) gives other
  schemes; not tested.
- The B sub-step has no entropy; transport adds curvature, not convexity.
  Whether it certifies more easily than S is an empirical question (Q2).
- The caps make "not certified" partly a budget statement, especially at
  γ ≥ 50, for all schemes alike; on a shared host, load can change outcomes
  near the cap.
- Selection: the order verdict is computed on cells gated by S's profile, and
  C2 on configurations with a usable margin; these subsets are reported.
- The certificate bounds a fixed-point residual, not the distance to the
  solution; conditioning is not controlled. The resolution floor is heuristic.
- Indeterminate dissipation comparisons pass C3; they are counted.
- TV is the only distance in the decision; W2 is not used.
- Platform: one host (KXKM-AI); trajectories on this path are not
  bit-portable across machines (ERRATUM §16).

## 12. Out of scope

Nothing in this study recomputes, replaces or amends a result of the paper.
Using a splitting step in any other driver needs a separate PI decision.

## Annex A. Independent review before the freeze

- **Reviewer:** Codex CLI 0.162.1, model `gpt-6-astra`, reasoning effort xhigh,
  sandbox workspace-write on a scratch clone of `b5fcabd`, 2026-10-30
  (about 14:41–14:52 CET), 139 153 tokens; brief and report kept outside the
  repository (scratch `splitting/codex/`). Static review: no test or computation
  was run by the reviewer.
- **Findings:** 3 BLOQUANT, 8 MAJEUR, 2 MINEUR groups. Each was checked against
  the source before any change; all BLOQUANT and MAJEUR findings were
  confirmed and fixed or bounded in this revision (a verification pass by an
  independent Claude critic agent on `f8cbe96` found all 13 corrections
  resolved and no new blocking or major issue; its one minor remark, the
  undocumented `step_error` status, is fixed in §4):
  1. Q3 ignored missing slopes (B): missing exponents now count against the
     majority over gated cells (§7), test added.
  2. Margin coverage depended on X (B): usability is computed from S and R only;
     a failing X is a C1/C2 failure, not a missing margin; failures decide
     before "undecided" (§7), test added.
  3. Job errors were re-run and dropped, duplicates overwritten (B): attempt
     policy and completeness (§4), tests added.
  4. The margin was presented as accuracy (M): requalified as proximity to S,
     degenerate margins excluded (§5–6).
  5. The exponent was presented as an order, and the 3-point slope ignored its
     middle point (M): requalified (§2, §5), adjacent-exponent consistency.
  6. The floor was justified by the KKT residual, with wrong sub-step counts
     (M): declared heuristic, counts 3/6/9 (§6).
  7. Overlapping brackets were called equality; missing brackets passed (M):
     four states, missing fails C3 (§5, §7).
  8. The pilot kept `Q_final` / `Q_explicit` despite "timing only"; "blind for
     Lie" was too strong (M): pilot whitelist (code), disclosure (§9, below).
  9. Timeout scope and lost diagnostics (M): separate bracket cap, partial
     sub-steps kept, step errors recorded, worker test (§4).
  10. No completeness check; S missing counted as S failing (M): §4, §7.
  11. Nulls and scope wording; relation to R11 (M): §1, §2, §8, §11.
  Minor: stratified rates, median in every branch, cost ratio on step-1
  certification, A-step solve count, sufficient (not necessary) conditions for
  order 2, curvature of B, role of Q in C3: fixed.
- **Exposure of the pilot.** The pilot was launched from `b5fcabd` (before this
  revision). Its records contain, besides statuses and wall times, the
  sub-problem objective values `Q_final` and `Q_explicit` of each first step on
  seed 999 (the pilot code at `b5fcabd` did not whitelist them). They were not
  read: the only fields read from the pilot records were `status`, the
  sub-step statuses, `wall` and `bracket_wall` (script in the pilot note).
  Certification statuses of first steps on seed 999 were therefore seen before
  this revision of §7. The revision kept every constant of the first version
  and added two conventions, the usability bound ½ of the margin and the
  adjacent-exponent bands; both answer counter-examples of the review, and
  neither could be informed by the pilot, which computed no error, margin,
  exponent or dissipation state.
