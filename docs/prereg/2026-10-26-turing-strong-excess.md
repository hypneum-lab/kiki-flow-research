# Preregistration (L1): specialisation excess over Gibbs in the strong-Turing regime

- **Date:** 2026-10-26, written and tagged before any run of the driver.
- **Tag:** `prereg-turing-strong-20261026` (annotated, pushed before the run).
- **Level:** L1 (internal preregistration, frozen in the repository
  by a pushed tag; not deposited on OSF).
- **Decision:** PI, QCM of 2026-10-25 20:51 (issue #31).
- **Driver:** `experiments/turing_strong_excess_2026_10_26.py` (frozen
  at the tagged commit), tests in
  `tests/test_turing_strong_excess_2026_10_26.py`.
- **Host:** electron-server only, `nice 19`.

## 1. Background and the observation being tested

On 2026-10-25 the PI made the excess over the Gibbs equilibrium of the
potentials the primary specialisation statistic (`ERRATUM.md` §13).
That statistic was computed on the entropic-JKO sweep of PR #46 (seeds
0–9; `results/excess_over_gibbs_2026_10_25/`).

- The excess was negative at every couple with α ≤ 20 and γ ≤ 10.
- It was positive only with a strong Turing term (γ ≥ 20) and,
  marginally, at α = 30.
- After subtracting the paired α = 0 control, a single couple stayed
  positive: (α, γ) = (5, 50), with +0.223 bits [+0.132, +0.308].

That observation was found by looking at 27 couples. It is therefore
**exploratory**: a post-hoc selection, with no correction for
multiplicity. This preregistration tests it on a grid fixed in advance
and on new seeds.

## 2. Hypothesis

H1. In the strong-Turing regime, the corrected slow step adds
specialisation beyond the equilibrium imposed by the potentials, even
after the finite-particle bias is removed. Formally, for at least one
cell of the grid, the primary excess d (section 4) is greater than 0.

H0 (per cell). The mean of d is ≤ 0.

## 3. Design (fixed in advance)

- **Grid:** γ ∈ {20, 30, 50, 70, 100} × α ∈ {2, 5, 10}, i.e. 15 test
  cells, plus the α = 0 control at each γ (5 couples, not tested).
  Total: 20 couples.
- **Seeds:** 100, 101, …, 109, 10 per couple. They are new and disjoint
  from seeds 0–9 of PR #46, so the test does not reuse the data that
  generated the hypothesis.
- **Model:** the problem of
  `experiments/rigorous_specialisation_2026_10_24.py`.
  - 4 species, grid 16 on [−2, 2], uniform prior.
  - Potentials: phono α(x + 1.5), lex α(x − 1.5), syntax α|x|,
    sem −α|x|.
  - Turing strength: 0.1 γ.
  - Particles: 500, `MLXParticleSimulator` on the CPU.
  - Steps: 200 fast × 30 slow, h = 0.05.
- **Slow step:** the default since 2026-10-25, set explicitly in the
  driver.
  - `FullJKOSolver(scheme="jko_entropic")`, Sinkhorn reg = h² = 0.0025,
    `max_iter` = 1000.
  - `MultiscaleLoop(reinject=True)`.
  - This is the configuration of PR #46.
- **β:** not a parameter; the entropic step ignores it.
- **Runs:** 200 in total, resumable, 6 worker processes.
- **Estimated cost:** about 30 min wall (#46 measured 21–40 s per run).
  The 6 h cap of the decision is far.

## 4. Statistic and analysis (fixed in advance)

For each test cell (α, γ) and each seed s:

    d_s = [gap_s(α, γ) − gap_Gibbs(α)] − gap_s(0, γ)

The terms are:

- **gap** = log2(16) − mean over the four species of the final entropy,
  in bits.
- **gap_Gibbs(α)** = the gap of ρ_i ∝ prior_i e^(−V_i). It is
  deterministic, does not depend on γ, and is 0 at α = 0.
- **gap_s(0, γ)** = the paired α = 0 control, same γ and same seed.
  Its equilibrium gap is 0, so its measured gap is the finite-particle
  bias of the slow step's anchor.

Per cell:

- the mean of d over the 10 seeds;
- a 95 % percentile bootstrap CI over seeds (200 000 resamples, rng
  seed 20261026);
- a one-sided bootstrap p-value, p = (1 + #{bootstrap means ≤ 0}) /
  (B + 1).

**Multiplicity:** Holm step-down over the 15 cells, family-wise level
0.05, one-sided.

**Decision rule, per cell:**

- **"excess > 0"** if the Holm-adjusted p is < 0.05. Equivalently, the
  one-sided bootstrap lower bound at that cell's Holm level is > 0.
- **"inconclusive"** otherwise. A cell is never declared "excess ≤ 0"
  or "no effect": the test is one-sided and the null is not proven by
  failure to reject.

**Overall verdict:**

- **"Supported"** if at least one cell is "excess > 0" after Holm. The
  claim is then restricted to the cells that pass.
- **"Not concluding"** otherwise.

**Descriptive only, not part of any decision:**

- the excess without the α = 0 control;
- the excess over the minimiser of the full T2 energy (J_sym coupling
  plus the Turing term). That minimiser is computed by a damped fixed
  point, which **does not converge for γ ≥ 20** (residual 0.17–0.33 on
  2026-10-25; reported per cell). The Turing term is non-smooth, and
  the reference is unreliable in exactly this regime.
- per-species entropies.

## 5. Interpretation limits, stated in advance

- The Turing term γ·0.1·Σ_{i<k} ∫|∇ρ_i · ∇ρ_k| is part of the energy F.
  A positive excess over the Gibbs state of the **potentials alone** can
  therefore reflect the equilibrium of the full energy, not something
  the dynamics adds beyond equilibrium. With the full-energy reference
  unavailable (section 4), a "supported" verdict will read: "the final
  state is more specialised than the equilibrium of the potentials,
  under a strong Turing term". It will not read: "specialisation
  emerges beyond the equilibrium of F".
- The α = 0 control removes the finite-particle bias only to the extent
  that the bias does not depend on α.
- 30 slow steps (t = 1.5): no claim about stationarity.
- Only the 15 cells and the 10 preregistered seeds count. Any other
  run is exploratory and will be labelled as such.

## 6. Deviations

Any deviation from this document (failed runs, reruns, changed code)
will be listed in the result note,
`docs/notes/2026-10-26-turing-strong-excess.md`, with its reason. The
analysis is the one coded in the tagged driver; it will not be changed
after the run.

## 7. Outputs

- `results/turing_strong_excess_2026_10_26/`: `partial.jsonl` (one
  record per run), `excess.json`, `summary.md`.
- Note: `docs/notes/2026-10-26-turing-strong-excess.md`.
- Draft PR « exp(jko): preregistered strong-Turing excess over Gibbs ».
- A comment on issue #31.
