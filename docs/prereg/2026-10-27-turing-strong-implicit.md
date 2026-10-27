# Preregistration (L1): strong-Turing excess over Gibbs under the implicit step

- **Date:** 2026-10-27. Written, committed and tagged before any run of the
  driver.
- **Tag:** `prereg-turing-strong-implicit-20261027`, annotated, pushed
  before the run.
- **Level:** L1 (internal preregistration, frozen in the repository by a
  pushed tag; not deposited on OSF).
- **Decision:** PI, 2026-10-27 16:31 (issue #31).
- **Driver:** `experiments/turing_strong_implicit_2026_10_27.py`, frozen at
  the tagged commit. It reuses `experiments/implicit_sweep_2026_10_27.py`
  (`run_couple`) and the bootstrap and Holm functions of
  `experiments/turing_strong_excess_2026_10_26.py`.
- **Host:** electron-server only, `nice 19`, at most 4 workers.

## 1. What has been seen (declared)

**PR #48 (seeds 100–109).** The 2026-10-26 L1 test on the same grid
declared "supported" for a single cell, (α, γ) = (5, 30): +0.248 bits
[+0.102, +0.390], Holm p = 0.0064. That verdict was **withdrawn on
2026-10-27** (ERRATUM §14). Under strong Turing the slow step then used
(entropic, coupling and Turing terms linearised explicitly) does not make
F decrease. The final states lay far above the minimum of F and above the
Gibbs state of the potentials, so the measured excess was an artefact.

**The current step.** The slow step is now the default since 2026-10-27
(ERRATUM §15): the coupling and Turing terms are treated implicitly, with
a safeguard that halves the step if F would increase. Its dissipation
check found 0 increases of F over 960 grid steps, for γ from 0 to 100
(`results/turing_implicit_check_2026_10_27/`).

The present test is therefore **a new test of the same question with a
dissipative step**, not a replication of PR #48. The PR #48 results and
the explicit-step values at γ ≥ 10 of PR #46 / #47 have been seen. They
motivate the question but are not evidence for it.

## 2. Hypothesis

H1: in the strong-Turing regime, the final state is more specialised than
the equilibrium imposed by the potentials, beyond what the same Turing
term produces without potentials. Formally, d > 0 (section 4) in at least
one cell.

H0 (per cell): the mean of d is ≤ 0.

## 3. Design (fixed in advance)

- **Grid:** γ ∈ {20, 30, 50, 70, 100} × α ∈ {2, 5, 10}, i.e. 15 test
  cells, plus the α = 0 control at each γ (5 couples, not tested). Total:
  20 couples.
- **Seeds:** 300, 301, …, 309 (10 per couple). They are new and disjoint
  from seeds 0–9 (PR #46), 100–109 (PR #48) and 200–209 (planned for the
  full-energy test, never run).
- **Model:** the problem of PR #46 / #48.
  - Grid 16 on [−2, 2], 4 species, uniform prior.
  - Potentials: phono α(x + 1.5), lex α(x − 1.5), syntax α|x|, sem −α|x|.
  - Turing strength: 0.1 γ.
  - Particles: 500, MLX simulator on the CPU.
  - Steps: 200 fast × 30 slow, h = 0.05, with reinjection.
- **Slow step:** `FullJKOSolver(scheme="jko_entropic",
  nonseparable="implicit")`, Sinkhorn reg = h², `max_iter` = 1000.
- **Runs:** 200, resumable from `partial.jsonl`.
- **Estimated cost:** 30.4 CPU hours from the timed probe
  (`results/implicit_eta_probe_2026_10_27/probe.json`), i.e. about 7.6 h
  wall time on 4 workers. The run starts after the recomputation of
  PR #46 and of the paper-scale run, in the same queue.

## 4. Statistic and analysis (fixed in advance)

For each test cell and seed s:

    d_s = [gap_s(α, γ) − gap_Gibbs(α)] − gap_s(0, γ)

The terms are:

- **gap** = log2(16) − mean final entropy over the four species, in bits;
- **gap_Gibbs(α)** = the gap of ρ_i ∝ prior_i e^(−V_i);
- **gap_s(0, γ)** = the gap of the paired α = 0 control (same γ, same
  seed).

Per cell:

- the mean of d over the seeds;
- a 95 % percentile bootstrap CI over seeds (200 000 resamples, rng seed
  20261028);
- a one-sided bootstrap p-value, (1 + #{bootstrap means ≤ 0}) / (B + 1).

Multiplicity: Holm step-down over the 15 cells, one-sided, family-wise
level 0.05.

Decision rule:

- **Per cell:** "excess > 0" if the Holm-adjusted p is < 0.05;
  "inconclusive" otherwise. A cell is never declared "no effect".
- **Overall:** "supported", restricted to the cells that pass, if at least
  one cell passes; "not concluding" otherwise.

Descriptive only: the excess without the α = 0 control, and the per-run
runtime.

## 5. Interpretation limits, stated in advance

- **The α = 0 control is not a pure bias at large γ.** Under PR #48 its gap
  grew with γ (0.22 → 2.59 bits): the Turing term structures the species
  on its own. For large γ the primary statistic therefore subtracts a
  Turing-driven structure, and negative values there do not mean "less
  specialised than equilibrium".
- **The equilibrium of the full energy is not the reference.** The Turing
  term is part of F, but its minimiser could not be computed to
  convergence for γ ≥ 20 (PR #49). A "supported" verdict will read: "more
  specialised than the equilibrium of the potentials, beyond the same
  Turing term without potentials". It will not read: "beyond the
  equilibrium of F".
- **30 slow steps (t = 1.5).** No claim about stationarity.
- **Scope.** Only these 15 cells and seeds 300–309 count.

## 6. Deviations

Any deviation (failed or interrupted runs, reruns, changed code) will be
listed with its reason in the result note,
`docs/notes/2026-10-28-turing-strong-implicit.md`. The analysis is the one
coded in the tagged driver.

## 7. Outputs

- `results/turing_strong_implicit_2026_10_27/`: `partial.jsonl`,
  `excess.json`, `summary.md`.
- The note above, a draft PR and a comment on issue #31.
