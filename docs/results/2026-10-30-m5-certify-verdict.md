# M5 `certify`: verdict (2026-10-30)

Rule: `docs/prereg/2026-10-30-m5-certify-reading-rule.md`, tag
`prereg-m5-certify-20261030` (tag object `61dd69e`, commit `f2f146e`),
fixed before any configuration with gamma <= 30 had been computed.
Code executed: `31feab6` (main, clean checkout on both hosts).
Records: `results/implicit_jko_certified_2026_10_27/m5_pooled_20261030/`.

## Primary verdict (gamma <= 30, 60 configurations): **failed**

50 of the 60 configurations are certified; 10 are not. The rule has no
tolerance band: a single uncertified configuration in this stratum makes
the primary verdict `failed`.

## Certification by gamma (pooled, 96 configurations, each exactly once)

| gamma | certified | not certified | host |
|------:|----------:|--------------:|------|
| 0   | 12 | 0  | macM3 |
| 5   | 12 | 0  | macM3 |
| 10  | 11 | 1  | macM3 |
| 20  | 10 | 2  | macM3 |
| 30  | 5  | 7  | macM3 |
| 50  | 3  | 9  | macM3 |
| 70  | 3  | 9  | electron-server 3, macM3 9 |
| 100 | 1  | 11 | electron-server 10, macM3 2 |

## The 10 failures of the primary stratum

Grid index in the driver order; residual factors are those of the last
recorded step (residual / tolerance, `kkt_tol` 1e-6, `log_tol` 1e-5).

| idx | (alpha, gamma, h, seed) | status | kkt/tol | log/tol |
|----:|-------------------------|--------|--------:|--------:|
| 37 | (2, 30, 0.05, 501)  | max_outer, step 2           | 1.99    | 2.17    |
| 41 | (5, 30, 0.05, 501)  | max_outer, step 2           | 216     | 225     |
| 42 | (5, 30, 0.025, 500) | line_search, step 1         | 0.77    | 1.30    |
| 43 | (5, 30, 0.025, 501) | max_outer, step 2           | 4.52    | 3.42    |
| 44 | (10, 30, 0.05, 500) | inner_not_converged, step 1 | 3.4e4   | 3.5e4   |
| 45 | (10, 30, 0.05, 501) | inner_not_converged, step 3 | 1.7e5   | 1.5e5   |
| 47 | (10, 30, 0.025, 501)| line_search, step 1         | 2.21    | 3.66    |
| 55 | (5, 20, 0.025, 501) | max_outer, step 1           | 2.09    | 1.94    |
| 56 | (10, 20, 0.05, 500) | inner_not_converged, step 1 | 7.4e3   | 7.5e3   |
| 68 | (10, 10, 0.05, 500) | inner_not_converged, step 2 | 5.7e4   | 3.0e4   |

By status: 4 `max_outer`, 2 `line_search`, 4 `inner_not_converged`.
Certified share in the primary stratum by alpha: 19/20 (2), 16/20 (5),
15/20 (10); by h: 26/30 (0.025), 24/30 (0.05); by seed: 26/30 (500),
24/30 (501).

Per the rule, these 10 configurations go to the declared secondary
analysis (budget-only re-run). Its result will be reported next to this
verdict and does not replace it.

## Stress stratum (gamma >= 50, 36 configurations): descriptive, no verdict

7 of 36 certified. Failing statuses: 19 `max_outer`, 8
`inner_not_converged`, 2 `line_search`. Certified by alpha: 5/12 (2),
2/12 (5), 0/12 (10); by h: 4/18 (0.025), 3/18 (0.05); by seed: 5/18
(500), 2/18 (501). For `max_outer` and `line_search` failures, kkt/tol
ranges from 1.09 to 7.5e3 (median 37.5).

## Both strata (descriptive)

- F decreases (`F_out < F_in`) on all 219 computed steps.
- `Q_final <= Q_explicit` on all 219 computed steps.
- Neither is a certification criterion: a step can lower F and Q_h and
  still not be a certified stationary point, which is what the 39
  uncertified configurations show.
- Wall time per configuration (median / max): macM3 gamma 0: 2 s / 2 s;
  5: 4 / 112 s; 10: 13 / 195 s; 20: 85 / 2081 s; 30: 425 / 3382 s;
  50: 1603 / 6219 s; 70: 2137 / 9130 s; 100: 4822 s (n = 2).
  electron-server gamma 70: 5727 / 8712 s (n = 3); 100: 9500 / 21222 s
  (n = 10), on a saturated host.

## Execution and platforms

- electron-server (Linux x86_64, numpy with OpenBLAS) ran the driver's
  `certify` from 2026-10-30T01:13Z and was stopped by PI decision at
  13:57 CET with 13 records (10 at gamma 100, 3 at gamma 70).
- macM3 (macOS 27.0.1 arm64, Apple M3 Pro, numpy with Accelerate) ran the
  other 83 configurations through an out-of-repo wrapper
  (`certify_shard.py`) that calls the driver's own `_guarded(certify_job,
  ...)` on a filtered list; no parameter, tolerance or budget is changed.
  Same Python 3.14.4, numpy 2.4.4, scipy 1.17.1, POT 0.9.6.post1 on both
  hosts; macM3 environment from `uv.lock` with hashes (`uv sync --frozen
  --extra dev`, uv 0.13.5). The record identity (`RUN_KEY`: solver
  version, delta, N_STEPS, JKOStep defaults, commit) is identical on both
  hosts.
- macM3 run 1 (2026-10-30T12:29Z) was stopped cleanly for a reboot; its
  46 records were intact (no truncated line). Configurations in flight
  at the stop had no record and were recomputed in run 2
  (14:11:51Z to 19:28:06Z, exit 0).
- Pooling (`merge_certify.py`): 96 records, each configuration exactly
  once, every record tagged with its host platform.
- Correction: the run 1 line `end exit=0` in the macM3 `main.log` is not
  an exit status: `$?` was expanded after the `$(date)` substitution and
  so reflected `date`. Run 1 was in fact stopped for the reboot. Run 2
  captures the status before logging.

Information, not a reservation on the verdict: the whole primary stratum
ran on macM3 (Accelerate BLAS), electron-server uses OpenBLAS. Floating-
point results can differ between the two, and some failures sit close to
the tolerances (idx 37, 42, 43, 47, 55: factors 0.77 to 4.5). The rule
counts records from both hosts the same; the verdict stands as stated.

## `inner_not_converged` records: what the fields mean

Example: idx 68, (10, 10, 0.05, 500), step 2 records
`inner_not_converged` with `inner_max_iterations` 172 217 and
`inner_max_residual` 5.0e-14, below the 200 000 cap and the 1e-10
marginal tolerance.

- The status is set only when `SinkhornNotConvergedError` reaches
  `JKOStep._implicit_solve` (`master_equation.py`, `except` around the
  outer loop). With `ot_calls = 0` it comes from
  `jko_entropic_prox_certified` (`wasserstein_ops.py`), which raises when,
  after its scaling loop (at most `inner_n_iter` = 200 000 iterations,
  early stop when the change of log b is below 1e-13), the second-marginal
  residual exceeds `inner_marginal_tol` (1e-10) or its log form exceeds
  `inner_log_tol` (1e-9).
- `inner_max_iterations` and `inner_max_residual` are updated only after
  a prox call returns, so they are the maxima over the successful calls
  of the step, not the values of the call that failed.
- The failing call's own info is in `inner_error`, and `certify_job`
  drops that field from the record. Whether the failing call hit the
  iteration cap or stopped with a residual above tolerance is therefore
  not recorded, for this and the 11 other `inner_not_converged` records.
- Reaching the cap is not by itself a failure: several successful calls
  report 200 000 iterations; the certificate is decided by the residuals.

The secondary re-run will record `inner_error`, as a declared deviation
from the `31feab6` driver's record format (logging only; solver,
certificate and tolerances unchanged).

## Scope

The recomputation of the paper's results stays frozen; it needs a
separate PI decision after M5, as stated in the rule.
