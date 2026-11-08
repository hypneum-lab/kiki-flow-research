# M5 `certify`: reading rule (pre-registered 2026-10-30)

Status: FIXED before any `certify` result at gamma <= 30 exists.
Decided by the PI on 2026-10-30 (13:30 CET).

**L2 deposit (annotation added 2026-11-08):** restricted Zenodo record [10.5281/zenodo.24054064](https://doi.org/10.5281/zenodo.24054064) (Hypneum Lab lot of 2026-11-08, file `prereg-m5-certify-20261030.zip`, SHA-256 `fa6a09bfcfc510976cfa0d5bbc0c7466adef5396d45779ef65d58c8ed0cac789`) gives the freeze tag `prereg-m5-certify-20261030` (commit `f2f146e618501f63b99fce1bd1396402355b6eca`, `swh:1:rev:f2f146e618501f63b99fce1bd1396402355b6eca`) a verifiable deposit date, level L2 of PROJECT-MANAGEMENT section 4. The deposit holds this file as it was at the tag; anything dated after the tag in this live file is not in the deposit.

## Context

M5 validates the certified implicit JKO step merged in #51
(`31feab6`). The `certify` subcommand of
`experiments/implicit_jko_certified_validation_2026_10_27.py` runs the
grid gamma in {0, 5, 10, 20, 30, 50, 70, 100} x alpha in {2, 5, 10} x
h in {0.05, 0.025} x seeds {500, 501}, 3 steps per configuration
(96 configurations). Its docstring calls the records descriptive; this
note adds the reading rule.

State when this rule was fixed (electron-server run started
2026-10-29T18:13Z, `certify` since 2026-10-30T01:13Z): 13
configurations done, all at gamma in {70, 100}; 3 certified on all
3 steps. **No configuration with gamma <= 30 had been computed.**

## Definitions

- A configuration is **certified** when it has exactly 3 recorded
  steps and every step has `status == "converged"`.
- Any other outcome counts as **not certified**: a step with another
  status (`max_outer`, `inner_not_converged`, `line_search`, ...), fewer
  than 3 steps, or a `job_error` record.
- Code, tolerances and budgets are those of `31feab6`, unchanged
  (`kkt_tol` 1e-6, `log_tol` 1e-5, `inner_log_tol` 1e-9, delta 1e-5).

## Primary rule (criterion stratum: gamma <= 30, 60 configurations)

- **passed** if all 60 configurations with gamma in {0, 5, 10, 20, 30}
  are certified;
- **failed** otherwise, reported with the list of configurations not
  certified and the failing status of each.

There is no tolerance band: one uncertified configuration in this
stratum makes the primary verdict `failed`.

## Stress stratum (gamma >= 50, 36 configurations)

Descriptive only, no verdict: certification rate by gamma, alpha, h and
seed, with the failing statuses and the residual-to-tolerance factors.

## Reported in both strata (descriptive)

- whether F decreases on every computed step (`F_out < F_in`);
- `Q_final` versus `Q_explicit`;
- wall time per configuration, and the host platform of each record.

## Execution and pooling

The grid may be split between electron-server (Linux x86_64) and macM3
(macOS arm64), both from a clean checkout of `31feab6`. Each record
carries its platform. Records from both hosts count the same. Each of
the 96 configurations must appear exactly once in the pooled set; a
missing configuration counts as not certified.

## Secondary analysis (declared now, outside the primary verdict)

Every configuration not certified, in either stratum, is re-run with a
**budget-only** change: inner iterations 200 000 -> 1 000 000,
`max_outer` 500 -> 2 000, and a recorded residual history. The solver,
its certificate and all tolerances stay unchanged. The secondary result
is reported next to the primary verdict and never replaces it.

## Out of scope

No recomputation of the paper's results is started by this rule; that
needs a separate PI decision after M5.
