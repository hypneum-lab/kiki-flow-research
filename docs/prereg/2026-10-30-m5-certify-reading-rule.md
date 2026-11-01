# M5 `certify`: reading rule (pre-registered 2026-10-30)

Status: FIXED before any `certify` result at gamma <= 30 exists.
Decided by the PI on 2026-10-30 (13:30 CET).

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
