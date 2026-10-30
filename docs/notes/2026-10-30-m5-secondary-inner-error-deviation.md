# M5 secondary analysis: `inner_error` kept (declared deviation)

Status: declared deviation, decided by the PI on 2026-10-30 (20:35 CET),
written before the secondary re-run is launched.
Scope: the secondary analysis of the M5 `certify` reading rule only
(`docs/prereg/2026-10-30-m5-certify-reading-rule.md`, tag
`prereg-m5-certify-20261030`, section "Secondary analysis"). The primary
verdict and its records are not affected.

## What changes

Nothing that can change a result. No parameter, tolerance, budget or code
path moves: the secondary re-run keeps the budget-only change declared by
the rule (inner iterations 200 000 -> 1 000 000, `max_outer` 500 -> 2 000,
recorded residual history), and the solver and its certificate stay those
of `31feab6`.

The only change is that one **diagnostic field is kept**: for every step,
the re-run record stores the full `inner_error` string that `JKOStep`
writes into `last_info` when an inner certified solve raises
`SinkhornNotConvergedError` (status `inner_not_converged`).

## Why

`certify_job` in
`experiments/implicit_jko_certified_validation_2026_10_27.py` drops this
field from its records (`k != "inner_error"`), and the
`inner_max_iterations`, `inner_max_residual` and `inner_max_log_residual`
statistics of `_map` are updated only after a call **succeeds**. A
`certify.jsonl` record with `status == "inner_not_converged"` therefore
does not say which inner call failed or why: the iteration budget
(`inner_n_iter` reached) or the certificate (marginal L1 residual above
`inner_marginal_tol` = 1e-10, or relative log residual above
`inner_log_tol` = 1e-9, after an early stop on `delta < inner_tol`).
Reading the reported maxima as those of the failing call is wrong. In
(alpha 10, gamma 10, h 0.05, seed 500), for example, the record shows
172 217 iterations and a residual of 5e-14 although the failing call used
all 200 000 iterations and ended at an L1 residual of 1.77e-10. Without the
field, the secondary analysis could not tell budget failures from
precision-floor failures for the `inner_not_converged` configurations.

## How (no change to the tagged code)

The field is kept by the secondary-analysis wrapper, which lives outside
the repository (`m5sec_worker.py`, run on macM3 in
`~/scratch/m5-secondary/`). The wrapper calls the driver's own
`certify_job` unchanged and reads `inner_error` from `JKOStep.last_info`
after every step, before the driver discards it, together with a scalar
summary of the failing inner call (iterations, budget, last delta, L1 and
log residuals) that it already recorded. No file of the repository is
modified; `31feab6` stays clean (`git status --porcelain` empty), and every
record keeps the driver's `RUN_KEY` (commit, dirty flag, diff hash).

Before the re-run is launched, two gates run with the new wrapper. With
the default budgets, the wrapper must reproduce existing macM3
`certify.jsonl` records bit for bit, and it must match the driver run
without the wrapper. The re-run starts only if both hold.

## Records already produced

The interrupted KXKM run (2026-10-30, 14:45-15:20 CET, stopped by PI
decision) already stored `inner_error`, truncated to 600 characters, and
the structured summary of the failing call. Its only completed
configuration (alpha 5, gamma 100, h 0.025, seed 500, status
`line_search`) has no inner error. For the primary records, the same
messages are visible only in the run logs (`UserWarning: implicit JKO step
not certified`, electron-server `logs/m5_certify.log` and macM3
`logs/certify_shard*.log`). They are not part of `certify.jsonl` and the
primary verdict does not use them.
