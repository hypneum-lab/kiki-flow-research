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

## Addendum 2026-10-31: execution hosts

This addendum records where the secondary re-run is executed. It changes no
parameter, tolerance, budget or code, and neither the wrapper nor the
`inner_error` field.

- 2026-10-30, 14:45 to 15:20 CET: KXKM-AI (Linux x86_64), stopped by PI
  decision because its CPU load slowed another campaign. One configuration
  had completed and is kept with its platform: (5, 100, 0.025, 500).
- From 2026-10-30, 20:48 CET: macM3 (Apple M3 Pro, macOS 27.0.1 arm64),
  38 configurations queued, 6 workers.
- PI decision 2026-10-31, 17:50 CET: the 10 configurations of the macM3
  queue that had not started run on Studio (Apple M3 Ultra, 32 cores,
  macOS 27.2 arm64). The 6 already running on macM3 finish there. On
  macM3, only the dispatcher was stopped; the running workers were left
  untouched.
- Configurations (alpha, gamma, h, seed) by host:
  - macM3, 28: (2, 100, 0.025, 500); (2, 100, 0.025, 501);
    (5, 100, 0.05, 501); (2, 100, 0.05, 501); (10, 100, 0.05, 500);
    (5, 100, 0.05, 500); (10, 100, 0.025, 500); (10, 100, 0.025, 501);
    (2, 70, 0.05, 501); (10, 10, 0.05, 500); (10, 30, 0.05, 501);
    (10, 30, 0.05, 500); (10, 30, 0.025, 501); (5, 30, 0.025, 500);
    (10, 20, 0.05, 500); (5, 20, 0.025, 501); (5, 30, 0.05, 501);
    (2, 30, 0.05, 501); (10, 50, 0.025, 500); (10, 50, 0.05, 500);
    (5, 30, 0.025, 501); (10, 50, 0.05, 501); (10, 50, 0.025, 501);
    (5, 50, 0.05, 500); (2, 50, 0.025, 500); (2, 50, 0.05, 501);
    (5, 50, 0.025, 501); (5, 50, 0.05, 501).
  - Studio, 10: (10, 70, 0.05, 500); (10, 70, 0.025, 501);
    (5, 70, 0.05, 501); (10, 70, 0.05, 501); (10, 70, 0.025, 500);
    (5, 70, 0.05, 500); (10, 100, 0.05, 501); (2, 70, 0.025, 500);
    (5, 100, 0.025, 501); (5, 70, 0.025, 501).
- Studio environment: the uv binary and the uv-managed CPython 3.14.4 were
  copied from macM3 (no download), with a clean clone at `31feab6` and
  `uv sync --frozen --extra dev --python 3.14.4`. The installed packages
  match macM3 except for the path of the editable install.
- Studio gate before launch (2026-10-31, 17:52 to 17:54 CET), the same as
  on macM3. It passed:
  - the 46 certified-path tests pass;
  - with the default budgets, the wrapper reproduces the macM3 records
    (10, 10, 0.05, 500) (44 fields) and (10, 10, 0.05, 501) (66 fields)
    bit for bit, and matches the driver run without the wrapper;
  - `inner_error` is kept in full.
