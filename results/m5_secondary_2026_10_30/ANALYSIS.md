# M5 secondary analysis: budget-only re-run (descriptive)

Rule: `docs/prereg/2026-10-30-m5-certify-reading-rule.md`, tag
`prereg-m5-certify-20261030`, section "Secondary analysis". Deviation:
`docs/notes/2026-10-30-m5-secondary-inner-error-deviation.md`
(`inner_error` kept, execution hosts). This result is reported next to
the primary verdict and never replaces it. It is descriptive: the rule
fixes no threshold or verdict for it.

## What was run

Of the 96 pooled configurations, 39 are not certified under the primary
rule (12 `inner_not_converged`, 23 `max_outer`, 4 `line_search`). Each one
is re-run from its initial state with only two budgets changed: inner
iterations 200 000 -> 1 000 000 and `max_outer` 500 -> 2 000. Code
`31feab6` is unchanged and so are all tolerances. The residual history and
`inner_error` are recorded. Hosts: KXKM (1 configuration), macM3 (27, plus
1 still running) and Studio (10).

Two checks were made before each launch:
- with the default budgets, the wrapper reproduces existing records bit
  for bit (ES lines on KXKM with ES-aligned arithmetic; macM3 lines on
  macM3 and on Studio);
- it matches the driver run without the wrapper on the same host.

## Result (snapshot 2026-11-01, 38 of 39 completed)

- 10 of the 38 completed configurations are certified with the larger
  budget (3 steps, all `converged`).
- Criterion stratum (gamma <= 30, 10 completed re-runs): 4 certified,
  2 still failing on budget, 4 on the precision floor.
- Stress stratum (gamma >= 50, 28 completed re-runs): 6 certified, 6 budget,
  11 precision floor, 5 stagnation.
- Still running: (alpha 10, gamma 50, h 0.05, seed 500), stress stratum,
  more than 28 h. Its row is added when it ends (`table.md` is then
  regenerated).

The table, configuration by configuration, is in `table.md`. Its "outer N"
  is the number of residual evaluations (last outer index k = N - 1).

## How to read it

- **The primary verdict is unchanged.** The criterion stratum (gamma
  <= 30) stays `failed`. Certification with a larger budget is a different
  computation: once a budget binds, the iterates differ. Where the
  arithmetic matches the primary host, the re-run repeats the primary path
  exactly only in the 2 cases where no budget was reached (both line_search).
  It is not a corrected primary result.
- **Precision floor** (15 of 28 not certified) is the most frequent
  remaining failure. The line search finds no anchor weight that gives a
  sufficient decrease of Q, while the median of the last five accepted
  decreases (1.2e-13 to 4.5e-12) is below the estimated accuracy of Q
  (inner marginal residual x maximal cost / 2h, 1.2e-11 to 1.1e-9). These
  failures are not limited by the outer budget, but this run does not show
  that more budget could not help: in some of them, e.g.
  (10, 10, 0.05, 500), inner solves still reach their limit. The outer residuals stop up to about 8 times above
  their tolerances.
- **Budget** (8): the outer residual still decreases at `max_outer` = 2 000
  (last-quarter ratio < 0.5), or an inner solve uses all 1 000 000
  iterations and stops at an L1 residual of 1.0e-10 to 1.6e-10 (tolerance
  1e-10). These cases might certify with even more budget; that is not
  tested here.
- **Stagnation** (5, all at gamma >= 50): the outer residual stays flat or
  grows over the last quarter of 2 000 iterations, about 35 to 10 000
  times above its tolerance.
- The classification thresholds (last-quarter ratio 0.5; Q accuracy
  max(1e3 eps |Q|, residual x 16 / 2h)) were chosen after the first
  results and are declared as such. They label the failures; they decide
  nothing.

## Anomaly (alpha 10, gamma 10, h 0.05, seed 500)

The primary record reads `inner_not_converged` with
`inner_max_iterations` = 172 217 (below 200 000) and `inner_max_residual`
= 5e-14. This is not a labelling bug. Those maxima are updated only after
an inner call **succeeds**, and `certify_job` drops `inner_error` from the
record. The failing call (outer iteration 12 of step 2) used all 200 000
iterations and stopped at an L1 marginal residual of 1.77e-10, above the
tolerance of 1e-10 (log residual 5.85e-10 < 1e-9, last delta 5.85e-10).
The macM3 run log contains the same message (same wall time, 75.89 s), and
the instrumented re-run with the default budgets reproduces it. With the
larger budget, step 2 gets past that solve but ends in `line_search` at
outer index k = 411 (412th evaluation; kkt 2.1e-6, precision floor), so the
configuration stays not certified.

Evidence: `anomaly/macm3_primary_log_excerpt.txt` (the primary run log line,
with its `inner_error`) and `anomaly/macm3_default_budget_validation.jsonl.gz`
(the default-budget re-run on macM3, which is bit-identical to the primary
record and carries `inner_error` and the structured failing call). The
solver code is `kiki_flow_core/master_equation.py` (`_map`,
`_implicit_solve`) at `31feab6`.

## Verification

The counts, the table and the figures of this note were recomputed
independently from the raw records by Codex (gpt-6.1-sol, reasoning
effort high, 2026-11-01), which saw this note and the table but not the
script until after its own recount. Report: `codex_verification.md`. It
found no BLOQUANT and no wrong count or status. Its 1 MAJEUR (raw evidence
for the anomaly missing from the bundle) and 3 MINEUR findings (median
wording, outer counter convention, the claim that a larger budget cannot
help) were checked against the records and fixed above.
