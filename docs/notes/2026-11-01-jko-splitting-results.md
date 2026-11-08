# Lie vs Strang splitting: confirmatory results (2026-11-01)

Preregistration `docs/prereg/2026-10-30-jko-splitting-lie-strang.md` (tag
`prereg-jko-splitting-20261030`), read with its two declared deviations
(`2026-10-30-jko-splitting-deviation-host.md`: macM3 instead of KXKM-AI;
`2026-10-31-jko-splitting-osnice-deviation.md`: the 263 metadata job errors
were set aside and all 576 jobs relaunched).

## Run and integrity

- macM3, commit `6741181` (clean), 4 workers, nice 19, CPU only,
  2026-10-31 07:31:28Z → 22:46:13Z; GO "PI 2026-10-30 15:19 CET, QCM, session
  electron-fb".
- 576/576 jobs, one record each, all of kind trajectory, no `job_error`, no
  duplicate, no missing job: `analyse` gives final (not provisional) verdicts.
- Records and logs: `results/jko_splitting_2026_10_30/confirm/` with
  `SHA256SUMS` (`confirm.jsonl` SHA-256
  `256c1d4196dcb32d9b5029dc7d4e7202a85d277996de5ca8c6a065f91598efed`); copy on
  electron-server `~/archives/jko-splitting-confirm-20261031/` (12/12 checksums
  OK).
- Independent recomputation of §5–§7 (Codex `gpt-6-astra`, own NumPy script,
  no repository code): identical numbers and verdicts
  (`confirm/codex-verification-report.md`, `confirm/codex_recompute.py.txt` (verbatim, stored as text)).

## Verdicts under the frozen rule (§7)

**Q1, substitution (γ ≤ 30, 60 configurations): Lie not substitutable;
Strang not substitutable.** Substitution of the certified implicit step S by
a splitting step is not established.

| | C1 not certified on 3 steps | C2 fail (of 12 usable margins) | C3 fail | dissipation states on certified steps |
|---|---|---|---|---|
| Lie | 14 | 2 (first step not certified) | 30 | 53 proven, 88 missing, 0 violated |
| Strang | 8 | 11 (e_X > s_S) | 25 | 103 proven, 59 missing, 0 violated |

Each scheme fails C1 and C2 independently of C3. All C3 failures are
**missing** brackets (Q_h brackets not certified, all at h = 0.025), not
demonstrated violations: no violation of the discrete dissipation inequality
was found; the inequality is simply not established at h = 0.025. Only 12 of
the 60 configurations had a usable margin (38 degenerate, 10 absent).

**Q2, coverage (γ ≥ 50):** Lie rescues no configuration (0 certified on
3 steps). Strang rescues 3: (α, γ, seed, h) = (5, 50, 500, 0.05),
(5, 70, 500, 0.05), (5, 100, 500, 0.05), all steps proven. Rescue means
certified with no detected violation, not accurate (no reference there).

**Q3, local exponent: not identifiable for Lie and for Strang.** No cell
passes the control gate (0/30; threshold 15): S's class is "neither" in 20
cells and "unresolved" in 10; its median adjacent exponents are negative
(−0.72, −0.03), so the self-convergence defect does not decrease with h on
this ladder. This is the declared null of §8: the set-up cannot establish an
exponent; it is neither evidence for nor against second order of Strang.

## Descriptive (not in the decision)

- Certification on step 1 of S by γ (of 12): 12, 12, 12, 10, 9 at γ ≤ 30;
  6, 3, 1 at γ = 50, 70, 100.
- Median solver wall-time ratio X/S on step 1 (both certified): Lie 0.07,
  Strang 0.22.
- Median e_X / s_S on the usable margins: Lie 0.23, Strang 1.06.

## Calibration remarks (hypneum-nulls)

The verdicts are deterministic on a fixed grid (no sampling statistic, no
threshold to calibrate against a null). Q3's control (S) could fail and did:
the gate is informative, and its failure makes the order question
"non concluant", not refuted. Q1 rests on 12 usable margins out of 60 for C2;
C1 alone already decides "not substitutable" for both schemes.
