# Deviation: first confirmatory attempt lost to a metadata error (2026-10-31)

**Status:** declared deviation from `docs/prereg/2026-10-30-jko-splitting-lie-strang.md`
(tag `prereg-jko-splitting-20261030`, commit `09ec144`, unchanged). Written and
pushed before the relaunch. PI decision of 2026-10-31 08:24 CET.

## What happened

The confirmatory run started on macM3 at 2026-10-30T19:50:23Z (after the end
of M5 and NH-03, see `docs/notes/2026-10-30-jko-splitting-deviation-host.md`)
from a clean checkout of `09ec144`. Every job ended as a `job_error` of kind
`code`:

    PermissionError: [Errno 1] Operation not permitted

raised by `os.nice(0)` in `host_info()` (`experiments/jko_splitting_2026_10_30.py`,
line 133 at `09ec144`), which `trajectory_job` calls when it builds the record,
i.e. after the computation of the trajectory. The launcher process already ran
at niceness 5; `nice -n 19` put the workers at the lowest priority (20), where
`os.nice(0)` fails on macOS. The bug did not show in the tests or the pilots,
which ran at a niceness where the call succeeds.

On PI order the run was stopped on 2026-10-31 at about 07:25Z (confirm process,
its 4 workers and the waiting launcher; the M5 secondary replay was not
touched). 263 of the 576 jobs had been recorded, all as this `job_error`; no
trajectory was stored. Their records and the run log are kept as trace on
macM3: `results/jko_splitting_2026_10_30/confirm.jsonl.failed-osnice-20261031`
and `~/scratch/jko-splitting/confirm.log.failed-osnice-20261031`. They contain
no result.

## Correction

Metadata only: `host_info()` reads the priority with
`os.getpriority(os.PRIO_PROCESS, 0)` and records `None` if that raises an
`OSError`, instead of failing the job. A test simulates `os.nice` and
`os.getpriority` raising. No other change: the schemes, the solvers, the grid,
the seeds, the step sizes, the caps, the decision rule and the analysis are
those of `09ec144`.

## Consequences for the preregistration

- §4 says a `job_error` of kind `code` is the observation of its job and is
  never re-run. That rule is suspended for this attempt only: the 263 errors
  come from a defect of the record writer that is independent of the
  computation and of its outcome (it fails every job identically), so they
  carry no information about the schemes. The failed records are not pooled
  with the new run: the relaunch writes a new `confirm.jsonl`, and its records
  carry a new `git_commit` in their `RUN_KEY`, so the analysis cannot mix them.
- All 576 jobs are relaunched, on macM3, 4 workers, nice 19, with the same
  `--go` reference, at the commit of this correction.
- The code identity of the confirmatory records is therefore the correction
  commit, not `09ec144`. The diff between the two is: this note, the
  `host_info()` change and its test, the earlier host-deviation note, and the
  merge of `main` into the branch (PRs #54 and #55: CI runner on macos-15 and a
  ruff exclusion of `results/` in `pyproject.toml`), which touch no computed
  path.
