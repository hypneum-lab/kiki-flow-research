# Deviation: host of the splitting confirmatory run (2026-10-30)

**Status:** declared deviation from `docs/prereg/2026-10-30-jko-splitting-lie-strang.md`
§4 ("Host: KXKM-AI"), written and pushed before any confirmatory computation.
The tag `prereg-jko-splitting-20261030` (commit `09ec144`) is unchanged; the
confirmatory run uses the code of that commit.

## Decision

PI, 2026-10-30 15:19 CET (QCM, session electron-fb): GO for the confirmatory
run, on **macM3** instead of KXKM-AI, after the end of the M5 run and of the
NH-03 confirmatory run on that host. The step cap stays 1 800 s, as
preregistered.

## What changes

| | preregistered | run |
|---|---|---|
| host | KXKM-AI, Intel Core i7-14700KF, Linux x86_64 | macM3, Apple M3 Pro (12 cores), macOS 27.0.1 arm64 |
| environment | `uv sync --locked`, uv 0.13.5 | same lock, uv 0.13.5 for aarch64-apple-darwin (`~/.local/bin/uv`, SHA-256 `7dad6047467a9d6b081c033cb3efed9ba39a4799c698d09e6cd598f3a6a32338`) |
| directory | `/ai/saisail/jko-splitting/` | `~/scratch/jko-splitting/` |
| workers, niceness | 4, nice 19, CPU only | 4, nice 19, CPU only, `caffeinate`, `nohup` |
| pool start method | fork (Linux default) | spawn (macOS default) |

Unchanged: code (`09ec144`, clean checkout), grid, seeds, caps, decision rule,
analysis, `--go` reference ("PI 2026-10-30 15:19 CET, QCM, session
electron-fb").

## Reason

The timing pilot on KXKM-AI (13:39–14:04Z, nice 19, CPU only) slowed the
overlapping E1-small cells from about 177 s to 254–345 s
(`docs/notes/2026-10-30-jko-splitting-pilot.md`). The PI decided not to run
anything more on KXKM-AI.

## Consequences

- The pilot and its cost estimate were measured on KXKM-AI, not on the
  confirmatory host. Wall times on macM3 differ (M5 steps on macM3 were much
  faster than on electron-server); the estimate of the prereg §10 is not
  re-derived for macM3.
- Certification outcomes near the 1 800 s cap depend on the host's speed and
  load. The secondary M5 replay will run at the same time on macM3 (at most
  6 workers), so the load differs from the pilot's.
- Trajectories on this path are not bit-portable across machines (ERRATUM
  §16); the confirmatory records come from macM3 only, and each record carries
  its host (`host` field).
- The pilot statuses (KXKM, seed 999) are not a prediction of macM3 outcomes.
