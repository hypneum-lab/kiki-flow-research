# Splitting pilot: timing only (2026-10-30)

**Status.** Timing pilot of the preregistration
`docs/prereg/2026-10-30-jko-splitting-lie-strang.md`. **No scientific
conclusion is drawn from it**: it measures wall time and records whether a
step finished, to size the confirmatory run. Statuses on one seed of five
cells say nothing about certification rates, accuracy, dissipation or order.

## Set-up

- Host: KXKM-AI (Intel Core i7-14700KF, 28 logical CPUs), user `electron`,
  `/ai/saisail/jko-splitting/`, tmux `jko-split-pilot`. Shared machine: the M5
  secondary analysis (4 processes, nice 19) and the E1-small confirmatory run
  (GPU, tmux `e1s-conf`) ran at the same time; neither was touched. Load
  average 12–14 during the pilot.
- CPU only: `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=
  MKL_NUM_THREADS=1`, `nice -n 19`, 4 workers, wrapped in `timeout 58m`
  (launcher `results/jko_splitting_2026_10_30/run_pilot.sh`).
- Code: commit `b5fcabd` (branch `exp/jko-splitting`), clean checkout.
- Environment: `uv sync --locked --python 3.14` with uv 0.13.5
  (`~/.local/bin/uv`, SHA-256
  `8a70db72685b7138121bfcda71fa1a00cb709ed9f1fdc4ecf1dd871e8e167e52`),
  CPython 3.14.9, NumPy 2.4.4, POT 0.9.6.post1, SciPy 1.17.1; uv cache and
  Python under `/ai/saisail/jko-splitting/`.
- Jobs: `experiments/jko_splitting_2026_10_30.py pilot`: seed 999 (not a
  confirmatory seed), cells (α, γ) ∈ {(2, 0), (10, 5), (5, 30), (2, 70),
  (10, 100)}, schemes S, Lie, Strang, h ∈ {0.1, 0.05, 0.025, 0.0125}, one
  step each, cap 240 s per step: 60 jobs.
- Run: 2026-10-30 13:39:12Z → 14:03:45Z (24.5 min), exit code 0, 60/60 jobs
  recorded (`results/jko_splitting_2026_10_30/pilot_20261030T133912Z.jsonl`,
  SHA-256 `13dd0d58b18eb2d295f6bfc8e960dc9ed7a75f1cda2125d13866d277f2b41b58`).
- Fields read: `status`, sub-step statuses, `wall`, `bracket_wall`, by
  `results/jko_splitting_2026_10_30/pilot_summary.py`. The records also hold
  sub-problem objective values (`Q_final`, `Q_explicit`, residuals) because the
  pilot code at `b5fcabd` did not whitelist its fields (fixed in `17f26e1`);
  they were not read.

## Wall time per first step

`step wall` is the solve, `bracket wall` the two Q_h brackets (computed at
h ∈ {0.05, 0.025} only, timed, values discarded). A `timeout` is the 240 s
pilot cap, not the 1 800 s confirmatory cap.

| (alpha, gamma) | scheme | h | status | step wall (s) | bracket wall (s) | sub-step statuses |
|---|---|---|---|---|---|---|
| (2, 0) | implicit | 0.1 | converged | 0.2 | 0.0 |  |
| (2, 0) | implicit | 0.05 | converged | 0.5 | 1.2 |  |
| (2, 0) | implicit | 0.025 | converged | 0.9 | 0.2 |  |
| (2, 0) | implicit | 0.0125 | converged | 1.0 | 0.0 |  |
| (2, 0) | lie | 0.1 | converged | 1.3 | 0.0 | B:converged,A:converged |
| (2, 0) | lie | 0.05 | converged | 0.1 | 1.2 | B:converged,A:converged |
| (2, 0) | lie | 0.025 | converged | 0.2 | 0.2 | B:converged,A:converged |
| (2, 0) | lie | 0.0125 | converged | 0.3 | 0.0 | B:converged,A:converged |
| (2, 0) | strang | 0.1 | converged | 0.2 | 0.0 | A:converged,B:converged,A:converged |
| (2, 0) | strang | 0.05 | converged | 0.3 | 1.2 | A:converged,B:converged,A:converged |
| (2, 0) | strang | 0.025 | converged | 0.7 | 1.1 | A:converged,B:converged,A:converged |
| (2, 0) | strang | 0.0125 | converged | 1.3 | 0.0 | A:converged,B:converged,A:converged |
| (10, 5) | implicit | 0.1 | line_search | 208.0 | 0.0 |  |
| (10, 5) | implicit | 0.05 | converged | 0.5 | 1.2 |  |
| (10, 5) | implicit | 0.025 | converged | 1.1 | 0.3 |  |
| (10, 5) | implicit | 0.0125 | timeout | 240.0 | 0.0 |  |
| (10, 5) | lie | 0.1 | converged | 6.7 | 0.0 | B:converged,A:converged |
| (10, 5) | lie | 0.05 | converged | 0.1 | 1.2 | B:converged,A:converged |
| (10, 5) | lie | 0.025 | converged | 0.2 | 0.2 | B:converged,A:converged |
| (10, 5) | lie | 0.0125 | converged | 0.3 | 0.0 | B:converged,A:converged |
| (10, 5) | strang | 0.1 | converged | 1.2 | 0.0 | A:converged,B:converged,A:converged |
| (10, 5) | strang | 0.05 | converged | 0.3 | 1.2 | A:converged,B:converged,A:converged |
| (10, 5) | strang | 0.025 | converged | 0.7 | 0.2 | A:converged,B:converged,A:converged |
| (10, 5) | strang | 0.0125 | converged | 1.3 | 0.0 | A:converged,B:converged,A:converged |
| (5, 30) | implicit | 0.1 | converged | 20.3 | 0.0 |  |
| (5, 30) | implicit | 0.05 | converged | 151.0 | 1.2 |  |
| (5, 30) | implicit | 0.025 | timeout | 240.0 | 0.0 |  |
| (5, 30) | implicit | 0.0125 | timeout | 240.0 | 0.0 |  |
| (5, 30) | lie | 0.1 | timeout | 240.0 | 0.0 |  |
| (5, 30) | lie | 0.05 | inner_not_converged | 72.9 | 0.0 | B:inner_not_converged |
| (5, 30) | lie | 0.025 | timeout | 240.0 | 0.0 |  |
| (5, 30) | lie | 0.0125 | converged | 0.3 | 0.0 | B:converged,A:converged |
| (5, 30) | strang | 0.1 | timeout | 240.0 | 0.0 |  |
| (5, 30) | strang | 0.05 | converged | 127.2 | 1.2 | A:converged,B:converged,A:converged |
| (5, 30) | strang | 0.025 | converged | 3.5 | 0.2 | A:converged,B:converged,A:converged |
| (5, 30) | strang | 0.0125 | converged | 1.6 | 0.0 | A:converged,B:converged,A:converged |
| (2, 70) | implicit | 0.1 | timeout | 240.0 | 0.0 |  |
| (2, 70) | implicit | 0.05 | timeout | 240.0 | 0.0 |  |
| (2, 70) | implicit | 0.025 | inner_not_converged | 83.6 | 0.0 |  |
| (2, 70) | implicit | 0.0125 | timeout | 240.0 | 0.0 |  |
| (2, 70) | lie | 0.1 | timeout | 240.0 | 0.0 |  |
| (2, 70) | lie | 0.05 | inner_not_converged | 80.6 | 0.0 | B:inner_not_converged |
| (2, 70) | lie | 0.025 | inner_not_converged | 103.0 | 0.0 | B:inner_not_converged |
| (2, 70) | lie | 0.0125 | inner_not_converged | 69.1 | 0.0 | B:inner_not_converged |
| (2, 70) | strang | 0.1 | timeout | 240.0 | 0.0 |  |
| (2, 70) | strang | 0.05 | inner_not_converged | 125.9 | 0.0 | A:converged,B:inner_not_converged |
| (2, 70) | strang | 0.025 | timeout | 240.0 | 0.0 |  |
| (2, 70) | strang | 0.0125 | inner_not_converged | 141.3 | 0.0 | A:converged,B:inner_not_converged |
| (10, 100) | implicit | 0.1 | timeout | 240.0 | 0.0 |  |
| (10, 100) | implicit | 0.05 | inner_not_converged | 105.2 | 0.0 |  |
| (10, 100) | implicit | 0.025 | inner_not_converged | 59.6 | 0.0 |  |
| (10, 100) | implicit | 0.0125 | inner_not_converged | 195.3 | 0.0 |  |
| (10, 100) | lie | 0.1 | timeout | 240.0 | 0.0 |  |
| (10, 100) | lie | 0.05 | timeout | 240.0 | 0.0 |  |
| (10, 100) | lie | 0.025 | underflow | 14.2 | 0.0 | B:underflow |
| (10, 100) | lie | 0.0125 | underflow | 6.5 | 0.0 | B:underflow |
| (10, 100) | strang | 0.1 | timeout | 240.0 | 0.0 |  |
| (10, 100) | strang | 0.05 | inner_not_converged | 78.0 | 0.0 | A:converged,B:inner_not_converged |
| (10, 100) | strang | 0.025 | inner_not_converged | 61.5 | 0.0 | A:converged,B:inner_not_converged |
| (10, 100) | strang | 0.0125 | underflow | 10.6 | 0.0 | A:converged,B:underflow |

## Cost estimate of the confirmatory run

Model (`pilot_summary.py`): each of the 48 cells is represented by the pilot
cell of its γ class (γ 0 → (2, 0); 5, 10 → (10, 5); 20, 30 → (5, 30); 50, 70
→ (2, 70); 100 → (10, 100)); a step that finished in the pilot costs its pilot
time on every step of its trajectory (an upper bound: later steps were faster
in M5), brackets included; a failed step costs its pilot time once (the
trajectory stops); a pilot timeout costs between 240 s and the confirmatory
cap.

| cap per step | lower | central (one capped step, then stop) | upper (capped steps converge at the cap and the trajectory goes on) |
|---|---|---|---|
| 1 800 s (preregistered) | 17.6 CPU-h, 4.4 h wall | 90 CPU-h, 23 h wall | 150 CPU-h, 38 h wall |
| 600 s (option) | 17.6 CPU-h, 4.4 h wall | 34 CPU-h, 8.6 h wall | 54 CPU-h, 13.6 h wall |

Wall times assume 4 busy workers and perfect packing; the longest-first order
helps but the last jobs can leave workers idle. The estimate rests on one seed
per γ class, and S at γ ≥ 70 is known to vary from minutes to hours (M5). The
spread is dominated by how many capped steps would finish between 240 s and
the cap.

## Operations

- E1-small (tmux `e1s-conf`) was at 287 of an estimated 423 cells at 13:11Z
  (about 177 s per cell, end expected around 20:00Z); the pilot finished
  long before and did not use the GPU.
- **Interference observed.** Although at nice 19 and CPU-only, the pilot
  slowed E1-small: the four E1 cells that overlapped it took 254, 345, 284 and
  246 s instead of about 177 s (about 6 min of delay in total). E1 itself was
  not touched. The confirmatory run (hours, 4 workers) should therefore not
  run while E1-small or the NH-81 measurement is running on this host.
- The confirmatory run is not launched. It needs the PI's GO (`confirm --go`).
