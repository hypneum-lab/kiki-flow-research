# paper_rigorous — 84-minute full-W2-prox verification run

Artifact-only directory. Proves the qualitative claims from the fast
path (`paper/`) survive the rigorous setup: MLX Sinkhorn backend,
`apply_w2_prox=True`, 100 slow steps, 10k particles, 5 seeds.

## What lives here

- `figures/fig1_phase_seed{0..4}.{pdf,png}` — rigorous-path phase
  portraits. Only figure-1 series is reproduced here; the other figures
  (F-decay, Turing, CL-gap, sweep) are generated exclusively by the
  fast path.
- `stats.json` — minimal run metadata: seeds, slow-steps completed,
  backend config (`{"simulator": "mlx", "w2_prox": true}`).
- `run.log` — stdout/stderr of the last rigorous run. Keep for
  provenance; include `Sinkhorn did not converge` warnings — they are
  expected at the epsilon the rigorous path uses and do not invalidate
  the run as long as the invariants still pass at the end.

No LaTeX, no code, no BibTeX here. This is a pure "reviewer can verify"
artifact drop.

## Regenerating

Driven from the same `kiki_flow_core/track2_paper/paper_run.py` entry
point as the fast path, with `backend={"simulator": "mlx", "w2_prox":
true}` and `n_slow=100`. Expect ~84 min on GrosMac M5; roughly 5x
faster with the native MLX Sinkhorn vs POT reference. Output directory
is `paper_rigorous/`.

Never run this from an editor terminal that might be suspended —
resuming a suspended MLX process leaves the Metal context in a bad
state. Use a detached shell.

## Keeping in sync with `paper/`

- `stats.json` here and `stats.json` in `paper/` must report the same
  `n_seeds` and the same seed list. If they diverge, one of the two
  runs is stale.
- If a phase-portrait figure changes qualitatively between the fast and
  rigorous paths, that is a finding — investigate, don't just
  regenerate until they look similar.

## Anti-patterns (domain-specific)

- Committing a partial run (e.g. `n_slow_completed < 100` for any
  seed). A rigorous-path commit is all-or-nothing.
- Hand-editing `stats.json` to match the fast path. The whole point of
  this directory is that its numbers are independent.
- Deleting `run.log` because "it has warnings". The warnings are part
  of the provenance.
- Adding LaTeX sources, BibTeX, or figure generators here. Those live in
  `paper/` and `kiki_flow_core/track2_paper/figures/` respectively.
- Regenerating these artifacts on a different machine than the one
  recorded in `stats.json` / `run.log` without updating the machine
  label. Timing claims are machine-specific.
