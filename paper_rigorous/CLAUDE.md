# paper_rigorous — historical rigorous run (defective prox, superseded)

Artifact-only directory. It was meant to check that the qualitative
claims from the fast path (`paper/`) survive the rigorous setup:
`apply_w2_prox=True`, 100 slow steps, 10k particles, 5 seeds. Its prox
was defective (`ERRATUM.md` §1) and, with the corrected operator, the
two paths do not agree (§7, §9). These artefacts are kept for
provenance only; the corrected reruns are listed below.

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
true}` and `n_slow=100`. Write new runs to a new directory: this one
holds the historical artefacts and is not overwritten.

Erratum (2026-10-24/25): the committed artefacts here were produced
with the pre-`ffee5d9` POT `prox_w2`, whose output ignored the
reference. The ~84 min timing is for that old operator and the
"roughly 5x faster with the native MLX Sinkhorn" note is withdrawn.
Reruns with the corrected operator (M1 Max, MLX on CPU, 5 seeds):
`paper_rigorous_fixedprox/` (corrected prox, effective Sinkhorn reg
0.05, 12 min 42 s), `paper_rigorous_fixedeps/` (ε = 0.01 forwarded,
12 min 35 s), `paper_rigorous_maxiter/` (plus `max_iter` = 1000,
12 min 45 s). See `ERRATUM.md` §7, §9, §10.

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
