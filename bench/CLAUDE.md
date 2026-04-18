# bench — SLO ledger + T2 pair cache

Two kinds of artifact live here; they are not the same thing.

## JSONL ledgers (append-only)

- `T3_latency.jsonl` — streaming-surrogate p50/p99 measurements. One row
  per `{machine, weights, state_dim}` combination. Referenced in the
  README quantitative-results table.
- `sinkhorn_backend_bench.jsonl` — POT vs MLX Sinkhorn wall-clock on the
  rigorous setup. Backs the "~5x speedup" claim in the paper abstract.

Rules:

- Append only. A new measurement is a new line; never rewrite or delete
  historical rows. The ledger is the provenance for published numbers.
- Required fields per row: `ts` (unix seconds), `machine` (e.g.
  `GrosMac-M5`, `KXKM-AI`), plus the metric-specific fields already in
  use (`p50_ms`, `p99_ms`, `weights`, `state_dim` for latency;
  `pot_s`, `mlx_s`, `speedup`, `setup` for backend).
- Machine label matches the host; don't anonymize.
- Never commit a row where `p50_ms > p99_ms` or `speedup < 1` without
  a `note` field explaining the regression.

## `runs/T2_pairs*/` — T3 trainer input cache

Safetensors pairs produced by `scripts/dump_t2_pairs.py` and
`scripts/dump_hybrid_pairs.py`. These are regenerable and exist to cut
surrogate-training turnaround; treat them as a cache, not as committed
data.

- `T2_pairs/` — 32 pairs, `state_dim=16` (v0.1 surrogate).
- `T2_pairs_d128/` — paired with v0.2-d128 surrogate training.

If you change the pair schema (shapes, keys), bump the directory name
(`T2_pairs_d256/`) rather than overwriting — trained weights under
`kiki_flow_core/track3_deploy/weights/` are keyed to a specific schema.

## Anti-patterns (domain-specific)

- Editing JSONL rows in place to "fix a number". Write a new row; if the
  old row was wrong, add one with a `correction_of: <ts>` field.
- Committing a `.jsonl.bak` or a rewritten file with squashed history.
- Using `bench/` for ad-hoc experiment logs — those belong in a scratch
  directory that isn't tracked. Only publication-relevant measurements
  land here.
- Regenerating `runs/T2_pairs/` without also retraining and republishing
  the matching surrogate weights. Pairs and weights ship together.
