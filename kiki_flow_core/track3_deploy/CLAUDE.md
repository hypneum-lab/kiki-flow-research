# track3_deploy — streaming surrogate, advisory routing

T3 subpackage. Pure-NumPy production runtime: encode query, MLP forward, simplex projection, advisory publish. No `torch`, no live JKO, no MLX at inference. Discipline: robustness + fallback + bit-stable forward.

## Module map

| File / dir | Role |
|---|---|
| `streaming_runner.py` | `StreamingRunner.on_query` — encode/surrogate/project/normalize/advise |
| `kiki_flow_bridge.py` | `KikiFlowBridge` — lazy wrapper for `micro-kiki`, gated by `KIKI_FLOW_ENABLED=1` |
| `neural_surrogate.py` | Pure-NumPy 3-layer MLP, residual skip, weights via safetensors |
| `query_encoder.py` | MiniLM (384-d) + hash-based stub fallback + LRU cache |
| `state_projection.py` | `flatten`/`unflatten` between `FlowState.rho` dicts and 1D tensors |
| `query_conditioned_f.py` | T3-side `FreeEnergy` for the JKO oracle (offline / training-only) |
| `surrogate_trainer{,_v3}.py` | JAX trainer that distills T2 JKO → MLP |
| `train_g_jepa.py` | JEPA-style joint-embedding objective (v0.3 weights) |
| `jko_oracle_runner.py` | Offline oracle filling `data/jko_cache` with (pre, post, rho) triples |
| `sweep.py` | Hyperparameter sweep driver for trainer ablations |
| `weights/v{0.1,0.2-d128,0.3}.safetensors` | Shipped checkpoints; pair with matching `bench/runs/T2_pairs*` |
| `encoders/` | Encoder ablation: `base.TextEncoder` ABC + `hash_mlp` / `tiny_tf` / `distilled` |
| `data/` | Dataset construction: `jko_cache`, `corpus_builder`, `synth_qwen`, heuristic labelers, classes |
| `eval/kl_species.py` | Per-species KL of surrogate output vs JKO oracle ground truth |
| `export/to_numpy.py` | Strip JAX deps from a trained checkpoint → pure-NumPy safetensors |

`encoders/` and `data/` are covered by this file (parent routes).

## Diff vs Track 2

| Aspect | T2 | T3 |
|---|---|---|
| Math fidelity | paper-faithful (full W2 prox, log-Sinkhorn) | distilled MLP; simplified for latency |
| Backends | NumPy, JAX, MLX, POT | NumPy at inference; JAX in trainer only; no MLX |
| Stochasticity | seeded, deterministic | inference fully deterministic given weights |
| Failure mode | raise / warn | return `None` advisory (never raise into host) |
| Latency | minutes | p50/p99 logged in `bench/T3_latency.jsonl` |

T3 advisory must agree qualitatively with T2 on same input — drift is a paper finding, not a runtime bug. See `eval/kl_species.py` for the regression metric.

## Robustness contract (advisory-only)

- Inference returns dict; never raises into host. `on_query`: NaN delta → suppressed advisory; `state_dim` mismatch → passthrough; weights missing → `KikiFlowBridge` disabled, `route_advisory` returns `None`.
- After every surrogate forward: `clip >= 1e-6` then renormalize each species sum=1. Skipping either breaks `assert_invariants`.
- `KIKI_FLOW_ENABLED=1` is the **only** switch (default OFF). Do not add a second flag.
- `QueryEncoder` falls back to hash stub if `sentence-transformers` missing; stub deterministic via `hashlib.sha256` over UTF-8 bytes, no salt.

## Weights, caches, schema

- `weights/v{...}.safetensors` pinned to pair-cache schemas under `bench/runs/T2_pairs*` (see `bench/CLAUDE.md`). New weights without republished pairs break `surrogate_trainer.py` reproducibility.
- `data/jko_cache.JKOCache` keyed by `sha256(query_utf8)`; required-pair-keys frozen-set is on-disk contract — no rename without cache-dir bump.
- `export/to_numpy.py` is the only path JAX-trained → shipped runtime. No hand-port.

## Anti-patterns (T3-specific)

- Numerical regression vs T2 = silent paper drift. After touching `neural_surrogate.py`/`query_conditioned_f.py`/encoder, re-run `eval/kl_species.py` against T2 pairs and bump EC if KL outside published band.
- Swapping clip-then-renormalize order in `on_query` — order matters under finite precision.
- Hard-coded production paths. Take `Path` at construction or read `KIKI_FLOW_*` env; never `Path("/var/...")` at module top-level.
- New encoder without subclassing `TextEncoder` + `param_count`/`save`/`load` triple. ABC is the ablation contract.
- Editing checkpoint in place, or reusing a filename. Bump suffix (`v0.4-…`) and add a `bench/T3_latency.jsonl` row.
- Letting exceptions escape `KikiFlowBridge.route_advisory` — bridge is advisory; raising into `micro-kiki` violates host contract.
