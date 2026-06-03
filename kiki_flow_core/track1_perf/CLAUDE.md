# track1_perf — nightly offline consolidation

T1 subpackage. Eulerian-grid JKO over `MixedCanonicalSpecies` (4 canonical x N stacks; production = 10 stacks -> 40 hybrid species). Discipline: offline cadence, deterministic checkpoint, full hook surface (Aeon + MoE-LoRA + Routing), no latency budget.

## Module map

| File | Role |
|---|---|
| `offline_consolidator.py` | `run_once(config, aeon_fetcher, moe_snapshotter, advisory_publisher)` — single consolidation step; `main()` is the CLI stub, not the prod entry |
| `eulerian_grid_solver.py` | `EulerianGridSolver.step` — scheduler -> phono-loop source -> `AdvectionDiffusion.step_1d` -> `JKOStep.step` (in this order) |
| `phenomenological_f.py` | `T1FreeEnergy = alpha*<rho,V_curr> + beta*entropy + gamma*reaction(J_full)` — uses `species.coupling_matrix()` directly (no J_sym/J_asym split, that is T2-only) |
| `trigger_heuristic.py` | `DriftTrigger.should_fire(aeon, last_manifest)` — fires when new/total concepts in window > threshold (default 0.05 over 24h) |
| `checkpoint.py` | `save_checkpoint`/`load_checkpoint` — safetensors + sibling JSON manifest with `git_sha`, `tau`, `rho_keys` |
| `__init__.py` | Public re-exports for `run_once`, `EulerianGridSolver`, `T1FreeEnergy`, `DriftTrigger` |

## Diff vs T2 / T3

| Aspect | T1 | T2 | T3 |
|---|---|---|---|
| Species | `MixedCanonicalSpecies(stack_names)` -> 4*N hybrid | 4 canonical (`phono/lex/syntax/sem`) | flatten/unflatten of T2 dict |
| Solver | Eulerian grid, `JKOStep(apply_w2_prox=False, n_inner=10, h=0.05)` | Particle Langevin x JKO, optional W2-prox | distilled MLP (no JKO) |
| Cadence | offline / nightly, drift-triggered | scripted reproducibility runs | per-query streaming |
| Hooks | all three (`AeonAdapter`, `MoELoraAdapter`, `RoutingAdapter`) live | none | only `RoutingAdapter` via `KikiFlowBridge` |
| Failure mode | raise on bad state (offline, retryable) | raise / warn | swallow -> `None` advisory |
| Coupling term | `gamma * sum_{i,k} J[i,k] <rho_i, rho_k>` | `J_sym` only (J_asym vanishes) | n/a (no functional at inference) |

## Hybridation contract

- `stack_names` is the only knob that scales species count. `MixedCanonicalSpecies(["code","math"])` -> 8 species; production prod uses 10 stacks -> 40. The `40` figure in root `CLAUDE.md` and `README.md` is the prod claim, not the CLI default (`["code","math"]`).
- Species names are `f"{stack}::{canonical}"`. Anything iterating `state.rho` MUST tolerate the `::` separator; do not parse with `split("_")`.
- `T1FreeEnergy._coupling_2d` is cached at `__init__` from `species.coupling_matrix()`. If you mutate `stack_names` after construction, rebuild `T1FreeEnergy` — there is no invalidation.
- `EulerianGridSolver.step` only injects a phono source when `"phono" in name` (substring match). Renaming the canonical species breaks the loop silently.

## Cadence and side effects

- Designed for nightly cron, not mid-day. T1 holds the MoE-LoRA snapshot lock and writes `bench/runs/T1/latest.{safetensors,json}`; running concurrently with T3 production routing is contention, not a race fix.
- Checkpoint atomicity: `save_checkpoint` writes safetensors then JSON non-atomically. A crash between the two leaves a stale manifest — `load_checkpoint` will read mismatched `tau`. Acceptable offline; do not port to T3.
- `DriftTrigger.should_fire` is the only sanctioned scheduler entrypoint. Calling `run_once` on every cron tick regardless of drift wastes the JKO budget and inflates `tau` artificially (R1 reproducibility hashes embed `tau`).

## Anti-patterns (T1-specific)

- Importing `track1_perf` from `track2_paper` or `track3_deploy` (root track-isolation rule). T1 hook payloads are stack-shaped; T2/T3 expect 4-canonical shapes — even shared helpers leak schema.
- Setting `apply_w2_prox=True` to "make T1 paper-faithful". The full W2 prox at 40 species blows past the nightly window; if you need paper fidelity on hybrid species, that is a new track, not a flag flip.
- Hand-editing `bench/runs/T1/latest.json` (tau, git_sha, rho_keys) to skip a re-run. The manifest is the R1 hash anchor.
- Bumping `n_grid` (default 256) without re-baselining all downstream `bench/runs/T1*` checkpoints — grid resolution is part of the FC axis.
- Mid-day invocation while T3 is serving traffic: routing-advisory channel is shared; T1's `publish_advisory` will collide with T3's `route_advisory`.
- Replacing `MixedCanonicalSpecies` with raw 4-canonical to "simplify": that is T2. Use T2.
