# track1_perf — nightly offline consolidation

T1 subpackage. Eulerian-grid JKO over `MixedCanonicalSpecies` (4 canonical x N stacks; production = 10 stacks -> 40 hybrid species). Discipline: offline cadence, deterministic checkpoint, full hook surface (Aeon + MoE-LoRA + Routing), no latency budget.

## Module map

| File | Role |
|---|---|
| `offline_consolidator.py` | `run_once(config, aeon_fetcher, moe_snapshotter, advisory_publisher)` — single consolidation step; `main()` is the CLI stub, not the prod entry |
| `eulerian_grid_solver.py` | `EulerianGridSolver.step` — scheduler -> phono-loop source -> `AdvectionDiffusion.step_1d` -> `JKOStep.step` (in this order) |
| `phenomenological_f.py` | `T1FreeEnergy = alpha*<rho,V_curr> + beta*entropy + gamma*reaction(J_full)` — uses `species.coupling_matrix()` directly (no J_sym/J_asym split, that is T2-only) |
| `trigger_heuristic.py` | `DriftTrigger.should_fire(aeon, last_manifest)` — fires when new/total concepts in window > threshold_on (default 0.07 over 24h), hysteresis re-arm below threshold_off (default 0.05); optional cooldown/persisted state, see below |
| `checkpoint.py` | `save_checkpoint`/`load_checkpoint` — safetensors + sibling JSON manifest with `git_sha`, `tau`, `rho_keys`, `known_concepts`; `load_known_concepts` reads the latter (legacy manifest -> empty set) |
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

## DriftTrigger parameters

| Parameter | Default | Meaning |
|---|---|---|
| `threshold` (alias `threshold_on`) | 0.07 | fire when `rate > threshold` (strict) |
| `window_h` | 24 | Aeon window in hours |
| `threshold_off` | 0.05 | hysteresis (`None` = legacy memoryless): after a fire, disarmed until a tick sees `rate < threshold_off` (strict); must satisfy `0 <= threshold_off < threshold` |
| `cooldown_s` | `None` | refractory period: a fire also needs `clock() - last_fire >= cooldown_s`; a blocked tick does not disarm |
| `state_path` | `None` | JSON file holding `armed` + `last_fire_ts`, read at every `should_fire` and rewritten atomically (tmp + `os.replace`) on change; corrupt file -> `ValueError` |
| `clock` | `time.time` | wall clock (persisted across processes, so not monotonic); inject a fake for tests/sweeps |

- Defaults 0.07 / 0.05 / no cooldown adopted by PI decision 2026-10-24 from `scripts/drift_trigger_sweep.py` (synthetic AR(1) background 0.04 +- 0.012, 97 events): ~6 fires/yr vs 121 (open loop) for the old 0.05 memoryless trigger, recall 0.99, mean delay 0.76 tick. **Recalibrate on a recorded Aeon trace** (rule of thumb: `threshold_off` ~ background mean + 1 sigma, `threshold_on` ~ mean + 2.5 sigma).
- `threshold_off=None` (+ `cooldown_s=None`) reproduces the pre-2026-10-24 memoryless trigger exactly (tested). `DriftTrigger(threshold=t)` with `t <= 0.05` and no explicit `threshold_off` raises `ValueError`: pass `threshold_off` (or `None`).
- An empty window (no episodes / no concepts) returns `False` and leaves the state untouched.
- Persistence: a cron job that builds a new `DriftTrigger` per tick MUST pass `state_path` (e.g. `bench/runs/T1/drift_trigger_state.json`), otherwise the armed flag resets every tick and hysteresis is a no-op. With `state_path=None` the state stays in memory (right for a long-lived process) and `should_fire` emits a `RuntimeWarning` whenever hysteresis or cooldown is active; there is no default path on purpose (location is deployment-specific, and tests/sweeps must not write files). Do not store it in `latest.json`: that manifest is the R1 hash anchor and is only rewritten on a fire, while re-arming happens on non-firing ticks.
- `drift_rate(aeon, manifest)` and `update(rate)` are the two halves of `should_fire`; `scripts/drift_trigger_sweep.py` drives `update` on synthetic traces.
- `run_once` fetches the Aeon window before publishing the advisory and writes `known_concepts` = previous manifest set U concepts of the window, so `latest.json` can be passed straight to `should_fire`. Before 2026-10-24 the field did not exist and the rate was always 1.0. A pre-fix manifest (no field) reads as the empty set: the next drift check sees rate 1.0 (one fire at most), and the next `run_once` seeds the field.

## Anti-patterns (T1-specific)

- Importing `track1_perf` from `track2_paper` or `track3_deploy` (root track-isolation rule). T1 hook payloads are stack-shaped; T2/T3 expect 4-canonical shapes — even shared helpers leak schema.
- Setting `apply_w2_prox=True` to "make T1 paper-faithful". The full W2 prox at 40 species blows past the nightly window; if you need paper fidelity on hybrid species, that is a new track, not a flag flip.
- Hand-editing `bench/runs/T1/latest.json` (tau, git_sha, rho_keys, known_concepts) to skip a re-run. The manifest is the R1 hash anchor.
- Bumping `n_grid` (default 256) without re-baselining all downstream `bench/runs/T1*` checkpoints — grid resolution is part of the FC axis.
- Mid-day invocation while T3 is serving traffic: routing-advisory channel is shared; T1's `publish_advisory` will collide with T3's `route_advisory`.
- Replacing `MixedCanonicalSpecies` with raw 4-canonical to "simplify": that is T2. Use T2.
