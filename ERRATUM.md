# Erratum — W2 proximal operator defects (2026-10-24)

Status: **draft for the PI**. Items marked **[PI DECISION]** are not
settled by this document. No result JSON, PDF or zip has been edited or
regenerated; this file and the source-text corrections in the same PR
are the only changes.

## 1. The two defects

**D1 — POT `prox_w2`, 2026-04-17 (`029edba`) to 2026-06-04 (`ffee5d9`).**
The "transport gradient" was `transport @ cost_matrix.diagonal()`. The
squared-Euclidean self-distance diagonal is zero, so the term was zero
and the output never depended on `reference`. What remained was the
update `q <- q + step * epsilon * log(q / distribution)`, started at
`q = distribution`: an unstable fixed point. In practice:

- on well-populated, near-uniform histograms (the T2 runs) it returned
  almost exactly its input, so the "rigorous" JKO step reduced to the
  Euclidean projected-gradient step of the fast path;
- when some bins are close to zero it diverges away from `distribution`
  (repulsion), still without looking at `reference`.

Fixed in `ffee5d9` (Peyré entropic prox, Dykstra scaling).

**D2 — MLX `mlx_prox_w2`, 2026-04-18 (`04ed762`) to 2026-10-24
(`3fc35f6`, PR #16).** The MLX port also ignored `reference` and
returned `distribution` (outputs for opposite references were
bit-identical). Fixed in PR #16 by porting the Dykstra operator.

## 2. Dependency map

| Claim / artefact | Where | Produced by | Code path | Affected |
|---|---|---|---|---|
| "native MLX Sinkhorn accelerates the rigorous path by 5x / 5.13x" | `paper/main.tex` abstract and §Fast vs rigorous; `paper/arxiv_bundle/main.tex` (same lines); `paper/kiki-flow-arxiv-v0.5.zip`; `paper/main.pdf`; `PERFORMANCE.md`; `README.md`; `CHANGELOG.md`; `bench/CLAUDE.md`; `paper_rigorous/CLAUDE.md` | `bench/sinkhorn_backend_bench.jsonl:1` (`pot_s` 29.72, `mlx_s` 5.79; grid 16, 5 slow, 100 particles, 1 seed) | D2 (MLX side) timed against D1 (POT side) | **Yes — withdrawn.** Neither side ran the current algorithm. |
| `bench/sinkhorn_mlx/stats.json` + `figures/fig1_phase_seed0.*` | bench only (not in the paper) | `run_paper(..., sinkhorn_backend="mlx")` | D2 | **Yes.** Also the figure is a constant point (see §5). |
| `bench/sinkhorn_pot/stats.json` + figure | bench only | `run_paper(..., sinkhorn_backend="pot")` | D1 | **Yes.** |
| Rigorous runtime "84 min", "35x slower per JKO step", abstract "under ninety minutes" | `paper/main.tex` abstract and §Fast vs rigorous; `README.md` table; `PERFORMANCE.md` | `paper_rigorous/` run (`647af58`, re-run `5dad58e`, POT) | D1 | **Yes.** It times the old POT loop (500 Sinkhorn solves per prox call), not the corrected operator, which is much cheaper (§3). |
| `paper_rigorous/stats.json`, `paper_rigorous/figures/fig1_*` | §Phase portrait caption ("Rigorous-JKO variant") | `run_paper(use_w2_prox=True)`, POT | D1 | **Yes** for provenance. The figure content is unaffected only because it is constant (§5). |
| Fast-vs-rigorous agreement "KL = 0.000638 ± 0.000265 bits", "within ~0.06 %", "justifying its use for the headline figures" | `paper/main.tex` §Fast vs rigorous; arXiv bundle | `scripts/fast_vs_rigorous_kl.py` → `paper/fast_vs_rigorous_kl.json` (`6b09725`, 2026-04-18) | D1 (`FullJKOSolver`, default POT prox) | **Yes — the conclusion does not survive** (§3). |
| Fig. 4 and "entropic bias grows 0 → 8.3 bits, saturates at ε ≥ 0.05" | `paper/main.tex` §Entropic Sinkhorn bias; `README.md` table; arXiv bundle | `scripts/epsilon_sweep.py` → `paper/epsilon_sweep.json`, `fig4_kl_vs_epsilon` (`e0d0e31`, 2026-04-18) | D1, direct `prox_w2` call | **Yes — artefact.** Reproduced bit-for-bit with the old code; the outputs do not depend on the target (§3). |
| T3 surrogate "distilled from T2 rigorous-run trajectories"; latency p50 0.04 ms / p99 0.06 ms | `paper/main.tex` §Deployment latency; `README.md`; `PERFORMANCE.md` | `scripts/dump_t2_pairs.py` (`FullJKOSolver`, POT) → `bench/runs/T2_pairs/` → T3 weights; `bench/T3_latency.jsonl` | D1 | **Latency: no** (it depends only on the network size). **What the surrogate imitates: yes** (the pre-fix map). |
| Fast-path figures 2–3, aggregate statistics, 27/72-config sweeps, coupling-modes ablation, CL proxy and EWC benchmarks, λ-monotonicity, Track 1 | `paper/main.tex` §§ Aggregate, Sweep, Coupling, CL, λ-mono | `JKOStep(apply_w2_prox=False)` in `paper_run.py`, `hyperparam_sweep*.py`, `cl_benchmark*.py`, `check_lambda_monotonicity.py`, `offline_consolidator.py` | no prox | **No** (they never call `prox_w2`). |
| E1 / E1b (OU pre-registered bias test), E2 (propagation-of-chaos preconditions) | `paper/main.tex` §Discussion | `experiments/e1*.py`, `e2*.py` (2026-06-04, after `ffee5d9`) | own Dykstra prox / no prox | **No.** |
| Real-LLM continual-learning runs (E2–E4b) | `paper/main.tex` §CL real | `bench/cl_llm/` | no prox | **No.** |

## 3. Spot-check (one seed, labelled, NOT a paper result)

Run twice on 2026-10-24 on macM1 (CPU, about 2.5 min, identical output), script
`experiments/erratum_2026_10_24_spot_prox.py`. It copies the old
`prox_w2` verbatim from `029edba` and compares it with the current one.
The T2 part uses the NumPy `ParticleSimulator` instead of the MLX one
(Metal was busy), so the numbers are indicative only.

**(A) ε-sweep setup of `scripts/epsilon_sweep.py`** (source peak at
x = −0.99, target peak at x = +0.99):

| ε | old: KL vs ε=0.001 (nats) | old: output mean x | fixed: KL vs ε=0.001 (nats) | fixed: output mean x |
|---|---|---|---|---|
| 0.001 | 0 | −0.64 | 0 | 0.985 |
| 0.005 | 4.68 | 0.82 | 3.8e-10 | 0.985 |
| 0.01 | 8.02 | 1.73 | 4.7e-4 | 0.971 |
| 0.05 | 8.34 | 1.77 | 0.229 | 0.655 |
| 0.1 | 8.35 | 1.77 | 0.669 | 0.418 |

The old column matches `paper/epsilon_sweep.json` exactly. With the old
operator, swapping the target for another random target changes the
output by L1 = 0.0 (it ignores the target); with the fixed one, by
L1 = 0.44. The old outputs are pushed to the right edge of the grid
(mean x ≈ 1.77, beyond the target) by the repulsion, not toward a
"uniform barycenter" as the paper says. The paper also labels these
nats as bits.

**(B) Reduced fast-vs-rigorous setup** (grid 16, 500 particles,
200 fast × 20 slow, seed 0, `h = 0.05`):

| Comparison | KL summed over 4 species (nats) | L1 summed |
|---|---|---|
| rigorous (old prox) vs fast | 2.2e-4 | 0.033 |
| rigorous (fixed prox) vs fast | **0.219** | 1.04 |
| rigorous (fixed) vs rigorous (old) | 0.207 | 1.01 |

The old value is the same order as the published seed-0 value
(2.05e-4), which confirms that the published agreement measured
"Euclidean step vs nearly the same Euclidean step". With a working
prox, the gap is roughly 300–1000× larger.

Wall time on this setup: fast 1.5 s, rigorous with old POT prox
133.9 s, rigorous with fixed NumPy prox 7.1 s.

**Reproducibility CI.** The weekly `reproducibility.yml` runs on GitHub
(last one 2026-07-27, run 30258671097) all used commit `2ff8511`, which
predates `ffee5d9`. They reproduced the committed
`paper/epsilon_sweep.json` byte-for-byte and
`paper/fast_vs_rigorous_kl.json` to about 1e-14, so the published
numbers are faithful outputs of the old code. That workflow only checks
that JSON keys are present, so it would not flag a change in values. No
rerun with the fixed prox exists yet.

## 4. Corrected speedup statement

The 5.13× row is withdrawn. Per-call benchmark of the corrected Dykstra
prox (PR #16; 200 iterations, median of 7, Apple M1 Max, NumPy float64
vs MLX float32, parity checked before timing):

| n | NumPy | MLX | NumPy/MLX |
|---|---|---|---|
| 16 | 0.80 ms | 7.86 ms | 0.10 |
| 64 | 1.16 ms | 7.96 ms | 0.15 |
| 256 | 3.75 ms | 10.45 ms | 0.36 |
| 1024 | 41.35 ms | 31.35 ms | 1.32 |
| 2048 | 382.94 ms | 89.93 ms | 4.26 |

MLX is slower at the paper's grid sizes (16 and 64) and only faster from
n ≈ 1024. No end-to-end rigorous-path speedup has been measured with the
corrected code; none is claimed.

## 5. Other problems found during the audit (not caused by D1/D2)

1. **Figure 1 is a constant point.** `figures/phase_portrait.py` plots
   `rho.mean()`, which is always `1/GRID` for a normalised histogram.
   All ten PNGs in `paper/figures/fig1_*` and
   `paper_rigorous/figures/fig1_*` have the same SHA-1 (`a9a56d84…`) and
   show one point at (0.0156, 0.0156). The text "the trajectories
   explore distinct regions of the species simplex" is not supported.
   The caption says "Rigorous-JKO variant" while §Figure status says
   figures 1–3 come from the fast path.
2. **Units.** `fast_vs_rigorous_kl.py` and `epsilon_sweep.py` use the
   natural log; the paper reports bits. "KL = 0.000638 … within
   ~0.06 %" also reads a KL value as a percentage.
3. **Stale "Figure status" paragraph** says Fig. 4 uses synthetic data,
   while §Entropic Sinkhorn bias calls it measured.
4. **JKO output is not fed back.** `MultiscaleLoop.run` histograms the
   particles, applies one JKO step and stores the result, but never
   updates the particles from it. Each "slow step" is therefore one JKO
   step applied to a histogram of freely diffusing particles. This
   bounds how much any prox defect can change a final state, and it
   also bears on what the T2 "flow" figures show.
5. **Provenance details.** `bench/sinkhorn_backend_bench.jsonl:1` has
   `ts = 1713406400`, which decodes to 2024-04-18, two years before the
   commit (`ae01ab5`, 2026-04-18); the row looks hand-entered.
   `README.md` cited a non-existent `bench/T2_backend_speedup.jsonl`.
   `kiki_flow_core/track2_paper/CLAUDE.md` said the rigorous artefacts
   used `sinkhorn_backend="mlx"`, but `paper_rigorous/stats.json`
   records `"pot"` and `run.log` predates the MLX backend.

## 6. Public exposure

- **arXiv: not submitted.** `CITATION.cff` says "arXiv preprint
  forthcoming", and `README.md` says the v0.5 bundle "must be
  regenerated before submission". arXiv itself was not contacted.
- `paper/kiki-flow-arxiv-v0.5.zip` (2026-04-19) contains
  `arxiv_bundle/main.tex` (identical to the pre-PR
  `paper/arxiv_bundle/main.tex`, with the 5× and 5.13× claims, the KL
  agreement, the ε-sweep text and the constant Fig. 1) and a compiled
  `main.pdf`.
- The repository is **public** on GitHub. The 5× claim is in every
  `paper-v0.5-draft` … `paper-v0.11-draft` tag; the KL agreement from
  `paper-v0.7-draft`. GitHub releases `paper-v0.5-draft` and
  `paper-v0.7-draft` ship a `main.pdf`. Zenodo DOI
  `10.5281/zenodo.19666756` archives `paper-v0.8-draft`, which contains
  both claims.

## 7. Pending reruns (not run here)

`FullJKOSolver` hard-imports MLX, so T2 reruns need Apple Silicon
(electron-server cannot import it without a new dependency). Estimates
are from the spot-check timings.

| Rerun | Cost estimate | Host |
|---|---|---|
| `scripts/epsilon_sweep.py` (Fig. 4) | seconds | any Apple Silicon |
| `scripts/fast_vs_rigorous_kl.py`, 5 seeds | about 1–2 min | macM1 after the N10-C campaign, or Studio |
| `paper_rigorous/` (5 seeds × 100 slow × 10k particles, grid 64) | about 15–20 min on M-series with the MLX simulator: the fixed prox costs milliseconds per call at n = 64, and the 5× more inner gradient steps (`n_inner` 50 vs 10) dominate, as in the spot-check (7.1 s vs 1.5 s) | macM1 after N10-C, or Studio |
| End-to-end NumPy vs MLX rigorous timing (to replace the 5.13× row with a real one, if wanted) | about 10–20 min | same machine for both sides |
| `scripts/dump_t2_pairs.py` + T3 retraining | pairs: seconds; retraining: as the original | macM1 / Studio |

## 8. [PI DECISION] items

1. Withdraw or re-measure the end-to-end MLX vs NumPy rigorous timing.
2. Rerun `paper_rigorous/` and `fast_vs_rigorous_kl.py` with the fixed
   prox before editing the agreement paragraph, or remove that paragraph
   now.
3. Rerun `epsilon_sweep.py` (and fix units) or drop Fig. 4, given that
   E1b already covers the entropic bias properly.
4. Fix or drop Fig. 1 (constant plot).
5. Public notice: a note on the GitHub releases / a new Zenodo version,
   or only this file.
6. Regenerate `paper/main.pdf` and the v0.5 bundle, or mark the bundle
   superseded.
