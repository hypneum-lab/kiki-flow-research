# Erratum — W2 proximal operator defects (2026-10-24)

Status: sections 1–6 were written before the reruns (PR #18); section 7
has the rerun results and section 8 the PI decisions (2026-10-24).
No historical result JSON or zip has been edited: the reruns are
written to new files. `paper/main.pdf` was rebuilt from the corrected
source (§8, item 6); earlier PDFs remain in the tags and releases.

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

## 7. Reruns with the corrected prox (done 2026-10-24)

Host macM1 (Apple M1 Max, `MacBookPro.lan`), Python 3.14.4, MLX pinned
to the CPU device (Metal was busy with another campaign), `nice -n 19`,
one job at a time. Driver: `experiments/erratum_reruns_2026_10_24.py`
at `8a92b56` (Fig. 1 subcommand at `979b7e7`). Seeds 0–4. Each output
JSON records its provenance. The historical outputs are kept unchanged.

| Quantity | Old (defective prox) | New (corrected prox) | New file |
|---|---|---|---|
| Fast vs rigorous, reduced setup (grid 16, 500 particles, 20 slow), KL(rigorous‖fast) summed over 4 species, 5 seeds | 0.000638 ± 0.000265 (max 0.000937), reported as "bits", is nats | **0.218 ± 0.026 nats** (max 0.262); per seed 0.232, 0.186, 0.262, 0.206, 0.202 | `paper/erratum_2026_10_24/fast_vs_rigorous_kl_fixed.json` (43 s) |
| Same at paper scale (grid 64, 10k particles, 100 slow), final step | not measured | **0.262 ± 0.014 nats** | `paper/erratum_2026_10_24/fig1_species_moments_summary.json` |
| Rigorous runtime, 5 seeds × 100 slow × 10k particles | 84 min (GrosMac M5, MLX GPU simulator, old POT prox loop) | **12 min 42 s** (M1 Max, MLX on CPU) | `paper_rigorous_fixedprox/stats.json` |
| Fast runtime, same scale | 2 min 31 s (GrosMac M5, GPU) | 3 min 37 s (M1 Max, CPU) | `paper_rigorous_fixedprox/fast/stats.json` |
| Final entropy per species (bits), flat potentials | fast 5.99998 | fast 5.99998; **rigorous 5.896–5.912** (5.86–5.87 at step 1) | same |
| ε-sweep KL(ρ_ε‖ρ_0.001) at ε = 0.005 / 0.01 / 0.05 / 0.1 | 4.68 / 8.02 / 8.34 / 8.35 (nats, reported as bits) | **3.8e-10 / 4.7e-4 / 0.229 / 0.669 nats** | `paper/erratum_2026_10_24/epsilon_sweep_fixed.json`, `paper/figures/fig4_kl_vs_epsilon_fixed.*` |

Reading:

- **The fast path is not a numerical stand-in for the rigorous path.**
  With a working prox, the two differ by about 0.2–0.26 nats. The fast
  path sits at the uniform histogram (5.99998 bits). The rigorous path,
  whose prox anchors each update to the particle histogram, stays
  around 5.90 bits. Results obtained only with the fast path (Figs.
  2–3, aggregate statistics, specialisation sweeps, coupling ablation,
  CL proxy) describe the projected-gradient scheme. Whether they hold
  for the entropic-W2 scheme has not been tested.
- **New Fig. 1** (`paper/figures/fig1_species_moments.*`) plots
  per-species mean position and entropy over slow steps for both paths.
  It shows no species separation under flat potentials: seed-mean
  positions stay within about ±0.012 and the seed bands overlap. The
  "trajectories explore distinct regions" text is removed.
- In the ε-sweep test, ε sets both the prox strength and the Sinkhorn
  regularisation, so Fig. 4 is not a clean entropic-bias measurement.
  The paper now says so and points to the E1b control.

Still pending:

| Item | Cost | Host |
|---|---|---|
| T3 pairs (`dump_t2_pairs.py`) + retraining with the fixed prox | pairs: seconds; retraining: as the original | macM1 / Studio |
| Specialisation sweep on the rigorous path (does the 2.04-bit gap hold?) | 72 configs × rigorous cost; estimate hours on CPU | macM1 / Studio |
| End-to-end NumPy vs MLX rigorous timing | 10–20 min | one machine for both sides |

## 8. Decisions (PI, 2026-10-24)

1. MLX speed claim: withdrawn (PR #18).
2. Reruns of `fast_vs_rigorous_kl`, `paper_rigorous/` and the ε-sweep:
   done (§7). The agreement paragraph is rewritten with the new numbers.
3. Fig. 4: replaced by the re-measured curve, in nats, with the
   ε-conflation caveat.
4. Fig. 1: replaced by per-species moments.
5. Public notice: a note on the GitHub releases and a new Zenodo
   version. Drafted, **not published**; awaiting PI confirmation.
6. PDF / bundle: `paper/main.pdf` rebuilt from `paper/main.tex` at
   `087d1f6` on 2026-10-24 (electron-server, Tectonic 0.17.0, TeX
   bundle `tlextras-2022.0r0`, 16 pages, no undefined reference or
   citation). The v0.5 arXiv bundle and zip are marked superseded and
   left unchanged: its source still has the withdrawn KL agreement,
   the old Fig. 1 and the old epsilon sweep, so rebuilding its PDF
   alone would republish them.
7. **[PI DECISION]** Whether to rerun the specialisation sweep on the
   rigorous path before claim (ii) of the abstract is kept. The
   abstract now says the claim is about the fast path.
