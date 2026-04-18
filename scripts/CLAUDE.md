# scripts — paper reproducibility entry points

Every script here is a deterministic producer of one or more artifacts
consumed by `paper/` (JSON + PDFs). Running a script must be idempotent
given the same seeds.

## Script -> artifact map

| Script | Produces | Referenced in |
|---|---|---|
| `cl_benchmark.py` | `paper/cl_benchmark.json`, `paper/figures/fig5_cl_gap.*` | README results table |
| `cl_benchmark_ewc.py` | `paper/cl_benchmark_ewc.json`, `fig5_cl_gap_ewc.*` | paper appendix |
| `epsilon_sweep.py` | `paper/epsilon_sweep.json`, `fig4_kl_vs_epsilon.*` | Sinkhorn-bias claim |
| `hyperparam_sweep.py` | `paper/hyperparam_sweep.json`, `fig6_heatmap.*` | specialization claim |
| `hyperparam_sweep_dense.py` | `paper/hyperparam_sweep_dense.json` | denser follow-up sweep |
| `dump_t2_pairs.py` | `bench/runs/T2_pairs/*.safetensors` | T3 surrogate trainer input |
| `dump_hybrid_pairs.py` | `bench/runs/T2_pairs_d128/*.safetensors` | T3 v0.2-d128 trainer input |

Figures live under `paper/figures/`, emitted via the generators in
`kiki_flow_core/track2_paper/figures/` — scripts call those, never write
PDFs directly.

## Rules

- Seeds are declared at module top (`SEEDS = [0, 1, 2, 3, 4]`). Don't
  randomize silently; don't shrink the list to "make it faster".
- Invocation form is `PYTHONPATH=. uv run python scripts/<name>.py` —
  `pyproject.toml` already has `pythonpath = ["."]` for pytest, but
  scripts run outside pytest and need the explicit env var.
- Paths are relative to repo root (e.g. `Path("paper/cl_benchmark.json")`).
  Never use absolute paths; never compute paths from `__file__` gymnastics.
- Output JSON schema must stay backward-compatible with existing files in
  `paper/` and `paper_rigorous/` — the LaTeX source reads them. If a
  schema change is unavoidable, update `paper/main.tex` in the same commit.

## Anti-patterns (domain-specific)

- Adding a script that mutates an existing JSON file rather than rewriting
  it. JSON outputs are full snapshots of a run; partial updates break
  provenance.
- Embedding seeds inside the loop body (`rng = np.random.default_rng(0)`
  hard-coded). Accept `seed: int` as a parameter, expose `SEEDS` at the
  top of the module.
- Using `print(...)` for the result and forgetting the JSON write — the
  terminal output is a diagnostic, the JSON file is the artifact.
- Calling `plt.show()` in a reproducibility script. Save to disk only.
- Long-running sweeps with no checkpointing. If the script takes more
  than ~10 min, write incremental partial results so a crash doesn't
  destroy N hours of compute (see `hyperparam_sweep_dense.py` for the
  pattern).
