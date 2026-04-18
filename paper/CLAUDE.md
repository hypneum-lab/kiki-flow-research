# paper — LaTeX source + fast-path figures + result JSONs

Publication directory. Contains the paper source (`main.tex`,
`references.bib`), the figures it embeds, the JSON result files that
back every numerical claim, and a packaged `arxiv_bundle/` +
`kiki-flow-arxiv-v0.4.zip`.

This is the **fast-path** run (target ~2 min 31 s on GrosMac M5). The
companion **rigorous** run (84 min) lives in `paper_rigorous/`.

## Layout

- `main.tex` / `main.pdf` — paper source + latest build.
- `references.bib` — BibTeX; add entries here, not inline in `.tex`.
- `figures/fig{1..6}_*.{pdf,png}` — 5 seeds of phase portraits, F-decay,
  Turing, KL-vs-epsilon, CL-gap, hyperparam heatmap. Emitted by the
  generators in `kiki_flow_core/track2_paper/figures/` via the scripts
  in `scripts/`. Never hand-edit.
- `stats.json`, `cl_benchmark.json`, `cl_benchmark_ewc.json`,
  `epsilon_sweep.json`, `hyperparam_sweep.json`,
  `hyperparam_sweep_dense.json` — single source of truth for every
  measured number in the paper.
- `trajectories/` — cached particle trajectories for phase portraits.
- `arxiv_bundle/` + `kiki-flow-arxiv-v0.4.zip` — packaged submission.
  Regenerate both together; never hand-edit one.

## Editing discipline

- LaTeX build: `tectonic main.tex` (or any TeX distribution). Don't
  commit `.aux`, `.log`, `.out`, `.synctex.gz`; the `.gitignore` should
  already drop them.
- Every numerical claim in prose must trace to one of the JSONs above or
  to a row in `bench/*.jsonl`. If you add a claim, cite the source file
  in a comment: `% from paper/hyperparam_sweep.json`.
- Figure references: `\includegraphics{figures/figN_name}` without
  extension — LaTeX picks PDF, the PNGs exist for the README and for
  web previews.
- If a result JSON changes, rebuild the PDF in the same commit. A commit
  that updates a JSON but not the PDF leaves the paper inconsistent
  with its claims.

## Fast path vs rigorous path

The two tracks answer different questions:

- **Fast (this dir)**: "Is the flow tractable and does the qualitative
  behaviour hold at fast-path resolution?" — what goes into the paper
  body.
- **Rigorous (`paper_rigorous/`)**: "Do the numbers survive the
  full W2-prox + MLX Sinkhorn path with 100 slow steps?" — what backs
  the companion appendix / reviewer rebuttal material.

When a claim appears in both, both JSONs must agree to the precision
reported in prose.

## Anti-patterns (domain-specific)

- Hand-editing numbers in prose to "round nicely" without re-running the
  script. The scripts are deterministic — the number is what it is.
- Adding a figure by dropping a PDF in `figures/` without a matching
  generator under `kiki_flow_core/track2_paper/figures/`. Every figure
  must be regeneratable from scratch.
- Committing `main.pdf` without running `tectonic` on a clean checkout
  first. Stale PDFs diverge from `main.tex` silently.
- Editing the `arxiv_bundle/` contents directly. Regenerate the bundle.
- Inlining bibliographic entries in `\bibitem{}` — use `references.bib`.
