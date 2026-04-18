# Announcement drafts

Draft posts for X / Bluesky / LinkedIn / Hacker News. Copy, edit to
your voice, fire when ready.

---

## X / Bluesky thread (5 posts)

**1/5**
New open-source release: kiki-flow-research. We model an LLM's state
as 4 activation densities (phono / lex / syntax / sem) à la
Levelt-Baddeley, evolve them as a Wasserstein gradient flow,
extract a 0.04 ms streaming surrogate. Paper + code + figures:
https://github.com/electron-rare/kiki-flow-research

**2/5**
Key number: 5 seeds × 100 JKO steps × 10k Langevin particles on
commodity Apple Silicon in 2 min 31 s (fast path) or 84 min
(rigorous with POT Sinkhorn prox). MLX on Metal + numpy + JAX
wired together. Figures are deterministic and reproducible.

**3/5**
Honest findings: under flat potentials the flow stays uniform
(entropy ≈ max). But with asymmetric per-species potentials at
(α=5, β=1), mean entropy drops 1.40 bits below max — that is
emergent species specialization. We report both regimes.

**4/5**
The naive consolidation we tested reduces forgetting on the first
task (0.90 vs 0.29 baseline) but over-stabilizes and hurts later
tasks (0.004 vs 0.81). Paper reports this mixed-sign result
explicitly. EWC-style Fisher weighting is next.

**5/5**
Three implementation tracks, 93 tests, 95 % coverage, strict
ruff + mypy. MIT licence. Looking for: a real CL benchmark
(issue #1), denser hyperparameter sweep with EWC (#2), native
MLX Sinkhorn (#3), end-to-end micro-kiki integration (#4).

---

## LinkedIn / longer-form post

> **Public release: kiki-flow-research — a Wasserstein gradient flow
> engine for LLM consolidation, grounded in the Levelt-Baddeley model
> of language production**
>
> The idea in one sentence: represent an LLM's internal state as four
> psycholinguistic activation densities and let them evolve under a
> Wasserstein-regularized gradient flow. The JKO scheme gives an
> implicit-time discretization; entropic Sinkhorn (POT) gives the
> proximal step numerically tractable; MLX on Apple Silicon makes
> particle simulation 2.6× faster than pure numpy.
>
> Three implementation tracks in one repo:
> - **Track 1** (perf): nightly offline consolidation over 40 hybrid
>   species for a MoE-LoRA deployment.
> - **Track 2** (paper): 10 000 Langevin particles coupled to a full
>   JKO step, producing the paper's figures.
> - **Track 3** (deploy): pure-NumPy 3-layer MLP surrogate that meets
>   a 10 ms p50 streaming-inference budget (measured p50 = 0.04 ms).
>
> Highlights from the v0.4 paper draft:
> - Flow is tractable (2 min 31 s fast / 84 min rigorous on 5 seeds).
> - Emergent species specialization under asymmetric potentials
>   (entropy gap 1.40 bits below max).
> - Honest mixed-sign continual-learning finding: naive consolidation
>   preserves the first task but over-stabilizes at its expense.
>
> MIT licence. 93 tests, 95 % coverage, strict ruff + mypy. Looking
> for collaborators on a real CL benchmark, an EWC-style refinement,
> a native MLX Sinkhorn, and an end-to-end micro-kiki integration.
>
> https://github.com/electron-rare/kiki-flow-research

---

## Hacker News Show HN

**Title**: `Show HN: Kiki-flow – Wasserstein gradient flow over
Levelt-Baddeley species for LLM routing`

**Body**:

> I released kiki-flow-research, a small research repo that models an
> LLM's internal state as four psycholinguistic activation densities
> (phono / lex / syntax / sem) evolving under a Wasserstein gradient
> flow. The JKO scheme makes this implicit-time tractable; entropic
> Sinkhorn (via POT) gives the proximal step a closed form; MLX on
> Apple Silicon makes 10 000 Langevin particles affordable.
>
> There are three implementations at different tradeoff points: a
> nightly offline consolidator for MoE-LoRA stacks, a rigorous
> N-particle paper-grade version, and a 3-layer MLP surrogate that
> streams at p50 = 0.04 ms on a GrosMac M5.
>
> The paper draft (v0.4) reports honest mixed-sign findings: the flow
> is numerically tractable and induces emergent specialization under
> asymmetric potentials, but naive prior-averaged consolidation
> over-stabilizes on the first task at the expense of later ones. A
> real continual-learning benchmark replacing the current
> distributional proxy is open as issue #1.
>
> MIT licence. 93 tests. Strict ruff + mypy. Feedback welcome.
>
> Repo: https://github.com/electron-rare/kiki-flow-research
> Paper: https://github.com/electron-rare/kiki-flow-research/releases/tag/paper-v0.4-draft

---

## arXiv submission draft (tweak before upload)

**Primary category**: cs.LG (Machine Learning)
**Secondary**: cs.CL (Computation and Language), math.OC (Optimization
and Control)

**Abstract**: (copy from `paper/main.tex`, 200 words)

**Comments field**: "Draft; code, data, and figures released under MIT
at https://github.com/electron-rare/kiki-flow-research tag
paper-v0.4-draft. Feedback and issues welcome on GitHub."

Remember: arXiv accepts LaTeX source with figures; upload
`paper/main.tex` + `paper/references.bib` + all PNG files in
`paper/figures/`.
