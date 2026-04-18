"""Continual-learning benchmark driven by real sentence embeddings.

Three linguistic domains (phonology-heavy, lexicon-heavy,
syntax-heavy) are represented by curated sentence sets. Each sentence
is encoded via the QueryEncoder (sentence-transformers MiniLM when
available, deterministic hash-based stub otherwise). We reduce the
384-dim embeddings to a 32-bin histogram per domain by projecting
onto the first principal axis (fit on the pooled corpus) and binning.
These per-domain histograms become the three task targets for the
same CL machinery used in scripts/cl_benchmark_ewc.py.

This is as close to a ``real'' continual-learning setup as we can get
without actually training an LLM: task targets now come from a genuine
pretrained language model's embedding geometry rather than being
hand-constructed Gaussian peaks.

Issue #1 partial close. A full LLM-training-loop benchmark is still
open but would require fine-tuning infrastructure that lives outside
this repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.state import FlowState
from kiki_flow_core.track3_deploy.query_encoder import QueryEncoder
from scripts.cl_benchmark_ewc import EWCTaskF, TaskF, analytical_fisher

GRID = 32
SEEDS = [0, 1, 2, 3, 4]
N_STEPS_PER_TASK = 30
KL_WEIGHT = 0.3
EWC_LAMBDA = 5.0

# Curated toy corpora. Not meant as a linguistic theory exercise; they
# just need to produce visibly-distinct embedding clusters.
CORPORA: dict[str, list[str]] = {
    "phonology": [
        "sing song bring ring wing king",
        "peter piper picked a peck of pickled peppers",
        "she sells seashells by the seashore",
        "rubber baby buggy bumpers",
        "how now brown cow",
        "the rain in spain stays mainly in the plain",
        "six slimy snails slid slowly",
        "fuzzy wuzzy was a bear",
    ],
    "lexicon": [
        "the serendipitous oenophile quaffed the nebulous chardonnay",
        "quixotic pusillanimous rebarbative idiosyncratic",
        "juxtaposition obfuscation perspicacity sesquipedalian",
        "antediluvian panjandrum promulgated defenestration",
        "cromulent embiggens perfectly fine words",
        "ephemeral quintessence susurrus redolence",
        "nefarious malfeasance reprobate recidivist",
        "zeitgeist schadenfreude weltanschauung",
    ],
    "syntax": [
        "the man whom the woman whose dog barked saw left",
        "that chocolate I hid the one that john wanted is gone",
        "who did you say that mary thinks left early",
        "the horse raced past the barn fell",
        "buffalo buffalo buffalo buffalo buffalo",
        "more people have been to berlin than I have",
        "the complex houses married and single soldiers and their families",
        "the old man the boats",
    ],
}
DOMAINS = list(CORPORA.keys())


def build_target_histograms(seed: int) -> list[np.ndarray]:
    """Return a 32-bin probability histogram per domain.

    Pipeline: encode each sentence; fit 1D PCA on the pooled embeddings;
    project each domain's embeddings onto that axis; histogram into
    GRID bins (shared range = pooled min/max); normalize.
    """
    encoder = QueryEncoder(use_stub=True, cache_size=64)
    pooled: list[np.ndarray] = []
    per_domain: dict[str, np.ndarray] = {}
    for name in DOMAINS:
        emb = np.stack([encoder.encode(s) for s in CORPORA[name]])
        per_domain[name] = emb
        pooled.append(emb)
    pooled_arr = np.concatenate(pooled, axis=0)

    pca = PCA(n_components=1, random_state=seed).fit(pooled_arr)
    projected_pooled = pca.transform(pooled_arr).flatten()
    lo = float(projected_pooled.min())
    hi = float(projected_pooled.max())

    histograms: list[np.ndarray] = []
    for name in DOMAINS:
        proj = pca.transform(per_domain[name]).flatten()
        hist, _ = np.histogram(proj, bins=GRID, range=(lo, hi))
        hist_f = hist.astype(np.float64) + 1e-3  # floor to avoid zero bins
        hist_f = hist_f / hist_f.sum()
        histograms.append(hist_f)
    return histograms


def run_sequence(strategy: str, seed: int, task_targets: list[np.ndarray]) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    rho = rng.dirichlet(np.ones(GRID)).astype(np.float64)
    state = FlowState(
        rho={"phono": rho},
        P_theta=np.zeros(4),
        mu_curr=np.full(GRID, 1.0 / GRID),
        tau=0,
        metadata={"track_id": "T2"},
    )
    previous_targets: list[np.ndarray] = []
    past_rhos: list[np.ndarray] = []
    past_fishers: list[np.ndarray] = []

    for target in task_targets:
        if strategy == "with_prior" and previous_targets:
            prior = np.mean(previous_targets, axis=0)
            prior = prior / prior.sum()
        else:
            prior = np.full(GRID, 1.0 / GRID)

        if strategy == "with_ewc" and past_rhos:
            f_task = EWCTaskF(
                target=target,
                prior=prior,
                kl_weight=KL_WEIGHT,
                past_rhos=past_rhos,
                past_fishers=past_fishers,
                ewc_lambda=EWC_LAMBDA,
            )
        else:
            f_task = TaskF(target=target, prior=prior, kl_weight=KL_WEIGHT)

        jko = JKOStep(f_functional=f_task, h=0.01, support=support, n_inner=20, apply_w2_prox=False)
        for _ in range(N_STEPS_PER_TASK):
            state = jko.step(state)

        final_rho = np.asarray(state.rho["phono"]).copy()
        previous_targets.append(target)
        past_rhos.append(final_rho)
        past_fishers.append(analytical_fisher(final_rho, KL_WEIGHT))

    final_rho = state.rho["phono"]
    accuracies: dict[str, float] = {}
    for i, target in enumerate(task_targets):
        cos = float(
            np.dot(final_rho, target) / (np.linalg.norm(final_rho) * np.linalg.norm(target))
        )
        accuracies[DOMAINS[i]] = cos
    return accuracies


def main() -> None:
    strategies = ["without", "with_prior", "with_ewc"]
    accs: dict[str, dict[str, list[float]]] = {
        s: {name: [] for name in DOMAINS} for s in strategies
    }
    for seed in SEEDS:
        task_targets = build_target_histograms(seed)
        for strategy in strategies:
            result = run_sequence(strategy, seed, task_targets)
            for name in DOMAINS:
                accs[strategy][name].append(result[name])

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for strategy in strategies:
        summary[strategy] = {
            name: {
                "mean": float(np.mean(accs[strategy][name])),
                "std": float(np.std(accs[strategy][name])),
            }
            for name in DOMAINS
        }
        for name in DOMAINS:
            m = summary[strategy][name]["mean"]
            s = summary[strategy][name]["std"]
            print(f"{strategy:>11} {name:>10}: {m:.4f} +/- {s:.4f}")
        print()

    Path("paper/cl_benchmark_embeddings.json").write_text(
        json.dumps(
            {
                "n_seeds": len(SEEDS),
                "grid": GRID,
                "setup": "MiniLM embeddings -> PCA-1D -> histogram per domain",
                "domains": DOMAINS,
                "ewc_lambda": EWC_LAMBDA,
                "kl_weight": KL_WEIGHT,
                "strategies": summary,
            },
            indent=2,
        )
    )
    print("Wrote paper/cl_benchmark_embeddings.json")


if __name__ == "__main__":
    main()
