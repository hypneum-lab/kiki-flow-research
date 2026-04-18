"""Multi-modal continual-learning proxy.

Extends scripts/cl_benchmark_ewc.py with richer task targets: each task
is a mixture of 2-3 Gaussian peaks rather than a single peak, which
better approximates the structure of real linguistic tasks where a
stack's output distribution typically has several modes. Compares the
three strategies (without / with_prior / with_ewc) on this harder
setup.

Does not close issue #1 by itself — a real LLM benchmark is still
pending — but demonstrates that the existing consolidation machinery
scales to non-trivial target structure and produces an actionable
sensitivity curve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.state import FlowState
from scripts.cl_benchmark_ewc import EWCTaskF, TaskF, analytical_fisher

GRID = 32
SEEDS = [0, 1, 2, 3, 4]
N_STEPS_PER_TASK = 30
KL_WEIGHT = 0.3
EWC_LAMBDA = 5.0


def multimodal_target(task_id: int) -> np.ndarray:
    """Each task has 2-3 Gaussian peaks at distinct grid indices,
    modelling the structure of a linguistic task with multiple
    salient outputs (e.g., several plausible word completions)."""
    centers_by_task = [
        [5, 12],  # task 0: bimodal at 5, 12
        [8, 16, 24],  # task 1: trimodal
        [20, 28],  # task 2: bimodal at 20, 28
    ]
    weights_by_task = [
        [0.6, 0.4],
        [0.3, 0.4, 0.3],
        [0.5, 0.5],
    ]
    x = np.arange(GRID)
    peak = np.zeros(GRID)
    for c, w in zip(centers_by_task[task_id], weights_by_task[task_id], strict=True):
        peak += w * np.exp(-0.5 * ((x - c) / 1.5) ** 2)
    return peak / peak.sum()


def run_sequence(strategy: str, seed: int) -> dict[str, float]:
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

    for task_id in range(3):
        target = multimodal_target(task_id)
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
    for i in range(3):
        target_i = multimodal_target(i)
        cos = float(
            np.dot(final_rho, target_i) / (np.linalg.norm(final_rho) * np.linalg.norm(target_i))
        )
        accuracies[f"task{i}"] = cos
    return accuracies


def main() -> None:
    strategies = ["without", "with_prior", "with_ewc"]
    accs: dict[str, dict[str, list[float]]] = {
        s: {"task0": [], "task1": [], "task2": []} for s in strategies
    }
    for seed in SEEDS:
        for strategy in strategies:
            result = run_sequence(strategy, seed)
            for k in ["task0", "task1", "task2"]:
                accs[strategy][k].append(result[k])

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for strategy in strategies:
        summary[strategy] = {
            k: {
                "mean": float(np.mean(accs[strategy][k])),
                "std": float(np.std(accs[strategy][k])),
            }
            for k in ["task0", "task1", "task2"]
        }
        for k in ["task0", "task1", "task2"]:
            m = summary[strategy][k]["mean"]
            s = summary[strategy][k]["std"]
            print(f"{strategy:>11} {k}: {m:.4f} +/- {s:.4f}")
        print()

    Path("paper/cl_benchmark_multimodal.json").write_text(
        json.dumps(
            {
                "n_seeds": len(SEEDS),
                "grid": GRID,
                "setup": "multimodal targets (2-3 Gaussian peaks per task)",
                "ewc_lambda": EWC_LAMBDA,
                "kl_weight": KL_WEIGHT,
                "strategies": summary,
            },
            indent=2,
        )
    )
    print("Wrote paper/cl_benchmark_multimodal.json")


if __name__ == "__main__":
    main()
