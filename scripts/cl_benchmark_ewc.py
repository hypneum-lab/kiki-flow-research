"""Continual-learning proxy with EWC-style Fisher regularization.

Issue #2 part (2). Replaces the naive prior-averaged consolidation in
scripts/cl_benchmark.py with a Fisher-weighted L2 penalty on past
task-final rhos. For our quadratic+KL task free energy, the Fisher
diagonal is computed analytically as ``2 + kl_weight / rho``, capturing
which grid coordinates are ``rigid`` (important for the previous task,
high Fisher) vs ``plastic`` (low Fisher, free to change).

Compares three arms on the same 3-task sequence:
  * ``without``: uniform prior between tasks (baseline forgetting).
  * ``with_prior``: prior = mean of past targets (the v0.4 naive
    baseline, preserves task 0 but over-stabilizes).
  * ``with_ewc``: EWC Fisher-weighted penalty.

Writes paper/cl_benchmark_ewc.json and updates paper/figures/fig5_cl_gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kiki_flow_core.master_equation import FreeEnergy, JKOStep
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.figures.continual_learning_gap import (
    make_continual_learning_gap,
)

GRID = 16
SEEDS = [0, 1, 2, 3, 4]
N_STEPS_PER_TASK = 30
KL_WEIGHT = 0.3
EWC_LAMBDA = 5.0


def target_distribution(task_id: int) -> np.ndarray:
    centers = [3, 8, 13]
    x = np.arange(GRID)
    peak = np.exp(-0.5 * ((x - centers[task_id]) / 1.2) ** 2)
    return peak / peak.sum()


class TaskF(FreeEnergy):
    """Base F: quadratic attraction to target + KL regularization to prior."""

    def __init__(self, target: np.ndarray, prior: np.ndarray, kl_weight: float) -> None:
        self.target = target
        self.prior = prior
        self.kl_weight = kl_weight

    def value(self, state: FlowState) -> float:
        r = state.rho["phono"]
        r_safe = np.clip(r, 1e-12, None)
        prior_safe = np.clip(self.prior, 1e-12, None)
        attr = float(((r - self.target) ** 2).sum())
        kl = float((r_safe * np.log(r_safe / prior_safe)).sum())
        return attr + self.kl_weight * kl

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        r = state.rho[species_name]
        r_safe = np.clip(r, 1e-12, None)
        prior_safe = np.clip(self.prior, 1e-12, None)
        grad_attr = 2.0 * (r - self.target)
        grad_kl = self.kl_weight * (np.log(r_safe / prior_safe) + 1.0)
        out: np.ndarray = grad_attr + grad_kl
        return out


class EWCTaskF(TaskF):
    """TaskF + Fisher-weighted L2 penalty on past task-final rhos."""

    def __init__(
        self,
        target: np.ndarray,
        prior: np.ndarray,
        kl_weight: float,
        past_rhos: list[np.ndarray],
        past_fishers: list[np.ndarray],
        ewc_lambda: float,
    ) -> None:
        super().__init__(target=target, prior=prior, kl_weight=kl_weight)
        self.past_rhos = past_rhos
        self.past_fishers = past_fishers
        self.ewc_lambda = ewc_lambda

    def value(self, state: FlowState) -> float:
        base = super().value(state)
        r = state.rho["phono"]
        ewc = 0.0
        for past_r, fisher in zip(self.past_rhos, self.past_fishers, strict=True):
            ewc += float((fisher * (r - past_r) ** 2).sum())
        return base + self.ewc_lambda * ewc

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        base_grad = super().grad_rho(state, species_name, eps)
        r = state.rho[species_name]
        ewc_grad = np.zeros_like(r)
        for past_r, fisher in zip(self.past_rhos, self.past_fishers, strict=True):
            ewc_grad = ewc_grad + 2.0 * fisher * (r - past_r)
        out: np.ndarray = base_grad + self.ewc_lambda * ewc_grad
        return out


def analytical_fisher(rho: np.ndarray, kl_weight: float) -> np.ndarray:
    """Diagonal Fisher of TaskF at rho: Hessian of the quadratic+KL free energy.

    Quadratic term |rho - target|^2 contributes 2 * I to the Hessian.
    KL term kl_weight * rho log(rho/prior) contributes kl_weight / rho.
    """
    rho_s = np.clip(rho, 1e-6, None)
    return 2.0 + kl_weight / rho_s


def run_sequence(strategy: str, seed: int) -> dict[str, float]:
    """Run the 3-task sequence with one of three strategies."""
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
        target = target_distribution(task_id)
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

        # Record end-of-task state
        final_rho = np.asarray(state.rho["phono"]).copy()
        previous_targets.append(target)
        past_rhos.append(final_rho)
        past_fishers.append(analytical_fisher(final_rho, KL_WEIGHT))

    final_rho = state.rho["phono"]
    accuracies: dict[str, float] = {}
    for i in range(3):
        target_i = target_distribution(i)
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

    # Update fig5 to show EWC alongside the baseline arms
    without_means = [summary["without"][k]["mean"] for k in ["task0", "task1", "task2"]]
    with_ewc_means = [summary["with_ewc"][k]["mean"] for k in ["task0", "task1", "task2"]]
    make_continual_learning_gap(
        tasks=["phonology", "lexicon", "syntax"],
        with_consolidation=with_ewc_means,
        without_consolidation=without_means,
        out_dir=Path("paper/figures"),
        filename="fig5_cl_gap_ewc",
    )

    Path("paper/cl_benchmark_ewc.json").write_text(
        json.dumps(
            {
                "n_seeds": len(SEEDS),
                "ewc_lambda": EWC_LAMBDA,
                "kl_weight": KL_WEIGHT,
                "strategies": summary,
            },
            indent=2,
        )
    )
    print("Wrote paper/cl_benchmark_ewc.json and paper/figures/fig5_cl_gap_ewc.*")


if __name__ == "__main__":
    main()
