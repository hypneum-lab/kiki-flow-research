"""Erratum spot-check (2026-10-24): old vs fixed ``prox_w2``. NOT a paper result.

Compares the pre-ffee5d9 POT ``prox_w2`` (body copied verbatim from 029edba)
with the current Dykstra ``prox_w2`` on:

(A) the epsilon-sweep setup of ``scripts/epsilon_sweep.py``;
(B) one seed of the reduced setup of ``scripts/fast_vs_rigorous_kl.py``,
    using the NumPy ``ParticleSimulator`` instead of the MLX one;
(C) sensitivity of each operator to its ``reference`` argument.

Prints JSON to stdout and writes nothing. See ERRATUM.md section 3.
"""

from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import ot

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy
from kiki_flow_core.track2_paper.particle_simulator import ParticleSimulator
from kiki_flow_core.wasserstein_ops import prox_w2 as prox_fixed

GRID = 16
EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1]


def prox_old(
    distribution: np.ndarray,
    reference: np.ndarray,
    epsilon: float,
    support: np.ndarray,
    n_iter: int = 500,
    step_size: float = 0.1,
) -> np.ndarray:
    """Verbatim body of ``prox_w2`` at 029edba (pre-ffee5d9), kept for the audit."""
    cost_matrix = ot.dist(support, support, metric="sqeuclidean")
    q = distribution.copy()
    reg = max(epsilon, 1e-3)
    for _ in range(n_iter):
        transport = ot.sinkhorn(
            q, reference, cost_matrix, reg=reg, numItermax=200, method="sinkhorn_log"
        )
        grad = (transport @ cost_matrix.diagonal()) - epsilon * np.log(q / distribution + 1e-12)
        q = q - step_size * grad
        q = np.clip(q, 1e-12, None)
        q = q / q.sum()
    return q


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return float((p * np.log(p / q)).sum())


def part_a(support: np.ndarray) -> dict[str, object]:
    x = support[:, 0]
    d = np.exp(-0.5 * ((x + 1) / 0.5) ** 2)
    d /= d.sum()
    r = np.exp(-0.5 * ((x - 1) / 0.5) ** 2)
    r /= r.sum()
    out: dict[str, object] = {"mean_x": {"distribution": float(d @ x), "reference": float(r @ x)}}
    for name, fn in [("old", prox_old), ("fixed", prox_fixed)]:
        outs = [fn(d, reference=r, epsilon=e, support=support, n_iter=50) for e in EPSILONS]
        out[name] = {
            "kl_vs_eps0.001_nats": [kl(o, outs[0]) for o in outs],
            "mean_x_out": [float(o @ x) for o in outs],
        }
    return out


def run_t2(kind: str, seed: int = 0) -> tuple[dict[str, np.ndarray], float]:
    species = CanonicalSpecies()
    names = species.species_names()
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    f_func = T2FreeEnergy(
        species=species,
        potentials={n: np.zeros(GRID) for n in names},
        prior={n: np.full(GRID, 1.0 / GRID) for n in names},
        turing_strength=0.1,
    )
    if kind == "fast":
        jko = JKOStep(f_functional=f_func, h=0.05, support=support, n_inner=10)
    else:
        jko = FullJKOSolver(f_functional=f_func, h=0.05, support=support, epsilon=0.05)
        jko.prox_fn = prox_old if kind == "rig_old" else prox_fixed
    sim = ParticleSimulator(species=species, n_particles=500, latent_dim=2, seed=seed)
    loop = MultiscaleLoop(sim=sim, jko=jko, n_fast=200, n_slow=20, support=support)
    t0 = time.perf_counter()
    manifest = loop.run(seed=seed)
    wall = time.perf_counter() - t0
    final = manifest["trajectory"][-1]
    return {n: np.asarray(final.rho[n]) for n in names}, wall


def part_b() -> dict[str, object]:
    res = {k: run_t2(k) for k in ["fast", "rig_old", "rig_fixed"]}
    out: dict[str, object] = {"wall_s": {k: v[1] for k, v in res.items()}}
    for a, b in [("rig_old", "fast"), ("rig_fixed", "fast"), ("rig_fixed", "rig_old")]:
        pa, pb = res[a][0], res[b][0]
        out[f"KL({a}||{b})_nats_sum4"] = sum(kl(pa[n], pb[n]) for n in pa)
        out[f"L1({a},{b})_sum4"] = sum(float(np.abs(pa[n] - pb[n]).sum()) for n in pa)
    return out


def part_c(support: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(0)
    d, r1, r2 = (rng.dirichlet(np.ones(GRID)) for _ in range(3))
    o1, o2 = (prox_old(d, r, 0.05, support, n_iter=100) for r in (r1, r2))
    f1, f2 = (prox_fixed(d, r, 0.05, support, n_iter=100) for r in (r1, r2))
    return {
        "L1_old(r1)-old(r2)": float(np.abs(o1 - o2).sum()),
        "L1_fixed(r1)-fixed(r2)": float(np.abs(f1 - f2).sum()),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    report = {"A": part_a(support), "B": part_b(), "C": part_c(support)}
    json.dump(report, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
