"""Convergence of the rigorous-path Dykstra prox in its iteration count.

Until 2026-10-25 ``JKOStep.step`` called the prox with ``n_iter=100`` and
``FullJKOSolver``/``MLXFullJKOSolver`` stored ``max_iter`` (default 1000)
without forwarding it. This driver records the prox inputs of real
rigorous-path runs and replays each call at several iteration counts,
reporting how far the 100-iteration output is from convergence:

* ``marginal_l1``: the Sinkhorn/Dykstra stopping criterion, i.e. the L1
  violation of the second-marginal constraint ``P^T 1 = reference`` after
  one more first-marginal projection (zero at the fixed point);
* ``dq_l1_vs_ref``: L1 distance between the n-iteration output and the
  output at ``N_REF`` iterations;
* ``kl_vs_ref_nats``: KL(q_ref || q_n).

Inputs are recorded from (a) the paper-scale rigorous run (grid 64,
10 000 particles, epsilon = 0.01, seed 0) and (b) the rigorous half of the
specialisation sweep (grid 16, 500 particles, epsilon = 0.01, seed 0) at
the points the paper quotes. MLX is pinned to the CPU device. Writes
``paper/erratum_2026_10_25_maxiter/convergence.json`` (a new file).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import mlx.core as mx
import numpy as np
import ot

from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver
from kiki_flow_core.track2_paper.mlx_particle_simulator import MLXParticleSimulator
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop
from kiki_flow_core.wasserstein_ops import prox_w2

ITERS = [10, 25, 50, 100, 200, 500, 1000, 2000]
N_REF = 5000
OUT = Path("paper/erratum_2026_10_25_maxiter/convergence.json")
SPEC_POINTS = [(15.0, 0.2, 0.0), (10.0, 1.0, 0.0), (30.0, 1.0, 0.0), (1.0, 0.3, 0.0)]


def _load(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dykstra(
    dist: np.ndarray, ref: np.ndarray, h: float, reg: float, support: np.ndarray, n_iter: int
) -> tuple[np.ndarray, float]:
    """Same loop as ``wasserstein_ops.prox_w2``; also returns the marginal residual."""
    cost = ot.dist(support, support, metric="sqeuclidean")
    sink_reg = max(reg, 1e-3)
    lam = sink_reg / (sink_reg + 2.0 * max(h, 0.0))
    gibbs = np.exp(-cost / sink_reg)
    d = np.clip(dist, 1e-300, None)
    r = np.clip(ref, 1e-300, None)
    a, b = np.ones_like(d), np.ones_like(r)
    for _ in range(n_iter):
        a = d ** (1.0 - lam) * np.maximum(gibbs @ b, 1e-300) ** (lam - 1.0)
        b = r / np.maximum(gibbs @ a, 1e-300)
    q = a * (gibbs @ b)
    a_next = d ** (1.0 - lam) * np.maximum(gibbs @ b, 1e-300) ** (lam - 1.0)
    residual = float(np.abs(b * (gibbs @ a_next) - r).sum())
    return q / q.sum(), residual


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p_s, q_s = np.clip(p, 1e-300, None), np.clip(q, 1e-300, None)
    return float((p_s * np.log(p_s / q_s)).sum())


def _recorder(solver: FullJKOSolver, keep: list[dict[str, Any]], every: int) -> None:
    inner = solver.prox_fn
    count = {"n": 0}

    def spy(distribution: np.ndarray, **kwargs: Any) -> np.ndarray:
        if count["n"] % every == 0:
            keep.append(
                {
                    "dist": np.array(distribution, dtype=np.float64),
                    "ref": np.array(kwargs["reference"], dtype=np.float64),
                    "h": float(kwargs["epsilon"]),
                    "reg": float(kwargs.get("reg", kwargs["epsilon"])),
                    "support": np.asarray(kwargs["support"], dtype=np.float64),
                    "call": count["n"],
                }
            )
        count["n"] += 1
        return inner(distribution, **kwargs)

    solver.prox_fn = spy  # type: ignore[assignment]


def record_paper(n_slow: int, every: int) -> list[dict[str, Any]]:
    err = _load("experiments_erratum_reruns", "erratum_reruns_2026_10_24.py")
    species, support, f_func = err.setup(64)
    jko = FullJKOSolver(f_functional=f_func, h=0.05, support=support, epsilon=0.01)
    calls: list[dict[str, Any]] = []
    _recorder(jko, calls, every)
    sim = MLXParticleSimulator(species=species, n_particles=10_000, latent_dim=2, seed=0)
    MultiscaleLoop(sim=sim, jko=jko, n_fast=1000, n_slow=n_slow, support=support).run(seed=0)
    return calls


def record_spec(every: int) -> dict[str, list[dict[str, Any]]]:
    spec = _load("experiments_rigorous_specialisation", "rigorous_specialisation_2026_10_24.py")
    out: dict[str, list[dict[str, Any]]] = {}
    for alpha, beta, gamma in SPEC_POINTS:
        species, support, f_func = spec.problem(alpha, gamma)
        jko = spec.make_jko(f_func, support, beta, rigorous=True)
        calls: list[dict[str, Any]] = []
        _recorder(jko, calls, every)
        sim = MLXParticleSimulator(species=species, n_particles=500, latent_dim=2, seed=0)
        loop = MultiscaleLoop(sim=sim, jko=jko, n_fast=200, n_slow=30, support=support)
        loop.run(seed=0)
        out[f"spec({alpha:g},{beta:g},{gamma:g})"] = calls
    return out


def analyse(calls: list[dict[str, Any]]) -> dict[str, Any]:
    per_iter: dict[int, dict[str, list[float]]] = {
        n: {"marginal_l1": [], "dq_l1_vs_ref": [], "kl_vs_ref_nats": []} for n in ITERS
    }
    for c in calls:
        args = (c["dist"], c["ref"], c["h"], c["reg"], c["support"])
        q_ref, _ = dykstra(*args, N_REF)
        q100_repo = prox_w2(
            c["dist"],
            reference=c["ref"],
            epsilon=c["h"],
            support=c["support"],
            n_iter=100,
            reg=c["reg"],
        )
        q100, _ = dykstra(*args, 100)
        assert np.allclose(q100, q100_repo, atol=1e-14), "probe loop != prox_w2"
        for n in ITERS:
            q, res = dykstra(*args, n)
            per_iter[n]["marginal_l1"].append(res)
            per_iter[n]["dq_l1_vs_ref"].append(float(np.abs(q - q_ref).sum()))
            per_iter[n]["kl_vs_ref_nats"].append(_kl(q_ref, q))
    return {
        "n_calls": len(calls),
        "grid": int(calls[0]["support"].shape[0]),
        "h": calls[0]["h"],
        "reg": calls[0]["reg"],
        "per_iter": {
            str(n): {
                k: {"max": float(np.max(v)), "median": float(np.median(v))} for k, v in d.items()
            }
            for n, d in per_iter.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-slow", type=int, default=100)
    parser.add_argument("--every", type=int, default=20)
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)
    t0 = time.perf_counter()
    groups = {"paper_scale_seed0": record_paper(args.n_slow, args.every)}
    groups.update(record_spec(every=6))
    result = {name: analyse(calls) for name, calls in groups.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "iters": ITERS,
                "n_ref": N_REF,
                "groups": result,
                "params": {"paper_n_slow": args.n_slow, "record_every": args.every},
                "runtime_s": time.perf_counter() - t0,
            },
            indent=2,
        )
    )
    for name, r in result.items():
        row = r["per_iter"]
        print(name, r["n_calls"], "calls")
        for n in ITERS:
            print(
                f"  n={n:5d} res={row[str(n)]['marginal_l1']['max']:.3e} "
                f"dq={row[str(n)]['dq_l1_vs_ref']['max']:.3e}"
            )


if __name__ == "__main__":
    main()
