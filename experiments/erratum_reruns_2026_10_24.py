"""Erratum reruns (2026-10-24) with the corrected ``prox_w2`` (ffee5d9).

Reruns the three artefacts that depended on the old, reference-blind prox,
writing to NEW files so the historical outputs stay untouched:

  eps       -> paper/erratum_2026_10_24/epsilon_sweep_fixed.json (+ fig4 *_fixed)
  fvr       -> paper/erratum_2026_10_24/fast_vs_rigorous_kl_fixed.json
  rigorous  -> paper_rigorous_fixedprox/  (stats.json + trajectories/)
  fast      -> paper_rigorous_fixedprox/fast/  (fast path, same host/device)

MLX is pinned to the CPU device (``mx.set_default_device(mx.cpu)``) so the
runs never touch Metal. KL values are in nats (natural log).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)

from kiki_flow_core.master_equation import JKOStep  # noqa: E402
from kiki_flow_core.species import CanonicalSpecies  # noqa: E402
from kiki_flow_core.track2_paper.figures.kl_vs_epsilon import make_kl_vs_epsilon  # noqa: E402
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver  # noqa: E402
from kiki_flow_core.track2_paper.mlx_particle_simulator import MLXParticleSimulator  # noqa: E402
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop  # noqa: E402
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy  # noqa: E402
from kiki_flow_core.track2_paper.paper_run import (  # noqa: E402
    _aggregate_rho_stats,
    _persist_trajectory,
)
from kiki_flow_core.wasserstein_ops import prox_w2  # noqa: E402

ERR_DIR = Path("paper/erratum_2026_10_24")
RIG_DIR = Path("paper_rigorous_fixedprox")
SEEDS = [0, 1, 2, 3, 4]


def provenance() -> dict[str, Any]:
    sha = (
        os.environ.get("KFR_COMMIT")
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    )
    return {
        "git_commit": sha or "unknown (copy without .git; see PR)",
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mlx_default_device": str(mx.default_device()),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prox": "kiki_flow_core.wasserstein_ops.prox_w2 (Dykstra, post-ffee5d9)",
        "kl_units": "nats",
    }


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p_s = np.clip(p, 1e-12, None)
    q_s = np.clip(q, 1e-12, None)
    return float((p_s * np.log(p_s / q_s)).sum())


def setup(grid: int) -> tuple[CanonicalSpecies, np.ndarray, T2FreeEnergy]:
    species = CanonicalSpecies()
    names = species.species_names()
    support = np.linspace(-2, 2, grid).reshape(-1, 1)
    f_func = T2FreeEnergy(
        species=species,
        potentials={n: np.zeros(grid) for n in names},
        prior={n: np.full(grid, 1.0 / grid) for n in names},
        turing_strength=0.1,
    )
    return species, support, f_func


# ---------------------------------------------------------------- eps sweep
def run_eps() -> None:
    """Same setup as scripts/epsilon_sweep.py, corrected prox."""
    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]
    support = np.linspace(-2, 2, 16).reshape(-1, 1)
    x = support[:, 0]
    dist = np.exp(-0.5 * ((x + 1.0) / 0.5) ** 2)
    dist /= dist.sum()
    ref = np.exp(-0.5 * ((x - 1.0) / 0.5) ** 2)
    ref /= ref.sum()
    t0 = time.perf_counter()
    outs = [prox_w2(dist, reference=ref, epsilon=e, support=support, n_iter=50) for e in epsilons]
    kls = [kl(o, outs[0]) for o in outs]
    ERR_DIR.mkdir(parents=True, exist_ok=True)
    make_kl_vs_epsilon(
        epsilons,
        [max(v, 1e-12) for v in kls],
        out_dir=Path("paper/figures"),
        filename="fig4_kl_vs_epsilon_fixed",
    )
    payload = {
        "epsilons": epsilons,
        "kl_values_nats": kls,
        "mean_x_out": [float(o @ x) for o in outs],
        "mean_x_distribution": float(dist @ x),
        "mean_x_reference": float(ref @ x),
        "params": {"grid": 16, "n_iter": 50, "setup": "peaked-left vs peaked-right 1D prox"},
        "runtime_s": time.perf_counter() - t0,
        "provenance": provenance(),
    }
    (ERR_DIR / "epsilon_sweep_fixed.json").write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------- fast vs rigorous
def _final(jko: JKOStep, seed: int, grid: int, n_particles: int, n_fast: int, n_slow: int):
    species, support, _ = setup(grid)
    sim = MLXParticleSimulator(species=species, n_particles=n_particles, latent_dim=2, seed=seed)
    loop = MultiscaleLoop(sim=sim, jko=jko, n_fast=n_fast, n_slow=n_slow, support=support)
    return loop.run(seed=seed)["trajectory"]


def run_fvr() -> None:
    """Same setup as scripts/fast_vs_rigorous_kl.py, corrected prox."""
    grid, n_particles, n_fast, n_slow = 16, 500, 200, 20
    per_seed = []
    t0 = time.perf_counter()
    for seed in SEEDS:
        _, support, f_func = setup(grid)
        fast_jko = JKOStep(f_functional=f_func, h=0.05, support=support, n_inner=10)
        rig_jko = FullJKOSolver(f_functional=f_func, h=0.05, support=support, epsilon=0.05)
        fast = _final(fast_jko, seed, grid, n_particles, n_fast, n_slow)[-1].rho
        rig = _final(rig_jko, seed, grid, n_particles, n_fast, n_slow)[-1].rho
        kl_sp = {n: kl(np.asarray(rig[n]), np.asarray(fast[n])) for n in fast}
        per_seed.append({"seed": seed, "kl_per_species": kl_sp, "total_kl": sum(kl_sp.values())})
        print(f"seed={seed} total KL(rigorous||fast) = {per_seed[-1]['total_kl']:.6f} nats")
    vals = [s["total_kl"] for s in per_seed]
    payload = {
        "n_seeds": len(SEEDS),
        "params": {"grid": grid, "n_particles": n_particles, "n_fast": n_fast, "n_slow": n_slow},
        "per_seed": per_seed,
        "mean_total_kl": float(np.mean(vals)),
        "std_total_kl": float(np.std(vals)),
        "max_total_kl": float(np.max(vals)),
        "runtime_s": time.perf_counter() - t0,
        "provenance": provenance(),
    }
    ERR_DIR.mkdir(parents=True, exist_ok=True)
    (ERR_DIR / "fast_vs_rigorous_kl_fixed.json").write_text(json.dumps(payload, indent=2))


# ------------------------------------------------------- 5-seed full runs
def run_full(kind: str, seeds: list[int], n_slow: int, out_dir: Path) -> None:
    """Rigorous (FullJKOSolver, epsilon=0.01) or fast (JKOStep n_inner=10) at paper scale."""
    grid, n_particles, n_fast = 64, 10_000, 1000
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = []
    trajectories = []
    t_all = time.perf_counter()
    for seed in seeds:
        _, support, f_func = setup(grid)
        jko: JKOStep
        if kind == "rigorous":
            jko = FullJKOSolver(f_functional=f_func, h=0.05, support=support, epsilon=0.01)
        else:
            jko = JKOStep(f_functional=f_func, h=0.05, support=support, n_inner=10)
        t0 = time.perf_counter()
        traj = _final(jko, seed, grid, n_particles, n_fast, n_slow)
        dt = time.perf_counter() - t0
        _persist_trajectory(traj, out_dir / "trajectories" / f"seed{seed}.safetensors")
        trajectories.append(traj)
        per_seed.append({"seed": seed, "n_slow_completed": len(traj), "runtime_s": dt})
        print(f"{kind} seed={seed} n_slow={len(traj)} {dt:.1f}s", flush=True)
    stats = {
        "n_seeds": len(seeds),
        "per_seed": per_seed,
        "params": {"grid": grid, "n_particles": n_particles, "n_fast": n_fast, "n_slow": n_slow},
        "backend": {
            "simulator": "mlx (cpu device)",
            "w2_prox": kind == "rigorous",
            "prox": "numpy Dykstra prox_w2 (post-ffee5d9)" if kind == "rigorous" else None,
            "epsilon": 0.01 if kind == "rigorous" else None,
        },
        "aggregate": _aggregate_rho_stats(trajectories),
        "runtime_s": time.perf_counter() - t_all,
        "provenance": provenance(),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["eps", "fvr", "rigorous", "fast"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n-slow", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.what == "eps":
        run_eps()
    elif args.what == "fvr":
        run_fvr()
    else:
        default = RIG_DIR if args.what == "rigorous" else RIG_DIR / "fast"
        run_full(args.what, args.seeds, args.n_slow, args.out or default)


if __name__ == "__main__":
    main()
