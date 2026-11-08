"""Specialisation sweep on the rigorous path (2026-10-24 erratum follow-up).

Reruns the 72-configuration dense sweep of ``scripts/hyperparam_sweep_dense.py``
(same grid, potentials, beta -> n_inner map, 500 particles, 30 slow steps,
grid 16) with ``FullJKOSolver``: the projected-gradient substeps followed by
the corrected entropic W2 prox (``wasserstein_ops.prox_w2``, Dykstra,
post-ffee5d9). The fast path is rerun on the same host and device as a
control. Seed 0 is the seed of the historical sweep; seeds 1-4 are added to
estimate seed-to-seed spread.

Outputs go to NEW files under ``paper/rigorous_specialisation_2026_10_24/``;
``paper/hyperparam_sweep_dense.json`` is never touched. MLX is pinned to the
CPU device. Entropies are in bits.
"""

from __future__ import annotations

import argparse
import itertools
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

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver
from kiki_flow_core.track2_paper.mlx_particle_simulator import MLXParticleSimulator
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy

GRID = 16
MAX_ENTROPY_BITS = float(np.log2(GRID))
ALPHAS = [0.0, 1.0, 3.0, 5.0, 8.0, 10.0]
BETAS = [0.0, 0.5, 1.0, 2.0]
GAMMAS = [0.0, 1.0, 5.0]
SEEDS = [0, 1, 2, 3, 4]
N_PARTICLES = 500
N_FAST = 200
N_SLOW = 30
H = 0.05
CANONICAL = (10.0, 1.0, 0.0)
OUT_DIR = Path("paper/rigorous_specialisation_2026_10_24")
PATHS = ("fast", "rigorous")


def output_paths() -> dict[str, Path]:
    return {
        "fast": OUT_DIR / "sweep_dense_fast_rerun.json",
        "rigorous": OUT_DIR / "sweep_dense_rigorous.json",
        "summary": OUT_DIR / "summary.json",
        "partial": OUT_DIR / "partial.jsonl",
    }


def configs() -> list[tuple[float, float, float]]:
    return list(itertools.product(ALPHAS, BETAS, GAMMAS))


def provenance() -> dict[str, Any]:
    sha = (
        os.environ.get("KFR_COMMIT")
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    )
    return {
        "git_commit": sha or "unknown",
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mlx_default_device": str(mx.default_device()),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prox": "kiki_flow_core.wasserstein_ops.prox_w2 (Dykstra, post-ffee5d9)",
        "prox_params": {"epsilon": H, "n_iter": 100, "note": "JKOStep passes epsilon=h"},
        "entropy_units": "bits",
    }


def problem(alpha: float, gamma: float) -> tuple[CanonicalSpecies, np.ndarray, T2FreeEnergy]:
    """Same asymmetric potentials as ``scripts/hyperparam_sweep_dense.py``."""
    species = CanonicalSpecies()
    names = species.species_names()
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    x = support[:, 0]
    potentials = {
        "phono": alpha * (x + 1.5),
        "lex": alpha * (x - 1.5),
        "syntax": alpha * np.abs(x),
        "sem": -alpha * np.abs(x),
    }
    prior = {n: np.full(GRID, 1.0 / GRID) for n in names}
    f_func = T2FreeEnergy(
        species=species,
        potentials=potentials,
        prior=prior,
        turing_strength=gamma * 0.1,
        turing_smoothing=None,
    )
    return species, support, f_func


def make_jko(
    f_func: T2FreeEnergy,
    support: np.ndarray,
    beta: float,
    rigorous: bool,
    epsilon: float | None = None,
) -> JKOStep:
    """``epsilon`` is the Sinkhorn regularisation of the rigorous prox
    (``FullJKOSolver.epsilon``, forwarded as ``reg``); ``None`` keeps the
    solver default (0.01). The JKO step ``H`` stays the proximal pull. The
    fast path has no prox and ignores ``epsilon``.
    """
    n_inner = max(1, int(10 * beta)) if beta > 0 else 1
    if rigorous:
        eps_kw: dict[str, float] = {} if epsilon is None else {"epsilon": epsilon}
        return FullJKOSolver(
            scheme="legacy_split",
            f_functional=f_func,
            h=H,
            support=support,
            n_inner=n_inner,
            **eps_kw,
        )
    return JKOStep(f_functional=f_func, h=H, support=support, n_inner=n_inner, apply_w2_prox=False)


def run_one(
    alpha: float, beta: float, gamma: float, rigorous: bool, seed: int, n_slow: int = N_SLOW
) -> dict[str, float]:
    species, support, f_func = problem(alpha, gamma)
    jko = make_jko(f_func, support, beta, rigorous)
    sim = MLXParticleSimulator(species=species, n_particles=N_PARTICLES, latent_dim=2, seed=seed)
    loop = MultiscaleLoop(
        reinject=False, sim=sim, jko=jko, n_fast=N_FAST, n_slow=n_slow, support=support
    )
    final = loop.run(seed=seed)["trajectory"][-1]
    out = {}
    for n, rho in final.rho.items():
        rho_s = np.clip(np.asarray(rho, dtype=np.float64), 1e-12, None)
        out[n] = float(-(rho_s * np.log2(rho_s)).sum())
    return out


def _key(path: str, seed: int, cfg: tuple[float, float, float]) -> str:
    return f"{path}|{seed}|{cfg[0]}|{cfg[1]}|{cfg[2]}"


def run_all(seeds: list[int]) -> None:
    paths = output_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    if paths["partial"].exists():
        for line in paths["partial"].read_text().splitlines():
            rec = json.loads(line)
            done[_key(rec["path"], rec["seed"], (rec["alpha"], rec["beta"], rec["gamma"]))] = rec
    with paths["partial"].open("a") as fh:
        for seed in seeds:
            for path in PATHS:
                for cfg in configs():
                    if _key(path, seed, cfg) in done:
                        continue
                    t0 = time.perf_counter()
                    ent = run_one(*cfg, rigorous=path == "rigorous", seed=seed)
                    h_mean = float(np.mean(list(ent.values())))
                    rec = {
                        "path": path,
                        "seed": seed,
                        "alpha": cfg[0],
                        "beta": cfg[1],
                        "gamma": cfg[2],
                        "mean_entropy_bits": h_mean,
                        "gap_bits": MAX_ENTROPY_BITS - h_mean,
                        "per_species": ent,
                        "runtime_s": time.perf_counter() - t0,
                    }
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    done[_key(path, seed, cfg)] = rec
                    print(f"{path} seed={seed} cfg={cfg} H={h_mean:.4f}", flush=True)
    write_outputs(list(done.values()), seeds)


def _stats(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def write_outputs(records: list[dict[str, Any]], seeds: list[int]) -> None:
    paths = output_paths()
    prov = provenance()
    params = {"grid": GRID, "n_particles": N_PARTICLES, "n_fast": N_FAST, "n_slow": N_SLOW}
    grid = {"alpha": ALPHAS, "beta": BETAS, "gamma": GAMMAS}
    table: dict[str, dict[tuple[float, float, float], list[float]]] = {p: {} for p in PATHS}
    for path in PATHS:
        recs = sorted(
            (r for r in records if r["path"] == path),
            key=lambda r: (r["seed"], r["alpha"], r["beta"], r["gamma"]),
        )
        for r in recs:
            table[path].setdefault((r["alpha"], r["beta"], r["gamma"]), []).append(
                r["mean_entropy_bits"]
            )
        seed0 = [r for r in recs if r["seed"] == 0]
        best0 = min(seed0, key=lambda r: r["mean_entropy_bits"]) if seed0 else None
        paths[path].write_text(
            json.dumps(
                {
                    "path": path,
                    "max_entropy_bits": MAX_ENTROPY_BITS,
                    "n_configs": len(configs()),
                    "seeds": seeds,
                    "grid": grid,
                    "params": params,
                    "best_seed0": best0,
                    "results": recs,
                    "runtime_s": float(sum(r["runtime_s"] for r in recs)),
                    "provenance": prov,
                },
                indent=2,
            )
        )
    rows = []
    for cfg in configs():
        fast_v, rig_v = table["fast"].get(cfg, []), table["rigorous"].get(cfg, [])
        if not fast_v or not rig_v:
            continue
        diff = [r - f for r, f in zip(rig_v, fast_v, strict=True)]
        rows.append(
            {
                "alpha": cfg[0],
                "beta": cfg[1],
                "gamma": cfg[2],
                "fast_H_seed0": fast_v[0],
                "rigorous_H_seed0": rig_v[0],
                "fast_H": _stats(fast_v),
                "rigorous_H": _stats(rig_v),
                "rigorous_minus_fast_H": _stats(diff),
            }
        )

    def best(path: str) -> dict[str, Any]:
        key = f"{path}_H"
        r = min(rows, key=lambda row: row[key]["mean"])
        return {
            "alpha": r["alpha"],
            "beta": r["beta"],
            "gamma": r["gamma"],
            "H_mean": r[key]["mean"],
            "H_sd": r[key]["sd"],
            "gap_mean": MAX_ENTROPY_BITS - r[key]["mean"],
        }

    canon = next(r for r in rows if (r["alpha"], r["beta"], r["gamma"]) == CANONICAL)
    summary = {
        "metric": "mean over 4 species of final entropy (bits); gap = log2(16) - H",
        "max_entropy_bits": MAX_ENTROPY_BITS,
        "seeds": seeds,
        "best_by_seed_mean": {p: best(p) for p in PATHS},
        "canonical_point": canon,
        "n_configs_rigorous_below_fast": sum(r["rigorous_minus_fast_H"]["mean"] < 0 for r in rows),
        "rows": rows,
        "provenance": prov,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2))


def main() -> None:
    # Set here, not at import, so importing this module (tests) leaves the
    # process-wide MLX device untouched.
    mx.set_default_device(mx.cpu)
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    run_all(args.seeds)


if __name__ == "__main__":
    main()
