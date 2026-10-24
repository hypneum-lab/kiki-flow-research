"""Reruns after forwarding ``FullJKOSolver.max_iter`` to the prox.

Until the fix, ``JKOStep.step`` called the prox with ``n_iter=100`` and
``FullJKOSolver`` stored ``max_iter`` without passing it on. Every rigorous
run cited by the paper builds ``FullJKOSolver`` with the default
``max_iter = 1000``, so all of them ran 100 Dykstra iterations where 1000
were configured. This driver reruns them with the fixed solver:

  fvr       -> paper/erratum_2026_10_25_maxiter/fast_vs_rigorous_kl_maxiter.json
               (reduced setup, epsilon = 0.05, fast path rerun alongside)
  rigorous  -> paper_rigorous_maxiter/ (paper scale, epsilon = 0.01)
  fig1      -> paper/erratum_2026_10_25_maxiter/fig1_species_moments_summary_maxiter.json,
               fig1 *_maxiter (fast trajectories reused: the fast path has no prox)
  spec      -> paper/rigorous_specialisation_wide_2026_10_25_maxiter/ (rigorous
               path of the 138-point, 10-seed grid; fast records copied)
  spec-compare -> comparison.json in the same directory: the rerun records
               present so far against the epsilon rerun, same seeds
  e1        -> no file: checks that E1 (own single-projection prox) reproduces
               ``experiments/results/e1_epsilon_sweep_var.json`` exactly

New files only. MLX is pinned to the CPU device in ``main``. KL in nats,
entropies in bits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import mlx.core as mx
import numpy as np

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver


def _load(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    path = Path("experiments") / filename
    if not path.exists():  # imported from the tests, cwd may differ
        path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reg = _load("experiments_erratum_reg_reruns", "erratum_reg_reruns_2026_10_24.py")
wide = reg.wide

SEEDS = [0, 1, 2, 3, 4]
ERR_DIR = Path("paper/erratum_2026_10_25_maxiter")
RIG_DIR = Path("paper_rigorous_maxiter")
SPEC_DIR = Path("paper/rigorous_specialisation_wide_2026_10_25_maxiter")
OLD_FAST_TRAJ = reg.OLD_FAST_TRAJ
MAX_ITER = 1000


def output_paths() -> dict[str, Path]:
    return {
        "fvr": ERR_DIR / "fast_vs_rigorous_kl_maxiter.json",
        "fig1_summary": ERR_DIR / "fig1_species_moments_summary_maxiter.json",
        "rigorous": RIG_DIR,
        "spec_records": SPEC_DIR / "sweep_wide_records.json",
        "spec_summary": SPEC_DIR / "summary.json",
        "spec_partial": SPEC_DIR / "partial.jsonl",
    }


def provenance() -> dict[str, Any]:
    prov = reg.provenance()
    prov["prox_params"] = {
        "epsilon": "h = 0.05 (JKO step, proximal pull)",
        "reg": "FullJKOSolver.epsilon (Sinkhorn entropic regularisation)",
        "n_iter": f"FullJKOSolver.max_iter = {MAX_ITER} (forwarded)",
    }
    return prov


# ------------------------------------------------------------------ fvr
def run_fvr() -> None:
    """Same setup and seeds as ``erratum_reruns_2026_10_24.py fvr``."""
    err = reg._erratum()
    grid, n_particles, n_fast, n_slow = 16, 500, 200, 20
    per_seed = []
    t0 = time.perf_counter()
    for seed in SEEDS:
        _, support, f_func = err.setup(grid)
        fast_jko = JKOStep(f_functional=f_func, h=0.05, support=support, n_inner=10)
        rig_jko = FullJKOSolver(f_functional=f_func, h=0.05, support=support, epsilon=0.05)
        assert rig_jko.prox_n_iter == MAX_ITER
        fast = err._final(fast_jko, seed, grid, n_particles, n_fast, n_slow)[-1].rho
        rig = err._final(rig_jko, seed, grid, n_particles, n_fast, n_slow)[-1].rho
        kl_sp = {n: err.kl(np.asarray(rig[n]), np.asarray(fast[n])) for n in fast}
        per_seed.append({"seed": seed, "kl_per_species": kl_sp, "total_kl": sum(kl_sp.values())})
        print(f"seed={seed} total KL(rigorous||fast) = {per_seed[-1]['total_kl']:.6f}", flush=True)
    vals = [s["total_kl"] for s in per_seed]
    ERR_DIR.mkdir(parents=True, exist_ok=True)
    output_paths()["fvr"].write_text(
        json.dumps(
            {
                "n_seeds": len(SEEDS),
                "params": {
                    "grid": grid,
                    "n_particles": n_particles,
                    "n_fast": n_fast,
                    "n_slow": n_slow,
                    "epsilon": 0.05,
                },
                "per_seed": per_seed,
                "mean_total_kl": float(np.mean(vals)),
                "std_total_kl": float(np.std(vals)),
                "max_total_kl": float(np.max(vals)),
                "runtime_s": time.perf_counter() - t0,
                "provenance": provenance(),
            },
            indent=2,
        )
    )


# ------------------------------------------------------- rigorous + fig1
def run_rigorous(n_slow: int) -> None:
    reg._erratum().run_full("rigorous", SEEDS, n_slow, RIG_DIR)


def run_fig1(n_slow: int) -> None:
    """Same summary as ``erratum_reg_reruns_2026_10_24.py fig1``, new paths."""
    from kiki_flow_core.species import CanonicalSpecies  # noqa: PLC0415
    from kiki_flow_core.track2_paper.figures.species_moments import (  # noqa: PLC0415
        make_species_moments,
        species_moments,
    )

    err = reg._erratum()
    x = np.linspace(-2, 2, 64)
    runs = {
        "fast": err._load(OLD_FAST_TRAJ, SEEDS, n_slow),
        "rigorous": err._load(RIG_DIR / "trajectories", SEEDS, n_slow),
    }
    make_species_moments(
        runs, x, out_dir=Path("paper/figures"), filename="fig1_species_moments_maxiter"
    )
    names = CanonicalSpecies().species_names()
    summary: dict[str, Any] = {"seeds": SEEDS, "n_slow": n_slow, "entropy_units": "bits"}
    for label, arr in runs.items():
        mean, ent = species_moments(arr, x)
        summary[label] = {
            n: {
                "mean_x_final": float(mean[:, -1, i].mean()),
                "mean_x_final_sd": float(mean[:, -1, i].std()),
                "entropy_bits_first": float(ent[:, 0, i].mean()),
                "entropy_bits_final": float(ent[:, -1, i].mean()),
                "entropy_bits_final_sd": float(ent[:, -1, i].std()),
            }
            for i, n in enumerate(names)
        }
    fin = [
        sum(reg._kl(runs["rigorous"][s, -1, i], runs["fast"][s, -1, i]) for i in range(len(names)))
        for s in range(len(SEEDS))
    ]
    summary["final_kl_rigorous_vs_fast_nats_sum4"] = {
        "per_seed": fin,
        "mean": float(np.mean(fin)),
        "std": float(np.std(fin)),
    }
    summary["fast_trajectories"] = str(OLD_FAST_TRAJ)
    summary["provenance"] = provenance()
    ERR_DIR.mkdir(parents=True, exist_ok=True)
    output_paths()["fig1_summary"].write_text(json.dumps(summary, indent=2))


# ------------------------------------------------------------------ spec
def run_spec() -> None:
    paths = output_paths()
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    old = json.loads(wide.output_paths()["records"].read_text())["results"]
    done: dict[str, dict[str, Any]] = {}
    for r in reg.fast_records(old):
        done[wide.key("fast", r["seed"], (r["alpha"], r["beta"], r["gamma"]))] = {
            **r,
            "source": f"{wide.OUT_DIR} (fast path has no prox)",
        }
    if paths["spec_partial"].exists():
        for line in paths["spec_partial"].read_text().splitlines():
            r = json.loads(line)
            done[wide.key(r["path"], r["seed"], (r["alpha"], r["beta"], r["gamma"]))] = r
    jobs = [j for j in wide.todo(set(done)) if j[0] == "rigorous"]
    print(f"{len(jobs)} rigorous runs", flush=True)
    with paths["spec_partial"].open("a") as fh:
        for _, seed, p in jobs:
            t0 = time.perf_counter()
            ent = wide.base.run_one(*p, rigorous=True, seed=seed)
            h_mean = float(np.mean(list(ent.values())))
            rec = {
                "path": "rigorous",
                "seed": seed,
                "alpha": p[0],
                "beta": p[1],
                "gamma": p[2],
                "mean_entropy_bits": h_mean,
                "gap_bits": wide.base.MAX_ENTROPY_BITS - h_mean,
                "per_species": ent,
                "runtime_s": time.perf_counter() - t0,
                "source": f"this driver (reg = 0.01, n_iter = {MAX_ITER})",
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[wide.key("rigorous", seed, p)] = rec
            print(f"rigorous seed={seed} p={p} H={h_mean:.4f}", flush=True)
    records = sorted(
        done.values(), key=lambda r: (r["path"], r["seed"], r["alpha"], r["beta"], r["gamma"])
    )
    prov = provenance()
    paths["spec_records"].write_text(
        json.dumps({"n_records": len(records), "results": records, "provenance": prov}, indent=1)
    )
    paths["spec_summary"].write_text(
        json.dumps(wide.summarise(records) | {"provenance": prov}, indent=1)
    )


def compare_spec() -> None:
    """Compare the rerun records present in ``partial.jsonl`` with the epsilon rerun."""
    old = json.loads((reg.SPEC_DIR / "sweep_wide_records.json").read_text())["results"]
    old_by = {wide.key(r["path"], r["seed"], (r["alpha"], r["beta"], r["gamma"])): r for r in old}
    new = [json.loads(line) for line in output_paths()["spec_partial"].read_text().splitlines()]
    d_gap, d_h = [], []
    by_point: dict[tuple[float, float, float], list[float]] = {}
    for r in new:
        p = (r["alpha"], r["beta"], r["gamma"])
        o = old_by[wide.key("rigorous", r["seed"], p)]
        d_gap.append(r["gap_bits"] - o["gap_bits"])
        d_h.extend(r["per_species"][n] - o["per_species"][n] for n in r["per_species"])
        by_point.setdefault(p, []).append(d_gap[-1])
    point_means = [float(np.mean(v)) for v in by_point.values()]
    out = {
        "n_runs": len(new),
        "seeds": sorted({r["seed"] for r in new}),
        "n_points": len(by_point),
        "baseline": str(reg.SPEC_DIR / "sweep_wide_records.json"),
        "max_abs_delta_gap_bits_per_run": float(np.max(np.abs(d_gap))),
        "mean_delta_gap_bits_per_run": float(np.mean(d_gap)),
        "max_abs_delta_point_mean_gap_bits": float(np.max(np.abs(point_means))),
        "max_abs_delta_species_entropy_bits": float(np.max(np.abs(d_h))),
        "note": "new (max_iter = 1000 forwarded) minus old (n_iter = 100), same seeds",
        "provenance": provenance(),
    }
    (SPEC_DIR / "comparison.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "provenance"}, indent=2))


# ------------------------------------------------------------------- e1
def run_e1() -> None:
    e1 = _load("experiments_e1_epsilon_sweep_var", "e1_epsilon_sweep_var.py")
    committed = json.loads(Path("experiments/results/e1_epsilon_sweep_var.json").read_text())
    support = np.linspace(-e1.X_MAX, e1.X_MAX, e1.GRID).reshape(-1, 1)
    x = support[:, 0]
    worst = 0.0
    for row in committed["rows"]:
        var_k = e1.run_one(row["epsilon"], support, x)["var_K"]
        worst = max(worst, abs(var_k - row["var_K"]))
        print(f"eps={row['epsilon']:.1e} var_K new={var_k!r} committed={row['var_K']!r}")
    print(f"max |delta var_K| = {worst!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fvr", "rigorous", "fig1", "spec", "spec-compare", "e1"])
    ap.add_argument("--n-slow", type=int, default=100)
    args = ap.parse_args()
    mx.set_default_device(mx.cpu)
    warnings.filterwarnings("ignore")
    {
        "fvr": run_fvr,
        "rigorous": lambda: run_rigorous(args.n_slow),
        "fig1": lambda: run_fig1(args.n_slow),
        "spec": run_spec,
        "spec-compare": compare_spec,
        "e1": run_e1,
    }[args.what]()


if __name__ == "__main__":
    main()
