"""Reruns after forwarding ``FullJKOSolver.epsilon`` as the Sinkhorn reg.

Until the fix, ``JKOStep.step`` called the prox with ``epsilon = h`` and the
prox's default ``reg = epsilon``, so every ``FullJKOSolver`` ran with
``reg = h = 0.05`` whatever its configured ``epsilon``. Runs configured with
``epsilon = 0.05`` are unchanged by the fix. This driver reruns the outputs
whose configured epsilon was 0.01, plus the epsilon sweep with epsilon as
the Sinkhorn reg only (prox pull fixed at the JKO step h):

  eps       -> paper/erratum_2026_10_24_reg/epsilon_sweep_reg.json, fig4 *_reg
  rigorous  -> paper_rigorous_fixedeps/ (paper scale, epsilon = 0.01)
  fig1      -> paper/erratum_2026_10_24_reg/fig1_species_moments_summary_reg.json,
               fig1 *_reg (fast trajectories reused: the fast path has no prox)
  spec      -> paper/rigorous_specialisation_wide_2026_10_24_reg/ (rigorous path
               rerun on the 138-point, 10-seed grid; fast records copied)

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

from kiki_flow_core.wasserstein_ops import prox_w2


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


wide = _load(
    "experiments_rigorous_specialisation_wide", "rigorous_specialisation_wide_2026_10_24.py"
)

H = 0.05
EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1]
SEEDS = [0, 1, 2, 3, 4]
ERR_DIR = Path("paper/erratum_2026_10_24_reg")
RIG_DIR = Path("paper_rigorous_fixedeps")
OLD_FAST_TRAJ = Path("paper_rigorous_fixedprox/fast/trajectories")
SPEC_DIR = Path("paper/rigorous_specialisation_wide_2026_10_24_reg")


def output_paths() -> dict[str, Path]:
    return {
        "eps": ERR_DIR / "epsilon_sweep_reg.json",
        "fig1_summary": ERR_DIR / "fig1_species_moments_summary_reg.json",
        "rigorous": RIG_DIR,
        "spec_records": SPEC_DIR / "sweep_wide_records.json",
        "spec_summary": SPEC_DIR / "summary.json",
        "spec_partial": SPEC_DIR / "partial.jsonl",
    }


def provenance() -> dict[str, Any]:
    prov = wide.base.provenance()
    prov["prox_params"] = {
        "epsilon": "h = 0.05 (JKO step, proximal pull)",
        "reg": "FullJKOSolver.epsilon (Sinkhorn entropic regularisation)",
        "n_iter": 100,
    }
    return prov


# ------------------------------------------------------------------ eps
def _eps_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = np.linspace(-2, 2, 16).reshape(-1, 1)
    x = support[:, 0]
    dist = np.exp(-0.5 * ((x + 1.0) / 0.5) ** 2)
    ref = np.exp(-0.5 * ((x - 1.0) / 0.5) ** 2)
    return support, dist / dist.sum(), ref / ref.sum()


def eps_outputs(epsilons: list[float]) -> list[np.ndarray]:
    """One prox step per Sinkhorn reg; the proximal pull stays at h."""
    support, dist, ref = _eps_problem()
    return [
        prox_w2(dist, reference=ref, epsilon=H, support=support, n_iter=50, reg=e) for e in epsilons
    ]


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p_s, q_s = np.clip(p, 1e-12, None), np.clip(q, 1e-12, None)
    return float((p_s * np.log(p_s / q_s)).sum())


def run_eps() -> None:
    from kiki_flow_core.track2_paper.figures.kl_vs_epsilon import (  # noqa: PLC0415
        make_kl_vs_epsilon,
    )

    support, dist, ref = _eps_problem()
    x = support[:, 0]
    t0 = time.perf_counter()
    outs = eps_outputs(EPSILONS)
    kls = [_kl(o, outs[0]) for o in outs]
    make_kl_vs_epsilon(
        EPSILONS,
        [max(v, 1e-12) for v in kls],
        out_dir=Path("paper/figures"),
        filename="fig4_kl_vs_epsilon_reg",
    )
    ERR_DIR.mkdir(parents=True, exist_ok=True)
    output_paths()["eps"].write_text(
        json.dumps(
            {
                "epsilons": EPSILONS,
                "kl_values_nats": kls,
                "mean_x_out": [float(o @ x) for o in outs],
                "mean_x_distribution": float(dist @ x),
                "mean_x_reference": float(ref @ x),
                "params": {
                    "grid": 16,
                    "n_iter": 50,
                    "prox_epsilon": H,
                    "swept": "reg (Sinkhorn entropic regularisation)",
                    "setup": "peaked-left vs peaked-right 1D prox",
                },
                "runtime_s": time.perf_counter() - t0,
                "provenance": provenance(),
            },
            indent=2,
        )
    )


# ------------------------------------------------------- rigorous + fig1
def _erratum() -> ModuleType:
    return _load("experiments_erratum_reruns", "erratum_reruns_2026_10_24.py")


def run_rigorous(n_slow: int) -> None:
    _erratum().run_full("rigorous", SEEDS, n_slow, RIG_DIR)


def run_fig1(n_slow: int) -> None:
    from kiki_flow_core.species import CanonicalSpecies  # noqa: PLC0415
    from kiki_flow_core.track2_paper.figures.species_moments import (  # noqa: PLC0415
        make_species_moments,
        species_moments,
    )

    err = _erratum()
    x = np.linspace(-2, 2, 64)
    runs = {
        "fast": err._load(OLD_FAST_TRAJ, SEEDS, n_slow),
        "rigorous": err._load(RIG_DIR / "trajectories", SEEDS, n_slow),
    }
    make_species_moments(
        runs, x, out_dir=Path("paper/figures"), filename="fig1_species_moments_reg"
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
        sum(_kl(runs["rigorous"][s, -1, i], runs["fast"][s, -1, i]) for i in range(len(names)))
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
def fast_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r["path"] == "fast"]


def run_spec() -> None:
    paths = output_paths()
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    old = json.loads(wide.output_paths()["records"].read_text())["results"]
    done: dict[str, dict[str, Any]] = {}
    for r in fast_records(old):
        done[wide.key("fast", r["seed"], (r["alpha"], r["beta"], r["gamma"]))] = {
            **r,
            "source": f"{wide.OUT_DIR} (fast path has no prox)",
        }
    if paths["spec_partial"].exists():
        for line in paths["spec_partial"].read_text().splitlines():
            r = json.loads(line)
            done[wide.key(r["path"], r["seed"], (r["alpha"], r["beta"], r["gamma"]))] = r
    jobs = [j for j in wide.todo(set(done)) if j[0] == "rigorous"]
    print(f"{len(jobs)} rigorous runs, estimate {wide.estimate_seconds(jobs) / 3600:.2f} h")
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
                "source": "this driver (reg = 0.01)",
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["eps", "rigorous", "fig1", "spec"])
    ap.add_argument("--n-slow", type=int, default=100)
    args = ap.parse_args()
    mx.set_default_device(mx.cpu)
    warnings.filterwarnings("ignore")
    {
        "eps": run_eps,
        "rigorous": lambda: run_rigorous(args.n_slow),
        "fig1": lambda: run_fig1(args.n_slow),
        "spec": run_spec,
    }[args.what]()


if __name__ == "__main__":
    main()
