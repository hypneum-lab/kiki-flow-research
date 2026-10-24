"""Widened specialisation sweep, fast vs rigorous path (2026-10-24).

Extends ``experiments/rigorous_specialisation_2026_10_24.py`` (same problem,
same ``run_one``) along the three grid axes, with 10 seeds:

- core:  the original 72 points (alpha {0,1,3,5,8,10} x beta {0,0.5,1,2}
         x gamma {0,1,5}); seeds 0-4 are reused from the first sweep
         (same driver code), seeds 5-9 are new;
- beta:  beta {0.2, 0.3, 4, 8} at every core alpha, gamma = 0;
- alpha: alpha {15, 20, 30} x beta {0, 0.2, 0.5, 1, 2, 4}, gamma = 0;
- gamma: gamma {10, 20, 50} x alpha {0, 5, 8, 10} x beta {0.5, 1}.

beta only enters through ``n_inner = max(1, int(10 * beta))``, so beta
values below 0.2 are equivalent to beta = 0 and are not swept. Every point
has an alpha = 0 control at the same (beta, gamma). Outputs go to NEW files
under ``paper/rigorous_specialisation_wide_2026_10_24/``. MLX is pinned to
the CPU device in ``main``. Entropies are in bits.
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


def _load_base() -> ModuleType:
    name = "experiments_rigorous_specialisation_2026_10_24"
    if name in sys.modules:
        return sys.modules[name]
    path = Path("experiments/rigorous_specialisation_2026_10_24.py")
    if not path.exists():  # imported from the tests, cwd may differ
        path = Path(__file__).with_name("rigorous_specialisation_2026_10_24.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()

SEEDS = list(range(10))
REUSE_SEEDS = {0, 1, 2, 3, 4}
REUSE_SOURCE = "paper/rigorous_specialisation_2026_10_24 (driver c55c784)"
PATHS = base.PATHS
OUT_DIR = Path("paper/rigorous_specialisation_wide_2026_10_24")
SLICES: dict[str, tuple[list[float], list[float], list[float]]] = {
    "core": (base.ALPHAS, base.BETAS, base.GAMMAS),
    "beta": (base.ALPHAS, [0.2, 0.3, 4.0, 8.0], [0.0]),
    "alpha": ([15.0, 20.0, 30.0], [0.0, 0.2, 0.5, 1.0, 2.0, 4.0], [0.0]),
    "gamma": ([0.0, 5.0, 8.0, 10.0], [0.5, 1.0], [10.0, 20.0, 50.0]),
}
# Seconds per run, measured in the first sweep on the same host:
# t = T0 + T1 * n_inner (both paths within 5 %).
T0 = 0.18
T1 = 0.165
# Two-sided 95 % Student t quantiles by sample size n (df = n - 1).
T_CRIT_975 = {5: 2.776, 10: 2.262}

Point = tuple[float, float, float]


def output_paths() -> dict[str, Path]:
    return {
        "records": OUT_DIR / "sweep_wide_records.json",
        "summary": OUT_DIR / "summary.json",
        "partial": OUT_DIR / "partial.jsonl",
    }


def n_inner(beta: float) -> int:
    return max(1, int(10 * beta)) if beta > 0 else 1


def slices_of(p: Point) -> list[str]:
    return [s for s, (a, b, g) in SLICES.items() if p[0] in a and p[1] in b and p[2] in g]


def points() -> list[Point]:
    pts: set[Point] = set()
    for a_s, b_s, g_s in SLICES.values():
        pts.update((a, b, g) for a in a_s for b in b_s for g in g_s)
    # alpha = 0 control for every point
    pts.update({(0.0, b, g) for _, b, g in list(pts)})
    return sorted(pts)


def key(path: str, seed: int, p: Point) -> str:
    return f"{path}|{seed}|{p[0]}|{p[1]}|{p[2]}"


def reuse_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    core = set(base.configs())
    out = []
    for r in records:
        p = (r["alpha"], r["beta"], r["gamma"])
        if r["seed"] in REUSE_SEEDS and p in core:
            out.append({**r, "source": REUSE_SOURCE})
    return out


def _first_sweep_records() -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for p in ("fast", "rigorous"):
        f = base.output_paths()[p]
        if f.exists():
            recs += json.loads(f.read_text())["results"]
    return reuse_records(recs)


def reused_keys() -> set[str]:
    return {
        key(r["path"], r["seed"], (r["alpha"], r["beta"], r["gamma"]))
        for r in _first_sweep_records()
    }


def todo(done: set[str]) -> list[tuple[str, int, Point]]:
    return [
        (path, seed, p)
        for seed in SEEDS
        for path in PATHS
        for p in points()
        if key(path, seed, p) not in done
    ]


def estimate_seconds(jobs: list[tuple[str, int, Point]]) -> float:
    return float(sum(T0 + T1 * n_inner(p[1]) for _, _, p in jobs))


def run_all() -> None:
    paths = output_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    for r in _first_sweep_records():
        done[key(r["path"], r["seed"], (r["alpha"], r["beta"], r["gamma"]))] = r
    if paths["partial"].exists():
        for line in paths["partial"].read_text().splitlines():
            r = json.loads(line)
            done[key(r["path"], r["seed"], (r["alpha"], r["beta"], r["gamma"]))] = r
    jobs = todo(set(done))
    print(f"{len(jobs)} runs to do, estimate {estimate_seconds(jobs) / 3600:.2f} h", flush=True)
    with paths["partial"].open("a") as fh:
        for path, seed, p in jobs:
            t0 = time.perf_counter()
            ent = base.run_one(*p, rigorous=path == "rigorous", seed=seed)
            h_mean = float(np.mean(list(ent.values())))
            rec = {
                "path": path,
                "seed": seed,
                "alpha": p[0],
                "beta": p[1],
                "gamma": p[2],
                "mean_entropy_bits": h_mean,
                "gap_bits": base.MAX_ENTROPY_BITS - h_mean,
                "per_species": ent,
                "runtime_s": time.perf_counter() - t0,
                "source": "this driver",
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[key(path, seed, p)] = rec
            print(f"{path} seed={seed} p={p} H={h_mean:.4f}", flush=True)
    write_outputs(list(done.values()))


def _ci(vals: np.ndarray) -> dict[str, float]:
    sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    t_q = T_CRIT_975.get(int(vals.size), float("nan"))
    half = t_q * sd / np.sqrt(vals.size) if vals.size > 1 else 0.0
    m = float(vals.mean())
    return {"mean": m, "sd": sd, "ci95_lo": m - half, "ci95_hi": m + half, "n": int(vals.size)}


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[tuple[str, Point], dict[int, dict[str, Any]]] = {}
    for r in records:
        by.setdefault((r["path"], (r["alpha"], r["beta"], r["gamma"])), {})[r["seed"]] = r
    rows = []
    for p in points():
        row: dict[str, Any] = {
            "alpha": p[0],
            "beta": p[1],
            "gamma": p[2],
            "n_inner": n_inner(p[1]),
            "slices": slices_of(p) or ["control"],
        }
        h_seed: dict[str, np.ndarray] = {}
        for path in PATHS:
            recs = by.get((path, p), {})
            ctrl = by.get((path, (0.0, p[1], p[2])), {})
            seeds = sorted(set(recs) & set(ctrl))
            h = np.array([recs[s]["mean_entropy_bits"] for s in seeds])
            h0 = np.array([ctrl[s]["mean_entropy_bits"] for s in seeds])
            h_seed[path] = h
            species = recs[seeds[0]]["per_species"] if seeds else {}
            row[path] = {
                "H": _ci(h),
                "gap": _ci(base.MAX_ENTROPY_BITS - h),
                "specialisation_vs_alpha0": _ci(h0 - h),
                "per_species_H": {
                    n: _ci(np.array([recs[s]["per_species"][n] for s in seeds])) for n in species
                },
                "seeds": seeds,
            }
        if h_seed["fast"].size and h_seed["fast"].size == h_seed["rigorous"].size:
            row["rigorous_minus_fast_H"] = _ci(h_seed["rigorous"] - h_seed["fast"])
        rows.append(row)

    def beta_trend(path: str, alpha: float) -> dict[str, Any]:
        pts = [r for r in rows if r["alpha"] == alpha and r["gamma"] == 0.0 and r["beta"] > 0]
        pts.sort(key=lambda r: r["beta"])
        if len(pts) < 2:  # noqa: PLR2004
            return {}
        x = np.log2([r["beta"] for r in pts])
        y = np.array([r[path]["H"]["mean"] for r in pts])
        slope = float(np.polyfit(x, y, 1)[0])
        return {
            "betas": [r["beta"] for r in pts],
            "H_mean": y.tolist(),
            "slope_H_per_log2_beta": slope,
            "argmin_beta": pts[int(np.argmin(y))]["beta"],
        }

    alphas = sorted({r["alpha"] for r in rows})
    trends = {path: {str(a): beta_trend(path, a) for a in alphas} for path in PATHS}
    best = {path: min(rows, key=lambda r, path=path: r[path]["H"]["mean"]) for path in PATHS}
    return {
        "metric": "H = mean over 4 species of final entropy (bits); gap = 4 - H; "
        "specialisation_vs_alpha0 = H(alpha=0, same beta, gamma, seed) - H, paired by seed",
        "ci": "mean +/- t(0.975, n-1) * sd / sqrt(n) over seeds",
        "max_entropy_bits": base.MAX_ENTROPY_BITS,
        "seeds": SEEDS,
        "slices": {k: {"alpha": a, "beta": b, "gamma": g} for k, (a, b, g) in SLICES.items()},
        "n_points": len(rows),
        "best_point": {
            path: {k: best[path][k] for k in ("alpha", "beta", "gamma")}
            | {"H": best[path][path]["H"], "gap": best[path][path]["gap"]}
            for path in PATHS
        },
        "beta_trend_gamma0": trends,
        "rows": rows,
    }


def write_outputs(records: list[dict[str, Any]]) -> None:
    paths = output_paths()
    prov = base.provenance()
    records = sorted(
        records, key=lambda r: (r["path"], r["seed"], r["alpha"], r["beta"], r["gamma"])
    )
    params = {
        "grid": base.GRID,
        "n_particles": base.N_PARTICLES,
        "n_fast": base.N_FAST,
        "n_slow": base.N_SLOW,
    }
    paths["records"].write_text(
        json.dumps(
            {
                "params": params,
                "n_records": len(records),
                "runtime_s_new_runs": float(
                    sum(r["runtime_s"] for r in records if r.get("source") == "this driver")
                ),
                "results": records,
                "provenance": prov,
            },
            indent=1,
        )
    )
    summary = summarise(records) | {"params": params, "provenance": prov}
    paths["summary"].write_text(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true", help="print the runtime estimate and exit")
    args = ap.parse_args()
    if args.estimate:
        jobs = todo(reused_keys())
        print(f"{len(points())} points, {len(jobs)} runs, {estimate_seconds(jobs) / 3600:.2f} h")
        return
    mx.set_default_device(mx.cpu)
    warnings.filterwarnings("ignore")
    run_all()


if __name__ == "__main__":
    main()
