"""Rigorous specialisation sweep under the entropic JKO step of F.

Reruns the "rigorous" half of the widened specialisation sweep
(``experiments/rigorous_specialisation_wide_2026_10_24.py``: 138 points
(alpha, beta, gamma), seeds 0-9, grid 16, 500 particles, 200 fast x 30 slow
steps, h = 0.05) with the corrected slow step validated in ERRATUM.md section
12 (PI decision of 2026-10-25):

* ``FullJKOSolver(scheme="jko_entropic")`` with Sinkhorn reg = h^2 (0.0025)
  and the solver's default ``max_iter = 1000``;
* ``MultiscaleLoop(reinject=True)``: the JKO output is fed back into the
  particles after each slow step.

beta only set the number of Euclidean sub-steps of the legacy split
(``n_inner = max(1, int(10 beta))``); the entropic scheme ignores
``n_inner``, so for given (alpha, gamma, seed) every beta gives the same run.
The driver therefore runs each distinct (alpha, gamma) once per seed and
maps it onto every beta of the grid (33 couples x 10 seeds = 330 runs); a
test pins the beta invariance. The fast path (no prox, unchanged) is not
rerun: its records are read from the committed widened sweep.

Sensitivity: the two key points of the epsilon sweep, (10, 1, 0) and
(8, 0.5, 5), and their alpha = 0 controls, are also run at reg = 0.01 (the
paper's epsilon), seeds 0-4.

Outputs (new files): ``results/jko_entropic_sweep_2026_10_25/{records.json,
summary.json, summary.md}`` and ``partial.jsonl`` (resumable). MLX pinned to
the CPU; ``--workers`` processes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver
from kiki_flow_core.track2_paper.mlx_particle_simulator import MLXParticleSimulator
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop


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


base = _load(
    "experiments_rigorous_specialisation_2026_10_24", "rigorous_specialisation_2026_10_24.py"
)
wide = _load(
    "experiments_rigorous_specialisation_wide_2026_10_24",
    "rigorous_specialisation_wide_2026_10_24.py",
)

Point = tuple[float, float, float]
Couple = tuple[float, float]

SEEDS = list(range(10))
SENS_SEEDS = [0, 1, 2, 3, 4]
SENS_REG = 0.01
KEY_POINTS: list[Point] = [(10.0, 1.0, 0.0), (8.0, 0.5, 5.0)]
H = base.H
MAIN_REG = H * H
MAX_ITER = 1000
GRID = base.GRID
MAX_ENTROPY_BITS = float(np.log2(GRID))
RHO_FLOOR = 1e-12
OUT_DIR = Path("results/jko_entropic_sweep_2026_10_25")
LEGACY = Path("paper/rigorous_specialisation_wide_2026_10_24_reg/sweep_wide_records.json")
T_CRIT_975 = {5: 2.776, 10: 2.262}


def output_paths() -> dict[str, Path]:
    return {
        "records": OUT_DIR / "records.json",
        "summary": OUT_DIR / "summary.json",
        "summary_md": OUT_DIR / "summary.md",
        "partial": OUT_DIR / "partial.jsonl",
    }


def points() -> list[Point]:
    return list(wide.points())


def couples(pts: list[Point]) -> list[Couple]:
    return sorted({(a, g) for a, _, g in pts})


def make_solver(alpha: float, gamma: float, reg: float) -> tuple[Any, np.ndarray, FullJKOSolver]:
    species, support, f_func = base.problem(alpha, gamma)
    jko = FullJKOSolver(
        f_functional=f_func,
        h=H,
        support=support,
        epsilon=reg,
        max_iter=MAX_ITER,
        scheme="jko_entropic",
    )
    return species, support, jko


def entropy_bits(rho: np.ndarray) -> float:
    r = np.clip(np.asarray(rho, dtype=np.float64), RHO_FLOOR, None)
    return float(-(r * np.log2(r)).sum())


def run_couple(
    alpha: float, gamma: float, seed: int, reg: float, n_slow: int = base.N_SLOW
) -> dict[str, Any]:
    import mlx.core as mx  # noqa: PLC0415  (per-process import, CPU stream)

    t0 = time.perf_counter()
    with mx.stream(mx.cpu):
        species, support, jko = make_solver(alpha, gamma, reg)
        sim = MLXParticleSimulator(
            species=species, n_particles=base.N_PARTICLES, latent_dim=2, seed=seed
        )
        loop = MultiscaleLoop(
            sim=sim, jko=jko, n_fast=base.N_FAST, n_slow=n_slow, support=support, reinject=True
        )
        final = loop.run(seed=seed)["trajectory"][-1]
    per = {n: entropy_bits(r) for n, r in final.rho.items()}
    mean_h = float(np.mean(list(per.values())))
    return {
        "alpha": alpha,
        "gamma": gamma,
        "seed": seed,
        "reg": reg,
        "mean_entropy_bits": mean_h,
        "gap_bits": MAX_ENTROPY_BITS - mean_h,
        "per_species": per,
        "rho": {n: [float(v) for v in np.asarray(r)] for n, r in final.rho.items()},
        "runtime_s": time.perf_counter() - t0,
    }


def jobs(seeds: list[int]) -> list[tuple[float, float, int, float]]:
    main = [(a, g, s, MAIN_REG) for a, g in couples(points()) for s in seeds]
    sens_couples = sorted({(a, g) for a, _, g in KEY_POINTS} | {(0.0, g) for _, _, g in KEY_POINTS})
    sens = [(a, g, s, SENS_REG) for a, g in sens_couples for s in SENS_SEEDS if s in seeds]
    return main + sens


def _key(a: float, g: float, s: int, reg: float) -> str:
    return f"{a}|{g}|{s}|{reg}"


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean()) if n else float("nan")
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    half = T_CRIT_975.get(n, 1.96) * sd / np.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95_lo": mean - half, "ci95_hi": mean + half, "n": n}


def legacy_records() -> list[dict[str, Any]]:
    if not LEGACY.exists():
        return []
    return list(json.loads(LEGACY.read_text())["results"])


def summarise(
    records: list[dict[str, Any]], legacy: list[dict[str, Any]], reg: float, seeds: list[int]
) -> list[dict[str, Any]]:
    """One row per wide-grid point: new gap and S, legacy rigorous and fast gaps."""
    by = {(r["alpha"], r["gamma"], r["seed"]): r for r in records if r["reg"] == reg}
    leg: dict[tuple[str, float, float, float, int], dict[str, Any]] = {
        (r["path"], r["alpha"], r["beta"], r["gamma"], r["seed"]): r for r in legacy
    }
    rows = []
    pts = points() if reg == MAIN_REG else [*KEY_POINTS]
    for a, b, g in pts:
        new_gap, new_s, old_gap, fast_gap, old_s = [], [], [], [], []
        used = []
        for s in seeds:
            r, c = by.get((a, g, s)), by.get((0.0, g, s))
            if r is None or c is None:
                continue
            used.append(s)
            new_gap.append(r["gap_bits"])
            new_s.append(c["mean_entropy_bits"] - r["mean_entropy_bits"])
            lr, lc = leg.get(("rigorous", a, b, g, s)), leg.get(("rigorous", 0.0, b, g, s))
            if lr is not None:
                old_gap.append(lr["gap_bits"])
                if lc is not None:
                    old_s.append(lc["mean_entropy_bits"] - lr["mean_entropy_bits"])
            lf = leg.get(("fast", a, b, g, s))
            if lf is not None:
                fast_gap.append(lf["gap_bits"])
        rows.append(
            {
                "alpha": a,
                "beta": b,
                "gamma": g,
                "seeds": used,
                "entropic_gap": stats(new_gap),
                "entropic_S": stats(new_s),
                "legacy_rigorous_gap": stats(old_gap),
                "legacy_rigorous_S": stats(old_s),
                "fast_gap": stats(fast_gap),
            }
        )
    return rows


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True, check=False
        )
        return out.stdout.strip()
    except OSError:
        return "unknown"


def _f(st: dict[str, float]) -> str:
    if not st["n"]:
        return "—"
    return f"{st['mean']:.3f} ± {st['sd']:.3f}"


def write_summary_md(summary: dict[str, Any], path: Path) -> None:
    meta = summary["meta"]
    lines = [
        "# Specialisation sweep under the entropic JKO step (2026-10-25)",
        "",
        f"Driver `{meta['script']}` at `{meta['git_commit']}`, host `{meta['host']}`, "
        f"Python {meta['python']}, {meta['timestamp_utc']}, {meta['runs']} runs, "
        f"{meta['wall_s']:.0f} s wall, {meta['workers']} workers.",
        "",
        "Problem of `experiments/rigorous_specialisation_2026_10_24.py` (grid 16, 500 "
        "particles, 200 fast x 30 slow steps, h = 0.05), points and seeds of the widened "
        'sweep. Slow step: `FullJKOSolver(scheme="jko_entropic")`, Sinkhorn reg = h^2 = '
        "0.0025, max_iter 1000, `MultiscaleLoop(reinject=True)`. beta has no effect under "
        "this scheme (it only set n_inner of the legacy split), so each (alpha, gamma) run "
        "serves every beta. Gap = log2(16) - mean H over the four species (bits); S = "
        "H(alpha = 0, same gamma, seed) - H. Legacy columns: rigorous path (legacy split, "
        "epsilon = 0.01) and fast path from "
        "`paper/rigorous_specialisation_wide_2026_10_24_reg/`. Mean ± sd over seeds.",
        "",
        "## Key points",
        "",
        "| (alpha, beta, gamma) | entropic gap (reg h^2) | S | entropic gap (reg 0.01, 5 seeds) "
        "| legacy rigorous gap | legacy S | fast gap |",
        "|---|---|---|---|---|---|---|",
    ]
    main = {(r["alpha"], r["beta"], r["gamma"]): r for r in summary["rows_main"]}
    sens = {(r["alpha"], r["beta"], r["gamma"]): r for r in summary["rows_sensitivity"]}
    best_legacy = max(
        (r for r in summary["rows_main"] if r["legacy_rigorous_gap"]["n"]),
        key=lambda r: r["legacy_rigorous_gap"]["mean"],
    )
    best_new = max(summary["rows_main"], key=lambda r: r["entropic_gap"]["mean"])
    for p in [*KEY_POINTS, (best_legacy["alpha"], best_legacy["beta"], best_legacy["gamma"])]:
        r = main[p]
        se = sens.get(p)
        lines.append(
            f"| {p} | {_f(r['entropic_gap'])} | {_f(r['entropic_S'])} "
            f"| {_f(se['entropic_gap']) if se else '—'} | {_f(r['legacy_rigorous_gap'])} "
            f"| {_f(r['legacy_rigorous_S'])} | {_f(r['fast_gap'])} |"
        )
    lines += [
        "",
        f"Best entropic point: ({best_new['alpha']}, any beta, {best_new['gamma']}), gap "
        f"{_f(best_new['entropic_gap'])}. Best legacy rigorous point: "
        f"({best_legacy['alpha']}, {best_legacy['beta']}, {best_legacy['gamma']}), gap "
        f"{_f(best_legacy['legacy_rigorous_gap'])}.",
        "",
        "## All (alpha, gamma) couples (beta-invariant under the entropic scheme)",
        "",
        "| alpha | gamma | entropic gap | S | legacy rigorous gap, range over beta | fast gap, "
        "range over beta |",
        "|---|---|---|---|---|---|",
    ]
    seen: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for r in summary["rows_main"]:
        seen.setdefault((r["alpha"], r["gamma"]), []).append(r)
    for (a, g), rows in sorted(seen.items()):
        leg = [r["legacy_rigorous_gap"]["mean"] for r in rows if r["legacy_rigorous_gap"]["n"]]
        fst = [r["fast_gap"]["mean"] for r in rows if r["fast_gap"]["n"]]
        rng = lambda v: f"{min(v):.3f}–{max(v):.3f}" if v else "—"  # noqa: E731
        lines.append(
            f"| {a:g} | {g:g} | {_f(rows[0]['entropic_gap'])} | {_f(rows[0]['entropic_S'])} "
            f"| {rng(leg)} | {rng(fst)} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    ap.add_argument("--n-slow", type=int, default=base.N_SLOW)
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    paths = output_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    if paths["partial"].exists():
        for line in paths["partial"].read_text().splitlines():
            r = json.loads(line)
            done[_key(r["alpha"], r["gamma"], r["seed"], r["reg"])] = r
    todo = [j for j in jobs(seeds) if _key(*j) not in done]
    print(f"{len(done)} done, {len(todo)} to run, {args.workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, paths["partial"].open("a") as fh:
        futs = {
            pool.submit(run_couple, a, g, s, reg, args.n_slow): (a, g, s, reg)
            for a, g, s, reg in todo
        }
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            done[_key(r["alpha"], r["gamma"], r["seed"], r["reg"])] = r
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(todo)}] alpha={r['alpha']} gamma={r['gamma']} seed={r['seed']} "
                f"reg={r['reg']} gap={r['gap_bits']:.3f} {r['runtime_s']:.1f}s",
                flush=True,
            )
    records = list(done.values())
    legacy = legacy_records()
    summary = {
        "meta": {
            "script": "experiments/jko_entropic_sweep_2026_10_25.py",
            "git_commit": _git_commit(),
            "host": platform.node(),
            "python": platform.python_version(),
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "runs": len(records),
            "wall_s": time.perf_counter() - t0,
            "workers": args.workers,
            "scheme": "jko_entropic",
            "reg_main": MAIN_REG,
            "reg_sensitivity": SENS_REG,
            "max_iter": MAX_ITER,
            "reinject": True,
            "seeds": seeds,
            "legacy_source": str(LEGACY),
        },
        "rows_main": summarise(records, legacy, MAIN_REG, seeds),
        "rows_sensitivity": summarise(
            records, legacy, SENS_REG, [s for s in SENS_SEEDS if s in seeds]
        ),
    }
    paths["records"].write_text(
        json.dumps({"n_records": len(records), "results": records}, indent=1), encoding="utf-8"
    )
    paths["summary"].write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_summary_md(summary, paths["summary_md"])
    print(f"wrote {OUT_DIR} ({summary['meta']['wall_s']:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
