"""Preregistered test: specialisation excess over Gibbs in the strong-Turing regime.

Preregistration: ``docs/prereg/2026-10-26-turing-strong-excess.md`` (tag
``prereg-turing-strong-20261026``, pushed before any run of this driver).

Grid fixed in advance: gamma in {20, 30, 50, 70, 100} x alpha in {2, 5, 10}
(15 cells), plus the alpha = 0 control at every gamma; seeds 100-109 (new,
disjoint from seeds 0-9 of the 2026-10-25 sweep that motivated the test).
Problem and slow step identical to ``experiments/jko_entropic_sweep_2026_10_25.py``
(grid 16, 500 particles, 200 fast x 30 slow steps, h = 0.05, entropic JKO step
of F with reg = h^2, max_iter 1000, reinjection).

Primary statistic, per cell (alpha, gamma) and seed s:
    d_s = [gap_s(alpha, gamma) - gap_Gibbs(alpha)] - gap_s(0, gamma)
with gap_Gibbs the gap of rho_i ∝ prior_i exp(-V_i) (it does not depend on
gamma; it is 0 at alpha = 0). Per cell: mean over seeds, 95 % percentile
bootstrap CI over seeds, one-sided bootstrap p-value
p = (1 + #{bootstrap means <= 0}) / (B + 1). Holm correction over the 15
cells at family-wise level 0.05 (one-sided). A cell is "excess > 0" iff its
Holm-adjusted p < 0.05 (equivalently, the one-sided bootstrap lower bound at
the Holm level excludes 0); otherwise "inconclusive". Descriptive only: the
excess without the alpha = 0 control, and the excess over the minimiser of the
full T2 energy (its damped fixed point does not converge for gamma >= 20).

``run`` executes the 200 runs (resumable, ``--workers`` processes);
``analyse`` writes ``results/turing_strong_excess_2026_10_26/{excess.json,
summary.md}``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


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


sweep = _load("experiments_jko_entropic_sweep_2026_10_25", "jko_entropic_sweep_2026_10_25.py")
excess = _load("experiments_excess_over_gibbs_2026_10_25", "excess_over_gibbs_2026_10_25.py")

GAMMAS = [20.0, 30.0, 50.0, 70.0, 100.0]
ALPHAS = [2.0, 5.0, 10.0]
SEEDS = list(range(100, 110))
FORBIDDEN_SEEDS = set(range(10))  # used by the sweep that motivated the test
FAMILY_LEVEL = 0.05
N_BOOT = 200_000
BOOT_SEED = 20261026
REG = sweep.MAIN_REG
OUT_DIR = Path("results/turing_strong_excess_2026_10_26")

Cell = tuple[float, float]


def output_paths() -> dict[str, Path]:
    return {
        "partial": OUT_DIR / "partial.jsonl",
        "excess": OUT_DIR / "excess.json",
        "summary": OUT_DIR / "summary.md",
    }


def cells() -> list[Cell]:
    return [(a, g) for g in GAMMAS for a in ALPHAS]


def couples() -> list[Cell]:
    return sorted({*cells(), *((0.0, g) for g in GAMMAS)})


def jobs() -> list[tuple[float, float, int]]:
    return [(a, g, s) for a, g in couples() for s in SEEDS]


def _key(a: float, g: float, s: int) -> str:
    return f"{a}|{g}|{s}"


def bootstrap(values: list[float], rng: np.random.Generator) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    means = arr[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)),
        "ci95_lo": float(np.quantile(means, 0.025)),
        "ci95_hi": float(np.quantile(means, 0.975)),
        "p_one_sided": float((1 + np.sum(means <= 0.0)) / (N_BOOT + 1)),
        "n": int(n),
        "_means": means,
    }


def holm(pvalues: list[float], level: float = FAMILY_LEVEL) -> tuple[list[float], list[bool]]:
    """Holm step-down: adjusted p-values and rejections at ``level``."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return [float(v) for v in adjusted], [bool(v < level) for v in adjusted]


def run(workers: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = output_paths()["partial"]
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add(_key(r["alpha"], r["gamma"], r["seed"]))
    todo = [j for j in jobs() if _key(*j) not in done]
    print(f"{len(done)} done, {len(todo)} to run, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a") as fh:
        futs = [pool.submit(sweep.run_couple, a, g, s, REG) for a, g, s in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(todo)}] alpha={r['alpha']} gamma={r['gamma']} seed={r['seed']} "
                f"gap={r['gap_bits']:.3f} {r['runtime_s']:.1f}s",
                flush=True,
            )


def analyse(records: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["alpha"], r["gamma"], r["seed"]): r["gap_bits"] for r in records}
    rng = np.random.default_rng(BOOT_SEED)
    refs = excess.references(cells())
    rows = []
    for a, g in cells():
        gg = refs[(a, g)]["gap_gibbs"]
        primary = [by[(a, g, s)] - gg - by[(0.0, g, s)] for s in SEEDS]
        raw = [by[(a, g, s)] - gg for s in SEEDS]
        fmin = [by[(a, g, s)] - refs[(a, g)]["gap_fmin"] for s in SEEDS]
        rows.append(
            {
                "alpha": a,
                "gamma": g,
                "gap": float(np.mean([by[(a, g, s)] for s in SEEDS])),
                "gap_alpha0": float(np.mean([by[(0.0, g, s)] for s in SEEDS])),
                "gap_gibbs": gg,
                "primary": bootstrap(primary, rng),
                "excess_no_control": bootstrap(raw, rng),
                "excess_fmin": bootstrap(fmin, rng),
                "fmin_residual": refs[(a, g)]["fmin_delta"],
                "fmin_iter": refs[(a, g)]["fmin_iter"],
            }
        )
    adjusted, reject = holm([r["primary"]["p_one_sided"] for r in rows])
    order = np.argsort([r["primary"]["p_one_sided"] for r in rows])
    m = len(rows)
    for rank, idx in enumerate(order):
        r = rows[idx]
        level = FAMILY_LEVEL / (m - rank)  # Holm level of this step
        r["holm_level"] = float(level)
        r["holm_lower_bound"] = float(np.quantile(r["primary"]["_means"], level))
    for r, p_adj, rej in zip(rows, adjusted, reject, strict=True):
        r["p_holm"] = p_adj
        r["verdict"] = "excess > 0" if rej else "inconclusive"
        for key in ("primary", "excess_no_control", "excess_fmin"):
            r[key].pop("_means", None)
    return {
        "meta": {
            "script": "experiments/turing_strong_excess_2026_10_26.py",
            "prereg": "docs/prereg/2026-10-26-turing-strong-excess.md",
            "prereg_tag": "prereg-turing-strong-20261026",
            "git_commit": subprocess.run(
                ["git", "rev-parse", "--short=8", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "host": platform.node(),
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seeds": SEEDS,
            "grid": {"alpha": ALPHAS, "gamma": GAMMAS},
            "reg": REG,
            "n_boot": N_BOOT,
            "family_level": FAMILY_LEVEL,
            "n_records": len(records),
        },
        "rows": rows,
        "n_positive": int(sum(reject)),
    }


def _fmt(st: dict[str, float]) -> str:
    return f"{st['mean']:+.3f} [{st['ci95_lo']:+.3f}, {st['ci95_hi']:+.3f}]"


def write_summary(res: dict[str, Any], path: Path) -> None:
    lines = [
        "# Strong-Turing excess over Gibbs — preregistered test (2026-10-26)",
        "",
        f"Preregistration `{res['meta']['prereg']}` (tag `{res['meta']['prereg_tag']}`). "
        f"Driver at `{res['meta']['git_commit']}`, host `{res['meta']['host']}`, "
        f"{res['meta']['timestamp_utc']}, {res['meta']['n_records']} runs, seeds 100-109.",
        "",
        "Primary: d = [gap − gap_Gibbs] − gap(alpha = 0, same gamma, seed), bits; mean "
        "[95 % bootstrap CI over seeds]; one-sided bootstrap p; Holm over 15 cells at 0.05.",
        "",
        "| alpha | gamma | gap | gap(α=0) | gap_Gibbs | primary excess | p | p Holm "
        "| Holm lower bound | verdict | excess w/o control | excess vs Fmin (descr.) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        lines.append(
            f"| {r['alpha']:g} | {r['gamma']:g} | {r['gap']:.3f} | {r['gap_alpha0']:.3f} "
            f"| {r['gap_gibbs']:.3f} | {_fmt(r['primary'])} | {r['primary']['p_one_sided']:.2g} "
            f"| {r['p_holm']:.2g} | {r['holm_lower_bound']:+.3f} | {r['verdict']} "
            f"| {_fmt(r['excess_no_control'])} | {_fmt(r['excess_fmin'])} |"
        )
    lines += [
        "",
        f"Cells with excess > 0 after Holm: {res['n_positive']} / {len(res['rows'])}.",
        "",
        "The minimiser of the full T2 energy is a damped fixed point that does not converge "
        "for gamma >= 20 (residual in `excess.json`); the last column is descriptive only.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=6)
    sub.add_parser("analyse")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args.workers)
        return 0
    records = [json.loads(x) for x in output_paths()["partial"].read_text().splitlines()]
    res = analyse(records)
    output_paths()["excess"].write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_summary(res, output_paths()["summary"])
    print(output_paths()["summary"].read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
