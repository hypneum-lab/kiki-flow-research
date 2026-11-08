"""Preregistered test: strong-Turing excess over Gibbs under the implicit step.

Preregistration: ``docs/prereg/2026-10-27-turing-strong-implicit.md`` (tag
``prereg-turing-strong-implicit-20261027``, pushed before any run of this
driver). It replaces the 2026-10-26 test (PR #48), whose "supported" verdict
was withdrawn as an artefact of the non-dissipative explicit step (ERRATUM.md
section 14).

Grid fixed in advance: gamma in {20, 30, 50, 70, 100} x alpha in {2, 5, 10}
(15 cells) plus the alpha = 0 control at every gamma; seeds 300-309 (new,
disjoint from seeds 0-9 of PR #46, 100-109 of PR #48 and 200-209 planned for
the unrun full-energy test). Problem as in PR #46 / #48 (grid 16, 500
particles, 200 fast x 30 slow steps, h = 0.05); slow step: entropic JKO step
of F with the coupling and Turing terms treated implicitly (the default since
2026-10-27, ERRATUM section 15), Sinkhorn reg = h^2, max_iter 1000,
reinjection.

Primary statistic, per cell and seed: d = [gap - gap_Gibbs(alpha)] -
gap(0, gamma, seed); per cell mean, 95 % percentile bootstrap CI and one-sided
bootstrap p over seeds (200 000 resamples); Holm over the 15 cells at 0.05
(one-sided). "excess > 0" iff Holm-adjusted p < 0.05, otherwise
"inconclusive". Descriptive: the excess without the control and the change of
F over the run (dissipation record).

``run`` executes the 200 runs (resumable via ``partial.jsonl``); ``analyse``
writes ``excess.json`` and ``summary.md`` under
``results/turing_strong_implicit_2026_10_27/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


def _load(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    path = Path("experiments") / filename
    if not path.exists():
        path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


imp = _load("experiments_implicit_sweep_2026_10_27", "implicit_sweep_2026_10_27.py")
excess = _load("experiments_excess_over_gibbs_2026_10_25", "excess_over_gibbs_2026_10_25.py")
prev = _load("experiments_turing_strong_excess_2026_10_26", "turing_strong_excess_2026_10_26.py")

GAMMAS = [20.0, 30.0, 50.0, 70.0, 100.0]
ALPHAS = [2.0, 5.0, 10.0]
SEEDS = list(range(300, 310))
FORBIDDEN_SEEDS = set(range(10)) | set(range(100, 110)) | set(range(200, 210))
FAMILY_LEVEL = 0.05
BOOT_SEED = 20261028
REG = imp.MAIN_REG
OUT_DIR = Path("results/turing_strong_implicit_2026_10_27")

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


def jobs() -> list[tuple[float, float, int, float]]:
    # Largest gamma first: the slowest runs start early.
    order = sorted(couples(), key=lambda c: -c[1])
    return [(a, g, s, REG) for a, g in order for s in SEEDS]


def run(workers: int) -> None:
    imp.run_jobs(jobs(), output_paths()["partial"], workers, imp.sweep.base.N_SLOW)


def analyse(records: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["alpha"], r["gamma"], r["seed"]): r["gap_bits"] for r in records}
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for a, g in cells():
        gg = excess.gap_of(excess.gibbs_potentials(excess.problem(a, g)))
        primary = [by[(a, g, s)] - gg - by[(0.0, g, s)] for s in SEEDS]
        raw = [by[(a, g, s)] - gg for s in SEEDS]
        rows.append(
            {
                "alpha": a,
                "gamma": g,
                "gap": float(np.mean([by[(a, g, s)] for s in SEEDS])),
                "gap_alpha0": float(np.mean([by[(0.0, g, s)] for s in SEEDS])),
                "gap_gibbs": gg,
                "primary": prev.bootstrap(primary, rng),
                "excess_no_control": prev.bootstrap(raw, rng),
            }
        )
    adjusted, reject = prev.holm([r["primary"]["p_one_sided"] for r in rows])
    order = np.argsort([r["primary"]["p_one_sided"] for r in rows])
    for rank, idx in enumerate(order):
        r = rows[idx]
        level = FAMILY_LEVEL / (len(rows) - rank)
        r["holm_lower_bound"] = float(np.quantile(r["primary"]["_means"], level))
    for r, p_adj, rej in zip(rows, adjusted, reject, strict=True):
        r["p_holm"] = p_adj
        r["verdict"] = "excess > 0" if rej else "inconclusive"
        for key in ("primary", "excess_no_control"):
            r[key].pop("_means", None)
    return {
        "meta": {
            "script": "experiments/turing_strong_implicit_2026_10_27.py",
            "prereg": "docs/prereg/2026-10-27-turing-strong-implicit.md",
            "prereg_tag": "prereg-turing-strong-implicit-20261027",
            "git_commit": subprocess.run(
                ["git", "rev-parse", "--short=8", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "host": platform.node(),
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seeds": SEEDS,
            "n_records": len(records),
        },
        "rows": rows,
        "n_positive": int(sum(reject)),
    }


def _fmt(st: dict[str, float]) -> str:
    return f"{st['mean']:+.3f} [{st['ci95_lo']:+.3f}, {st['ci95_hi']:+.3f}]"


def write_summary(res: dict[str, Any], path: Path) -> None:
    meta = res["meta"]
    lines = [
        "# Strong-Turing excess over Gibbs, implicit step — preregistered (2026-10-27)",
        "",
        f"Preregistration `{meta['prereg']}` (tag `{meta['prereg_tag']}`); driver at "
        f"`{meta['git_commit']}`, {meta['n_records']} runs, seeds 300-309, "
        f"{meta['timestamp_utc']}.",
        "",
        "| α | γ | gap | gap(α=0) | gap_Gibbs | primary excess | p | p Holm | Holm LB "
        "| verdict | excess w/o control |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        lines.append(
            f"| {r['alpha']:g} | {r['gamma']:g} | {r['gap']:.3f} | {r['gap_alpha0']:.3f} "
            f"| {r['gap_gibbs']:.3f} | {_fmt(r['primary'])} | {r['primary']['p_one_sided']:.2g} "
            f"| {r['p_holm']:.2g} | {r['holm_lower_bound']:+.3f} | {r['verdict']} "
            f"| {_fmt(r['excess_no_control'])} |"
        )
    lines += [
        "",
        f"Cells with excess > 0 after Holm: {res['n_positive']} / {len(res['rows'])}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyse"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
