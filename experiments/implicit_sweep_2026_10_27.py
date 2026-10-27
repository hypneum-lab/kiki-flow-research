"""Specialisation sweep and excess over Gibbs under the implicit default (2026-10-27).

Recomputation of PR #46 (``experiments/jko_entropic_sweep_2026_10_25.py``) and
of the excess analysis of PR #47 (``experiments/excess_over_gibbs_2026_10_25.py``)
with the slow step that is the default since 2026-10-27 (ERRATUM.md section
15): entropic JKO step of F with the coupling and Turing terms treated
implicitly (``nonseparable="implicit"``), Sinkhorn reg = h^2, max_iter 1000,
reinjection. Same problem, points, couples, seeds (0-9) and sensitivity runs
(reg 0.01 at the epsilon-sweep key points, seeds 0-4) as PR #46; beta has no
effect under the entropic scheme.

``run`` appends one JSON line per run to ``partial.jsonl`` (resumable: done runs
are skipped); ``analyse`` writes ``records.json``, ``summary.json``,
``summary.md`` (gaps next to the entropic-explicit values of PR #46 and the
legacy values) and ``excess.json`` / ``excess_summary.md`` (excess over the
Gibbs equilibrium of the potentials, bootstrap CI over seeds, alpha = 0
control), all under ``results/implicit_sweep_2026_10_27/``.
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


sweep = _load("experiments_jko_entropic_sweep_2026_10_25", "jko_entropic_sweep_2026_10_25.py")
excess = _load("experiments_excess_over_gibbs_2026_10_25", "excess_over_gibbs_2026_10_25.py")

SEEDS = sweep.SEEDS
MAIN_REG = sweep.MAIN_REG
SENS_REG = sweep.SENS_REG
OUT_DIR = Path("results/implicit_sweep_2026_10_27")
EXPLICIT_RECORDS = Path("results/jko_entropic_sweep_2026_10_25/records.json")
NONSEPARABLE = "implicit"
GAMMA_NOT_INTERPRETABLE = 10.0  # explicit values at gamma >= 10 withdrawn (ERRATUM section 14)


def output_paths(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    return {
        "partial": out_dir / "partial.jsonl",
        "records": out_dir / "records.json",
        "summary": out_dir / "summary.json",
        "summary_md": out_dir / "summary.md",
        "excess": out_dir / "excess.json",
        "excess_md": out_dir / "excess_summary.md",
    }


def run_couple(
    alpha: float, gamma: float, seed: int, reg: float, n_slow: int = sweep.base.N_SLOW
) -> dict[str, Any]:
    """One multiscale run of the PR #46 problem under the implicit slow step."""
    import mlx.core as mx  # noqa: PLC0415

    from kiki_flow_core.track2_paper.mlx_particle_simulator import (  # noqa: PLC0415
        MLXParticleSimulator,
    )
    from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop  # noqa: PLC0415

    t0 = time.perf_counter()
    with mx.stream(mx.cpu):
        species, support, f_func = sweep.base.problem(alpha, gamma)
        jko = FullJKOSolver(
            f_functional=f_func,
            h=sweep.H,
            support=support,
            epsilon=reg,
            max_iter=sweep.MAX_ITER,
            scheme="jko_entropic",
            nonseparable=NONSEPARABLE,
        )
        sim = MLXParticleSimulator(
            species=species, n_particles=sweep.base.N_PARTICLES, latent_dim=2, seed=seed
        )
        loop = MultiscaleLoop(
            sim=sim, jko=jko, n_fast=sweep.base.N_FAST, n_slow=n_slow, support=support
        )
        final = loop.run(seed=seed)["trajectory"][-1]
    per = {n: sweep.entropy_bits(r) for n, r in final.rho.items()}
    mean_h = float(np.mean(list(per.values())))
    return {
        "alpha": alpha,
        "gamma": gamma,
        "seed": seed,
        "reg": reg,
        "nonseparable": NONSEPARABLE,
        "mean_entropy_bits": mean_h,
        "gap_bits": sweep.MAX_ENTROPY_BITS - mean_h,
        "per_species": per,
        "rho": {n: [float(v) for v in np.asarray(r)] for n, r in final.rho.items()},
        "runtime_s": time.perf_counter() - t0,
    }


def _key(a: float, g: float, s: int, reg: float) -> str:
    return f"{a}|{g}|{s}|{reg}"


def run_jobs(
    jobs: list[tuple[float, float, int, float]], path: Path, workers: int, n_slow: int
) -> None:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add(_key(r["alpha"], r["gamma"], r["seed"], r["reg"]))
    todo = [j for j in jobs if _key(*j) not in done]
    print(f"{len(done)} done, {len(todo)} to run, {workers} workers", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a") as fh:
        futs = [pool.submit(run_couple, a, g, s, reg, n_slow) for a, g, s, reg in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(todo)}] alpha={r['alpha']} gamma={r['gamma']} seed={r['seed']} "
                f"reg={r['reg']:.4g} gap={r['gap_bits']:.3f} {r['runtime_s']:.0f}s",
                flush=True,
            )


def analyse(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    paths = output_paths(out_dir)
    records = [json.loads(x) for x in paths["partial"].read_text().splitlines()]
    paths["records"].write_text(
        json.dumps({"n_records": len(records), "results": records}, indent=1), encoding="utf-8"
    )
    explicit = json.loads(EXPLICIT_RECORDS.read_text())["results"]
    legacy = sweep.legacy_records()
    rows_new = sweep.summarise(records, legacy, MAIN_REG, SEEDS)
    rows_old = sweep.summarise(explicit, legacy, MAIN_REG, SEEDS)
    old_by = {(r["alpha"], r["beta"], r["gamma"]): r for r in rows_old}
    for r in rows_new:
        r["explicit_gap_pr46"] = old_by[(r["alpha"], r["beta"], r["gamma"])]["entropic_gap"]
    summary = {
        "meta": {
            "script": "experiments/implicit_sweep_2026_10_27.py",
            "git_commit": subprocess.run(
                ["git", "rev-parse", "--short=8", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "host": platform.node(),
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "runs": len(records),
            "nonseparable": NONSEPARABLE,
            "reg_main": MAIN_REG,
            "seeds": SEEDS,
        },
        "rows_main": rows_new,
        "rows_sensitivity": sweep.summarise(
            records, legacy, SENS_REG, [s for s in sweep.SENS_SEEDS if s in SEEDS]
        ),
    }
    paths["summary"].write_text(json.dumps(summary, indent=1), encoding="utf-8")
    _write_summary_md(summary, paths["summary_md"])
    ent = [r for r in records if r["reg"] == MAIN_REG]
    couples = sorted({(r["alpha"], r["gamma"]) for r in ent})
    # Gibbs reference of the potentials only; the full-energy fixed point of PR #47
    # does not converge for gamma >= 20 (ERRATUM section 14) and is not recomputed.
    refs = {
        c: {
            "gap_gibbs": excess.gap_of(excess.gibbs_potentials(excess.problem(*c))),
            "gap_fmin": float("nan"),
        }
        for c in couples
    }
    exc = {
        "meta": {"source": str(paths["partial"]), "nonseparable": NONSEPARABLE},
        "implicit": excess.analyse(ent, refs, excess.BOOT_SEED),
        "explicit_pr46": excess.analyse(
            [r for r in explicit if r["reg"] == MAIN_REG], refs, excess.BOOT_SEED
        ),
    }
    paths["excess"].write_text(json.dumps(exc, indent=1), encoding="utf-8")
    _write_excess_md(exc, paths["excess_md"])
    return summary


def _f(st: dict[str, float]) -> str:
    return "—" if not st["n"] else f"{st['mean']:.3f} ± {st['sd']:.3f}"


def _ci(st: dict[str, float]) -> str:
    return "—" if not st["n"] else f"{st['mean']:+.3f} [{st['lo']:+.3f}, {st['hi']:+.3f}]"


def _write_summary_md(summary: dict[str, Any], path: Path) -> None:
    meta = summary["meta"]
    lines = [
        "# Specialisation sweep under the implicit default (2026-10-27)",
        "",
        f"`{meta['script']}` at `{meta['git_commit']}`, {meta['runs']} runs, "
        f"{meta['timestamp_utc']}. Entropic JKO step of F, coupling and Turing terms "
        "implicit (ERRATUM §15), reg = h², reinjection; problem, points and seeds of PR #46. "
        "Gap in bits, mean ± sd over seeds.",
        "",
        "| α | γ | gap (implicit) | S vs α=0 | gap (entropic explicit, #46) "
        "| legacy rigorous, range over β | fast, range over β |",
        "|---|---|---|---|---|---|---|",
    ]
    seen: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for r in summary["rows_main"]:
        seen.setdefault((r["alpha"], r["gamma"]), []).append(r)
    for (a, g), rows in sorted(seen.items()):
        leg = [r["legacy_rigorous_gap"]["mean"] for r in rows if r["legacy_rigorous_gap"]["n"]]
        fst = [r["fast_gap"]["mean"] for r in rows if r["fast_gap"]["n"]]

        def rng(v: list[float]) -> str:
            return f"{min(v):.3f}–{max(v):.3f}" if v else "—"

        flag = " (γ ≥ 10: #46 value not interpretable, §14)" if g >= GAMMA_NOT_INTERPRETABLE else ""
        lines.append(
            f"| {a:g} | {g:g} | {_f(rows[0]['entropic_gap'])} | {_f(rows[0]['entropic_S'])} "
            f"| {_f(rows[0]['explicit_gap_pr46'])}{flag} | {rng(leg)} | {rng(fst)} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_excess_md(exc: dict[str, Any], path: Path) -> None:
    lines = [
        "# Excess over the Gibbs equilibrium, implicit default (2026-10-27)",
        "",
        "Excess = gap − gap_Gibbs (bits), gap_Gibbs of ρ_i ∝ prior_i e^(−V_i); mean "
        "[95 % bootstrap CI over seeds]; `ctrl` subtracts the paired α = 0 gap.",
        "",
        "| α | γ | gap | gap_Gibbs | excess (implicit) | verdict | excess − α0 control "
        "| excess (entropic explicit, #46) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    old = {(r["alpha"], r["gamma"]): r for r in exc["explicit_pr46"]}
    for r in exc["implicit"]:
        if r["alpha"] == 0.0:
            continue
        o = old.get((r["alpha"], r["gamma"]))
        lines.append(
            f"| {r['alpha']:g} | {r['gamma']:g} | {r['gap']['mean']:.3f} | {r['gap_gibbs']:.3f} "
            f"| {_ci(r['excess'])} | {r['verdict']} | {_ci(r['excess_ctrl'])} "
            f"| {_ci(o['excess']) if o else '—'} |"
        )
    n_pos = sum(1 for r in exc["implicit"] if r["alpha"] > 0 and r["verdict"] == "excess > 0")
    n_all = sum(1 for r in exc["implicit"] if r["alpha"] > 0)
    lines += ["", f"Implicit: {n_pos} / {n_all} couples (α > 0) with excess > 0.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyse"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-slow", type=int, default=sweep.base.N_SLOW)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run_jobs(
            sweep.jobs(SEEDS), output_paths(args.out_dir)["partial"], args.workers, args.n_slow
        )
    else:
        print(json.dumps(analyse(args.out_dir)["meta"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
