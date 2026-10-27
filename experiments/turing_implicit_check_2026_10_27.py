"""Verification of the implicit treatment of the Turing term (2026-10-27).

PI decision of 2026-10-27 06:48 (ERRATUM.md section 14): the default slow step
(entropic JKO step of F, non-separable terms linearised at rho_tau) is not
dissipative for gamma >= 10. ``JKOStep(nonseparable="implicit")`` iterates the
linearisation point to the step's own output and halves the step if F would
increase. This driver checks it, without changing any default.

* ``dissipation``: for gamma in {0, 5, 10, 20, 30, 50, 70, 100}, alpha in
  {2, 5, 10}, h in {0.05, 0.025} and two seeds (initial states drawn from
  Dirichlet(1) per species with seeds 500 and 501), ``N_STEPS`` slow steps of
  ``FullJKOSolver(nonseparable="implicit")`` on the grid (problem of
  ``experiments/rigorous_specialisation_2026_10_24.py``, reg = h^2), counting
  the steps where F increases (by more than ``INCREASE_TOL``). The explicit
  step is run on the same states for comparison. Criterion: 0 increases.
* ``unchanged``: for gamma <= 5 the implicit and explicit steps must agree
  within tolerance: (a) per-step max TV between the two outputs from the same
  input on the grid (gamma in {0, 1, 5}, alpha in {2, 5, 10}); (b) the
  specialisation gap at the paper's key points (alpha, gamma) = (10, 0) and
  (8, 5), seeds 0-4, full multiscale runs (500 particles, 200 fast x 30 slow,
  reinjection) with the implicit step, against the explicit runs of PR #46
  (``results/jko_entropic_sweep_2026_10_25/records.json``, same seeds).
  Tolerances: max TV <= ``TV_TOL``; |mean gap difference| <= ``GAP_TOL`` bits.

Outputs: ``results/turing_implicit_check_2026_10_27/{dissipation.jsonl,
unchanged.json, summary.json, summary.md}``. electron-server only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from kiki_flow_core.state import FlowState
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


excess = _load("experiments_excess_over_gibbs_2026_10_25", "excess_over_gibbs_2026_10_25.py")
sweep = _load("experiments_jko_entropic_sweep_2026_10_25", "jko_entropic_sweep_2026_10_25.py")

GAMMAS = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0]
ALPHAS = [2.0, 5.0, 10.0]
HS = [0.05, 0.025]
INIT_SEEDS = [500, 501]
N_STEPS = 10
INCREASE_TOL = 1e-12
PROX_TOL = 1e-12
TV_GAMMAS = [0.0, 1.0, 5.0]
TV_TOL = 1e-3
GAP_TOL = 0.05
KEY_POINTS = [(10.0, 0.0), (8.0, 5.0)]
KEY_SEEDS = [0, 1, 2, 3, 4]
GRID = excess.GRID
OUT_DIR = Path("results/turing_implicit_check_2026_10_27")
REF_SWEEP = Path("results/jko_entropic_sweep_2026_10_25/records.json")


def output_paths() -> dict[str, Path]:
    return {
        "dissipation": OUT_DIR / "dissipation.jsonl",
        "unchanged": OUT_DIR / "unchanged.json",
        "summary": OUT_DIR / "summary.json",
        "summary_md": OUT_DIR / "summary.md",
    }


def _state(rho: dict[str, np.ndarray]) -> FlowState:
    return FlowState(
        rho=rho, P_theta=np.zeros(4), mu_curr=np.array([1.0]), tau=0, metadata={"track_id": "T2"}
    )


def _support() -> np.ndarray:
    return np.linspace(-2, 2, GRID).reshape(-1, 1)


def solver(f: Any, h: float, mode: str) -> FullJKOSolver:
    return FullJKOSolver(
        f_functional=f, h=h, support=_support(), nonseparable=mode, prox_tol=PROX_TOL
    )


def initial_state(f: Any, seed: int) -> FlowState:
    rng = np.random.default_rng(seed)
    return _state({n: rng.dirichlet(np.ones(GRID)) for n in f.species.species_names()})


def dissipation_job(alpha: float, gamma: float, h: float, seed: int) -> dict[str, Any]:
    f = excess.problem(alpha, gamma)
    out: dict[str, Any] = {"alpha": alpha, "gamma": gamma, "h": h, "seed": seed}
    for mode in ("implicit", "explicit"):
        jko = solver(f, h, mode)
        state = initial_state(f, seed)
        values = [f.value(state)]
        halvings = []
        tic = time.perf_counter()
        for _ in range(N_STEPS):
            state = jko.step(state)
            values.append(f.value(state))
            halvings.append(jko.last_halvings)
        rises = [b - a for a, b in zip(values, values[1:], strict=False)]
        out[mode] = {
            "F": values,
            "n_increase": int(sum(1 for d in rises if d > INCREASE_TOL)),
            "max_increase": float(max(rises)),
            "halvings": halvings,
            "seconds": time.perf_counter() - tic,
        }
    return out


def tv_job(alpha: float, gamma: float) -> dict[str, Any]:
    f = excess.problem(alpha, gamma)
    state = initial_state(f, INIT_SEEDS[0])
    imp, exp_ = solver(f, 0.05, "implicit"), solver(f, 0.05, "explicit")
    tvs = []
    for _ in range(N_STEPS):
        a, b = imp.step(state), exp_.step(state)
        tvs.append(max(0.5 * float(np.abs(a.rho[n] - b.rho[n]).sum()) for n in a.rho))
        state = b
    return {"alpha": alpha, "gamma": gamma, "max_tv": float(max(tvs)), "tv": tvs}


def key_job(alpha: float, gamma: float, seed: int) -> dict[str, Any]:
    import mlx.core as mx  # noqa: PLC0415

    from kiki_flow_core.track2_paper.mlx_particle_simulator import (  # noqa: PLC0415
        MLXParticleSimulator,
    )
    from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop  # noqa: PLC0415

    with mx.stream(mx.cpu):
        species, support, f = sweep.base.problem(alpha, gamma)
        jko = FullJKOSolver(
            f_functional=f, h=sweep.H, support=support, nonseparable="implicit", prox_tol=PROX_TOL
        )
        sim = MLXParticleSimulator(
            species=species, n_particles=sweep.base.N_PARTICLES, latent_dim=2, seed=seed
        )
        loop = MultiscaleLoop(
            sim=sim, jko=jko, n_fast=sweep.base.N_FAST, n_slow=sweep.base.N_SLOW, support=support
        )
        final = loop.run(seed=seed)["trajectory"][-1]
    mean_h = float(np.mean([sweep.entropy_bits(r) for r in final.rho.values()]))
    return {
        "alpha": alpha,
        "gamma": gamma,
        "seed": seed,
        "gap_bits": sweep.MAX_ENTROPY_BITS - mean_h,
    }


def run_dissipation(workers: int) -> None:
    path = output_paths()["dissipation"]
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["alpha"], r["gamma"], r["h"], r["seed"]))
    jobs = [
        (a, g, h, s)
        for g in reversed(GAMMAS)
        for a in ALPHAS
        for h in HS
        for s in INIT_SEEDS
        if (a, g, h, s) not in done
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a") as fh:
        futs = [pool.submit(dissipation_job, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(jobs)}] a={r['alpha']} g={r['gamma']} h={r['h']} s={r['seed']} "
                f"implicit inc={r['implicit']['n_increase']} "
                f"({r['implicit']['seconds']:.0f}s, halvings {max(r['implicit']['halvings'])}) "
                f"explicit inc={r['explicit']['n_increase']}",
                flush=True,
            )


def run_unchanged(workers: int) -> None:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tv_f = [pool.submit(tv_job, a, g) for g in TV_GAMMAS for a in ALPHAS]
        key_f = [pool.submit(key_job, a, g, s) for a, g in KEY_POINTS for s in KEY_SEEDS]
        tv = [f.result() for f in tv_f]
        key = [f.result() for f in key_f]
    ref = json.loads(REF_SWEEP.read_text())["results"]
    by_ref = {
        (r["alpha"], r["gamma"], r["seed"]): r["gap_bits"]
        for r in ref
        if r["reg"] == sweep.MAIN_REG
    }
    rows = []
    for a, g in KEY_POINTS:
        new = [r["gap_bits"] for r in key if (r["alpha"], r["gamma"]) == (a, g)]
        old = [by_ref[(a, g, s)] for s in KEY_SEEDS]
        diff = [n - o for n, o in zip(new, old, strict=True)]
        rows.append(
            {
                "alpha": a,
                "gamma": g,
                "gap_implicit": float(np.mean(new)),
                "gap_explicit_pr46": float(np.mean(old)),
                "diff_mean": float(np.mean(diff)),
                "diff_sd": float(np.std(diff, ddof=1)),
                "within_tol": bool(abs(np.mean(diff)) <= GAP_TOL),
            }
        )
    out = {"tv": tv, "key_points": rows, "key_runs": key, "tv_tol": TV_TOL, "gap_tol": GAP_TOL}
    output_paths()["unchanged"].write_text(json.dumps(out, indent=1), encoding="utf-8")


def summarise() -> dict[str, Any]:
    recs = [json.loads(x) for x in output_paths()["dissipation"].read_text().splitlines()]
    unchanged = json.loads(output_paths()["unchanged"].read_text())
    by_gamma = {}
    for g in GAMMAS:
        sel = [r for r in recs if r["gamma"] == g]
        by_gamma[str(g)] = {
            "configs": len(sel),
            "implicit_increases": int(sum(r["implicit"]["n_increase"] for r in sel)),
            "explicit_increases": int(sum(r["explicit"]["n_increase"] for r in sel)),
            "steps": len(sel) * N_STEPS,
            "max_halvings": int(max(max(r["implicit"]["halvings"]) for r in sel)) if sel else None,
            "implicit_seconds_per_step": float(
                np.mean([r["implicit"]["seconds"] for r in sel]) / N_STEPS
            )
            if sel
            else None,
        }
    total_imp = sum(v["implicit_increases"] for v in by_gamma.values())
    tv_ok = all(t["max_tv"] <= TV_TOL for t in unchanged["tv"])
    gap_ok = all(r["within_tol"] for r in unchanged["key_points"])
    summary = {
        "by_gamma": by_gamma,
        "implicit_increases_total": total_imp,
        "n_records": len(recs),
        "dissipation_pass": total_imp == 0 and len(recs) == len(GAMMAS) * len(ALPHAS) * len(HS) * 2,
        "tv_max": float(max(t["max_tv"] for t in unchanged["tv"])),
        "tv_pass": tv_ok,
        "key_points": unchanged["key_points"],
        "gap_pass": gap_ok,
    }
    output_paths()["summary"].write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = [
        "# Implicit Turing treatment: dissipation and gamma <= 5 checks (2026-10-27)",
        "",
        f"{len(recs)} configurations x {N_STEPS} slow steps on the grid (alpha in {ALPHAS}, "
        f"h in {HS}, initial states Dirichlet seeds {INIT_SEEDS}), reg = h^2, Sinkhorn "
        f"early stop {PROX_TOL}. Steps with F increasing by more than {INCREASE_TOL}:",
        "",
        "| gamma | steps | implicit: F increases | explicit: F increases | max halvings "
        "| implicit s/step |",
        "|---|---|---|---|---|---|",
    ]
    for g, v in by_gamma.items():
        lines.append(
            f"| {float(g):g} | {v['steps']} | {v['implicit_increases']} "
            f"| {v['explicit_increases']} | {v['max_halvings']} "
            f"| {v['implicit_seconds_per_step']:.1f} |"
        )
    lines += [
        "",
        f"Dissipation criterion (0 increases, all configurations): "
        f"{'PASS' if summary['dissipation_pass'] else 'FAIL'}.",
        "",
        f"gamma <= 5, per-step max TV implicit vs explicit (same input): {summary['tv_max']:.2e} "
        f"(tolerance {TV_TOL}) — {'PASS' if tv_ok else 'FAIL'}.",
        "",
        "| key point (alpha, gamma) | gap implicit | gap explicit (PR #46) | diff mean ± sd "
        "| within ±0.05 |",
        "|---|---|---|---|---|",
    ]
    for r in unchanged["key_points"]:
        lines.append(
            f"| ({r['alpha']:g}, {r['gamma']:g}) | {r['gap_implicit']:.3f} "
            f"| {r['gap_explicit_pr46']:.3f} | {r['diff_mean']:+.3f} ± {r['diff_sd']:.3f} "
            f"| {'yes' if r['within_tol'] else 'no'} |"
        )
    lines.append("")
    output_paths()["summary_md"].write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dissipation", "unchanged", "summary"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "dissipation":
        run_dissipation(args.workers)
    elif args.cmd == "unchanged":
        run_unchanged(args.workers)
    else:
        print(json.dumps(summarise(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
