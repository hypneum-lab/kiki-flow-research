"""Excess over the minimiser of the full T2 energy, strong-Turing grid (2026-10-27).

Follow-up of ``docs/prereg/2026-10-26-turing-strong-excess.md`` (PR #48), whose
full-energy reference (a damped fixed point) did not converge for gamma >= 20.

The PI asked for the entropic JKO step iterated to stationarity. The pilot
probe (``results/full_energy_excess_2026_10_26/pilot_jko_probe.log``) shows it
cannot serve here: from the Gibbs state at (5, 30), F *increases* step after
step and the iterates oscillate with period 2 and a one-cell W2 move (h = 0.5,
reg 1e-3 and 1e-2), because the non-smooth Turing term is linearised at
rho_tau (explicit) and, on the 16-point grid, a small Sinkhorn reg pins mass
to the grid while a large one biases the fixed point by O(reg / 2h). The
minimiser of F does not depend on the metric, so the reference is computed by
a monotone method instead. A first mirror descent with the finite-difference
gradient of ``T2FreeEnergy`` stalled before stationarity at gamma = 30 (the
Turing term sum_{i<k} sum_x |grad rho_i grad rho_k| is not smooth), so:

* the Turing term is smoothed, |u| -> sqrt(u^2 + delta^2) - delta, with its
  exact gradient (``np.gradient`` is a fixed linear operator D), and delta is
  continued over ``DELTAS`` (1e-2 ... 1e-6), each stage warm-started from the
  previous one; F_delta -> F as delta -> 0 (|F_delta - F| <= n_terms * delta);
* each stage: mirror descent (exponentiated gradient, KL geometry) on F_delta
  over the product of simplices, rho_i <- rho_i exp(-t g_i) / Z_i, exact
  gradient (potential, KL to the prior, 2 J_sym rho, smoothed Turing); a step
  is accepted only if F_delta decreases (t halves on rejection, grows by 1.25
  on acceptance);
* stop, per stage: for ``PATIENCE`` consecutive accepted steps, decrease of
  F_delta < ``F_TOL`` and max over species of the exact W2 between successive
  iterates < ``W2_TOL``; at most ``MAX_STEPS`` steps; a stage also converges
  when the line search stalls at machine precision with the first-order
  stationarity residual on the simplex, max_i max_x rho_i(x) |g_i(x) -
  <g_i, rho_i>|, below ``RES_TOL`` (added after the pilot probe: the descent
  reached the same F to 1e-8 from every initialisation but stopped on the
  floating-point floor before five "calm" steps); an initialisation converges
  iff every stage
  converges and the gap moves by less than ``DELTA_STABILITY`` between the
  last two deltas;
* initialisations: the Gibbs state of the potentials, the uniform state and
  Dirichlet(1) draws (``init_seeds``); the reference is the converged
  initialisation of lowest F; the dispersion of the converged finals (F range,
  gap range, max pairwise W2) is reported; distinct minima are flagged when the
  converged finals differ by more than ``MULTI_W2`` in W2;
* a cell's reference "converges" iff at least one initialisation converges.

Subcommands: ``pilot`` (convergence check with two pilot init seeds on the 15
cells plus validation cells where the old fixed point converged, before the
preregistration is frozen), ``reference`` (the preregistered reference),
``run`` (the simulations), ``analyse`` (primary statistic, bootstrap, Holm).
Outputs under ``results/full_energy_excess_2026_10_26/``.
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
import ot

from kiki_flow_core.state import FlowState


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
turing = _load("experiments_turing_strong_excess_2026_10_26", "turing_strong_excess_2026_10_26.py")

GAMMAS = [20.0, 30.0, 50.0, 70.0, 100.0]
ALPHAS = [2.0, 5.0, 10.0]
SEEDS = list(range(200, 210))
FORBIDDEN_SEEDS = set(range(10)) | set(range(100, 110))
PILOT_INIT_SEEDS = [[1001], [1002]]
REF_INIT_SEEDS = [300, 301, 302]
VALIDATION_CELLS = [(5.0, 10.0), (10.0, 0.0), (8.0, 5.0)]  # old fixed point converged there
T0 = 0.5
T_MIN = 1e-14
W2_TOL = 1e-6
F_TOL = 1e-10
PATIENCE = 5
MAX_STEPS = 20000
MULTI_W2 = 1e-3
DELTAS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
DELTA_STABILITY = 1e-3
RES_TOL = 1e-6
FAMILY_LEVEL = 0.05
BOOT_SEED = 20261027
REG_SIM = sweep.MAIN_REG
GRID = excess.GRID
RHO_FLOOR = 1e-300
OUT_DIR = Path("results/full_energy_excess_2026_10_26")

Cell = tuple[float, float]


def output_paths() -> dict[str, Path]:
    return {
        "pilot": OUT_DIR / "pilot.json",
        "pilot_probe": OUT_DIR / "pilot_jko_probe.log",
        "reference": OUT_DIR / "reference.json",
        "partial": OUT_DIR / "partial.jsonl",
        "excess": OUT_DIR / "excess.json",
        "summary": OUT_DIR / "summary.md",
    }


def cells() -> list[Cell]:
    return [(a, g) for g in GAMMAS for a in ALPHAS]


def _state(rho: dict[str, np.ndarray]) -> FlowState:
    return FlowState(
        rho=rho, P_theta=np.zeros(4), mu_curr=np.array([1.0]), tau=0, metadata={"track_id": "T2"}
    )


def _support() -> np.ndarray:
    return np.linspace(-2, 2, GRID).reshape(-1, 1)


def _cost() -> np.ndarray:
    x = _support()[:, 0]
    return (x[:, None] - x[None, :]) ** 2


def inits(f: Any, init_seeds: list[int]) -> dict[str, dict[str, np.ndarray]]:
    names = f.species.species_names()
    out = {
        "gibbs": excess.gibbs_potentials(f),
        "uniform": {n: np.full(GRID, 1.0 / GRID) for n in names},
    }
    for s in init_seeds:
        rng = np.random.default_rng(s)
        out[f"dirichlet{s}"] = {n: rng.dirichlet(np.ones(GRID)) for n in names}
    return out


def w2_max(a: dict[str, np.ndarray], b: dict[str, np.ndarray], cost: np.ndarray) -> float:
    return max(
        float(np.sqrt(max(ot.emd2(a[n] / a[n].sum(), b[n] / b[n].sum(), cost), 0.0))) for n in a
    )


def gradient_operator(n: int = GRID) -> np.ndarray:
    """Matrix D with D @ v == np.gradient(v) (unit spacing, one-sided ends)."""
    return np.stack([np.gradient(e) for e in np.eye(n)], axis=1)


D_OP = gradient_operator()


class SmoothedT2:
    """T2 energy with the Turing |.| smoothed by sqrt(u^2 + delta^2) - delta, exact gradient."""

    def __init__(self, f: Any, delta: float) -> None:
        self.f = f
        self.delta = delta
        self.names = f.species.species_names()
        self.j_sym = f._J_sym
        self.s = f.turing_strength

    def value(self, state: FlowState) -> float:
        rhos = [state.rho[n] for n in self.names]
        total = 0.0
        for n, r in zip(self.names, rhos, strict=True):
            safe = np.clip(r, RHO_FLOOR, None)
            prior = np.clip(self.f.prior[n], RHO_FLOOR, None)
            total += float(np.dot(r, self.f.potentials[n]) + np.sum(safe * np.log(safe / prior)))
        for i, ri in enumerate(rhos):
            for k, rk in enumerate(rhos):
                total += float(self.j_sym[i, k] * np.dot(ri, rk))
        if self.s > 0.0:
            gs = [D_OP @ r for r in rhos]
            for i in range(len(gs)):
                for k in range(i + 1, len(gs)):
                    u = gs[i] * gs[k]
                    total += self.s * float(np.sum(np.sqrt(u * u + self.delta**2) - self.delta))
        return total

    def grad(self, state: FlowState) -> dict[str, np.ndarray]:
        rhos = [state.rho[n] for n in self.names]
        gs = [D_OP @ r for r in rhos]
        out = {}
        for i, n in enumerate(self.names):
            safe = np.clip(rhos[i], RHO_FLOOR, None)
            prior = np.clip(self.f.prior[n], RHO_FLOOR, None)
            g = self.f.potentials[n] + np.log(safe / prior) + 1.0
            g = g + 2.0 * sum(self.j_sym[i, k] * rhos[k] for k in range(len(rhos)))
            if self.s > 0.0:
                acc = np.zeros(GRID)
                for k in range(len(rhos)):
                    if k != i:
                        u = gs[i] * gs[k]
                        acc += u * gs[k] / np.sqrt(u * u + self.delta**2)
                g = g + self.s * (D_OP.T @ acc)
            out[n] = g
        return out


def descend_stage(energy: SmoothedT2, state: FlowState) -> tuple[FlowState, dict[str, Any]]:
    """Monotone mirror descent on F_delta from ``state`` until stationary (or MAX_STEPS)."""
    cost = _cost()
    f_cur = energy.value(state)
    t = T0
    calm = steps = rejected = 0
    w2_last = df_last = float("nan")
    tic = time.perf_counter()
    while steps < MAX_STEPS and calm < PATIENCE and t >= T_MIN:
        g = energy.grad(state)
        accepted = False
        while t >= T_MIN:
            new = {}
            for n, r in state.rho.items():
                logr = np.log(r) - t * (g[n] - g[n].min())
                w = np.exp(logr - logr.max())
                new[n] = np.clip(w / w.sum(), RHO_FLOOR, None)
            cand = _state(new)
            f_new = energy.value(cand)
            if f_new < f_cur:
                accepted = True
                break
            t *= 0.5
            rejected += 1
        if not accepted:
            break
        steps += 1
        w2_last = w2_max(cand.rho, state.rho, cost)
        df_last = f_cur - f_new
        calm = calm + 1 if (w2_last < W2_TOL and df_last < F_TOL) else 0
        state, f_cur = cand, f_new
        t *= 1.25
    g_end = energy.grad(state)
    residual = max(
        float(np.max(state.rho[n] * np.abs(g_end[n] - float(np.dot(g_end[n], state.rho[n])))))
        for n in state.rho
    )
    calm_ok = calm >= PATIENCE
    return state, {
        "delta": energy.delta,
        "converged": calm_ok or (t < T_MIN and residual < RES_TOL),
        "calm_criterion": calm_ok,
        "residual": residual,
        "stalled": t < T_MIN,
        "steps": steps,
        "rejected": rejected,
        "last_w2": w2_last,
        "last_dF": df_last,
        "F_delta": f_cur,
        "gap": excess.gap_of(state.rho),
        "seconds": time.perf_counter() - tic,
    }


def descend(f: Any, rho0: dict[str, np.ndarray]) -> dict[str, Any]:
    """Delta continuation of the smoothed mirror descent; reference = last delta."""
    clipped = {n: np.clip(np.asarray(r, float), RHO_FLOOR, None) for n, r in rho0.items()}
    state = _state({n: r / r.sum() for n, r in clipped.items()})
    stages = []
    for delta in DELTAS:
        state, info = descend_stage(SmoothedT2(f, delta), state)
        stages.append(info)
    stable = abs(stages[-1]["gap"] - stages[-2]["gap"]) < DELTA_STABILITY
    return {
        "converged": all(st["converged"] for st in stages) and stable,
        "stages_converged": [st["converged"] for st in stages],
        "gap_stable": stable,
        "steps": sum(st["steps"] for st in stages),
        "stages": stages,
        "F": f.value(state),
        "gap": excess.gap_of(state.rho),
        "rho": {n: [float(v) for v in r] for n, r in state.rho.items()},
        "seconds": sum(st["seconds"] for st in stages),
    }


def reference_cell(cell: Cell, init_seeds: list[int]) -> dict[str, Any]:
    a, g = cell
    f = excess.problem(a, g)
    cost = _cost()
    runs = {k: descend(f, r) for k, r in inits(f, init_seeds).items()}
    conv = {k: v for k, v in runs.items() if v["converged"]}
    out: dict[str, Any] = {
        "alpha": a,
        "gamma": g,
        "init_seeds": init_seeds,
        "runs": {k: {kk: vv for kk, vv in v.items() if kk != "rho"} for k, v in runs.items()},
        "converged": bool(conv),
        "gap_gibbs": excess.gap_of(excess.gibbs_potentials(f)),
    }
    if conv:
        best = min(conv, key=lambda k: conv[k]["F"])
        finals = {k: {n: np.asarray(r) for n, r in v["rho"].items()} for k, v in conv.items()}
        keys = list(finals)
        pair_w2 = [
            w2_max(finals[keys[i]], finals[keys[j]], cost)
            for i in range(len(keys))
            for j in range(i + 1, len(keys))
        ]
        out.update(
            {
                "best_init": best,
                "F_min": conv[best]["F"],
                "gap_fmin": conv[best]["gap"],
                "rho_fmin": conv[best]["rho"],
                "n_converged": len(conv),
                "F_range": float(max(v["F"] for v in conv.values()) - conv[best]["F"]),
                "gap_range": float(
                    max(v["gap"] for v in conv.values()) - min(v["gap"] for v in conv.values())
                ),
                "max_pair_w2": float(max(pair_w2)) if pair_w2 else 0.0,
                "multiple_minima": bool(pair_w2 and max(pair_w2) > MULTI_W2),
            }
        )
    return out


def _pool_map(fn: Any, args: list[tuple[Any, ...]], workers: int) -> list[Any]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, *a): i for i, a in enumerate(args)}
        out: list[Any] = [None] * len(args)
        for fut in as_completed(futs):
            r = fut.result()
            out[futs[fut]] = r
            print(
                f"ref alpha={r['alpha']} gamma={r['gamma']} seeds={r['init_seeds']} "
                f"converged={r['converged']} gap_fmin={r.get('gap_fmin', float('nan')):.4f} "
                f"steps={[v['steps'] for v in r['runs'].values()]}",
                flush=True,
            )
        return out


def pilot(workers: int) -> dict[str, Any]:
    allc = cells() + VALIDATION_CELLS
    jobs: list[tuple[Any, ...]] = [(c, s) for s in PILOT_INIT_SEEDS for c in allc]
    res = _pool_map(reference_cell, jobs, workers)
    m = len(allc)
    a, b = res[:m], res[m:]
    rows = []
    for ra, rb in zip(a, b, strict=True):
        ok = ra["converged"] and rb["converged"]
        rows.append(
            {
                "alpha": ra["alpha"],
                "gamma": ra["gamma"],
                "validation": (ra["alpha"], ra["gamma"]) in VALIDATION_CELLS,
                "converged_pilot_a": ra["converged"],
                "converged_pilot_b": rb["converged"],
                "n_converged_a": ra.get("n_converged", 0),
                "n_converged_b": rb.get("n_converged", 0),
                "F_min_a": ra.get("F_min"),
                "F_min_b": rb.get("F_min"),
                "gap_fmin_a": ra.get("gap_fmin"),
                "gap_fmin_b": rb.get("gap_fmin"),
                "gap_fmin_diff": abs(ra["gap_fmin"] - rb["gap_fmin"]) if ok else None,
                "multiple_minima_a": ra.get("multiple_minima"),
                "multiple_minima_b": rb.get("multiple_minima"),
                "F_range_a": ra.get("F_range"),
                "gap_range_a": ra.get("gap_range"),
                "max_steps": max(v["steps"] for v in [*ra["runs"].values(), *rb["runs"].values()]),
            }
        )
    return {"rows": rows, "details": {"pilot_a": a, "pilot_b": b}}


def reference(workers: int) -> list[dict[str, Any]]:
    return _pool_map(reference_cell, [(c, REF_INIT_SEEDS) for c in cells()], workers)


def run(workers: int) -> None:
    path = output_paths()["partial"]
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["alpha"], r["gamma"], r["seed"]))
    todo = [(a, g, s) for a, g in cells() for s in SEEDS if (a, g, s) not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a") as fh:
        futs = [pool.submit(sweep.run_couple, a, g, s, REG_SIM) for a, g, s in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(todo)}] {r['alpha']} {r['gamma']} {r['seed']} {r['gap_bits']:.3f}",
                flush=True,
            )


def analyse(records: list[dict[str, Any]], refs: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["alpha"], r["gamma"], r["seed"]): r["gap_bits"] for r in records}
    ref = {(r["alpha"], r["gamma"]): r for r in refs}
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for a, g in cells():
        rf = ref[(a, g)]
        row: dict[str, Any] = {
            "alpha": a,
            "gamma": g,
            "gap": float(np.mean([by[(a, g, s)] for s in SEEDS])),
            "ref_converged": rf["converged"],
            "gap_fmin": rf.get("gap_fmin"),
            "gap_gibbs": rf["gap_gibbs"],
            "multiple_minima": rf.get("multiple_minima"),
            "F_range": rf.get("F_range"),
            "gap_range": rf.get("gap_range"),
        }
        if rf["converged"]:
            row["primary"] = turing.bootstrap([by[(a, g, s)] - rf["gap_fmin"] for s in SEEDS], rng)
        else:
            row["primary"] = None
            row["verdict"] = "excluded (reference did not converge)"
        rows.append(row)
    incl = [r for r in rows if r["primary"] is not None]
    pvals = [r["primary"]["p_one_sided"] for r in incl]
    adjusted, reject = turing.holm(pvals) if incl else ([], [])
    for rank, idx in enumerate(np.argsort(pvals)):
        r = incl[idx]
        level = FAMILY_LEVEL / (len(incl) - rank)
        r["holm_lower_bound"] = float(np.quantile(r["primary"]["_means"], level))
    for r, p_adj, rej in zip(incl, adjusted, reject, strict=True):
        r["p_holm"] = p_adj
        r["verdict"] = "excess > 0" if rej else "inconclusive"
        r["primary"].pop("_means", None)
    return {
        "meta": {
            "script": "experiments/full_energy_excess_2026_10_26.py",
            "prereg": "docs/prereg/2026-10-27-full-energy-excess.md",
            "prereg_tag": "prereg-full-energy-20261027",
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
            "n_included": len(incl),
        },
        "rows": rows,
        "n_positive": int(sum(reject)),
    }


def _fmt(st: dict[str, float]) -> str:
    return f"{st['mean']:+.3f} [{st['ci95_lo']:+.3f}, {st['ci95_hi']:+.3f}]"


def write_summary(res: dict[str, Any], path: Path) -> None:
    meta = res["meta"]
    lines = [
        "# Excess over the full-energy minimiser, strong-Turing grid (2026-10-27)",
        "",
        f"Preregistration `{meta['prereg']}` (tag `{meta['prereg_tag']}`); driver at "
        f"`{meta['git_commit']}`, {meta['n_records']} runs, seeds 200-209, "
        f"{meta['n_included']} cells included.",
        "",
        "| alpha | gamma | gap | gap_Fmin | gap_Gibbs | reference | primary excess | p "
        "| p Holm | Holm LB | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        if r["primary"] is None:
            lines.append(
                f"| {r['alpha']:g} | {r['gamma']:g} | {r['gap']:.3f} | — | {r['gap_gibbs']:.3f} "
                f"| not converged | — | — | — | — | {r['verdict']} |"
            )
            continue
        ref = "multiple minima" if r["multiple_minima"] else "converged"
        lines.append(
            f"| {r['alpha']:g} | {r['gamma']:g} | {r['gap']:.3f} | {r['gap_fmin']:.3f} "
            f"| {r['gap_gibbs']:.3f} | {ref} | {_fmt(r['primary'])} "
            f"| {r['primary']['p_one_sided']:.2g} | {r['p_holm']:.2g} "
            f"| {r['holm_lower_bound']:+.3f} | {r['verdict']} |"
        )
    lines += [
        "",
        f"Cells with excess > 0 after Holm: {res['n_positive']} / {meta['n_included']} included.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "reference", "run", "analyse"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = output_paths()
    if args.cmd == "pilot":
        paths["pilot"].write_text(json.dumps(pilot(args.workers), indent=1), encoding="utf-8")
    elif args.cmd == "reference":
        paths["reference"].write_text(
            json.dumps(reference(args.workers), indent=1), encoding="utf-8"
        )
    elif args.cmd == "run":
        run(args.workers)
    else:
        records = [json.loads(x) for x in paths["partial"].read_text().splitlines()]
        refs = json.loads(paths["reference"].read_text())
        res = analyse(records, refs)
        paths["excess"].write_text(json.dumps(res, indent=1), encoding="utf-8")
        write_summary(res, paths["summary"])
        print(paths["summary"].read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
