"""Lie and Strang splitting of the entropic JKO step of T2 (preregistered 2026-10-30).

Preregistration: ``docs/prereg/2026-10-30-jko-splitting-lie-strang.md`` (the
decision rule, fixed before any confirmatory record exists). Schemes:
``kiki_flow_core/track2_paper/jko_splitting.py``.

Problems and inputs are those of the M5 ``certify`` check
(``experiments/implicit_jko_certified_validation_2026_10_27.py``, imported by
path so that nothing is re-implemented): grid 16 on [-2, 2], T2 energy with
Turing smoothing 1e-5, potentials alpha (x + 1.5), alpha (x - 1.5), alpha |x|,
-alpha |x|, Turing strength 0.1 gamma, Dirichlet(1) inputs drawn per species
from ``default_rng(seed)``.

Subcommands:

* ``pilot``: timing only, on seed ``PILOT_SEED`` (not a confirmatory seed),
  one step per (cell, scheme, h). Records status, wall time and iteration
  counts, and the wall time of the Q_h brackets; it stores no density, no
  F or Q_h value and computes no error.
* ``confirm``: the preregistered run. Refuses to start without ``--go`` (the
  reference of the PI's GO, recorded in every record). One job per (cell,
  scheme, h) trajectory; every step record stores the output densities, the
  certificates, F and, at the M5 step sizes, certified brackets of Q_h.
* ``analyse``: the preregistered analysis of the ``confirm`` records
  (metrics, gates, verdicts), written to ``summary.json``.

Each step runs under a wall-clock cap (``signal.alarm``, POSIX); a step that
hits it is recorded with ``status = "timeout"`` (not certified). Records are
JSONL, written as jobs complete, each with ``RUN_KEY`` (code identity and
settings); ``confirm`` resumes only from records with the same key.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sys
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from kiki_flow_core.master_equation import SOLVER_VERSION, ImplicitJKOError, JKOStep
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.jko_splitting import SPLIT_VERSION, SplitJKOStep
from kiki_flow_core.wasserstein_ops import SinkhornNotConvergedError

_M5_PATH = Path(__file__).resolve().parent / "implicit_jko_certified_validation_2026_10_27.py"
_spec = importlib.util.spec_from_file_location("m5_validation_driver", _M5_PATH)
assert _spec is not None and _spec.loader is not None
m5 = importlib.util.module_from_spec(_spec)
sys.modules["m5_validation_driver"] = m5
_spec.loader.exec_module(m5)

OUT_DIR = Path("results/jko_splitting_2026_10_30")
GRID = 16
GAMMAS = m5.GAMMAS  # [0, 5, 10, 20, 30, 50, 70, 100]
ALPHAS = m5.ALPHAS  # [2, 5, 10]
SEEDS = m5.INIT_SEEDS  # [500, 501]
PRIMARY_GAMMAS = (0.0, 5.0, 10.0, 20.0, 30.0)  # criterion stratum of the M5 reading rule
M5_HS = (0.05, 0.025)  # step sizes of the M5 grid: accuracy, certification, dissipation
LADDER = (0.1, 0.05, 0.025, 0.0125)  # step sizes of the order ladder
ORDER_HS = (0.1, 0.05, 0.025)  # s_Y(h) = TV(Y_h, Y_{h/2} o Y_{h/2}) at these h
SCHEMES = ("implicit", "lie", "strang")  # "implicit" = S_h, the certified full step
SPLITS = ("lie", "strang")
STEP_CAP_S = 1800  # wall-clock cap per step (all sub-steps), seconds
PILOT_SEED = 999
PILOT_CAP_S = 600
PILOT_CELLS = [(2.0, 0.0), (10.0, 5.0), (5.0, 30.0), (2.0, 70.0), (10.0, 100.0)]

# Decision constants (justified in the preregistration, section 6).
TV_FLOOR = 1e-5  # 10 x kkt_tol: below this a TV difference is not resolved by the solver
ORDER_THRESHOLD = 2.5  # midpoint of the competing slope predictions 2 and 3
CONTROL_BAND = (1.5, 2.5)  # control slope of S must be within 0.5 of its prediction 2
MIN_GATED_FRACTION = 0.5  # order verdict only if at least half of the cells pass the gate
MIN_MARGIN_FRACTION = 0.5  # C2 decidable only if the margin exists in half the configs
MAX_WORKERS = 4  # host rule (KXKM-AI, shared machine)


def n_steps(scheme: str, h: float) -> int:
    """Trajectory length of each (scheme, h) job, fixed by the preregistration."""
    if h == LADDER[0]:
        return 1
    if h == LADDER[-1]:
        return 2
    return 2 if scheme == "implicit" else 3


RUN_KEY: dict[str, Any] = {
    "solver_version": SOLVER_VERSION,
    "split_version": SPLIT_VERSION,
    "delta": m5.DELTA,
    "grid": GRID,
    "step_cap_s": STEP_CAP_S,
    "jko_defaults": m5._jko_defaults(),
    **m5._code_identity(),
}


class StepTimeoutError(RuntimeError):
    """A step hit the wall-clock cap."""


@contextmanager
def wall_cap(seconds: int) -> Iterator[None]:
    """Raise ``StepTimeoutError`` in the main thread after ``seconds`` (POSIX only)."""

    def _handler(signum: int, frame: Any) -> None:
        raise StepTimeoutError(f"step exceeded {seconds} s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def initial_state(alpha: float, gamma: float, seed: int) -> tuple[Any, FlowState]:
    f = m5.problem(GRID, alpha, gamma)
    rng = np.random.default_rng(seed)
    state = m5.state_of({n: rng.dirichlet(np.ones(GRID)) for n in f.species.species_names()})
    return f, state


def make_step(f: Any, scheme: str, h: float) -> JKOStep | SplitJKOStep:
    sup = m5.support(GRID)
    if scheme == "implicit":
        return JKOStep(f, h, sup, scheme="jko_entropic", on_failure="raise")
    return SplitJKOStep(f, h, sup, scheme, on_failure="raise")  # type: ignore[arg-type]


def _short_info(info: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly certificate summary of a JKOStep or SplitJKOStep ``last_info``."""
    keys = (
        "status",
        "outer_iterations",
        "prox_calls",
        "kkt_residual",
        "log_residual",
        "q_min",
        "inner_max_iterations",
        "inner_max_log_residual",
        "max_anchor_weight",
        "seconds",
        "Q_final",
        "Q_explicit",
    )
    out = {k: info[k] for k in keys if k in info}
    if "substeps" in info:
        out["substeps"] = [
            {"part": s["part"], "h_sub": s["h_sub"], **{k: s[k] for k in keys if k in s}}
            for s in info["substeps"]
        ]
        out["failed_part"] = info.get("failed_part")
    return out


def q_brackets(f: Any, h: float, state_in: FlowState, out: FlowState) -> dict[str, Any]:
    """Certified brackets of Q_h(out; in) and Q_h(in; in) for the full energy F."""
    jko = JKOStep(f, h, m5.support(GRID), scheme="jko_entropic")
    ref = {n: state_in.rho[n] for n in f.species.species_names()}
    q_out = {n: out.rho[n] for n in ref}
    try:
        lo_out, up_out = jko.objective_bounds(state_in, q_out, ref)
        lo_in, up_in = jko.objective_bounds(state_in, ref, ref)
    except SinkhornNotConvergedError as exc:
        return {"bracket": "not_certified", "error": str(exc)[:300]}
    return {"bracket": "ok", "Q_out": [lo_out, up_out], "Q_in": [lo_in, up_in]}


class Job(NamedTuple):
    """One trajectory of one scheme at one step size from one cell's input."""

    alpha: float
    gamma: float
    seed: int
    scheme: str
    h: float
    steps: int
    cap: int
    store: bool
    go: str | None


def trajectory_job(job: Job) -> dict[str, Any]:
    """Up to ``job.steps`` steps of one scheme from the cell's input; stops at a failure."""
    f, state = initial_state(job.alpha, job.gamma, job.seed)
    names = f.species.species_names()
    jko = make_step(f, job.scheme, job.h)
    records: list[dict[str, Any]] = []
    for k in range(job.steps):
        tic = time.perf_counter()
        new: FlowState | None
        try:
            with wall_cap(job.cap):
                new = jko.step(state)
            info = _short_info(jko.last_info)
        except ImplicitJKOError as exc:
            info, new = _short_info(exc.info), None
        except StepTimeoutError:
            info, new = {"status": "timeout"}, None
        info["wall"] = time.perf_counter() - tic
        info["k"] = k
        if new is not None and job.store:
            info["F_in"] = f.value(state)
            info["F_out"] = f.value(new)
            info["rho_out"] = {n: new.rho[n].tolist() for n in names}
            if job.h in M5_HS:
                info["dissipation"] = q_brackets(f, job.h, state, new)
        elif new is not None and job.h in M5_HS:  # pilot: time the brackets, keep no value
            tic_b = time.perf_counter()
            q_brackets(f, job.h, state, new)
            info["bracket_wall"] = time.perf_counter() - tic_b
        records.append(info)
        if new is None or info.get("status") != "converged":
            break
        state = new
    return {**job._asdict(), "n_steps": job.steps, "steps": records, **RUN_KEY}


def _guarded(job: Job) -> dict[str, Any]:
    try:
        return trajectory_job(job)
    except Exception as exc:  # noqa: BLE001 - every failure is recorded as data
        return {
            "job": list(job),
            "status": "job_error",
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "traceback": traceback.format_exc()[-4000:],
            **RUN_KEY,
        }


def run_jobs(jobs: list[Job], workers: int, path: Path) -> None:
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a") as fh:
        futs = {pool.submit(_guarded, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            st = [s.get("status") for s in r.get("steps", [])] or r.get("status")
            print(f"[{i}/{len(jobs)}] {tuple(futs[fut])[:5]} {st}", flush=True)


def confirm_jobs(go: str, done: set[tuple[Any, ...]]) -> list[Job]:
    return [
        Job(a, g, s, scheme, h, n_steps(scheme, h), STEP_CAP_S, True, go)
        for g in reversed(GAMMAS)  # longest first
        for a in ALPHAS
        for s in SEEDS
        for scheme in SCHEMES
        for h in LADDER
        if (a, g, s, scheme, h) not in done
    ]


def pilot_jobs() -> list[Job]:
    return [
        Job(a, g, PILOT_SEED, scheme, h, 1, PILOT_CAP_S, False, None)
        for a, g in PILOT_CELLS
        for scheme in SCHEMES
        for h in LADDER
    ]


# ----- analysis (preregistered) -------------------------------------------------------

Records = dict[tuple[Any, ...], dict[str, Any]]


def _same_run(record: dict[str, Any]) -> bool:
    return all(record.get(k) == v for k, v in RUN_KEY.items())


def load_records(path: Path) -> Records:
    """Confirmatory records by (alpha, gamma, seed, scheme, h); the last one wins."""
    out: Records = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "steps" in r and _same_run(r):
            out[(r["alpha"], r["gamma"], r["seed"], r["scheme"], r["h"])] = r
    return out


def _steps(records: Records, *key: Any) -> list[dict[str, Any]]:
    r = records.get(tuple(key))
    return [] if r is None else list(r["steps"])


def _certified(steps: list[dict[str, Any]], k: int) -> bool:
    return len(steps) > k and steps[k].get("status") == "converged"


def _rho(steps: list[dict[str, Any]], k: int) -> dict[str, np.ndarray] | None:
    """Output of step k, if every step up to k is certified (trajectories stop at a failure)."""
    if not all(_certified(steps, j) for j in range(k + 1)) or "rho_out" not in steps[k]:
        return None
    return {n: np.asarray(v, dtype=float) for n, v in steps[k]["rho_out"].items()}


def tv(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """Max over species of the total-variation distance."""
    return max(0.5 * float(np.abs(a[n] - b[n]).sum()) for n in a)


def _tv_or_none(a: dict[str, np.ndarray] | None, b: dict[str, np.ndarray] | None) -> float | None:
    return None if a is None or b is None else tv(a, b)


def slope(hs: list[float], values: list[float]) -> float:
    """Least-squares slope of log(values) against log(hs)."""
    return float(np.polyfit(np.log(np.asarray(hs)), np.log(np.asarray(values)), 1)[0])


def dissipation_ok(step: dict[str, Any]) -> bool | None:
    """C3 on one step: False on a certified violation or an uncertified bracket.

    Violation: lower(Q_h(out; in)) > upper(Q_h(in; in)). Overlapping brackets
    (equality within the certified precision) satisfy the inequality.
    """
    d = step.get("dissipation")
    if d is None:
        return None
    if d.get("bracket") != "ok":
        return False
    return not d["Q_out"][0] > d["Q_in"][1]


def _split_row(
    xs: list[dict[str, Any]],
    ref: dict[str, np.ndarray] | None,
    small_h_ref: dict[str, np.ndarray] | None,
    margin: float | None,
) -> dict[str, Any]:
    out = _rho(xs, 0)
    error = _tv_or_none(out, ref)
    return {
        "certified_3_steps": all(_certified(xs, k) for k in range(3)),
        "error_vs_S": error,
        "error_vs_small_h": _tv_or_none(out, small_h_ref),
        "accurate": None if error is None or margin is None else error <= margin,
        "dissipation_violation": any(dissipation_ok(st) is False for st in xs[:3]),
        "F_increase_steps": sum(1 for st in xs[:3] if "F_out" in st and st["F_out"] > st["F_in"]),
        "wall_step1": xs[0].get("wall") if xs else None,
    }


def config_rows(records: Records) -> list[dict[str, Any]]:
    """One row per M5 configuration (alpha, gamma, seed, h in M5_HS)."""
    rows = []
    for g in GAMMAS:
        for a in ALPHAS:
            for s in SEEDS:
                for h in M5_HS:
                    s_steps = _steps(records, a, g, s, "implicit", h)
                    ref = _rho(s_steps, 0)
                    small_h_ref = _rho(_steps(records, a, g, s, "implicit", h / 2), 1)
                    margin = _tv_or_none(ref, small_h_ref)
                    row: dict[str, Any] = {
                        "alpha": a,
                        "gamma": g,
                        "seed": s,
                        "h": h,
                        "stratum": "primary" if g in PRIMARY_GAMMAS else "stress",
                        "S_certified": ref is not None,
                        "small_h_reference": small_h_ref is not None,
                        "margin_s_S": margin,
                        "S_wall": s_steps[0].get("wall", 0.0) if s_steps else 0.0,
                    }
                    for x in SPLITS:
                        row[x] = _split_row(
                            _steps(records, a, g, s, x, h), ref, small_h_ref, margin
                        )
                    rows.append(row)
    return rows


def _cfg(c: dict[str, Any]) -> list[Any]:
    return [c["alpha"], c["gamma"], c["seed"], c["h"]]


def substitution_verdicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """C1-C3 on the primary stratum, coverage on the stress stratum (prereg section 7)."""
    primary = [c for c in rows if c["stratum"] == "primary"]
    stress = [c for c in rows if c["stratum"] == "stress"]
    verdicts: dict[str, Any] = {
        "implicit_stress_certified_step1": sum(1 for c in stress if c["S_certified"]),
        "implicit_primary_certified_step1": sum(1 for c in primary if c["S_certified"]),
    }
    for x in SPLITS:
        c1 = [c for c in primary if not c[x]["certified_3_steps"]]
        with_margin = [c for c in primary if c[x]["accurate"] is not None]
        c2 = [c for c in with_margin if not c[x]["accurate"]]
        c3 = [c for c in primary if c[x]["dissipation_violation"]]
        if len(with_margin) < MIN_MARGIN_FRACTION * len(primary):
            verdict = "undecided"
        elif c1 or c2 or c3:
            verdict = "not_substitutable"
        else:
            verdict = "substitutable"
        rescued = [
            c
            for c in stress
            if not c["S_certified"]
            and c[x]["certified_3_steps"]
            and not c[x]["dissipation_violation"]
        ]
        ratios = [
            c[x]["wall_step1"] / c["S_wall"]
            for c in rows
            if c["S_certified"] and c[x]["certified_3_steps"] and c["S_wall"] > 0
        ]
        rel = [c[x]["error_vs_S"] / c["margin_s_S"] for c in with_margin if c["margin_s_S"]]
        verdicts[x] = {
            "substitution": verdict,
            "c1_not_certified": [_cfg(c) for c in c1],
            "c2_configs_with_margin": len(with_margin),
            "c2_not_accurate": [_cfg(c) for c in c2],
            "c3_violations": [_cfg(c) for c in c3],
            "stress_certified": sum(1 for c in stress if c[x]["certified_3_steps"]),
            "stress_rescued": [_cfg(c) for c in rescued],
            "median_wall_ratio_vs_S": float(np.median(ratios)) if ratios else None,
            "median_error_over_margin": float(np.median(rel)) if rel else None,
        }
    return verdicts


def order_cells(records: Records) -> list[dict[str, Any]]:
    """Self-convergence defects s_Y(h) and slopes per primary cell, with the S gate."""
    cells = []
    for g in PRIMARY_GAMMAS:
        for a in ALPHAS:
            for s in SEEDS:
                cell: dict[str, Any] = {"alpha": a, "gamma": g, "seed": s}
                for y in SCHEMES:
                    vals = [
                        _tv_or_none(
                            _rho(_steps(records, a, g, s, y, h), 0),
                            _rho(_steps(records, a, g, s, y, h / 2), 1),
                        )
                        for h in ORDER_HS
                    ]
                    resolved = [v for v in vals if v is not None and v >= TV_FLOOR]
                    p = slope(list(ORDER_HS), resolved) if len(resolved) == len(vals) else None
                    cell[y] = {"s": vals, "slope": p}
                p_s = cell["implicit"]["slope"]
                cell["gate"] = p_s is not None and CONTROL_BAND[0] <= p_s <= CONTROL_BAND[1]
                cells.append(cell)
    return cells


def order_verdicts(cells: list[dict[str, Any]]) -> dict[str, Any]:
    gated = [c for c in cells if c["gate"]]
    out: dict[str, Any] = {"cells": len(cells), "gated": len(gated)}
    for x in SPLITS:
        slopes = [c[x]["slope"] for c in gated if c[x]["slope"] is not None]
        if len(gated) < MIN_GATED_FRACTION * len(cells) or not slopes:
            out[x] = {"verdict": "not_identifiable", "slopes": slopes}
            continue
        second = sum(1 for p in slopes if p >= ORDER_THRESHOLD)
        first = len(slopes) - second
        if second > first:
            verdict = "second_order_like"
        elif first > second:
            verdict = "first_order_like"
        else:
            verdict = "tied"
        out[x] = {
            "verdict": verdict,
            "slopes": slopes,
            "median_slope": float(np.median(slopes)),
            "n_second": second,
            "n_first": first,
        }
    return out


def analyse(records: Records) -> dict[str, Any]:
    """Metrics, gates and verdicts of the preregistration (sections 5 to 7)."""
    rows = config_rows(records)
    cells = order_cells(records)
    return {
        "configs": rows,
        "verdicts": substitution_verdicts(rows),
        "order_cells": cells,
        "order": order_verdicts(cells),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "confirm", "analyse"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--go", default=None, help="reference of the PI's GO (required by confirm)")
    args = ap.parse_args(argv)
    if args.workers > MAX_WORKERS:
        raise SystemExit(f"at most {MAX_WORKERS} workers")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "pilot":
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        path = OUT_DIR / f"pilot_{stamp}.jsonl"
        path.touch(exist_ok=False)
        run_jobs(pilot_jobs(), args.workers, path)
    elif args.cmd == "confirm":
        if not args.go:
            raise SystemExit("confirm needs --go <reference of the PI's GO>")
        path = OUT_DIR / "confirm.jsonl"
        run_jobs(confirm_jobs(args.go, set(load_records(path))), args.workers, path)
    else:
        summary = analyse(load_records(OUT_DIR / "confirm.jsonl"))
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        brief = {"verdicts": summary["verdicts"], "order": summary["order"]}
        print(json.dumps(brief, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
