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
  one step per (cell, scheme, h). Records, by whitelist, statuses, iteration
  counts and wall times (solver and Q_h brackets); no density, no F or Q_h
  value, no residual, no error.
* ``confirm``: the preregistered run. Refuses to start without ``--go`` (the
  reference of the PI's GO, recorded in every record). One job per (cell,
  scheme, h) trajectory; every step record stores the output densities, the
  certificates, F and, at the M5 step sizes, certified brackets of Q_h.
* ``analyse``: the preregistered analysis of the ``confirm`` records
  (metrics, gates, verdicts), written to ``summary.json``.

Caps. Each step's solve runs under ``STEP_CAP_S`` and its Q_h brackets under
``BRACKET_CAP_S`` (``signal.alarm``, POSIX; cooperative: a long native call
can delay the interruption). A capped solve is ``status = "timeout"`` (not
certified; the finished sub-steps of a splitting step are kept); capped
brackets are ``bracket = "timeout"`` (a missing measurement).

Attempts. Every job writes one JSONL record (success, ``job_error`` from the
job itself, or ``job_error`` of kind ``infrastructure`` when the worker pool
breaks), with the job identity and ``RUN_KEY`` (code identity and settings).
The first non-infrastructure record of a job is its observation; ``confirm``
never re-runs a job that has one, and re-runs infrastructure failures only
with ``--retry-infrastructure``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import signal
import sys
import time
import traceback
from collections import Counter, defaultdict
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
STEP_CAP_S = 1800  # wall-clock cap per step solve (all sub-steps), seconds
BRACKET_CAP_S = 600  # wall-clock cap on the Q_h brackets of one step, seconds
PILOT_SEED = 999
PILOT_CAP_S = 240  # 60 jobs x 240 s / 4 workers <= 1 h worst case
PILOT_CELLS = [(2.0, 0.0), (10.0, 5.0), (5.0, 30.0), (2.0, 70.0), (10.0, 100.0)]
MAX_WORKERS = 4  # host rule (KXKM-AI, shared machine)

# Decision constants (preregistration, section 6). They are conventions, stated as such.
TV_FLOOR = 1e-5  # heuristic resolution floor of a TV difference (not a certified bound)
DEGENERATE_MARGIN = 0.5  # margin unusable when s_S >= 0.5 * TV(S_h, rho0)
FIRST_BAND = (1.5, 2.5)  # local exponent predicted 2 (first order), +- half the gap
SECOND_BAND = (2.5, 3.5)  # local exponent predicted 3 (second order), +- half the gap
MIN_FRACTION = 0.5  # coverage needed for a verdict (margins, gated cells)


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
    "bracket_cap_s": BRACKET_CAP_S,
    "jko_defaults": m5._jko_defaults(),
    **m5._code_identity(),
}


def host_info() -> dict[str, Any]:
    """Where and how a record was produced (not part of the run key)."""
    import ot  # noqa: PLC0415 - version only

    return {
        "node": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pot": ot.__version__,
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nice": _niceness(),
    }


def _niceness() -> int | None:
    """Scheduling priority of this process, or None when it cannot be read.

    ``os.nice(0)`` raises PermissionError on macOS at the lowest priority; the
    priority is metadata only and must never fail a job.
    """
    try:
        return os.getpriority(os.PRIO_PROCESS, 0)
    except OSError:
        return None


class StepTimeoutError(RuntimeError):
    """A step hit the wall-clock cap."""


@contextmanager
def wall_cap(seconds: int) -> Iterator[None]:
    """Raise ``StepTimeoutError`` in the main thread after ``seconds`` (POSIX only)."""

    def _handler(signum: int, frame: Any) -> None:
        raise StepTimeoutError(f"exceeded {seconds} s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def probe_cap(sleep_s: float, cap_s: int) -> str:
    """Worker-side check of the cap (tests): 'timeout' or 'done'."""
    try:
        with wall_cap(cap_s):
            time.sleep(sleep_s)
    except StepTimeoutError:
        return "timeout"
    return "done"


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


FULL_KEYS = (
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
PILOT_KEYS = ("status", "outer_iterations", "prox_calls", "seconds")  # whitelist


def _short_info(info: dict[str, Any], pilot: bool) -> dict[str, Any]:
    """JSON-friendly summary of a JKOStep or SplitJKOStep ``last_info``."""
    keys = PILOT_KEYS if pilot else FULL_KEYS
    out = {k: info[k] for k in keys if k in info}
    if "substeps" in info:
        out["substeps"] = [
            {"part": s["part"], "h_sub": s["h_sub"], **{k: s[k] for k in keys if k in s}}
            for s in info["substeps"]
        ]
        out["failed_part"] = info.get("failed_part")
        out["active_part"] = info.get("active_part")
    return out


def q_brackets(f: Any, h: float, state_in: FlowState, out: FlowState) -> dict[str, Any]:
    """Certified brackets of Q_h(out; in) and Q_h(in; in) for the full energy F."""
    jko = JKOStep(f, h, m5.support(GRID), scheme="jko_entropic")
    ref = {n: state_in.rho[n] for n in f.species.species_names()}
    q_out = {n: out.rho[n] for n in ref}
    try:
        with wall_cap(BRACKET_CAP_S):
            lo_out, up_out = jko.objective_bounds(state_in, q_out, ref)
            lo_in, up_in = jko.objective_bounds(state_in, ref, ref)
    except SinkhornNotConvergedError as exc:
        return {"bracket": "not_certified", "error": str(exc)[:300]}
    except StepTimeoutError:
        return {"bracket": "timeout"}
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

    def key(self) -> tuple[Any, ...]:
        return (self.alpha, self.gamma, self.seed, self.scheme, self.h)


def _solve(jko: JKOStep | SplitJKOStep, state: FlowState, cap: int, pilot: bool) -> Any:
    """One capped step: (new state or None, record of the step)."""
    new: FlowState | None = None
    tic = time.perf_counter()
    try:
        with wall_cap(cap):
            new = jko.step(state)
        info = _short_info(jko.last_info, pilot)
    except ImplicitJKOError as exc:
        info = _short_info(exc.info, pilot)
    except StepTimeoutError:
        # a splitting step keeps its finished sub-steps; JKOStep publishes nothing mid-step
        info = _short_info(jko.last_info, pilot) if isinstance(jko, SplitJKOStep) else {}
        info["status"] = "timeout"
    except Exception as exc:  # noqa: BLE001 - recorded as data, earlier steps are kept
        info = {"status": "step_error", "error": f"{type(exc).__name__}: {exc}"[:2000]}
    info["solver_wall"] = time.perf_counter() - tic
    return new, info


def trajectory_job(job: Job) -> dict[str, Any]:
    """Up to ``job.steps`` steps of one scheme from the cell's input; stops at a failure."""
    f, state = initial_state(job.alpha, job.gamma, job.seed)
    names = f.species.species_names()
    jko = make_step(f, job.scheme, job.h)
    pilot = not job.store
    records: list[dict[str, Any]] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for k in range(job.steps):
        new, info = _solve(jko, state, job.cap, pilot)
        info["k"] = k
        if new is not None and info.get("status") == "converged":
            tic = time.perf_counter()
            if job.store:
                info["F_in"] = f.value(state)
                info["F_out"] = f.value(new)
                info["rho_out"] = {n: new.rho[n].tolist() for n in names}
                if job.h in M5_HS:
                    info["dissipation"] = q_brackets(f, job.h, state, new)
            elif job.h in M5_HS:  # pilot: time the brackets, keep no value
                info["bracket_status"] = q_brackets(f, job.h, state, new)["bracket"]
            info["bracket_wall"] = time.perf_counter() - tic
        records.append(info)
        if new is None or info.get("status") != "converged":
            break
        state = new
    return {
        **job._asdict(),
        "kind": "trajectory",
        "started": started,
        "host": host_info(),
        "steps": records,
        **RUN_KEY,
    }


def _error_record(job: Job, kind: str, exc: BaseException) -> dict[str, Any]:
    return {
        **job._asdict(),
        "kind": kind,
        "status": "job_error",
        "error": f"{type(exc).__name__}: {exc}"[:2000],
        "traceback": traceback.format_exc()[-4000:],
        "steps": [],
        **RUN_KEY,
    }


def _guarded(job: Job) -> dict[str, Any]:
    try:
        return trajectory_job(job)
    except Exception as exc:  # noqa: BLE001 - every failure is recorded as data
        return _error_record(job, "code", exc)


def run_jobs(jobs: list[Job], workers: int, path: Path) -> None:
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a") as fh:
        futs = {pool.submit(_guarded, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001 - broken pool, killed worker
                r = _error_record(futs[fut], "infrastructure", exc)
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            st = [s.get("status") for s in r.get("steps", [])] or r.get("status")
            print(f"[{i}/{len(jobs)}] {futs[fut].key()} {st}", flush=True)


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


# ----- attempts ------------------------------------------------------------------------

Records = dict[tuple[Any, ...], dict[str, Any]]


def _same_run(record: dict[str, Any]) -> bool:
    return all(record.get(k) == v for k, v in RUN_KEY.items())


def _record_key(r: dict[str, Any]) -> tuple[Any, ...]:
    return (r["alpha"], r["gamma"], r["seed"], r["scheme"], r["h"])


def load_attempts(path: Path) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Every record of this run, by job identity, in file order."""
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "scheme" in r and _same_run(r):
            out[_record_key(r)].append(r)
    return out


def primary_records(attempts: dict[tuple[Any, ...], list[dict[str, Any]]]) -> Records:
    """The observation of each job: its first record that is not an infrastructure failure."""
    out: Records = {}
    for key, rows in attempts.items():
        for r in rows:
            if r.get("kind") != "infrastructure":
                out[key] = r
                break
    return out


def done_keys(path: Path, retry_infrastructure: bool) -> set[tuple[Any, ...]]:
    attempts = load_attempts(path)
    if retry_infrastructure:
        return set(primary_records(attempts))
    return set(attempts)


def expected_keys() -> set[tuple[Any, ...]]:
    return {j.key() for j in confirm_jobs("", set())}


# ----- analysis (preregistered) -------------------------------------------------------


def _steps(records: Records, *key: Any) -> list[dict[str, Any]]:
    r = records.get(tuple(key))
    return [] if r is None else list(r.get("steps", []))


def _certified(steps: list[dict[str, Any]], k: int) -> bool:
    """Steps 0..k all converged with their densities recorded."""
    return len(steps) > k and all(
        steps[j].get("status") == "converged" and "rho_out" in steps[j] for j in range(k + 1)
    )


def _rho(steps: list[dict[str, Any]], k: int) -> dict[str, np.ndarray] | None:
    if not _certified(steps, k):
        return None
    return {n: np.asarray(v, dtype=float) for n, v in steps[k]["rho_out"].items()}


def tv(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """Max over species of the total-variation distance."""
    return max(0.5 * float(np.abs(a[n] - b[n]).sum()) for n in a)


def _tv_or_none(a: dict[str, np.ndarray] | None, b: dict[str, np.ndarray] | None) -> float | None:
    return None if a is None or b is None else tv(a, b)


def dissipation_state(step: dict[str, Any]) -> str:
    """Three-state comparison of Q_h(out; in) with Q_h(in; in), or ``missing``.

    ``proven``: upper(out) <= lower(in); ``violated``: lower(out) > upper(in);
    ``indeterminate``: overlapping brackets; ``missing``: no certified, finite,
    ordered brackets (not computed, not certified, timed out).
    """
    d = step.get("dissipation")
    if not d or d.get("bracket") != "ok":
        return "missing"
    lo_o, up_o = d["Q_out"]
    lo_i, up_i = d["Q_in"]
    if not all(np.isfinite([lo_o, up_o, lo_i, up_i])) or lo_o > up_o or lo_i > up_i:
        return "missing"
    if up_o <= lo_i:
        return "proven"
    if lo_o > up_i:
        return "violated"
    return "indeterminate"


def _split_row(
    xs: list[dict[str, Any]],
    *,
    rho0: dict[str, np.ndarray],
    ref: dict[str, np.ndarray] | None,
    small_h_ref: dict[str, np.ndarray] | None,
    margin_ok: bool,
    margin: float | None,
) -> dict[str, Any]:
    out = _rho(xs, 0)
    error = _tv_or_none(out, ref)
    states = [dissipation_state(st) for st in xs if st.get("status") == "converged"]
    if not margin_ok:
        c2 = None  # no usable margin: not counted
    elif error is None:
        c2 = "not_evaluable"  # X has no certified first step (a C1 failure too)
    else:
        c2 = "pass" if error <= margin else "fail"  # type: ignore[operator]
    return {
        "certified_step1": _certified(xs, 0),
        "certified_3_steps": _certified(xs, 2),
        "statuses": [st.get("status") for st in xs],
        "displacement": _tv_or_none(out, rho0),
        "error_vs_S": error,
        "error_vs_small_h": _tv_or_none(out, small_h_ref),
        "c2": c2,
        "near_tie": error is not None and margin is not None and abs(error - margin) < TV_FLOOR,
        "dissipation": dict(Counter(states)),
        "c3_fail": any(s in ("violated", "missing") for s in states),
        "F_increase_steps": sum(1 for st in xs[:3] if "F_out" in st and st["F_out"] > st["F_in"]),
        "solver_wall_step1": xs[0].get("solver_wall") if xs else None,
    }


def config_rows(records: Records) -> list[dict[str, Any]]:
    """One row per M5 configuration (alpha, gamma, seed, h in M5_HS)."""
    rows = []
    for g in GAMMAS:
        for a in ALPHAS:
            for s in SEEDS:
                _, state0 = initial_state(a, g, s)
                rho0 = dict(state0.rho)
                for h in M5_HS:
                    s_steps = _steps(records, a, g, s, "implicit", h)
                    ref = _rho(s_steps, 0)
                    small_h_ref = _rho(_steps(records, a, g, s, "implicit", h / 2), 1)
                    margin = _tv_or_none(ref, small_h_ref)
                    d_s = _tv_or_none(ref, rho0)
                    margin_ok = (
                        margin is not None and d_s is not None and margin < DEGENERATE_MARGIN * d_s
                    )
                    row: dict[str, Any] = {
                        "alpha": a,
                        "gamma": g,
                        "seed": s,
                        "h": h,
                        "stratum": "primary" if g in PRIMARY_GAMMAS else "stress",
                        "S_recorded": bool(s_steps),
                        "S_certified": ref is not None,
                        "S_status_step1": s_steps[0].get("status") if s_steps else None,
                        "small_h_reference": small_h_ref is not None,
                        "margin_s_S": margin,
                        "displacement_S": d_s,
                        "displacement_R": _tv_or_none(small_h_ref, rho0),
                        "margin_usable": margin_ok,
                        "S_solver_wall": s_steps[0].get("solver_wall") if s_steps else None,
                    }
                    for x in SPLITS:
                        row[x] = _split_row(
                            _steps(records, a, g, s, x, h),
                            rho0=rho0,
                            ref=ref,
                            small_h_ref=small_h_ref,
                            margin_ok=margin_ok,
                            margin=margin,
                        )
                    rows.append(row)
    return rows


def _cfg(c: dict[str, Any]) -> list[Any]:
    return [c["alpha"], c["gamma"], c["seed"], c["h"]]


def _rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Certification rates by gamma, alpha and h (S step 1; X step 1 and 3 steps)."""
    out: dict[str, Any] = {}
    for axis in ("gamma", "alpha", "h"):
        table: dict[str, Any] = {}
        for v in sorted({c[axis] for c in rows}):
            sub = [c for c in rows if c[axis] == v]
            table[str(v)] = {
                "n": len(sub),
                "S_step1": sum(c["S_certified"] for c in sub),
                **{f"{x}_step1": sum(c[x]["certified_step1"] for c in sub) for x in SPLITS},
                **{f"{x}_3_steps": sum(c[x]["certified_3_steps"] for c in sub) for x in SPLITS},
            }
        out[axis] = table
    return out


def substitution_verdicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Q1 (C1-C3) on the primary stratum and Q2 (rescue) on the stress stratum."""
    primary = [c for c in rows if c["stratum"] == "primary"]
    stress = [c for c in rows if c["stratum"] == "stress"]
    usable = [c for c in primary if c["margin_usable"]]
    verdicts: dict[str, Any] = {
        "primary_configs": len(primary),
        "usable_margins": len(usable),
        "degenerate_margins": [
            _cfg(c) for c in primary if c["margin_s_S"] is not None and not c["margin_usable"]
        ],
        "rates": _rates(rows),
    }
    for x in SPLITS:
        c1 = [c for c in primary if not c[x]["certified_3_steps"]]
        c2_fail = [c for c in usable if c[x]["c2"] in ("fail", "not_evaluable")]
        c3 = [c for c in primary if c[x]["c3_fail"]]
        if c1 or c2_fail or c3:  # a certain failure decides first
            verdict = "not_substitutable"
        elif len(usable) < MIN_FRACTION * len(primary):
            verdict = "undecided"
        else:
            verdict = "substitutable"
        rescued = [
            c
            for c in stress
            if c["S_recorded"]
            and not c["S_certified"]
            and c[x]["certified_3_steps"]
            and not c[x]["c3_fail"]
        ]
        ratios = [
            c[x]["solver_wall_step1"] / c["S_solver_wall"]
            for c in rows
            if c["S_certified"] and c[x]["certified_step1"] and (c["S_solver_wall"] or 0) > 0
        ]
        rel = [
            c[x]["error_vs_S"] / c["margin_s_S"]
            for c in usable
            if c[x]["error_vs_S"] is not None and c["margin_s_S"]
        ]
        verdicts[x] = {
            "substitution": verdict,
            "c1_not_certified": [_cfg(c) for c in c1],
            "c2_fail": [_cfg(c) for c in c2_fail],
            "c2_near_ties": [_cfg(c) for c in usable if c[x]["near_tie"]],
            "c3_fail": [_cfg(c) for c in c3],
            "stress_certified_3_steps": sum(1 for c in stress if c[x]["certified_3_steps"]),
            "stress_rescued": [_cfg(c) for c in rescued],
            "stress_rescued_with_indeterminate_dissipation": [
                _cfg(c) for c in rescued if c[x]["dissipation"].get("indeterminate")
            ],
            "median_wall_ratio_vs_S": float(np.median(ratios)) if ratios else None,
            "median_error_over_margin": float(np.median(rel)) if rel else None,
        }
    return verdicts


def _band(p: float) -> str:
    if FIRST_BAND[0] <= p < FIRST_BAND[1]:
        return "first"
    if SECOND_BAND[0] <= p <= SECOND_BAND[1]:
        return "second"
    return "neither"


def order_cells(records: Records) -> list[dict[str, Any]]:
    """Self-convergence defects s_Y(h), adjacent exponents and classes per primary cell."""
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
                    entry: dict[str, Any] = {"s": vals, "adjacent": None, "slope": None}
                    if all(v is not None and v >= TV_FLOOR for v in vals):
                        v = np.asarray(vals, dtype=float)
                        entry["adjacent"] = [
                            float(np.log2(v[0] / v[1])),
                            float(np.log2(v[1] / v[2])),
                        ]
                        entry["slope"] = float(
                            np.polyfit(np.log(np.asarray(ORDER_HS)), np.log(v), 1)[0]
                        )
                        bands = {_band(p) for p in entry["adjacent"]}
                        entry["class"] = bands.pop() if len(bands) == 1 else "neither"
                    else:
                        entry["class"] = "unresolved"
                    cell[y] = entry
                cell["gate"] = cell["implicit"]["class"] == "first"
                cells.append(cell)
    return cells


def order_verdicts(cells: list[dict[str, Any]]) -> dict[str, Any]:
    gated = [c for c in cells if c["gate"]]
    out: dict[str, Any] = {"cells": len(cells), "gated": len(gated)}
    for x in SPLITS:
        counts = Counter(c[x]["class"] for c in gated)
        slopes = [c[x]["slope"] for c in gated if c[x]["slope"] is not None]
        verdict = "not_identifiable"
        if len(gated) >= MIN_FRACTION * len(cells):
            for cls, name in (("first", "first_order_like"), ("second", "second_order_like")):
                if counts[cls] > len(gated) / 2:
                    verdict = name
        out[x] = {
            "verdict": verdict,
            "classes": dict(counts),
            "slopes": slopes,
            "median_slope": float(np.median(slopes)) if slopes else None,
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


def analyse_file(path: Path) -> dict[str, Any]:
    """``analyse`` on the observations of a records file, with completeness and provenance."""
    attempts = load_attempts(path)
    records = primary_records(attempts)
    missing = sorted(expected_keys() - set(records))
    summary = analyse(records)
    summary["provenance"] = {
        "records_file": str(path),
        "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
        "run_key": RUN_KEY,
        "complete": not missing,
        "missing_jobs": missing,
        "jobs_with_several_records": sorted(k for k, v in attempts.items() if len(v) > 1),
        "job_errors": sorted(k for k, r in records.items() if r.get("status") == "job_error"),
    }
    if missing:  # no final verdict on an incomplete set
        summary["provisional_verdicts"] = summary.pop("verdicts")
        summary["provisional_order"] = summary.pop("order")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "confirm", "analyse"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--go", default=None, help="reference of the PI's GO (required by confirm)")
    ap.add_argument("--retry-infrastructure", action="store_true")
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"workers must be between 1 and at most {MAX_WORKERS}")
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
        done = done_keys(path, args.retry_infrastructure)
        run_jobs(confirm_jobs(args.go, done), args.workers, path)
    else:
        summary = analyse_file(OUT_DIR / "confirm.jsonl")
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        brief = {k: v for k, v in summary.items() if k not in ("configs", "order_cells")}
        print(json.dumps(brief, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
