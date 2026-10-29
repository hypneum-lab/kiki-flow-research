"""Validation of the certified implicit JKO step (PR fix/implicit-jko-certified).

Codex review of PR #50 (2026-10-27): the damped Picard fixed point could
exhaust its iterations and be accepted silently (B1); "F decreases" is not "the
entropic JKO problem is solved" (B2). The step now minimises
    Q_h(q) = F(q) + sum_i OT_reg(q_i, rho_tau_i) / (2h)
by Bregman majorisation-minimisation with a line search on Q_h and certifies a
stationary point (KKT residual at the returned point) or fails explicitly.

Parts (all on electron-server, nice 19, at most 4 workers):

* ``reference``: small grids (4, 6, 8 points), T2 energy with declared Turing
  smoothing, inputs = Dirichlet draws, the Gibbs state of the potentials and a
  sparse particle histogram; the certified step is compared with an independent
  reference, L-BFGS-B on Q_h over a softmax parameterisation with the exact
  gradient grad F + f / (2h) (f = dual potential of OT_reg computed by POT,
  an implementation independent of the solver's), from three
  starts (uniform, input, step output), on the support [-0.5, 0.5] (identifiable
  dual potentials) with h = 0.2. Reported: status, TV(step, reference),
  Q_h(step) - Q_h(reference), KKT residual.
* ``certify``: the grid of the 2026-10-27 dissipation check, gamma in {0, 5, 10,
  20, 30, 50, 70, 100} x alpha in {2, 5, 10} x h in {0.05, 0.025} x 2 seeds
  (Dirichlet initial states, seeds 500 and 501), ``N_STEPS`` = 3 steps on grid 16:
  status of every step (converged / failure kind), KKT residual, outer
  iterations, inner residuals, seconds per step; F and Q_h along the steps are
  recorded descriptively (not a criterion).
* ``operating``: at the operating point (grid 16 on [-2, 2], h = 0.05, reg =
  h^2) for (10, 0) and (8, 5): Q_h at the certified step against absolute mass
  transfers between neighbouring bins, Q_h evaluated by certified brackets.
* ``smoothing``: Turing smoothing delta in {1e-4, 1e-5, 1e-6} at (5, 5), (5, 30)
  and (10, 100): change of the step output and of F with delta.

* ``b1``: the three cases of Codex B1 (grid 16, Dirichlet input of seed 500,
  h = 0.05, reg = h^2, delta = 1e-5): (8, 5), (5, 30), (10, 0); one certified
  step each, with status, KKT residual, iterations, Q_h and wall time.

Outputs (JSONL, one record per job written as it completes; a job that raises is
recorded with ``status = "job_error"`` and the batch goes on):
``results/implicit_jko_certified_2026_10_27/{b1_rerun_<UTC stamp>.jsonl,
reference.jsonl, operating.jsonl, certify.jsonl, smoothing.jsonl}``. Every record
carries ``RUN_KEY``: solver version, delta, N_STEPS, the effective default
settings of ``JKOStep`` (read from its signature) and the identity of the code
(git commit, dirty flag, SHA-256 of the tracked diff); ``certify`` resumes only
from records with the same key.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import ot
from scipy.optimize import minimize

from kiki_flow_core.master_equation import SOLVER_VERSION, ImplicitJKOError, JKOStep
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy
from kiki_flow_core.track2_paper.paper_run import _git_commit, _git_diff_sha256, _git_dirty
from kiki_flow_core.wasserstein_ops import SinkhornNotConvergedError

OUT_DIR = Path("results/implicit_jko_certified_2026_10_27")
DELTA = 1e-5
GAMMAS = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0]
ALPHAS = [2.0, 5.0, 10.0]
HS = [0.05, 0.025]
INIT_SEEDS = [500, 501]
N_STEPS = 3  # multi-step trajectories (dissipation along steps); up to ~400 s per step
REF_GRIDS = [4, 6, 8]
REF_CELLS = [(2.0, 0.0), (5.0, 5.0), (5.0, 30.0)]
OPERATING_CELLS = [(10.0, 0.0), (8.0, 5.0)]  # support [-2, 2], h = 0.05, reg = h^2, grid 16
SMOOTHING_DELTAS = [1e-4, 1e-5, 1e-6]
REF_H = 0.2  # reg = h^2 = 0.04
REF_HALF_WIDTH = 0.5  # reference support [-0.5, 0.5]: max cost / reg = 25, so the dual
# potential of the balanced problem is numerically identifiable (at cost / reg ~ 400 the
# kernel underflows and different potentials give the same plan)
SMOOTHING_CELLS = [(5.0, 5.0), (5.0, 30.0), (10.0, 100.0)]
FLOOR = 1e-300
POT_MARGINAL_TOL = 1e-8
# Operating-point check: each OT_reg bracket certified to OPERATING_GAP_TOL (attainable
# with the Newton polishing of entropic_ot; Sinkhorn alone stalls near 1e-6 here), so
# the summed Q_h uncertainty is <= 4 * 2 * OPERATING_GAP_TOL / (2h) = 8e-10. A transfer of
# t = OPERATING_TRANSFER (capped at half the donor's mass) changes Q_h by
# g t + O(t^2 H) with H >= 1 (entropy), i.e. by >= ~5e-9 at a stationary point: well
# above the uncertainty, so every comparison is decidable unless g t ~ -t^2 H / 2.
OPERATING_GAP_TOL = 1e-11
OPERATING_TRANSFER = 1e-4
# Everything that can change a result; a resumed run reuses a record only if all match.


def _jko_defaults() -> dict[str, Any]:
    """Effective default settings of ``JKOStep`` (its signature), JSON-normalised."""
    params = inspect.signature(JKOStep.__init__).parameters.values()
    defaults = {
        p.name: p.default
        for p in params
        if p.default is not inspect.Parameter.empty and p.name != "prox_fn"
    }
    normalised: dict[str, Any] = json.loads(json.dumps(defaults, default=repr))
    return normalised


RUN_KEY: dict[str, Any] = {
    "solver_version": SOLVER_VERSION,
    "delta": DELTA,
    "n_steps": N_STEPS,
    "jko_defaults": _jko_defaults(),
    "git_commit": _git_commit(),
    "git_dirty": _git_dirty(),
    "git_diff_sha256": _git_diff_sha256(),
}


def output_paths() -> dict[str, Path]:
    return {
        "reference": OUT_DIR / "reference.jsonl",
        "operating": OUT_DIR / "operating.jsonl",
        "certify": OUT_DIR / "certify.jsonl",
        "smoothing": OUT_DIR / "smoothing.jsonl",
        "summary": OUT_DIR / "summary.md",
    }


def state_of(rho: dict[str, np.ndarray]) -> FlowState:
    return FlowState(
        rho=rho, P_theta=np.zeros(4), mu_curr=np.array([1.0]), tau=0, metadata={"track_id": "T2"}
    )


def problem(grid: int, alpha: float, gamma: float, delta: float = DELTA) -> T2FreeEnergy:
    species = CanonicalSpecies()
    names = species.species_names()
    x = np.linspace(-2, 2, grid)
    potentials = {
        "phono": alpha * (x + 1.5),
        "lex": alpha * (x - 1.5),
        "syntax": alpha * np.abs(x),
        "sem": -alpha * np.abs(x),
    }
    prior = {n: np.full(grid, 1.0 / grid) for n in names}
    return T2FreeEnergy(
        species=species,
        potentials=potentials,
        prior=prior,
        turing_strength=0.1 * gamma,
        turing_smoothing=delta,
    )


def support(grid: int) -> np.ndarray:
    return np.linspace(-2, 2, grid).reshape(-1, 1)


def inputs(f: T2FreeEnergy, grid: int) -> dict[str, dict[str, np.ndarray]]:
    names = f.species.species_names()
    rng = np.random.default_rng(7)
    gibbs = {}
    for n in names:
        w = np.exp(-(f.potentials[n] - f.potentials[n].min()))
        gibbs[n] = w / w.sum()
    sparse = {}
    for n in names:
        counts = np.bincount(rng.integers(0, grid, size=6), minlength=grid).astype(float)
        sparse[n] = counts / counts.sum()  # zeros are legitimate bins
    return {
        "dirichlet": {n: rng.dirichlet(np.ones(grid)) for n in names},
        "gibbs": gibbs,
        "sparse_histogram": sparse,
    }


def grad_f(f: T2FreeEnergy, q: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    st = state_of(q)
    out = {}
    for n in f.species.species_names():
        safe = np.clip(q[n], FLOOR, None)
        out[n] = (
            f.potentials[n]
            + np.log(safe / np.clip(f.prior[n], FLOOR, None))
            + 1.0
            + f.nonseparable_grad(st, n)
        )
    return out


def pot_entropic_ot(
    q: np.ndarray, r: np.ndarray, cost: np.ndarray, reg: float
) -> tuple[float, np.ndarray]:
    """Independent OT_reg value and first-marginal dual potential via POT (log domain)."""
    plan, log = ot.sinkhorn(
        q, r, cost, reg, method="sinkhorn_log", numItermax=20_000, stopThr=1e-11, log=True
    )
    err = float(np.abs(plan.sum(axis=1) - q).sum() + np.abs(plan.sum(axis=0) - r).sum())
    if err > POT_MARGINAL_TOL:
        raise RuntimeError(f"POT Sinkhorn marginals not met ({err:.2e})")
    safe = np.where(plan > 0.0, np.log(np.maximum(plan, 1e-300)), 0.0)
    value = float(np.sum(plan * cost) + reg * np.sum(plan * (safe - 1.0)))
    return value, reg * np.asarray(log["log_u"], dtype=float)


def reference_minimiser(  # local reference (L-BFGS-B), no global guarantee
    f: T2FreeEnergy,
    ref: dict[str, np.ndarray],
    sup: np.ndarray,
    h: float,
    reg: float,
    starts: list[dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], float, bool]:
    """Independent reference: L-BFGS-B on Q_h(softmax(z)), exact gradient.

    Returns (q, Q_h(q), success). Raises ``RuntimeError`` when no start succeeds:
    a failed optimisation is not a reference.
    """
    names = f.species.species_names()
    grid = sup.shape[0]
    cost = ot.dist(sup, sup, metric="sqeuclidean")

    def unpack(z: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for i, n in enumerate(names):
            zi = z[i * grid : (i + 1) * grid]
            w = np.exp(zi - zi.max())
            out[n] = w / w.sum()
        return out

    def fun(z: np.ndarray) -> tuple[float, np.ndarray]:
        q = unpack(z)
        val = f.value(state_of(q))
        g = grad_f(f, q)
        grad = np.zeros_like(z)
        for i, n in enumerate(names):
            ot_val, pot = pot_entropic_ot(q[n], ref[n], cost, reg)
            val += ot_val / (2.0 * h)
            gq = g[n] + pot / (2.0 * h)
            grad[i * grid : (i + 1) * grid] = q[n] * (gq - float(np.dot(q[n], gq)))
        return float(val), grad

    best: tuple[dict[str, np.ndarray], float, bool] | None = None
    for start in starts:
        z0 = np.concatenate([np.log(np.clip(start[n], 1e-12, None)) for n in names])
        res = minimize(
            fun, z0, jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-10}
        )
        q = unpack(res.x)
        ok = bool(res.success)
        # a successful run beats a failed one; among equals, the lower value wins
        if best is None or (ok, -float(res.fun)) > (best[2], -best[1]):
            best = (q, float(res.fun), ok)
    if best is None or not best[2]:
        raise RuntimeError("L-BFGS-B reference failed from every start")
    return best


def tv(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    return max(0.5 * float(np.abs(a[n] - b[n]).sum()) for n in a)


def reference_job(grid: int, alpha: float, gamma: float, kind: str, h: float) -> dict[str, Any]:
    f = problem(grid, alpha, gamma)
    sup = np.linspace(-REF_HALF_WIDTH, REF_HALF_WIDTH, grid).reshape(-1, 1)
    rho = inputs(f, grid)[kind]
    jko = JKOStep(f, h, sup, scheme="jko_entropic", kkt_tol=1e-7, on_failure="warn")
    tic = time.perf_counter()
    out = jko.step(state_of(rho))
    info = dict(jko.last_info)
    q_step = {n: out.rho[n] for n in rho}
    names = f.species.species_names()
    uniform = {n: np.full(grid, 1.0 / grid) for n in names}
    q_ref, q_ref_val, ref_success = reference_minimiser(
        f, rho, sup, h, jko.entropic_reg, [uniform, rho, q_step]
    )
    cost = ot.dist(sup, sup, metric="sqeuclidean")
    q_step_val = f.value(state_of(q_step)) + sum(
        pot_entropic_ot(q_step[n], rho[n], cost, jko.entropic_reg)[0] for n in names
    ) / (2.0 * h)
    return {
        "grid": grid,
        "alpha": alpha,
        "gamma": gamma,
        "input": kind,
        "h": h,
        "status": info["status"],
        "kkt_residual": info["kkt_residual"],
        "outer_iterations": info["outer_iterations"],
        "tv_step_vs_reference": tv(q_step, q_ref),
        "Q_step_minus_Q_reference": q_step_val - q_ref_val,
        "reference_lbfgs_success": ref_success,
        "log_residual": info.get("log_residual"),
        "jko_overrides": {"kkt_tol": 1e-7},
        "seconds": time.perf_counter() - tic,
        **RUN_KEY,
    }


def operating_job(alpha: float, gamma: float) -> dict[str, Any]:
    """Operating point: independent check by certified Q_h brackets and mass transfers.

    At cost / reg up to 6400 the dual potentials are not identifiable, so the
    check compares certified brackets of Q_h (``JKOStep.objective_bounds``) at the
    step output and after absolute mass transfers between neighbouring bins
    (``OPERATING_TRANSFER``, capped at half the donor's mass, never negative). A
    transfer is a certified decrease when its upper bound is below the lower
    bound at the output, no decrease when its lower bound is above the upper
    bound, indeterminate otherwise. The step status is recorded first; a step
    that is not certified is a result, not a crash. The verdict
    (``operating_verdict``) is ``passed`` only when every transfer was compared
    and none decreases Q_h, ``failed`` on a certified decrease, ``inconclusive``
    otherwise.
    """
    f = problem(16, alpha, gamma)
    sup = support(16)
    rng = np.random.default_rng(500)
    rho = {n: rng.dirichlet(np.ones(16)) for n in f.species.species_names()}
    jko = JKOStep(f, 0.05, sup, scheme="jko_entropic", on_failure="warn")
    out = jko.step(state_of(rho))
    info = jko.last_info
    row: dict[str, Any] = {
        "alpha": alpha,
        "gamma": gamma,
        "status": info["status"],
        "kkt_residual": info["kkt_residual"],
        "log_residual": info["log_residual"],
        "q_min": info["q_min"],
        "step_seconds": info.get("seconds"),
        "ot_gap_tol": OPERATING_GAP_TOL,
        "transfer": OPERATING_TRANSFER,
        **RUN_KEY,
    }
    if info["status"] != "converged":
        row["check"] = "not_run: step not certified"
        return row
    q = {n: out.rho[n] for n in rho}
    state = state_of(rho)
    try:
        base_lo, base_up = jko.objective_bounds(state, q, rho, gap_tol=OPERATING_GAP_TOL)
    except SinkhornNotConvergedError as exc:
        row["check"] = "not_run: Q_h bracket not certified at the output"
        row["error"] = str(exc)[:400]
        return row
    counts = {"certified_decrease": 0, "no_decrease": 0, "indeterminate": 0, "not_certified": 0}
    worst = np.inf  # min over transfers of lower(pert) - upper(base)
    for n in rho:
        for a in range(15):
            for donor, receiver in ((a + 1, a), (a, a + 1)):
                t = min(OPERATING_TRANSFER, 0.5 * float(q[n][donor]))
                moved = q[n].copy()
                moved[receiver] += t
                moved[donor] -= t
                try:
                    lo, up = jko.objective_bounds(
                        state, {**q, n: moved}, rho, gap_tol=OPERATING_GAP_TOL
                    )
                except SinkhornNotConvergedError:
                    counts["not_certified"] += 1
                    continue
                worst = min(worst, lo - base_up)
                if up < base_lo:
                    counts["certified_decrease"] += 1
                elif lo > base_up:
                    counts["no_decrease"] += 1
                else:
                    counts["indeterminate"] += 1
    row.update(
        Q_lower=base_lo,
        Q_upper=base_up,
        transfers=counts,
        min_lower_pert_minus_upper_base=worst,
        check=operating_verdict(counts),
    )
    return row


def operating_verdict(counts: dict[str, int]) -> str:
    """Three-state verdict of the operating check.

    ``failed``: at least one transfer certifiably decreases Q_h. ``passed``: every
    transfer was compared and none decreases Q_h (all ``no_decrease``).
    ``inconclusive``: otherwise (some comparison missing or indeterminate); the
    absence of a demonstrated decrease is not a validation.
    """
    if counts["certified_decrease"] > 0:
        return "failed"
    total = sum(counts.values())
    if total > 0 and counts["no_decrease"] == total:
        return "passed"
    return "inconclusive"


def certify_job(alpha: float, gamma: float, h: float, seed: int) -> dict[str, Any]:
    f = problem(16, alpha, gamma)
    sup = support(16)
    rng = np.random.default_rng(seed)
    state = state_of({n: rng.dirichlet(np.ones(16)) for n in f.species.species_names()})
    jko = JKOStep(f, h, sup, scheme="jko_entropic", on_failure="warn")
    steps = []
    for _ in range(N_STEPS):
        f_in = f.value(state)
        try:
            new = jko.step(state)
        except ImplicitJKOError as exc:  # on_failure="warn" never raises; kept for safety
            steps.append({**exc.info, "F_in": f_in})
            break
        info = {k: v for k, v in jko.last_info.items() if k != "inner_error"}
        info.update(F_in=f_in, F_out=f.value(new))
        steps.append(info)
        if info["status"] != "converged":
            break
        state = new
    return {
        "alpha": alpha,
        "gamma": gamma,
        "h": h,
        "seed": seed,
        **RUN_KEY,
        "steps": steps,
    }


def b1_job(alpha: float, gamma: float) -> dict[str, Any]:
    f = problem(16, alpha, gamma)
    rng = np.random.default_rng(500)
    rho = {n: rng.dirichlet(np.ones(16)) for n in f.species.species_names()}
    jko = JKOStep(f, 0.05, support(16), scheme="jko_entropic", on_failure="warn")
    tic = time.perf_counter()
    out = jko.step(state_of(rho))
    info = {k: v for k, v in jko.last_info.items() if k != "inner_error"}
    info.update(
        **RUN_KEY,
        case=f"T2({alpha},{gamma}) delta={DELTA} h=0.05",
        wall=time.perf_counter() - tic,
        dF=f.value(out) - f.value(state_of(rho)),
    )
    return info


def smoothing_job(alpha: float, gamma: float) -> dict[str, Any]:
    outs = {}
    rng = np.random.default_rng(500)
    names = CanonicalSpecies().species_names()
    rho = {n: rng.dirichlet(np.ones(16)) for n in names}
    for delta in SMOOTHING_DELTAS:
        f = problem(16, alpha, gamma, delta)
        jko = JKOStep(f, 0.05, support(16), scheme="jko_entropic", on_failure="warn")
        out = jko.step(state_of(rho))
        outs[delta] = (
            {n: out.rho[n] for n in names},
            jko.last_info["status"],
            jko.last_info["seconds"],
            jko.last_info["kkt_residual"],
        )
    exact = problem(16, alpha, gamma, None)  # unsmoothed F, for the value only
    rows = []
    for delta in SMOOTHING_DELTAS:
        q, status, secs, res = outs[delta]
        rows.append(
            {
                "delta": delta,
                "status": status,
                "kkt_residual": res,
                "seconds": secs,
                "F_exact_at_output": exact.value(state_of(q)),
                "tv_vs_smallest_delta": tv(q, outs[SMOOTHING_DELTAS[-1]][0]),
            }
        )
    return {"alpha": alpha, "gamma": gamma, "rows": rows, **RUN_KEY}


def _guarded(fn: Any, *job: Any) -> dict[str, Any]:
    """Run one job; an exception becomes a record, never the end of the batch."""
    try:
        result: dict[str, Any] = fn(*job)
    except Exception as exc:  # noqa: BLE001 - every failure is recorded as data
        return {
            "job": list(job),
            "status": "job_error",
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "traceback": traceback.format_exc()[-4000:],
            **RUN_KEY,
        }
    return result


def run_pool(fn: Any, jobs: list[tuple[Any, ...]], workers: int, path: Path) -> list[Any]:
    """Run isolated jobs, appending each record to ``path`` (JSONL) as it completes.

    ``workers <= 1`` runs the jobs in this process (no pool).
    """
    out = []
    if workers <= 1:
        with path.open("a") as fh:
            for j in jobs:
                r = _guarded(fn, *j)
                fh.write(json.dumps(r, default=float) + "\n")
                fh.flush()
                out.append(r)
                print(json.dumps({"job": j, "status": r.get("status", "")}), flush=True)
        return out
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a") as fh:
        futs = {pool.submit(_guarded, fn, *j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            out.append(r)
            print(json.dumps({"job": futs[fut], "status": r.get("status", "")}), flush=True)
    return out


def _same_run(record: dict[str, Any]) -> bool:
    return all(record.get(k) == v for k, v in RUN_KEY.items())


def run_certify(workers: int) -> None:
    path = output_paths()["certify"]
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if _same_run(r) and "steps" in r:  # never reuse another solver or setting
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
        futs = {pool.submit(_guarded, certify_job, *j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            st = [s["status"] for s in r.get("steps", [])] or r.get("status")
            print(f"[{i}/{len(jobs)}] {futs[fut]} {st}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["b1", "reference", "operating", "certify", "smoothing"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "b1":
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        target = OUT_DIR / f"b1_rerun_{stamp}.jsonl"  # never overwrite the committed records
        target.touch(exist_ok=False)
        run_pool(b1_job, [(8.0, 5.0), (5.0, 30.0), (10.0, 0.0)], args.workers, target)
    elif args.cmd == "operating":
        run_pool(operating_job, OPERATING_CELLS, args.workers, output_paths()["operating"])
    elif args.cmd == "reference":
        jobs = [
            (g, a, gm, k, REF_H)
            for g in REF_GRIDS
            for a, gm in REF_CELLS
            for k in ("dirichlet", "gibbs", "sparse_histogram")
        ]
        run_pool(reference_job, jobs, args.workers, output_paths()["reference"])
    elif args.cmd == "certify":
        run_certify(args.workers)
    else:
        run_pool(smoothing_job, list(SMOOTHING_CELLS), args.workers, output_paths()["smoothing"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
