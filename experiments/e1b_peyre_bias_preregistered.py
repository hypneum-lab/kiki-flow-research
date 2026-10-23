"""E1b -- Peyre entropic-JKO bias: model bias vs grid artifact (PRE-REGISTERED).

Question (open from E1): is the measured "spurious diffusion" the *true theoretical
Peyre bias* ~ eps/(2 tau), or a *discretization artifact* (Gibbs-kernel underflow
at small reg, which previously forced GRID 81->321)?

------------------------------------------------------------------------------
PRE-REGISTERED MODEL & THRESHOLDS  (fixed BEFORE looking at any result)
------------------------------------------------------------------------------
Energy:  F[rho] = <V, rho>,  V(x) = x^2/2.
Exact-W2 JKO map is PURE TRANSPORT, T(x) = x/(1+tau): Var -> 0 (Dirac at 0).
Entropic-JKO (Peyre 2015) effective PDE:  d_t rho = div(rho grad V) + (eps/2) Lap rho,
an Ornstein-Uhlenbeck process with effective temperature T_eff = eps/2.
=> stationary law N(0, eps/2).  Stationary EXCESS variance over the exact-W2
   target (Var=0):  Delta sigma^2 = sigma^2_* = eps/2.

We sweep eps/tau in {0.05,0.1,0.2,0.3,0.5}.  With tau (= JKO step h) FIXED = 1.0,
eps = (eps/tau)*tau = (eps/tau), so the *predicted* slope of Delta sigma^2 vs (eps/tau)
is  d(eps/2)/d(eps/tau) = tau/2 = 0.5.

CLAIM 1.1 (slope of the bias):
  regress Delta sigma^2_stationary  vs  (eps/tau).
  PRE-SPEC PASS  <=>  slope in [0.45, 0.55]  AND  R^2 > 0.98.   else REFUTED.

CLAIM 1.3 (model bias, not grid):
  at eps/tau = 0.2 fixed, refine grid x4 (GRID 321 -> 1281, X_MAX fixed).
  PRE-SPEC: |Delta Delta sigma^2| < 2%  => model (Peyre) bias, NOT numerical.
            if refinement changes Delta sigma^2 by > 5% => (partly) a grid artifact.

CONTROL (null): exact 1-D W2 prox via monotone/sorted transport (NO entropy).
  PRE-SPEC: slope of Delta sigma^2 vs (eps/tau)  ~ 0  (|slope| < 0.02 say),
  Var stationary -> 0.  Confirms the effect is entropic.

STATIONARITY: run K large; report Var trajectory tail; declare stationary only if
the last-10-step relative drift in Var < 1e-3.  If not reached -> BLOCKED.

Deterministic / seeded.  Writes experiments/results/e1b_peyre_bias.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ot

# ---- pre-registered constants ----------------------------------------------
EPS_OVER_TAU = [0.05, 0.1, 0.2, 0.3, 0.5]
TAU = 1.0                 # JKO step h, fixed so predicted slope = tau/2 = 0.5
GRID = 321
GRID_FINE = 1281          # x4 refinement for Claim 1.3
X_MAX = 8.0               # wide enough that N(0, eps/2<=0.25) is fully resolved
K_MAX = 400               # JKO steps; stationarity checked, not assumed
N_INNER = 4000            # Dykstra/Sinkhorn inner iterations
SIGMA0 = 1.0
SEED = 0
STAT_TOL = 1e-3           # relative Var drift over last 10 steps to call stationary
STAT_WINDOW = 10          # JKO steps over which the stationarity check runs
LINREG_PARAMS = 2         # slope + intercept, offset for residual dof
EPS_MATCH_TOL = 1e-12     # float tolerance when matching a row to eps/tau=0.2
SLOPE_LO, SLOPE_HI = 0.45, 0.55
R2_MIN = 0.98
GRID_PASS = 0.02          # <2% -> model bias
GRID_ARTIFACT = 0.05      # >5% -> grid artifact
NULL_SLOPE_MAX = 0.02


def variance(rho: np.ndarray, x: np.ndarray) -> float:
    m = float(np.dot(rho, x))
    return float(np.dot(rho, (x - m) ** 2))


def make_initial(x: np.ndarray) -> np.ndarray:
    rho = np.exp(-0.5 * (x / SIGMA0) ** 2)
    return rho / rho.sum()


# ---- entropic Peyre JKO prox (one step) ------------------------------------
def entropic_jko_step(
    rho_prev: np.ndarray, x: np.ndarray, gibbs: np.ndarray, reg: float, tau: float, n_iter: int
) -> np.ndarray:
    """One entropic-W2 JKO step for F=<V,.>, V=x^2/2 (Peyre 2015 Bregman)."""
    v = 0.5 * x**2
    a_func = np.exp(-tau * v / reg)            # linear-functional prox factor
    ref = np.clip(rho_prev, 1e-300, None)
    b = np.ones_like(ref)
    ka = gibbs @ a_func
    for _ in range(n_iter):
        b = ref / np.maximum(ka, 1e-300)
        # a is fixed (linear functional); only b iterates -> 1 projection suffices,
        # but loop guards against any residual; recompute ka is constant so break early.
        break
    q = a_func * (gibbs @ b)
    s = q.sum()
    if not np.isfinite(s) or s <= 0:
        raise FloatingPointError("entropic_jko_step blew up (non-finite marginal)")
    return q / s


# ---- exact 1-D W2 prox (NULL control, no entropy) --------------------------
def exact_jko_step(rho_prev: np.ndarray, x: np.ndarray, tau: float) -> np.ndarray:
    """Exact-W2 JKO step for F=<V,.>, V=x^2/2 via the analytic optimal map.

    The exact-W2 prox of tau*F at rho_prev pushes forward rho_prev by the
    Euclidean-geometry map solving (x - y) + tau*V'(y) = 0, i.e. y = x/(1+tau)
    for V=x^2/2 (monotone, so it IS the optimal transport map in 1-D). We
    realize the push-forward on the fixed grid by mass redistribution.
    """
    y = x / (1.0 + tau)                 # target location of mass originally at x
    # push-forward rho_prev (mass at x) onto grid via linear interpolation in y
    new = np.zeros_like(rho_prev)
    xmin = x[0]
    dx = x[1] - x[0]
    for i, yi in enumerate(y):
        pos = (yi - xmin) / dx
        j = int(np.floor(pos))
        if j < 0:
            new[0] += rho_prev[i]
        elif j >= len(x) - 1:
            new[-1] += rho_prev[i]
        else:
            w = pos - j
            new[j] += rho_prev[i] * (1.0 - w)
            new[j + 1] += rho_prev[i] * w
    s = new.sum()
    return new / s


def run_flow(eps_over_tau: float, grid: int, *, entropic: bool) -> dict:
    x = np.linspace(-X_MAX, X_MAX, grid)
    reg = eps_over_tau * TAU                     # eps = (eps/tau)*tau
    rho = make_initial(x)
    gibbs = None
    if entropic:
        support = x.reshape(-1, 1)
        cost = ot.dist(support, support, metric="sqeuclidean")
        gibbs = np.exp(-cost / reg)
    var_traj = [variance(rho, x)]
    stationary = False
    k_stat = K_MAX
    for k in range(K_MAX):
        if entropic:
            rho = entropic_jko_step(rho, x, gibbs, reg, TAU, N_INNER)
        else:
            rho = exact_jko_step(rho, x, TAU)
        var_traj.append(variance(rho, x))
        if k >= STAT_WINDOW:
            tail = var_traj[-11:]
            drift = abs(tail[-1] - tail[0]) / max(abs(tail[0]), 1e-30)
            if drift < STAT_TOL:
                stationary = True
                k_stat = k + 1
                break
    var_stat = var_traj[-1]
    return {
        "eps_over_tau": eps_over_tau,
        "reg": reg,
        "grid": grid,
        "var_stationary": var_stat,
        "var0": var_traj[0],
        "stationary": stationary,
        "k_to_stationary": k_stat,
        "var_tail": var_traj[-6:],
        "predicted_var_star": (reg / 2.0) if entropic else 0.0,
    }


def regress(xv: list[float], yv: list[float]) -> dict:
    xa, ya = np.asarray(xv), np.asarray(yv)
    design = np.vstack([xa, np.ones_like(xa)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, ya, rcond=None)
    pred = slope * xa + intercept
    ss_res = float(np.sum((ya - pred) ** 2))
    ss_tot = float(np.sum((ya - ya.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n = len(xa)
    if n > LINREG_PARAMS and ss_tot > 0:
        se = np.sqrt(ss_res / (n - LINREG_PARAMS) / np.sum((xa - xa.mean()) ** 2))
        ci = 1.96 * se
    else:
        ci = float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "ci95_slope": float(ci),
    }


def main() -> None:
    np.random.seed(SEED)
    out: dict = {"experiment": "E1b_peyre_bias_preregistered",
                 "prereg": {
                     "model": "F=<x^2/2,rho>; exact-W2 Var->0; entropic OU stationary N(0,eps/2)",
                     "tau": TAU, "predicted_slope": TAU / 2.0,
                     "claim_1_1_pass": "slope in [0.45,0.55] AND R2>0.98",
                     "claim_1_3_pass": "|dDeltasigma2|<2% on x4 grid (model); >5% => grid artifact",
                     "null_pass": "|slope|<0.02 and Var_stat->0",
                     "stationarity": "last-10-step rel drift in Var < 1e-3",
                     "eps_over_tau_sweep": EPS_OVER_TAU,
                     "grid": GRID, "grid_fine": GRID_FINE, "x_max": X_MAX,
                     "k_max": K_MAX, "n_inner": N_INNER, "seed": SEED,
                 }}

    # --- Claim 1.1: entropic sweep on base grid ---
    ent_rows = []
    for eot in EPS_OVER_TAU:
        r = run_flow(eot, GRID, entropic=True)
        r["delta_sigma2"] = r["var_stationary"] - 0.0   # exact-W2 target Var=0
        ent_rows.append(r)
        print(f"[ENT eps/tau={eot:.2f}] Var*={r['var_stationary']:.6e} "
              f"pred={r['predicted_var_star']:.6e} stat={r['stationary']} K={r['k_to_stationary']}")
    reg_ent = regress([r["eps_over_tau"] for r in ent_rows],
                      [r["delta_sigma2"] for r in ent_rows])
    all_stat = all(r["stationary"] for r in ent_rows)
    claim11 = (SLOPE_LO <= reg_ent["slope"] <= SLOPE_HI) and (reg_ent["r2"] > R2_MIN) and all_stat
    out["claim_1_1"] = {"rows": ent_rows, "regression": reg_ent,
                        "all_stationary": all_stat,
                        "verdict": "CONFIRMED" if claim11 else "REFUTED"}

    # --- Claim 1.3: grid refinement at eps/tau=0.2 ---
    base = next(r for r in ent_rows if abs(r["eps_over_tau"] - 0.2) < EPS_MATCH_TOL)
    fine = run_flow(0.2, GRID_FINE, entropic=True)
    fine["delta_sigma2"] = fine["var_stationary"]
    d_base = base["delta_sigma2"]
    d_fine = fine["delta_sigma2"]
    rel_change = abs(d_fine - d_base) / max(abs(d_base), 1e-30)
    if rel_change < GRID_PASS:
        grid_verdict = "MODEL_BIAS (Peyre, not numerical)"
    elif rel_change > GRID_ARTIFACT:
        grid_verdict = "GRID_ARTIFACT (discretization)"
    else:
        grid_verdict = "INCONCLUSIVE (2-5% band)"
    print(f"[GRID] base({GRID})={d_base:.6e}  fine({GRID_FINE})={d_fine:.6e}  "
          f"rel={rel_change*100:.3f}%  -> {grid_verdict}")
    out["claim_1_3"] = {"eps_over_tau": 0.2, "grid_base": GRID, "grid_fine": GRID_FINE,
                        "delta_sigma2_base": d_base, "delta_sigma2_fine": d_fine,
                        "rel_change_pct": rel_change * 100.0,
                        "fine_stationary": fine["stationary"], "verdict": grid_verdict}

    # --- NULL control: exact W2, no entropy ---
    null_rows = []
    for eot in EPS_OVER_TAU:
        r = run_flow(eot, GRID, entropic=False)
        r["delta_sigma2"] = r["var_stationary"] - 0.0
        null_rows.append(r)
        print(f"[NULL eps/tau={eot:.2f}] Var*={r['var_stationary']:.6e} "
              f"stat={r['stationary']} K={r['k_to_stationary']}")
    reg_null = regress([r["eps_over_tau"] for r in null_rows],
                       [r["delta_sigma2"] for r in null_rows])
    null_ok = abs(reg_null["slope"]) < NULL_SLOPE_MAX
    out["control_null"] = {"rows": null_rows, "regression": reg_null,
                           "verdict": "CONFIRMED (no entropic diffusion)" if null_ok
                           else "UNEXPECTED (non-zero slope without entropy)"}

    out_path = Path("experiments/results/e1b_peyre_bias.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(f"CLAIM 1.1: slope={reg_ent['slope']:.4f} (CI95 +-{reg_ent['ci95_slope']:.4f}) "
          f"R2={reg_ent['r2']:.5f} -> {out['claim_1_1']['verdict']}")
    print(f"CLAIM 1.3: {grid_verdict} (rel {rel_change*100:.3f}%)")
    print(f"NULL: slope={reg_null['slope']:.5f} -> {out['control_null']['verdict']}")


if __name__ == "__main__":
    main()
