"""E1 — epsilon->0 sweep: quantify the spurious (alpha/2) Delta-rho entropic diffusion.

Backlog EXPERIMENT-BACKLOG.md / E1. Canonical 1D potential energy
    F[rho] = \\int V rho,  V(x) = x^2 / 2   (OU contraction)
whose exact-W2 JKO map is analytic, pure transport, NO diffusion:
    T(x) = x / (1 + tau),  so  Var(rho_K) = Var0 * (1 + tau)^(-2K).

We run FullJKOSolver at fixed JKO time-step h (== tau per step) and sweep the
Sinkhorn entropic regularization epsilon in {1e-2, 5e-3, 2e-3, 1e-3, 5e-4}
(backlog set; the task-requested subset {1e-2,5e-3,1e-3,5e-4} is included).

We measure Var(rho_K), compare to the analytic exact reference, and report the
stationary spurious variance floor, theory ~ epsilon/(2*tau), at production
epsilon=0.01.

Deterministic / seeded. Writes experiments/results/e1_epsilon_sweep_var.json.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import ot

from kiki_flow_core.master_equation import FreeEnergy
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver

# Backlog set (superset of the task-requested {1e-2,5e-3,1e-3,5e-4}).
EPSILONS = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4]
PRODUCTION_EPS = 1e-2
EPS_MATCH_TOL = 1e-15  # float tolerance when matching a row to PRODUCTION_EPS
GRID = 321        # fine enough that exp(-cost/reg) does not underflow for reg>=5e-4
X_MAX = 4.0
H = 0.05          # JKO time-step == tau per step (matches track2 production h=0.05)
K_STEPS = 20      # number of JKO steps
SEED = 0
SIGMA0 = 1.0      # initial Gaussian std on the grid


class PotentialF(FreeEnergy):
    """F[rho] = <rho, V> with V(x) = x^2 / 2 on a fixed grid.

    Analytic simplex gradient: delta F / delta rho_i = V(x_i), projected to the
    simplex tangent (mean-subtracted) so a constant shift does not move mass off
    the simplex. This is the canonical OU potential; the exact-W2 JKO flow is
    pure transport T(x) = x/(1+tau) with no diffusion.
    """

    def __init__(self, support: np.ndarray) -> None:
        x = support[:, 0]
        self._V = 0.5 * x**2

    def value(self, state: FlowState) -> float:
        total = 0.0
        for rho in state.rho.values():
            if rho.size == self._V.size:
                total += float(np.dot(rho, self._V))
        return total

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        # Project out the mean: simplex-tangent gradient (mass-conserving).
        return self._V - self._V.mean()


def variance(rho: np.ndarray, x: np.ndarray) -> float:
    m = float(np.dot(rho, x))
    return float(np.dot(rho, (x - m) ** 2))


def swept_prox_w2(
    distribution: np.ndarray,  # the (Euclidean-advected) guess from JKOStep -- ignored here
    reference: np.ndarray,     # rho_tau == the previous JKO iterate (true proximal anchor)
    epsilon: float,            # supplied by JKOStep as self.h -- ignored (we use `h` below)
    support: np.ndarray,
    n_iter: int = 2000,
    *,
    reg: float,      # the swept Sinkhorn entropic regularization (gamma)
    h: float,        # the true JKO time-step tau (proximal pull strength)
    step_size: float = 0.1,  # unused (Dykstra is step-free); kept for API parity
) -> np.ndarray:
    """One genuine entropic JKO step for F[rho] = <V, rho>, V(x) = x^2/2.

    Solves the FULL proximal problem in Wasserstein-2 geometry,
        rho^+ = argmin_q  OT_reg(q, reference) / (2*h) + <V, q>,
    via Peyre (2015) Dykstra / iterative Bregman projection on the Gibbs kernel
    ``K = exp(-cost/reg)``. The linear functional ``<V,.>`` enters in closed form
    as a multiplicative factor ``exp(-h * V / reg)`` on the first marginal; the
    second marginal is pinned to ``reference``.

    The Euclidean ``distribution`` produced by JKOStep's explicit advection is NOT
    used: that update lives in flat (Euclidean) geometry and does not implement the
    W2 transport T(x)=x/(1+tau). The whole JKO step is carried correctly here. The
    swept Sinkhorn reg ``reg`` is held independent of the time-step ``h``.
    """
    x = support[:, 0]
    potential_v = 0.5 * x**2
    cost_matrix = ot.dist(support, support, metric="sqeuclidean")
    sink_reg = max(reg, 1e-12)  # no production clamp: GRID is fine enough for reg>=5e-4
    gibbs = np.exp(-cost_matrix / sink_reg)
    ref = np.clip(reference, 1e-300, None)
    # Proximal factor of the linear functional h*<V,.> in the Gibbs geometry.
    a_func = np.exp(-h * potential_v / sink_reg)
    b = np.ones_like(ref)
    for _ in range(n_iter):
        # First marginal carries the linear-functional prox (fixed factor a_func).
        a = a_func
        ka = gibbs @ a
        # Second marginal pinned to the previous iterate `reference`.
        b = ref / np.maximum(ka, 1e-300)
    q = a_func * (gibbs @ b)
    s = q.sum()
    if not np.isfinite(s) or s <= 0:
        raise FloatingPointError("swept_prox_w2 JKO iteration did not converge")
    return q / s


def make_initial(support: np.ndarray) -> np.ndarray:
    x = support[:, 0]
    rho = np.exp(-0.5 * (x / SIGMA0) ** 2)
    return rho / rho.sum()


def run_one(eps: float, support: np.ndarray, x: np.ndarray) -> dict[str, float]:
    rho0 = make_initial(support)
    state = FlowState(
        rho={"phono": rho0.copy()},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )
    prox_fn = partial(swept_prox_w2, reg=eps, h=H)
    f = PotentialF(support)
    solver = FullJKOSolver(f_functional=f, h=H, support=support, epsilon=eps)
    solver.prox_fn = prox_fn  # inject decoupled prox

    var0 = variance(rho0, x)
    for _ in range(K_STEPS):
        state = solver.step(state)
    var_k = variance(state.rho["phono"], x)
    return {"var0": var0, "var_K": var_k}


def main() -> None:
    np.random.seed(SEED)
    support = np.linspace(-X_MAX, X_MAX, GRID).reshape(-1, 1)
    x = support[:, 0]

    rho0 = make_initial(support)
    var0 = variance(rho0, x)
    # Exact-W2 reference: pure transport contraction, no diffusion.
    var_exact = var0 * (1.0 + H) ** (-2 * K_STEPS)

    rows = []
    for eps in EPSILONS:
        res = run_one(eps, support, x)
        diff = res["var_K"] - var_exact
        rows.append(
            {
                "epsilon": eps,
                "var_K": res["var_K"],
                "var_exact": var_exact,
                "var_diff": diff,
                "floor_theory_eps_over_2tau": eps / (2.0 * H),
            }
        )
        print(
            f"eps={eps:.1e}  Var(rho_K)={res['var_K']:.6e}  "
            f"Var_exact={var_exact:.6e}  diff={diff:+.6e}  "
            f"eps/(2tau)={eps/(2*H):.4e}"
        )

    prod = next(r for r in rows if abs(r["epsilon"] - PRODUCTION_EPS) < EPS_MATCH_TOL)

    out = {
        "experiment": "E1",
        "params": {
            "grid": GRID,
            "x_max": X_MAX,
            "h_tau_per_step": H,
            "K_steps": K_STEPS,
            "seed": SEED,
            "sigma0": SIGMA0,
            "potential": "V(x)=x^2/2",
            "exact_reference": "T(x)=x/(1+tau), Var=Var0*(1+tau)^(-2K)",
        },
        "tau_real_per_step": H,
        "tau_effective_total": H * K_STEPS,
        "var0": var0,
        "var_exact": var_exact,
        "production_epsilon": PRODUCTION_EPS,
        "floor_at_production_eps": prod["floor_theory_eps_over_2tau"],
        "var_diff_at_production_eps": prod["var_diff"],
        "rows": rows,
    }
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "e1_epsilon_sweep_var.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(
        f"tau (real, per step) = {H};  spurious floor eps/(2tau) at eps=0.01 "
        f"= {prod['floor_theory_eps_over_2tau']:.4e}"
    )


if __name__ == "__main__":
    main()
