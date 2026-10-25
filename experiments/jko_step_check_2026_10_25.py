"""Is ``JKOStep`` a JKO step? Single-step identification on a 1-D support.

Companion to ``docs/notes/2026-10-25-jko-step-identification.md`` (NH-60,
issue #31, critique B5 of 2026-10-25). The step under test is
``JKOStep.step`` exactly as the rigorous path runs it (``FullJKOSolver``:
``n_inner`` projected Euclidean gradient sub-steps of F from rho_tau, then
``prox_w2`` with the KL pull toward the sub-step output and the entropic W2
anchor on rho_tau) and as the fast path runs it (no prox).

For each test energy F (quadratic potential + entropy, double well +
entropy, entropy alone) and each h of a grid, two references are computed
from the same rho_tau:

* the exact JKO minimiser ``argmin_rho F(rho) + W2^2(rho, rho_tau) / (2h)``
  on the simplex, with the exact (LP) W2 of POT, by Kelley cutting planes on
  the convex piecewise-affine ``W2^2(., rho_tau)`` (each cut is an LP dual
  potential of ``ot.emd``), with a duality-gap certificate; optionally
  cross-checked against a conic solve of the plan formulation when ``cvxpy``
  is importable (scratch venv only, not a project dependency);
* the entropic JKO minimiser of Peyre (2015) for the same F and the same
  Sinkhorn regularisation, by the generalised Sinkhorn scaling iteration.

Distances (TV, exact W2) between the step outputs and the references are
recorded as functions of h, ``n_inner``, ``reg`` and of the gradient
convention handed to ``JKOStep``; the dissipation inequality
``F(rho_1) + W2^2(rho_1, rho_0)/(2h) <= F(rho_0)`` and the mass on the
right of the barrier are recorded for every step. A short multi-step
trajectory compares the composite scheme with the exact JKO sequence at
fixed final time.

Deterministic (no random component). NumPy + SciPy + POT only (SciPy is a
transitive dependency of the lock through scikit-learn); no MLX, no JAX.
Outputs (new files): ``results/jko_step_check_2026_10_25/{results.json,
summary.md, fig_jko_step_check.png, fig_jko_step_check.pdf}``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import ot
import scipy
from scipy.linalg import expm
from scipy.optimize import minimize

from kiki_flow_core.master_equation import FreeEnergy, JKOStep
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver

try:  # optional conic cross-check of the exact reference (scratch venv only)
    import cvxpy as cp
except ImportError:  # pragma: no cover - cvxpy is not a project dependency
    cp = None

try:  # figure only; matplotlib is a dev extra
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

SPECIES = "rho"
RHO_FLOOR = 1e-12
TINY = 1e-300
BERNOULLI_LINEAR_TOL = 1e-12  # below this |z|, z / (e^z - 1) is taken as 1
GRID = 48
X_MIN, X_MAX = -1.0, 1.0
H_GRID = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
N_INNER_GRID = [1, 10, 50]
# Sinkhorn regularisation of the prox; None = the prox default (reg = h).
REG_GRID: list[float | None] = [0.005, 0.01, 0.05, None]
GRAD_MODES = ("numerical", "tangential", "raw")
PROX_N_ITER = 1000  # FullJKOSolver default max_iter (forwarded since 2026-10-25)
PAPER_H = 0.05
PAPER_REG = 0.01
PAPER_N_INNER = 50  # FullJKOSolver default n_inner
TRAJ_H = [0.1, 0.05, 0.025]
TRAJ_T = 0.5
EXACT_TOL = 1e-9
EXACT_MAX_CUTS = 600
ENTROPIC_N_ITER = 50000
ENTROPIC_TOL = 1e-13
WARM_GAMMA = 0.002
OUT_DIR = Path("results/jko_step_check_2026_10_25")

QUICK_H = [0.1, 0.05]
QUICK_N_INNER = [1, 50]
QUICK_REG: list[float | None] = [0.01, None]
QUICK_TRAJ_H = [0.1]
QUICK_TRAJ_T = 0.2

# Categorical palette, fixed slot order (dataviz reference instance).
COLOR_JKO = "#2a78d6"
COLOR_RIGOROUS = "#eb6834"
COLOR_FAST = "#1baf7a"
COLOR_PEYRE = "#eda100"
COLOR_RHO_TAU = "#8a8a85"
GRIDLINE = "#e1e0d9"
INK = "#3c3c3a"

Energy = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]
KLProx = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class EnergySpec:
    """F(rho) = <rho, potential> + theta * sum_i rho_i log rho_i, from rho_tau."""

    name: str
    label: str
    potential: np.ndarray
    theta: float
    rho_tau: np.ndarray
    description: str


@dataclass
class ExactJKO:
    """Exact JKO minimiser with its cutting-plane certificate."""

    rho: np.ndarray
    objective: float  # F(rho) + W2^2(rho, rho_tau)/(2h), exact W2
    lower_bound: float  # model minimum (lower bound up to the inner solver tolerance)
    gap: float
    n_cuts: int
    converged: bool
    dual_violation: float  # max_ij u_i + v_j - C_ij over the cuts (0 up to rounding)
    inner_failures: int


def output_paths(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    return {
        "results": out_dir / "results.json",
        "summary": out_dir / "summary.md",
        "figure_png": out_dir / "fig_jko_step_check.png",
        "figure_pdf": out_dir / "fig_jko_step_check.pdf",
    }


# ----------------------------------------------------------------------------
# Problem
# ----------------------------------------------------------------------------


def support(grid: int = GRID) -> np.ndarray:
    return np.linspace(X_MIN, X_MAX, grid)


def cost_matrix(x: np.ndarray) -> np.ndarray:
    return (x[:, None] - x[None, :]) ** 2


def gaussian(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    g = np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return g / g.sum()


def simplex(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=float), RHO_FLOOR, None)
    return v / v.sum()


def energies(x: np.ndarray) -> list[EnergySpec]:
    bimodal = simplex(0.7 * gaussian(x, -0.5, 0.1) + 0.3 * gaussian(x, 0.5, 0.1))
    return [
        EnergySpec(
            "quad",
            "quadratic potential + entropy",
            4.0 * x**2,
            0.1,
            simplex(gaussian(x, -0.5, 0.12)),
            "V = 4 x^2, theta = 0.1, rho_tau = N(-0.5, 0.12)",
        ),
        EnergySpec(
            "dw",
            "double well + entropy",
            (x**2 - 0.25) ** 2 / 0.0625,
            0.05,
            bimodal,
            "V = (x^2 - 1/4)^2 / (1/16) (wells at +-0.5, barrier 1), theta = 0.05, "
            "rho_tau = 0.7 N(-0.5, 0.1) + 0.3 N(0.5, 0.1)",
        ),
        EnergySpec(
            "ent",
            "entropy alone (T2 leading term)",
            np.zeros_like(x),
            1.0,
            bimodal,
            "V = 0, theta = 1 (KL to the uniform prior up to a constant), same bimodal rho_tau",
        ),
    ]


class PotentialEntropyF(FreeEnergy):
    """F(rho) = <rho, V> + theta * sum_i rho_i log rho_i on a fixed grid.

    ``grad_mode`` selects what ``JKOStep`` receives as the gradient:
    ``numerical`` keeps the base-class finite difference (what ``T2FreeEnergy``
    uses), ``tangential`` is its analytic limit dF_i - <dF, rho>, ``raw`` is
    the plain first variation dF_i = V_i + theta (log rho_i + 1).
    """

    def __init__(self, spec: EnergySpec, grad_mode: str = "numerical") -> None:
        if grad_mode not in GRAD_MODES:
            raise ValueError(f"grad_mode must be one of {GRAD_MODES}, got {grad_mode!r}")
        self.potential = spec.potential
        self.theta = spec.theta
        self.grad_mode = grad_mode

    def energy(self, rho: np.ndarray) -> float:
        safe = np.clip(rho, RHO_FLOOR, None)
        return float(np.dot(rho, self.potential) + self.theta * np.sum(safe * np.log(safe)))

    def first_variation(self, rho: np.ndarray) -> np.ndarray:
        safe = np.clip(rho, RHO_FLOOR, None)
        return self.potential + self.theta * (np.log(safe) + 1.0)

    def kl_prox(self) -> KLProx:
        """prox of tau * F in the KL geometry on R^n_+ (closed form)."""
        potential, theta = self.potential, self.theta

        def prox(p: np.ndarray, tau: float) -> np.ndarray:
            logp = np.log(np.maximum(p, TINY))
            return np.exp((logp - tau * potential - tau * theta) / (1.0 + tau * theta))

        return prox

    def value(self, state: FlowState) -> float:
        return self.energy(state.rho[SPECIES])

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        if self.grad_mode == "numerical":
            return super().grad_rho(state, species_name, eps)
        rho = state.rho[species_name]
        g = self.first_variation(rho)
        if self.grad_mode == "tangential":
            return g - float(np.dot(g, rho))
        return g


def kl_prox_to_target(target: np.ndarray) -> KLProx:
    """prox of tau * KL(. || target) in the KL geometry: p^(1/(1+tau)) target^(tau/(1+tau))."""
    safe_target = np.maximum(target, TINY)

    def prox(p: np.ndarray, tau: float) -> np.ndarray:
        lam = 1.0 / (1.0 + tau)
        return np.maximum(p, TINY) ** lam * safe_target ** (1.0 - lam)

    return prox


def _bernoulli(z: np.ndarray) -> np.ndarray:
    out = np.ones_like(z)
    nz = np.abs(z) > BERNOULLI_LINEAR_TOL
    out[nz] = z[nz] / np.expm1(z[nz])
    return out


def fp_generator(spec: EnergySpec, x: np.ndarray) -> np.ndarray:
    """Scharfetter-Gummel generator of d rho/dt = d/dx (rho V' + theta rho') on the grid.

    Finite volumes on cell masses with zero-flux ends; the Gibbs state exp(-V/theta)
    is stationary and mass is conserved (columns sum to zero). It is the standard
    discrete Fokker-Planck flow of F and is itself a gradient flow of the discrete
    F for a discrete transport metric (Maas 2011), so exp(h L) rho_tau is the
    grid-level "W2 flow of F over time h" that a JKO step approximates to O(h^2).
    """
    dx = float(x[1] - x[0])
    delta = np.diff(spec.potential) / spec.theta  # (V_{i+1} - V_i) / theta
    scale = spec.theta / dx**2
    rate_right = scale * _bernoulli(delta)  # mass rate i -> i + 1 per unit m_i
    rate_left = scale * _bernoulli(-delta)  # mass rate i + 1 -> i per unit m_{i+1}
    n = x.size
    gen = np.zeros((n, n))
    idx = np.arange(n - 1)
    gen[idx, idx] -= rate_right
    gen[idx + 1, idx] += rate_right
    gen[idx + 1, idx + 1] -= rate_left
    gen[idx, idx + 1] += rate_left
    return gen


def fp_flow(gen: np.ndarray, rho: np.ndarray, t: float) -> np.ndarray:
    """exp(t L) rho: the discrete Fokker-Planck flow over time t."""
    return simplex(expm(t * gen) @ np.asarray(rho, dtype=float))


# ----------------------------------------------------------------------------
# The step under test
# ----------------------------------------------------------------------------


def make_state(rho: np.ndarray) -> FlowState:
    return FlowState(
        rho={SPECIES: np.asarray(rho, dtype=float)},
        P_theta=np.zeros(1),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )


def kiki_step(
    f: FreeEnergy,
    rho_tau: np.ndarray,
    x: np.ndarray,
    h: float,
    n_inner: int,
    *,
    rigorous: bool,
    reg: float | None = None,
) -> np.ndarray:
    """One ``JKOStep`` on rho_tau: fast path (no prox) or rigorous path.

    The rigorous path is ``FullJKOSolver`` with ``epsilon = reg`` forwarded as
    the Sinkhorn regularisation and ``max_iter = PROX_N_ITER``; ``reg = None``
    keeps the prox default (Sinkhorn reg = h) through plain ``JKOStep``.
    """
    sup = x.reshape(-1, 1)
    if not rigorous:
        jko: JKOStep = JKOStep(
            f_functional=f, h=h, support=sup, n_inner=n_inner, apply_w2_prox=False
        )
    elif reg is None:
        jko = JKOStep(
            f_functional=f,
            h=h,
            support=sup,
            n_inner=n_inner,
            apply_w2_prox=True,
            prox_n_iter=PROX_N_ITER,
        )
    else:
        jko = FullJKOSolver(
            f_functional=f,
            h=h,
            support=sup,
            epsilon=reg,
            max_iter=PROX_N_ITER,
            n_inner=n_inner,
        )
    return jko.step(make_state(rho_tau)).rho[SPECIES]


# ----------------------------------------------------------------------------
# Distances
# ----------------------------------------------------------------------------


def w2sq_exact(p: np.ndarray, q: np.ndarray, cost: np.ndarray) -> float:
    a = np.asarray(p, dtype=float)
    b = np.asarray(q, dtype=float)
    return float(ot.emd2(a / a.sum(), b / b.sum(), cost))


def w2_exact(p: np.ndarray, q: np.ndarray, cost: np.ndarray) -> float:
    return float(np.sqrt(max(w2sq_exact(p, q, cost), 0.0)))


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def w2sq_quantile(p: np.ndarray, q: np.ndarray, x: np.ndarray) -> float:
    """Exact 1-D W2^2 from the quantile functions (independent of POT)."""
    cp_, cq = np.cumsum(p), np.cumsum(q)
    knots = np.unique(np.concatenate([[0.0], cp_, cq]))
    knots = knots[knots <= 1.0 + 1e-12]
    mids = 0.5 * (knots[1:] + knots[:-1])
    ip = np.minimum(np.searchsorted(cp_, mids), x.size - 1)
    iq = np.minimum(np.searchsorted(cq, mids), x.size - 1)
    return float(np.sum(np.diff(knots) * (x[ip] - x[iq]) ** 2))


# ----------------------------------------------------------------------------
# References
# ----------------------------------------------------------------------------


def _solve_model(
    energy: Energy,
    grad: Gradient,
    phis: np.ndarray,
    consts: np.ndarray,
    h: float,
    z0: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """min F(rho) + t/(2h) s.t. t >= <phi_k, rho> + c_k, rho on the simplex."""
    n = z0.size - 1
    k = phis.shape[0]
    ineq_jac_mat = np.hstack([-phis, np.ones((k, 1))])
    eq_jac_mat = np.append(np.ones(n), 0.0)[None, :]

    def fun(z: np.ndarray) -> float:
        return energy(z[:n]) + z[n] / (2.0 * h)

    def jac(z: np.ndarray) -> np.ndarray:
        return np.append(grad(z[:n]), 1.0 / (2.0 * h))

    def eq(z: np.ndarray) -> np.ndarray:
        return np.array([z[:n].sum() - 1.0])

    def eq_jac(_: np.ndarray) -> np.ndarray:
        return eq_jac_mat

    def ineq(z: np.ndarray) -> np.ndarray:
        return z[n] - phis @ z[:n] - consts

    def ineq_jac(_: np.ndarray) -> np.ndarray:
        return ineq_jac_mat

    res = minimize(
        fun,
        z0,
        jac=jac,
        method="SLSQP",
        bounds=[(RHO_FLOOR, 1.0)] * n + [(0.0, None)],
        constraints=[
            {"type": "eq", "fun": eq, "jac": eq_jac},
            {"type": "ineq", "fun": ineq, "jac": ineq_jac},
        ],
        options={"maxiter": 500, "ftol": 1e-14},
    )
    rho = simplex(res.x[:n])
    t_model = max(float(np.max(phis @ rho + consts)), 0.0)
    return np.append(rho, t_model), energy(rho) + t_model / (2.0 * h), bool(res.success)


def jko_exact(
    energy: Energy,
    grad: Gradient,
    rho_tau: np.ndarray,
    cost: np.ndarray,
    h: float,
    *,
    tol: float = EXACT_TOL,
    max_cuts: int = EXACT_MAX_CUTS,
    warm_starts: Sequence[np.ndarray] = (),
) -> ExactJKO:
    """argmin_rho F(rho) + W2^2(rho, rho_tau)/(2h) on the simplex, exact (LP) W2.

    Kelley cutting planes on the convex piecewise-affine W(rho) = W2^2(rho, rho_tau):
    the LP dual (u, v) of ``ot.emd`` at an iterate gives the cut <u, rho> + <v, rho_tau>,
    tight there and below W everywhere (dual feasibility). The inner model minimum is
    a lower bound (up to the SLSQP tolerance), the true objective at any iterate an
    upper bound; the loop stops when the gap is below ``tol``.
    """
    rho_tau = simplex(rho_tau)
    n = rho_tau.size
    phis: list[np.ndarray] = []
    consts: list[float] = []
    violation = 0.0

    def add_cut(rho: np.ndarray) -> float:
        nonlocal violation
        plan, log = ot.emd(rho, rho_tau, cost, log=True)
        u, v = np.asarray(log["u"], dtype=float), np.asarray(log["v"], dtype=float)
        w = float((plan * cost).sum())
        violation = max(violation, float((u[:, None] + v[None, :] - cost).max()))
        phis.append(u)
        consts.append(w - float(np.dot(u, rho)))
        return w

    best_rho = rho_tau.copy()
    best_obj = energy(best_rho) + add_cut(best_rho) / (2.0 * h)
    for start in warm_starts:
        candidate = simplex(start)
        obj = energy(candidate) + add_cut(candidate) / (2.0 * h)
        if obj < best_obj:
            best_obj, best_rho = obj, candidate
    z0 = np.append(best_rho, 2.0 * h * (best_obj - energy(best_rho)))
    lower = -np.inf
    converged = False
    failures = 0
    for _ in range(max_cuts):
        z, lower, ok = _solve_model(energy, grad, np.array(phis), np.array(consts), h, z0)
        failures += int(not ok)
        rho = z[:n]
        obj = energy(rho) + add_cut(rho) / (2.0 * h)
        if obj < best_obj:
            best_obj, best_rho = obj, rho
        z0 = z
        if best_obj - lower <= tol:
            converged = True
            break
    return ExactJKO(
        rho=best_rho,
        objective=best_obj,
        lower_bound=float(lower),
        gap=float(best_obj - lower),
        n_cuts=len(phis),
        converged=converged,
        dual_violation=violation,
        inner_failures=failures,
    )


def jko_entropic(
    prox: KLProx,
    rho_tau: np.ndarray,
    cost: np.ndarray,
    h: float,
    gamma: float,
    *,
    n_iter: int = ENTROPIC_N_ITER,
    tol: float = ENTROPIC_TOL,
) -> tuple[np.ndarray, dict[str, float]]:
    """Entropic JKO step of Peyre (2015): argmin_q W_gamma(q, rho_tau) + 2h F(q).

    Generalised Sinkhorn (dual block-coordinate ascent, Chizat et al. 2018) on
    pi = diag(a) K diag(b), K = exp(-C/gamma): a <- prox_{(2h/gamma) F}(K b) / (K b),
    b <- rho_tau / (K^T a). With ``prox = kl_prox_to_target(d)`` this is exactly
    the loop of ``wasserstein_ops.prox_w2(d, rho_tau, epsilon=h, reg=gamma)``.
    """
    gibbs = np.exp(-cost / gamma)
    tau = 2.0 * h / gamma
    a = np.ones_like(rho_tau)
    b = np.ones_like(rho_tau)
    delta = np.inf
    iterations = 0
    while iterations < n_iter and delta >= tol:
        iterations += 1
        kb = np.maximum(gibbs @ b, TINY)
        a = prox(kb, tau) / kb
        ka = np.maximum(gibbs @ a, TINY)
        b_new = rho_tau / ka
        delta = float(np.abs(np.log(np.maximum(b_new, TINY)) - np.log(np.maximum(b, TINY))).max())
        b = b_new
    q = a * (gibbs @ b)
    residual = float(np.abs((gibbs.T @ a) * b - rho_tau).sum())
    info = {"iterations": float(iterations), "marginal_residual": residual, "last_delta": delta}
    return simplex(q), info


def jko_exact_cvxpy(spec: EnergySpec, rho_tau: np.ndarray, cost: np.ndarray, h: float) -> Any:
    """Conic cross-check of the exact JKO step (plan formulation); None without cvxpy."""
    if cp is None:
        return None
    n = rho_tau.size
    plan = cp.Variable((n, n), nonneg=True)
    rho = cp.sum(plan, axis=1)
    objective = (
        cp.sum(cp.multiply(cost, plan)) / (2.0 * h)
        + spec.potential @ rho
        - spec.theta * cp.sum(cp.entr(rho))
    )
    problem = cp.Problem(cp.Minimize(objective), [cp.sum(plan, axis=0) == rho_tau])
    problem.solve(solver="CLARABEL")
    if rho.value is None:
        return None
    return simplex(np.asarray(rho.value, dtype=float))


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------


def step_metrics(
    rho0: np.ndarray,
    rho1: np.ndarray,
    energy: Energy,
    cost: np.ndarray,
    x: np.ndarray,
    h: float,
    ref_rho: np.ndarray,
    peyre_rho: np.ndarray,
    fast_rho: np.ndarray,
    flow_rho: np.ndarray,
) -> dict[str, float]:
    f0, f1 = energy(rho0), energy(rho1)
    w2sq_move = w2sq_exact(rho1, rho0, cost)
    w2_disp_jko = w2_exact(ref_rho, rho0, cost)
    w2_vs_jko = w2_exact(rho1, ref_rho, cost)
    w2_disp_flow = w2_exact(flow_rho, rho0, cost)
    w2_vs_flow = w2_exact(rho1, flow_rho, cost)
    return {
        "f_before": f0,
        "f_after": f1,
        "w2_move": float(np.sqrt(w2sq_move)),
        "dissipation_defect": f1 + w2sq_move / (2.0 * h) - f0,
        "tv_vs_jko": tv(rho1, ref_rho),
        "w2_vs_jko": w2_vs_jko,
        "rel_w2_vs_jko": w2_vs_jko / w2_disp_jko if w2_disp_jko > 0 else float("nan"),
        "tv_vs_peyre": tv(rho1, peyre_rho),
        "w2_vs_peyre": w2_exact(rho1, peyre_rho, cost),
        "tv_vs_fast": tv(rho1, fast_rho),
        "w2_vs_fast": w2_exact(rho1, fast_rho, cost),
        "tv_vs_flow": tv(rho1, flow_rho),
        "w2_vs_flow": w2_vs_flow,
        "rel_w2_vs_flow": w2_vs_flow / w2_disp_flow if w2_disp_flow > 0 else float("nan"),
        "mass_right": float(rho1[x > 0].sum()),
    }


def reference_record(
    spec: EnergySpec,
    f: PotentialEntropyF,
    x: np.ndarray,
    cost: np.ndarray,
    h: float,
    regs: Sequence[float | None],
    gen: np.ndarray,
) -> dict[str, Any]:
    rho_tau = spec.rho_tau
    rho_flow = fp_flow(gen, rho_tau, h)
    warm_fast = kiki_step(f, rho_tau, x, h, PAPER_N_INNER, rigorous=False)
    warm_peyre, _ = jko_entropic(f.kl_prox(), rho_tau, cost, h, WARM_GAMMA)
    t0 = time.perf_counter()
    exact = jko_exact(
        f.energy, f.first_variation, rho_tau, cost, h, warm_starts=[warm_peyre, warm_fast]
    )
    exact_s = time.perf_counter() - t0
    w2sq_disp = w2sq_exact(exact.rho, rho_tau, cost)
    peyre: dict[str, Any] = {}
    for reg in regs:
        gamma = h if reg is None else reg
        q, info = jko_entropic(f.kl_prox(), rho_tau, cost, h, gamma)
        peyre[reg_key(reg)] = {
            "gamma": gamma,
            "rho": q,
            "sinkhorn": info,
            "tv_vs_jko": tv(q, exact.rho),
            "w2_vs_jko": w2_exact(q, exact.rho, cost),
            "w2_vs_flow": w2_exact(q, rho_flow, cost),
            "f": f.energy(q),
            "dissipation_defect": f.energy(q)
            + w2sq_exact(q, rho_tau, cost) / (2.0 * h)
            - f.energy(rho_tau),
            "mass_right": float(q[x > 0].sum()),
        }
    record: dict[str, Any] = {
        "h": h,
        "rho_jko": exact.rho,
        "objective": exact.objective,
        "lower_bound": exact.lower_bound,
        "gap": exact.gap,
        "n_cuts": exact.n_cuts,
        "converged": exact.converged,
        "dual_violation": exact.dual_violation,
        "inner_failures": exact.inner_failures,
        "solve_seconds": exact_s,
        "f_rho_tau": f.energy(rho_tau),
        "f_jko": f.energy(exact.rho),
        "w2_disp_jko": float(np.sqrt(w2sq_disp)),
        "dissipation_defect": f.energy(exact.rho) + w2sq_disp / (2.0 * h) - f.energy(rho_tau),
        "mass_right_rho_tau": float(rho_tau[x > 0].sum()),
        "mass_right_jko": float(exact.rho[x > 0].sum()),
        "rho_flow": rho_flow,
        "w2_disp_flow": w2_exact(rho_flow, rho_tau, cost),
        "w2_jko_vs_flow": w2_exact(exact.rho, rho_flow, cost),
        "tv_jko_vs_flow": tv(exact.rho, rho_flow),
        "mass_right_flow": float(rho_flow[x > 0].sum()),
        "peyre": peyre,
    }
    record["rel_jko_vs_flow"] = (
        record["w2_jko_vs_flow"] / record["w2_disp_flow"] if record["w2_disp_flow"] > 0 else None
    )
    cross = jko_exact_cvxpy(spec, rho_tau, cost, h)
    if cross is not None:
        record["cvxpy"] = {
            "tv_vs_cutting_plane": tv(cross, exact.rho),
            "objective": f.energy(cross) + w2sq_exact(cross, rho_tau, cost) / (2.0 * h),
        }
    return record


def reg_key(reg: float | None) -> str:
    return "default(h)" if reg is None else f"{reg:g}"


def run_case(
    spec: EnergySpec,
    x: np.ndarray,
    cost: np.ndarray,
    *,
    h_grid: Sequence[float],
    n_inner_grid: Sequence[int],
    reg_grid: Sequence[float | None],
    grad_modes: Sequence[str],
    verbose: bool = True,
) -> dict[str, Any]:
    f_ref = PotentialEntropyF(spec, "numerical")
    gen = fp_generator(spec, x)
    references = []
    for h in h_grid:
        rec = reference_record(spec, f_ref, x, cost, h, reg_grid, gen)
        references.append(rec)
        if verbose:
            print(
                f"[{spec.name}] h={h:g} exact JKO: gap={rec['gap']:.2e} cuts={rec['n_cuts']} "
                f"{rec['solve_seconds']:.1f}s  W2(rho_tau, JKO)={rec['w2_disp_jko']:.4f} "
                f"W2(JKO, flow)={rec['w2_jko_vs_flow']:.4f} W2(rho_tau, flow)="
                f"{rec['w2_disp_flow']:.4f}",
                flush=True,
            )
    steps = []
    for rec in references:
        h = rec["h"]
        for grad_mode in grad_modes:
            f = PotentialEntropyF(spec, grad_mode)
            for n_inner in n_inner_grid:
                fast = kiki_step(f, spec.rho_tau, x, h, n_inner, rigorous=False)
                peyre_default = rec["peyre"].get(
                    reg_key(PAPER_REG), next(iter(rec["peyre"].values()))
                )
                base = {"h": h, "grad_mode": grad_mode, "n_inner": n_inner}
                steps.append(
                    {
                        **base,
                        "path": "fast",
                        "reg": None,
                        "rho": fast,
                        **step_metrics(
                            spec.rho_tau,
                            fast,
                            f.energy,
                            cost,
                            x,
                            h,
                            rec["rho_jko"],
                            peyre_default["rho"],
                            fast,
                            rec["rho_flow"],
                        ),
                    }
                )
                for reg in reg_grid:
                    rig = kiki_step(f, spec.rho_tau, x, h, n_inner, rigorous=True, reg=reg)
                    steps.append(
                        {
                            **base,
                            "path": "rigorous",
                            "reg": reg,
                            "reg_key": reg_key(reg),
                            "rho": rig,
                            **step_metrics(
                                spec.rho_tau,
                                rig,
                                f.energy,
                                cost,
                                x,
                                h,
                                rec["rho_jko"],
                                rec["peyre"][reg_key(reg)]["rho"],
                                fast,
                                rec["rho_flow"],
                            ),
                        }
                    )
        if verbose:
            print(f"[{spec.name}] h={h:g} steps done ({len(steps)} so far)", flush=True)
    return {
        "name": spec.name,
        "label": spec.label,
        "description": spec.description,
        "theta": spec.theta,
        "potential": spec.potential,
        "rho_tau": spec.rho_tau,
        "references": references,
        "steps": steps,
    }


def run_trajectory(
    spec: EnergySpec, x: np.ndarray, cost: np.ndarray, h: float, t_final: float
) -> dict[str, Any]:
    """Composite scheme vs exact JKO sequence vs fast path, same h, up to t_final."""
    n_steps = int(round(t_final / h))
    f = PotentialEntropyF(spec, "numerical")
    gen = fp_generator(spec, x)
    rho_rig = rho_fast = rho_jko = rho_flow = spec.rho_tau
    rows = []
    for k in range(1, n_steps + 1):
        rho_rig = kiki_step(f, rho_rig, x, h, PAPER_N_INNER, rigorous=True, reg=PAPER_REG)
        rho_fast = kiki_step(f, rho_fast, x, h, PAPER_N_INNER, rigorous=False)
        ref = jko_exact(f.energy, f.first_variation, rho_jko, cost, h, warm_starts=[rho_jko])
        rho_jko = ref.rho
        rho_flow = fp_flow(gen, spec.rho_tau, k * h)
        rows.append(
            {
                "step": k,
                "t": k * h,
                "gap": ref.gap,
                "w2_rig_vs_flow": w2_exact(rho_rig, rho_flow, cost),
                "w2_jko_vs_flow": w2_exact(rho_jko, rho_flow, cost),
                "w2_fast_vs_flow": w2_exact(rho_fast, rho_flow, cost),
                "w2_flow_from_rho_tau": w2_exact(rho_flow, spec.rho_tau, cost),
                "w2_rig_vs_jko": w2_exact(rho_rig, rho_jko, cost),
                "tv_rig_vs_jko": tv(rho_rig, rho_jko),
                "w2_fast_vs_jko": w2_exact(rho_fast, rho_jko, cost),
                "w2_rig_vs_fast": w2_exact(rho_rig, rho_fast, cost),
                "w2_jko_from_rho_tau": w2_exact(rho_jko, spec.rho_tau, cost),
                "f_rig": f.energy(rho_rig),
                "f_fast": f.energy(rho_fast),
                "f_jko": f.energy(rho_jko),
                "mass_right_rig": float(rho_rig[x > 0].sum()),
                "mass_right_jko": float(rho_jko[x > 0].sum()),
            }
        )
    return {
        "h": h,
        "t_final": n_steps * h,
        "n_steps": n_steps,
        "n_inner": PAPER_N_INNER,
        "reg": PAPER_REG,
        "rows": rows,
        "final": {"rigorous": rho_rig, "fast": rho_fast, "jko": rho_jko, "flow": rho_flow},
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        return "unknown"


def metadata() -> dict[str, Any]:
    return {
        "script": "experiments/jko_step_check_2026_10_25.py",
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pot": ot.__version__,
        "cvxpy": None if cp is None else cp.__version__,
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return [float(v) for v in obj.ravel()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def select_steps(case: dict[str, Any], **where: Any) -> list[dict[str, Any]]:
    out = []
    for s in case["steps"]:
        if all(s.get(k) == v for k, v in where.items()):
            out.append(s)
    return out


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{digits}g}"


def _main_table(case: dict[str, Any]) -> list[str]:
    lines = [
        "| h | W2(rho_tau, JKO) | W2(rho_tau, rig) | W2(rig, JKO) | rel. | TV(rig, JKO) "
        "| W2(rig, fast) | W2(rig, Peyre) | W2(Peyre, JKO) | defect rig | defect fast "
        "| defect JKO | mass right: tau / JKO / rig / fast | gap |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in case["references"]:
        h = rec["h"]
        rig = select_steps(
            case,
            h=h,
            path="rigorous",
            reg=PAPER_REG,
            n_inner=PAPER_N_INNER,
            grad_mode="numerical",
        )
        fast = select_steps(case, h=h, path="fast", n_inner=PAPER_N_INNER, grad_mode="numerical")
        if not rig or not fast:
            continue
        r, fa = rig[0], fast[0]
        pey = rec["peyre"].get(reg_key(PAPER_REG), {})
        lines.append(
            f"| {h:g} | {_fmt(rec['w2_disp_jko'])} | {_fmt(r['w2_move'])} | {_fmt(r['w2_vs_jko'])} "
            f"| {_fmt(r['rel_w2_vs_jko'], 3)} | {_fmt(r['tv_vs_jko'])} | {_fmt(r['w2_vs_fast'])} "
            f"| {_fmt(r['w2_vs_peyre'])} | {_fmt(pey.get('w2_vs_jko'))} "
            f"| {_fmt(r['dissipation_defect'], 3)} | {_fmt(fa['dissipation_defect'], 3)} "
            f"| {_fmt(rec['dissipation_defect'], 3)} "
            f"| {rec['mass_right_rho_tau']:.4f} / {rec['mass_right_jko']:.4f} "
            f"/ {r['mass_right']:.4f} / {fa['mass_right']:.4f} | {rec['gap']:.1e} |"
        )
    return lines


def _flow_table(case: dict[str, Any]) -> list[str]:
    lines = [
        "| h | W2(rho_tau, flow) | W2(JKO, flow) | rel. | W2(rig, flow) | rel. "
        "| W2(fast, flow) | rel. | W2(Peyre, flow) | rel. | mass right flow |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in case["references"]:
        h = rec["h"]
        rig = select_steps(
            case,
            h=h,
            path="rigorous",
            reg=PAPER_REG,
            n_inner=PAPER_N_INNER,
            grad_mode="numerical",
        )
        fast = select_steps(case, h=h, path="fast", n_inner=PAPER_N_INNER, grad_mode="numerical")
        if not rig or not fast:
            continue
        r, fa = rig[0], fast[0]
        disp = rec["w2_disp_flow"]
        pey = rec["peyre"].get(reg_key(PAPER_REG), {}).get("w2_vs_flow")
        pey_rel = pey / disp if pey is not None and disp > 0 else None
        lines.append(
            f"| {h:g} | {_fmt(disp)} | {_fmt(rec['w2_jko_vs_flow'])} "
            f"| {_fmt(rec.get('rel_jko_vs_flow'), 3)} | {_fmt(r['w2_vs_flow'])} "
            f"| {_fmt(r['rel_w2_vs_flow'], 3)} | {_fmt(fa['w2_vs_flow'])} "
            f"| {_fmt(fa['rel_w2_vs_flow'], 3)} | {_fmt(pey)} | {_fmt(pey_rel, 3)} "
            f"| {rec['mass_right_flow']:.4f} |"
        )
    return lines


def _config_table(case: dict[str, Any], h: float, reg_grid: Sequence[float | None]) -> list[str]:
    header = "| n_inner | fast | " + " | ".join(f"reg = {reg_key(r)}" for r in reg_grid) + " |"
    lines = [header, "|---|---|" + "---|" * len(reg_grid)]
    n_inners = sorted({s["n_inner"] for s in case["steps"]})
    for n_inner in n_inners:
        fast = select_steps(case, h=h, path="fast", n_inner=n_inner, grad_mode="numerical")
        cells = [_fmt(fast[0]["rel_w2_vs_jko"], 3) if fast else "—"]
        for reg in reg_grid:
            rig = select_steps(
                case, h=h, path="rigorous", reg=reg, n_inner=n_inner, grad_mode="numerical"
            )
            cells.append(_fmt(rig[0]["rel_w2_vs_jko"], 3) if rig else "—")
        lines.append(f"| {n_inner} | " + " | ".join(cells) + " |")
    return lines


def _grad_mode_lines(case: dict[str, Any], h: float) -> list[str]:
    lines = []
    for mode in GRAD_MODES:
        rig = select_steps(
            case, h=h, path="rigorous", reg=PAPER_REG, n_inner=PAPER_N_INNER, grad_mode=mode
        )
        if rig:
            r = rig[0]
            lines.append(
                f"- `{mode}`: W2(rig, JKO) = {_fmt(r['w2_vs_jko'])} "
                f"(rel. {_fmt(r['rel_w2_vs_jko'], 3)}), TV = {_fmt(r['tv_vs_jko'])}, "
                f"defect = {_fmt(r['dissipation_defect'], 3)}"
            )
    return lines


def _trajectory_lines(trajs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| h | steps | t | W2(rig, JKO) | TV(rig, JKO) | W2(fast, JKO) | W2(rig, fast) "
        "| W2(JKO, rho_tau) | W2(rig, flow) | W2(JKO, flow) | W2(fast, flow) "
        "| F rig / fast / JKO | mass right rig / JKO | max gap |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tr in trajs:
        last = tr["rows"][-1]
        lines.append(
            f"| {tr['h']:g} | {tr['n_steps']} | {last['t']:.3g} | {_fmt(last['w2_rig_vs_jko'])} "
            f"| {_fmt(last['tv_rig_vs_jko'])} | {_fmt(last['w2_fast_vs_jko'])} "
            f"| {_fmt(last['w2_rig_vs_fast'])} | {_fmt(last['w2_jko_from_rho_tau'])} "
            f"| {_fmt(last['w2_rig_vs_flow'])} | {_fmt(last['w2_jko_vs_flow'])} "
            f"| {_fmt(last['w2_fast_vs_flow'])} "
            f"| {_fmt(last['f_rig'])} / {_fmt(last['f_fast'])} / {_fmt(last['f_jko'])} "
            f"| {last['mass_right_rig']:.4f} / {last['mass_right_jko']:.4f} "
            f"| {max(r['gap'] for r in tr['rows']):.1e} |"
        )
    return lines


def write_summary(results: dict[str, Any], path: Path) -> None:
    meta = results["meta"]
    params = results["params"]
    lines = [
        "# Is `JKOStep` a JKO step? Single-step check (2026-10-25)",
        "",
        f"Script `{meta['script']}` at `{meta['git_commit']}`, host `{meta['host']}` "
        f"({meta['machine']}), Python {meta['python']}, NumPy {meta['numpy']}, "
        f"SciPy {meta['scipy']}, POT {meta['pot']}, cvxpy {meta['cvxpy'] or 'absent'}, "
        f"{meta['timestamp_utc']}, {results['runtime_seconds']:.0f} s.",
        "",
        f"Support: {params['grid']} points on [{X_MIN:g}, {X_MAX:g}], cost |x - y|^2. "
        f"Prox Dykstra iterations {params['prox_n_iter']}. `rig` = `FullJKOSolver` "
        f"(n_inner = {PAPER_N_INNER}, Sinkhorn reg = {PAPER_REG}, base-class numerical "
        "gradient as `T2FreeEnergy`), `fast` = `JKOStep` without prox, `JKO` = exact "
        "minimiser of F + W2^2(., rho_tau)/(2h) (cutting planes, exact LP W2), `Peyre` = "
        f"entropic JKO of F at gamma = reg = {PAPER_REG}. `rel.` = W2(rig, JKO) / "
        "W2(rho_tau, JKO). `defect` = F(rho_1) + W2^2(rho_1, rho_tau)/(2h) - F(rho_tau) "
        "(<= 0 when the dissipation inequality holds). `gap` = cutting-plane duality gap.",
        "",
    ]
    for case in results["cases"].values():
        ref_by_h = {r["h"]: r for r in case["references"]}
        lines += [f"## `{case['name']}` — {case['label']}", "", case["description"], ""]
        lines += _main_table(case)
        lines += [
            "",
            "Same step against the discrete Fokker-Planck flow exp(h L) rho_tau "
            "(Scharfetter-Gummel generator L of the W2 flow of F on the grid; `rel.` = "
            "distance / W2(rho_tau, flow)). The exact JKO step approximates the flow to "
            "O(h^2) until the grid pins it (displacement below one cell):",
            "",
        ]
        lines += _flow_table(case)
        h_cfg = PAPER_H if PAPER_H in ref_by_h else case["references"][0]["h"]
        lines += [
            "",
            f"Relative W2 error of the step vs exact JKO at h = {h_cfg:g}, by n_inner and reg "
            "(numerical gradient):",
            "",
        ]
        lines += _config_table(case, h_cfg, params["reg_grid"])
        lines += [
            "",
            f"Gradient convention at h = {h_cfg:g} (n_inner {PAPER_N_INNER}, reg {PAPER_REG}):",
            "",
        ]
        lines += _grad_mode_lines(case, h_cfg)
        cross = [r["cvxpy"]["tv_vs_cutting_plane"] for r in case["references"] if "cvxpy" in r]
        if cross:
            lines += [
                "",
                f"cvxpy/Clarabel cross-check of the exact reference: max TV "
                f"{max(cross):.2e} over {len(cross)} h values.",
            ]
        trajs = results.get("trajectories", {}).get(case["name"], [])
        if trajs:
            lines += [
                "",
                f"Trajectories (n_inner {PAPER_N_INNER}, reg {PAPER_REG}, final time):",
                "",
            ]
            lines += _trajectory_lines(trajs)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(results: dict[str, Any], png: Path, pdf: Path) -> None:
    if plt is None:  # pragma: no cover
        print("matplotlib absent, no figure", flush=True)
        return
    cases = list(results["cases"].values())
    fig, axes = plt.subplots(len(cases), 3, figsize=(13.5, 3.6 * len(cases)))
    axes = np.atleast_2d(axes)
    for row, case in enumerate(cases):
        _plot_error_panel(axes[row, 0], case)
        _plot_dissipation_panel(axes[row, 1], case)
        _plot_density_panel(axes[row, 2], case, results["params"])
        axes[row, 0].set_ylabel(f"{case['label']}\nW2(step, JKO) / W2(rho_tau, JKO)")
    for ax in axes.ravel():
        ax.grid(True, color=GRIDLINE, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8)
    fig.suptitle(
        "JKOStep vs exact JKO step (1-D, same rho_tau): error, dissipation, one step at h = "
        f"{PAPER_H:g}",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(png, dpi=160)
    fig.savefig(pdf)
    plt.close(fig)


def _series(case: dict[str, Any], key: str, **where: Any) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(select_steps(case, **where), key=lambda s: s["h"])
    return np.array([r["h"] for r in rows]), np.array([r[key] for r in rows])


def _plot_error_panel(ax: Any, case: dict[str, Any]) -> None:
    styles = {1: ":", 10: "--", 50: "-"}
    for n_inner, ls in styles.items():
        hs, ys = _series(
            case,
            "rel_w2_vs_jko",
            path="rigorous",
            reg=PAPER_REG,
            n_inner=n_inner,
            grad_mode="numerical",
        )
        if hs.size:
            ax.plot(
                hs,
                ys,
                ls,
                color=COLOR_RIGOROUS,
                lw=1.6,
                marker="o",
                ms=4,
                label=f"rigorous, n_inner={n_inner}",
            )
    hs, ys = _series(
        case, "rel_w2_vs_jko", path="fast", n_inner=PAPER_N_INNER, grad_mode="numerical"
    )
    if hs.size:
        ax.plot(
            hs,
            ys,
            "-",
            color=COLOR_FAST,
            lw=1.6,
            marker="s",
            ms=4,
            label=f"fast, n_inner={PAPER_N_INNER}",
        )
    refs = [
        r
        for r in sorted(case["references"], key=lambda r: r["h"])
        if reg_key(PAPER_REG) in r["peyre"] and r["w2_disp_jko"] > 0
    ]
    if refs:
        ax.plot(
            [r["h"] for r in refs],
            [r["peyre"][reg_key(PAPER_REG)]["w2_vs_jko"] / r["w2_disp_jko"] for r in refs],
            "-",
            color=COLOR_PEYRE,
            lw=1.6,
            marker="^",
            ms=4,
            label=f"Peyre entropic JKO of F, gamma={PAPER_REG}",
        )
    flow_refs = [
        r
        for r in sorted(case["references"], key=lambda r: r["h"])
        if r.get("rel_jko_vs_flow") is not None
    ]
    if flow_refs:
        ax.plot(
            [r["h"] for r in flow_refs],
            [r["rel_jko_vs_flow"] for r in flow_refs],
            "--",
            color=COLOR_JKO,
            lw=1.6,
            marker="d",
            ms=4,
            label="exact JKO vs FP flow / W2(rho_tau, flow)",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("h")
    ax.axhline(1.0, color=INK, lw=0.8, alpha=0.6)
    ax.legend(fontsize=7, frameon=False)


def _plot_dissipation_panel(ax: Any, case: dict[str, Any]) -> None:
    hs, ys = _series(
        case,
        "dissipation_defect",
        path="rigorous",
        reg=PAPER_REG,
        n_inner=PAPER_N_INNER,
        grad_mode="numerical",
    )
    ax.plot(hs, ys, "-", color=COLOR_RIGOROUS, lw=1.6, marker="o", ms=4, label="rigorous")
    hs, ys = _series(
        case, "dissipation_defect", path="fast", n_inner=PAPER_N_INNER, grad_mode="numerical"
    )
    ax.plot(hs, ys, "-", color=COLOR_FAST, lw=1.6, marker="s", ms=4, label="fast")
    refs = sorted(case["references"], key=lambda r: r["h"])
    ax.plot(
        [r["h"] for r in refs],
        [r["dissipation_defect"] for r in refs],
        "-",
        color=COLOR_JKO,
        lw=1.6,
        marker="d",
        ms=4,
        label="exact JKO",
    )
    ax.axhline(0.0, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("h")
    ax.set_ylabel("F(rho_1) + W2^2/(2h) - F(rho_0)  (<= 0 required)")
    ax.legend(fontsize=7, frameon=False)


def _plot_density_panel(ax: Any, case: dict[str, Any], params: dict[str, Any]) -> None:
    x = np.asarray(params["support"])
    refs = {r["h"]: r for r in case["references"]}
    h = PAPER_H if PAPER_H in refs else case["references"][0]["h"]
    rec = refs[h]
    rig = select_steps(
        case, h=h, path="rigorous", reg=PAPER_REG, n_inner=PAPER_N_INNER, grad_mode="numerical"
    )
    fast = select_steps(case, h=h, path="fast", n_inner=PAPER_N_INNER, grad_mode="numerical")
    ax.plot(x, np.asarray(case["rho_tau"]), "-", color=COLOR_RHO_TAU, lw=1.6, label="rho_tau")
    ax.plot(x, np.asarray(rec["rho_jko"]), "-", color=COLOR_JKO, lw=1.6, label="exact JKO")
    pey = rec["peyre"].get(reg_key(PAPER_REG))
    if pey is not None:
        ax.plot(
            x,
            np.asarray(pey["rho"]),
            "-",
            color=COLOR_PEYRE,
            lw=1.6,
            label=f"Peyre, gamma={PAPER_REG}",
        )
    if rig:
        ax.plot(
            x,
            np.asarray(rig[0]["rho"]),
            "-",
            color=COLOR_RIGOROUS,
            lw=1.6,
            label="rigorous (FullJKOSolver)",
        )
    if fast:
        ax.plot(x, np.asarray(fast[0]["rho"]), "--", color=COLOR_FAST, lw=1.6, label="fast")
    ax.set_xlabel("x")
    ax.set_ylabel(f"density after one step, h = {h:g}")
    ax.legend(fontsize=7, frameon=False)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="reduced grids (smoke run)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-trajectory", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="re-render summary and figure from an existing results.json, no computation",
    )
    return parser.parse_args(argv)


def render(results: dict[str, Any], out_dir: Path, *, figure: bool = True) -> dict[str, Path]:
    paths = output_paths(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(results, paths["summary"])
    if figure:
        make_figure(results, paths["figure_png"], paths["figure_pdf"])
    return paths


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.from_json is not None:
        results = json.loads(args.from_json.read_text(encoding="utf-8"))
        render(results, args.out_dir, figure=not args.no_figure)
        return results
    t_start = time.perf_counter()
    h_grid = QUICK_H if args.quick else H_GRID
    n_inner_grid = QUICK_N_INNER if args.quick else N_INNER_GRID
    reg_grid = QUICK_REG if args.quick else REG_GRID
    grad_modes = ("numerical",) if args.quick else GRAD_MODES
    traj_h = QUICK_TRAJ_H if args.quick else TRAJ_H
    traj_t = QUICK_TRAJ_T if args.quick else TRAJ_T
    x = support()
    cost = cost_matrix(x)
    results: dict[str, Any] = {
        "meta": metadata(),
        "params": {
            "grid": GRID,
            "support": x,
            "h_grid": list(h_grid),
            "n_inner_grid": list(n_inner_grid),
            "reg_grid": list(reg_grid),
            "grad_modes": list(grad_modes),
            "prox_n_iter": PROX_N_ITER,
            "exact_tol": EXACT_TOL,
            "exact_max_cuts": EXACT_MAX_CUTS,
            "trajectory_h": list(traj_h),
            "trajectory_t": traj_t,
            "quick": args.quick,
        },
        "cases": {},
        "trajectories": {},
    }
    for spec in energies(x):
        results["cases"][spec.name] = run_case(
            spec,
            x,
            cost,
            h_grid=h_grid,
            n_inner_grid=n_inner_grid,
            reg_grid=reg_grid,
            grad_modes=grad_modes,
            verbose=not args.quiet,
        )
        if not args.skip_trajectory:
            results["trajectories"][spec.name] = [
                run_trajectory(spec, x, cost, h, traj_t) for h in traj_h
            ]
            if not args.quiet:
                for tr in results["trajectories"][spec.name]:
                    last = tr["rows"][-1]
                    print(
                        f"[{spec.name}] trajectory h={tr['h']:g} t={last['t']:.3g}: "
                        f"W2(rig, JKO)={last['w2_rig_vs_jko']:.4f} "
                        f"W2(fast, JKO)={last['w2_fast_vs_jko']:.4f}",
                        flush=True,
                    )
    results["runtime_seconds"] = time.perf_counter() - t_start
    paths = output_paths(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths["results"].write_text(json.dumps(_jsonable(results), indent=1), encoding="utf-8")
    render(results, args.out_dir, figure=not args.no_figure)
    if not args.quiet:
        print(f"wrote {paths['results']} ({results['runtime_seconds']:.0f} s)", flush=True)
    return results


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
