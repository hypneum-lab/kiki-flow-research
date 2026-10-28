"""Certified implicit JKO step: regression tests from the Codex review of PR #50.

The implicit step minimises Q_h(q) = F(q) + sum_i OT_reg(q_i, rho_tau_i) / (2h)
by Bregman majorisation-minimisation with a line search on Q_h, certifies a
stationary point (KKT residual at the returned point) or fails explicitly
(``ImplicitJKOError``). No "F decreases" safeguard, no sub-stepping.
"""

from __future__ import annotations

import warnings

import numpy as np
import ot
import pytest

from kiki_flow_core.master_equation import (
    NONSEPARABLE_MODES,
    FreeEnergy,
    ImplicitJKOError,
    JKOStep,
    ZeroF,
)
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState, assert_invariants
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy, gradient_operator
from kiki_flow_core.track2_paper.paper_run import _make_jko
from kiki_flow_core.wasserstein_ops import (
    SinkhornNotConvergedError,
    entropic_ot,
    jko_entropic_prox_certified,
)

ENTROPIC_MIN_Q0 = 0.747012  # true entropic minimiser of the separable counter-example
SMALL_H_Q0 = 0.708128  # converged value (10 000 iterations) of the small-h case
ATOL = 1e-5
GRAD_TOL = 1e-6
STATIONARITY_TOL = 1e-5
PERTURBATION = 1e-3
PERTURBATION_SLACK = 1e-12


def _state(rho: dict[str, np.ndarray]) -> FlowState:
    return FlowState(
        rho=rho, P_theta=np.zeros(4), mu_curr=np.array([1.0]), tau=0, metadata={"track_id": "T2"}
    )


class EntropyTo(FreeEnergy):
    """F = KL(q || p) on one species, with its exact KL-prox."""

    def __init__(self, p: np.ndarray) -> None:
        self.p = p

    def value(self, state: FlowState) -> float:
        q = state.rho["a"]
        return float(np.sum(np.where(q > 0, q * np.log(np.maximum(q, 1e-300) / self.p), 0.0)))

    def kl_prox_log(self, state: FlowState, species_name: str):  # type: ignore[no-untyped-def]
        log_p = np.log(self.p)
        return lambda lp, tau: (lp + tau * log_p) / (1.0 + tau)


class Kink(FreeEnergy):
    """F = |q_0 - 1/4|: non-smooth, no exact gradient at the kink (Codex false stop)."""

    def value(self, state: FlowState) -> float:
        return abs(float(state.rho["a"][0]) - 0.25)

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        return np.array([np.sign(state.rho[species_name][0] - 0.25), 0.0])


def pot_entropic_ot(
    q: np.ndarray, r: np.ndarray, cost: np.ndarray, reg: float
) -> tuple[float, np.ndarray]:
    """Independent OT_reg value and first-marginal dual potential via POT (log domain)."""
    plan, log = ot.sinkhorn(
        q, r, cost, reg, method="sinkhorn_log", numItermax=20_000, stopThr=1e-11, log=True
    )
    safe = np.where(plan > 0.0, np.log(np.maximum(plan, 1e-300)), 0.0)
    value = float(np.sum(plan * cost) + reg * np.sum(plan * (safe - 1.0)))
    return value, reg * np.asarray(log["log_u"], dtype=float)


def _t2(grid: int, alpha: float, gamma: float, delta: float | None) -> T2FreeEnergy:
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
    return T2FreeEnergy(species, potentials, prior, 0.1 * gamma, turing_smoothing=delta)


def test_entropic_counterexample_returns_the_true_minimiser() -> None:
    """Codex B2: F rises at the true minimum of Q_h; no F guard may replace it."""
    state = _state({"a": np.array([0.8, 0.2])})
    jko = JKOStep(EntropyTo(state.rho["a"]), 1.0, np.array([[0.0], [1.0]]), scheme="jko_entropic")
    out = jko.step(state)
    assert jko.last_info["status"] == "converged"
    assert out.rho["a"][0] == pytest.approx(ENTROPIC_MIN_Q0, abs=ATOL)


def test_non_smooth_false_stop_fails_explicitly() -> None:
    """Codex B1: two identical outputs must not be taken as convergence."""
    state = _state({"a": np.array([0.8, 0.2])})
    jko = JKOStep(Kink(), 0.1, np.array([[0.0], [0.1]]), scheme="jko_entropic", max_outer=50)
    with pytest.raises(ImplicitJKOError) as err:
        jko.step(state)
    assert err.value.info["status"] in {"line_search", "max_outer"}
    assert err.value.info["kkt_residual"] > jko.kkt_tol


def test_small_h_inner_solve_is_certified_not_frozen() -> None:
    """Codex M3: a truncated inner prox must not be accepted as a result."""
    f = EntropyTo(np.array([0.8, 0.2]))
    jko = JKOStep(f, 1e-4, np.array([[0.0], [0.01]]), scheme="jko_entropic", reg=1e-8)
    out = jko.step(_state({"a": np.array([0.5, 0.5])}))
    assert jko.last_info["status"] == "converged"
    assert out.rho["a"][0] == pytest.approx(SMALL_H_Q0, abs=ATOL)


def test_inner_budget_exhaustion_raises() -> None:
    def prox(lp: np.ndarray, tau: float) -> np.ndarray:
        return (lp + tau * np.log(np.array([0.8, 0.2]))) / (1.0 + tau)

    with pytest.raises(SinkhornNotConvergedError):
        jko_entropic_prox_certified(
            prox, np.array([0.5, 0.5]), 1e-4, np.array([[0.0], [0.01]]), reg=1e-8, n_iter=5
        )
    with pytest.raises(SinkhornNotConvergedError):
        entropic_ot(
            np.array([0.3, 0.7]),
            np.array([0.6, 0.4]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            1e-4,
            n_iter=3,
        )


def test_failure_can_be_returned_with_a_warning_for_diagnostics() -> None:
    jko = JKOStep(
        Kink(),
        0.1,
        np.array([[0.0], [0.1]]),
        scheme="jko_entropic",
        max_outer=5,
        on_failure="warn",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jko.step(_state({"a": np.array([0.8, 0.2])}))
    assert jko.last_info["status"] != "converged"
    assert any("not certified" in str(w.message) for w in caught)


def test_invalid_inputs_and_settings() -> None:
    support = np.array([[0.0], [1.0]])
    jko = JKOStep(ZeroF(), 0.1, support, scheme="jko_entropic")
    with pytest.raises(ValueError, match="positive mass"):
        jko.step(_state({"a": np.zeros(2)}))
    off_grid = _state({"a": np.ones(3) / 3})
    np.testing.assert_array_equal(jko.step(off_grid).rho["a"], off_grid.rho["a"])
    assert jko.last_info["status"] == "converged"
    for bad in ({"kkt_tol": 0.0}, {"max_outer": 0}, {"l0": -1.0}, {"on_failure": "ignore"}):
        with pytest.raises(ValueError):
            JKOStep(ZeroF(), 0.1, support, scheme="jko_entropic", **bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        JKOStep(ZeroF(), float("nan"), support, scheme="jko_entropic")


def test_implicit_requires_a_declared_turing_smoothing() -> None:
    support = np.linspace(-2, 2, 6).reshape(-1, 1)
    assert NONSEPARABLE_MODES == ("explicit", "implicit")
    with pytest.raises(ValueError, match="smoothing"):
        FullJKOSolver(f_functional=_t2(6, 5.0, 5.0, None), h=0.05, support=support)
    explicit = FullJKOSolver(
        f_functional=_t2(6, 5.0, 5.0, None), h=0.05, support=support, nonseparable="explicit"
    )
    assert explicit.nonseparable == "explicit"
    assert _t2(6, 5.0, 0.0, None).implicit_ready
    assert _t2(6, 5.0, 5.0, 1e-5).implicit_ready
    solver = FullJKOSolver(
        f_functional=_t2(6, 5.0, 5.0, 1e-5),
        h=0.05,
        support=support,
        implicit_options={"kkt_tol": 1e-7, "max_outer": 77},
    )
    assert solver.kkt_tol == 1e-7  # noqa: PLR2004
    assert solver.max_outer == 77  # noqa: PLR2004


def test_smoothed_turing_gradient_is_exact_including_near_a_cusp() -> None:
    """Codex M2: the forward difference is not a derivative across sign changes."""
    f = _t2(4, 0.0, 1.0, 1e-6)
    names = f.species.species_names()
    rhos = [
        np.array([0.25003, 0.25001, 0.24999, 0.24997]),
        np.array([0.4, 0.3, 0.2, 0.1]),
        np.full(4, 0.25),
        np.full(4, 0.25),
    ]
    state = _state(dict(zip(names, rhos, strict=True)))
    grad = f.nonseparable_grad(state, names[0]) - 2.0 * f._grad_conservative(rhos)[0]
    eps = 1e-9
    fd = np.zeros(4)
    for i in range(4):
        up = [r.copy() for r in rhos]
        dn = [r.copy() for r in rhos]
        up[0][i] += eps
        dn[0][i] -= eps
        fd[i] = (f._turing_value(up) - f._turing_value(dn)) / (2 * eps)
    np.testing.assert_allclose(grad, fd, atol=GRAD_TOL)
    np.testing.assert_allclose(gradient_operator(4) @ rhos[1], np.gradient(rhos[1]))


def test_small_coupled_problem_is_stationary_by_an_independent_check() -> None:
    """Certified step checked with POT, independently of the solver's own Sinkhorn.

    At the returned q the gradient of Q_h, grad F + f / (2h) with f the POT dual
    potential of OT_reg(q_i, rho_tau_i), must be constant on each species'
    support (first-order condition on the simplex), and Q_h must not decrease
    along small tangent perturbations. The L-BFGS comparison against an
    independent minimiser is in experiments/implicit_jko_certified_validation_
    2026_10_27.py (too slow for the test suite).
    """
    grid, h = 4, 0.2
    f = _t2(grid, 2.0, 5.0, 1e-3)
    names = f.species.species_names()
    # Support [-0.5, 0.5]: max cost / reg = 25, so the dual potential of the
    # balanced problem is numerically identifiable (at cost / reg ~ 400 the
    # kernel underflows and different potentials give the same plan).
    support = np.linspace(-0.5, 0.5, grid).reshape(-1, 1)
    rng = np.random.default_rng(3)
    rho = {n: rng.dirichlet(np.ones(grid) * 3.0) for n in names}
    jko = JKOStep(f, h, support, scheme="jko_entropic", kkt_tol=1e-7, max_outer=200)
    out = jko.step(_state(rho))
    assert jko.last_info["status"] == "converged"
    assert_invariants(out)
    q = {n: out.rho[n] for n in names}
    cost = (support[:, 0][:, None] - support[:, 0][None, :]) ** 2
    reg = jko.entropic_reg
    st = _state(q)

    def q_value(qq: dict[str, np.ndarray]) -> float:
        val = f.value(_state(qq))
        for n in names:
            val += pot_entropic_ot(qq[n], rho[n], cost, reg)[0] / (2 * h)
        return val

    for n in names:
        _, pot = pot_entropic_ot(q[n], rho[n], cost, reg)
        g = (
            f.potentials[n]
            + np.log(q[n] / f.prior[n])
            + 1.0
            + f.nonseparable_grad(st, n)
            + pot / (2 * h)
        )
        assert float(np.max(q[n] * np.abs(g - float(np.dot(q[n], g))))) < STATIONARITY_TOL
    base = q_value(q)
    for _ in range(4):
        pert = {}
        for n in names:
            d = rng.standard_normal(grid)
            d -= d.mean()
            pert[n] = q[n] + PERTURBATION * d * q[n].min()
        assert q_value(pert) >= base - PERTURBATION_SLACK


def test_run_paper_forwards_nonseparable_in_all_branches() -> None:
    support = np.linspace(-2, 2, 6).reshape(-1, 1)
    f = _t2(6, 0.0, 1.0, 1e-5)
    for use_w2_prox, backend in ((False, "pot"), (True, "pot")):
        jko = _make_jko(f, support, use_w2_prox, backend, "jko_entropic", "explicit", None)
        assert jko.nonseparable == "explicit"
        assert jko.scheme == "jko_entropic"
    jko = _make_jko(f, support, False, "pot", None, "implicit", {"max_outer": 9})
    assert jko.scheme == "legacy_split"
    assert jko.max_outer == 9  # noqa: PLR2004
