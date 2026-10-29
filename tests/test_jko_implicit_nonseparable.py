"""Certified implicit JKO step: regression tests from the Codex review of PR #50.

The implicit step minimises Q_h(q) = F(q) + sum_i OT_reg(q_i, rho_tau_i) / (2h)
by Bregman majorisation-minimisation with a line search on Q_h, certifies a
stationary point (KKT residual at the returned point) or fails explicitly
(``ImplicitJKOError``). No "F decreases" safeguard, no sub-stepping.
"""

from __future__ import annotations

import warnings
from decimal import Decimal, getcontext

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
PERTURBATION = 1e-4  # absolute mass moved between two bins (all q entries > 1e-2 here)
PERTURBATION_SLACK = 1e-11


def _state(rho: dict[str, np.ndarray]) -> FlowState:
    return FlowState(
        rho=rho, P_theta=np.zeros(4), mu_curr=np.array([1.0]), tau=0, metadata={"track_id": "T2"}
    )


class EntropyTo(FreeEnergy):
    """F = KL(q || p) on one species, with its exact KL-prox."""

    implicit_ready = True  # exact KL-prox (kl_prox_log override)

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

    implicit_ready = True  # declares a (sub)gradient; non-smooth, so the step must fail

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


class Quad(FreeEnergy):
    """F = 8 x + 8 x^2 on q = (x, 1 - x), exact gradient (Codex boundary counter-example)."""

    implicit_ready = True

    def value(self, state: FlowState) -> float:
        x = float(state.rho["a"][0])
        return 8.0 * x + 8.0 * x * x

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        x = float(state.rho[species_name][0])
        return np.array([8.0 + 16.0 * x, 0.0])


class FDOnly(FreeEnergy):
    """Smooth energy relying on the inherited finite-difference gradient."""

    def value(self, state: FlowState) -> float:
        x = float(state.rho["a"][0])
        return 1000.0 * (x - 0.3) ** 2


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


def test_low_mass_false_stationarity_is_rejected() -> None:
    """Codex B1 on PR #51: an absolute TV residual accepted x = 1.3e-14 (x* = 1.13e-7).

    Zero cost, h = reg = 1: Q(x) = 8 x + 8 x^2 + (x log x + (1-x) log(1-x)) / 2 + c,
    so x* solves 8 + 16 x + log(x / (1-x)) / 2 = 0.
    """
    state = _state({"a": np.array([0.5, 0.5])})
    jko = JKOStep(Quad(), 1.0, np.array([[0.0], [0.0]]), scheme="jko_entropic", reg=1.0)
    try:
        out = jko.step(state)
    except ImplicitJKOError as err:  # an explicit failure is acceptable, a false pass is not
        assert err.info["status"] != "converged"
        return
    x = float(out.rho["a"][0])
    x_star = 1.0 / (1.0 + np.exp(16.0 + 32.0 * 1.13e-7))
    assert jko.last_info["log_residual"] < jko.log_tol
    assert abs(x - x_star) / x_star < 1e-3  # noqa: PLR2004
    assert abs(8.0 + 16.0 * x + 0.5 * np.log(x / (1.0 - x))) < 1e-3  # noqa: PLR2004


def test_energy_without_exact_gradient_is_refused_even_after_mutation() -> None:
    """Codex M1 on PR #51: the inherited finite difference must not be certified."""
    support = np.array([[0.0], [0.0]])
    with pytest.raises(ValueError, match="implicit_ready"):
        JKOStep(FDOnly(), 1.0, support, scheme="jko_entropic", reg=1.0)
    jko = JKOStep(FDOnly(), 1.0, support, scheme="jko_entropic", reg=1.0, nonseparable="explicit")
    jko.nonseparable = "implicit"
    with pytest.raises(ValueError, match="implicit_ready"):
        jko.step(_state({"a": np.array([0.3, 0.7])}))


def test_t2_entropy_has_no_floor_below_1e_12() -> None:
    """Codex M2 on PR #51: value() and the prox must minimise the same entropy."""
    f = _t2(4, 0.0, 0.0, 1e-5)  # smoothed energy: exact entropy
    names = f.species.species_names()
    rho = {n: np.full(4, 0.25) for n in names}
    tiny = np.array([1e-14, 0.0, 0.5, 0.5 - 1e-14])
    rho[names[0]] = tiny
    expected_kl = float(
        1e-14 * np.log(1e-14 / 0.25)
        + 0.5 * np.log(2.0)
        + (0.5 - 1e-14) * np.log((0.5 - 1e-14) / 0.25)
    )
    base = f.value(_state({**rho, names[0]: np.full(4, 0.25)}))
    assert f.value(_state(rho)) - base == pytest.approx(
        expected_kl + _coupling_delta(f, rho, names), abs=1e-13
    )


def _coupling_delta(f: T2FreeEnergy, rho: dict[str, np.ndarray], names: list[str]) -> float:
    rhos = [rho[n] for n in names]
    ref = [np.full(4, 0.25) for _ in names]
    j = f._J_sym

    def coupling(rs: list[np.ndarray]) -> float:
        return float(sum(j[i, k] * np.dot(rs[i], rs[k]) for i in range(4) for k in range(4)))

    return coupling(rhos) - coupling(ref)


class ShiftedEntropy(FreeEnergy):
    """F = 100 q_0 + sum q log q with its exact prox (Codex B2 on PR #51, review v2)."""

    @property
    def implicit_ready(self) -> bool:
        return True

    def value(self, state: FlowState) -> float:
        q = state.rho["a"]
        return float(np.sum(q * np.log(q)) + 100.0 * q[0])

    def kl_prox_log(self, state: FlowState, species_name: str):  # type: ignore[no-untyped-def]
        shift = np.array([100.0, 0.0])
        return lambda lp, tau: (lp - tau * shift) / (1.0 + tau)


def test_tiny_reference_mass_inner_solve_is_certified_in_relative_terms() -> None:
    """Review v2, B2: cost 1000, h = reg = 1, reference (1e-12, 1 - 1e-12).

    With one inner iteration the absolute marginal residual is 1e-12 (accepted
    before) but the relative residual on the tiny bin is about 26 in log: the
    step must fail. With the full budget the step is certified. The 2x2 problem
    is solved in closed form: the optimal plan has pi_01 pi_10 = exp(-2000)
    pi_00 pi_11, and the tangent derivative of Q_h is 1.5 log x - 400 -
    log(r_0 - x) / 2 for x < r_0, which vanishes at r_0 - x* ~ exp(-883): x*
    rounds to r_0 in double precision, and the step returns r_0 within a few
    ulps (the derivative there, about -410, is the log of a float-resolution
    gap of 8e-28).
    """
    r = np.array([1e-12, 1.0 - 1e-12])
    support = np.array([[0.0], [np.sqrt(1000.0)]])
    one = JKOStep(ShiftedEntropy(), 1.0, support, scheme="jko_entropic", reg=1.0, inner_n_iter=1)
    with pytest.raises(ImplicitJKOError) as err:
        one.step(_state({"a": r}))
    assert err.value.info["status"] == "inner_not_converged"
    full = JKOStep(ShiftedEntropy(), 1.0, support, scheme="jko_entropic", reg=1.0)
    x = float(full.step(_state({"a": r})).rho["a"][0])
    assert full.last_info["status"] == "converged"
    assert full.last_info["inner_max_log_residual"] <= full.inner_log_tol
    assert abs(x / r[0] - 1.0) < 1e-14  # noqa: PLR2004


def test_inner_certificate_is_relative_on_low_mass_bins() -> None:
    """An absolute marginal residual alone accepts a wrong low-mass column."""
    ref = np.array([1e-12, 1.0 - 1e-12])
    support = np.array([[0.0], [np.sqrt(1000.0)]])
    shift = np.array([100.0, 0.0])

    def prox(log_p: np.ndarray, tau: float) -> np.ndarray:
        return (log_p - tau * shift) / (1.0 + tau)

    _, info = jko_entropic_prox_certified(
        prox, ref, 1.0, support, reg=1.0, n_iter=1, marginal_log_tol=np.inf
    )
    assert info["marginal_residual"] <= 1e-10  # noqa: PLR2004  absolute test passes
    assert info["marginal_log_residual"] > 1.0  # relative error on the tiny bin
    with pytest.raises(SinkhornNotConvergedError):
        jko_entropic_prox_certified(prox, ref, 1.0, support, reg=1.0, n_iter=1)


def test_certified_entry_points_check_masses() -> None:
    """Review v2, M4 / m1: unequal masses are refused, not certified."""
    with pytest.raises(ValueError, match="equal masses"):
        entropic_ot(np.array([0.5, 0.5]), np.ones(2), np.zeros((2, 2)), 1.0, n_iter=10)
    with pytest.raises(ValueError, match="mass 1"):
        jko_entropic_prox_certified(lambda lp, t: lp, np.ones(2), 1.0, np.zeros((2, 1)), reg=1.0)


def test_operating_point_bracket_is_certified_by_newton_polishing() -> None:
    """Sinkhorn stalls near a gap of 1e-6 at cost / reg = 6400; the bracket must still close."""
    rng = np.random.default_rng(500)
    support = np.linspace(-2.0, 2.0, 16).reshape(-1, 1)
    cost = ((support - support.T) ** 2).astype(float)
    q, r = rng.dirichlet(np.ones(16)), rng.dirichlet(np.ones(16))
    _, _, info = entropic_ot(q, r, cost, 0.0025, gap_tol=1e-11)
    assert 0.0 <= info["upper"] - info["lower"] <= 1e-11  # noqa: PLR2004
    assert info["newton_steps"] > 0


def test_off_simplex_input_is_refused() -> None:
    jko = JKOStep(ZeroF(), 0.1, np.array([[0.0], [1.0]]), scheme="jko_entropic")
    with pytest.raises(ValueError, match="simplex"):
        jko.step(_state({"a": np.array([1.0, 1.0])}))


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
    # the historical energy keeps the floored entropy: refused even without Turing
    assert not _t2(6, 5.0, 0.0, None).implicit_ready
    assert _t2(6, 5.0, 0.0, 1e-5).implicit_ready
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
    along absolute mass transfers between any two bins. The L-BFGS comparison against an
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
        # unweighted: every bin, whatever its mass
        assert float(np.max(np.abs(g - float(np.dot(q[n], g))))) < STATIONARITY_TOL
    base = q_value(q)
    for n in names:  # absolute mass transfers between bins, both directions
        for a_bin in range(grid):
            for b_bin in range(grid):
                if a_bin == b_bin:
                    continue
                pert = dict(q)
                moved = q[n].copy()
                moved[a_bin] += PERTURBATION
                moved[b_bin] -= PERTURBATION
                pert[n] = moved
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


def test_ot_bracket_is_never_reversed_and_refuses_unreachable_tolerances() -> None:
    """Third review (N1): mismatched masses within MASS_TOL, tolerance below rounding."""
    value, _, info = entropic_ot(
        np.array([0.5, 0.5]), np.array([0.5, 0.5000000005]), np.zeros((2, 2)), 1.0
    )
    assert info["lower"] <= value <= info["upper"]
    assert info["gap"] >= 0.0
    cost = np.array([[1000.0, 1001.0], [1001.0, 1000.0]])
    with pytest.raises(SinkhornNotConvergedError):
        entropic_ot(np.array([0.3, 0.7]), np.array([0.6, 0.4]), cost, 0.1, gap_tol=1e-16)


def test_ot_bracket_slack_follows_the_terms_that_cancel() -> None:
    """Codex counter-example: huge cancelling terms, true value 1.9e-8.

    With q = r = (1/2, 1/2) and a constant cost C0, OT_reg = C0 - reg (1 + 2 ln 2)
    exactly (the plan is q x r). In double precision the result is lost in the
    cancellation of terms of order 1e8: the call must refuse a tight tolerance,
    and a loose one must bracket the true value.
    """
    getcontext().prec = 60
    c0, reg = 238629436.11198908, 1e8
    exact = float(Decimal(c0) - Decimal(reg) * (1 + 2 * Decimal(2).ln()))
    q = np.array([0.5, 0.5])
    cost = np.full((2, 2), c0)
    with pytest.raises(SinkhornNotConvergedError):
        entropic_ot(q, q, cost, reg)
    _, _, info = entropic_ot(q, q, cost, reg, gap_tol=1e-3)
    assert info["lower"] <= exact <= info["upper"]


def test_ot_bracket_contains_the_closed_form_value() -> None:
    """Separable cost C_ij = a_i + b_j: the optimal plan is q x r and
    OT_reg = <q, a> + <r, b> + reg (sum q log q + sum r log r - 1)."""
    q, r = np.array([0.3, 0.7]), np.array([0.6, 0.4])
    a, b = np.array([0.0, 2.0]), np.array([1.0, 3.0])
    reg = 0.1
    cost = a[:, None] + b[None, :]
    exact = float(q @ a + r @ b + reg * (np.sum(q * np.log(q)) + np.sum(r * np.log(r)) - 1.0))
    value, _, info = entropic_ot(q, r, cost, reg, gap_tol=1e-12)
    assert info["lower"] <= exact <= info["upper"]
    assert info["upper"] - info["lower"] <= 1e-12  # noqa: PLR2004
    assert value == pytest.approx(exact, abs=1e-12)


def test_ot_bracket_covers_subnormal_results() -> None:
    """Fourth review: reg = 1e-320, zero cost, OT_reg = -reg (1 + 2 ln 2) exactly."""
    q = np.array([0.5, 0.5])
    reg = 1e-320
    exact = float(-Decimal(reg) * (1 + 2 * Decimal(2).ln()))
    with pytest.raises(SinkhornNotConvergedError):
        entropic_ot(q, q, np.zeros((2, 2)), reg, gap_tol=1e-323)
    _, _, info = entropic_ot(q, q, np.zeros((2, 2)), reg, gap_tol=1e-300)
    assert info["lower"] <= exact <= info["upper"]


def test_ot_bracket_bound_does_not_overflow_for_finite_values() -> None:
    """Fifth review: huge reg with finite bounds must not give an infinite slack."""
    q = np.array([1e-10])
    _, _, info = entropic_ot(q, q, np.zeros((1, 1)), 5e306, gap_tol=1e290)
    assert np.isfinite(info["rounding_slack"])
