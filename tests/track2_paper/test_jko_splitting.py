"""Lie and Strang splitting of the entropic JKO step (prereg 2026-10-30).

Covers the composition order (analytic linear flows), the decomposition of
the smoothed T2 energy, the certified sub-steps, failure handling, and the
non-regression of the existing schemes (bit for bit).
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from kiki_flow_core.master_equation import ImplicitJKOError, JKOStep
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState, assert_invariants
from kiki_flow_core.track2_paper.jko_splitting import (
    SplitJKOStep,
    T2NonSeparablePart,
    T2SeparablePart,
    compose,
    split_sequence,
)
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy
from kiki_flow_core.wasserstein_ops import jko_entropic_prox_certified

ORDER_NS = (16, 32, 64, 128)  # steps over T = 1 in the analytic order test
ORDER_SLOPE_TOL = 0.15
DECOMPOSITION_TOL = 1e-12
PROX_TOL = 1e-12
SEPARABLE_KKT_TOL = 1e-9
SIMPLEX_TOL = 1e-9

# SHA-256 of the load-bearing sources at 31feab6 (merge of PR #51). The splitting
# is added beside them; this PR must not change them. A later, intentional change
# of these files updates these constants in the same commit.
SOURCES_31FEAB6 = {
    "kiki_flow_core/master_equation.py": (
        "4441eda65324d56b7acd419716b4df3f7facf567bd62f9f3412d9d087e46de4b"
    ),
    "kiki_flow_core/wasserstein_ops.py": (
        "d367701efd825ac4547a08fcf162d937ee0db736fbd4b7cdbc2b92341750c574"
    ),
    "kiki_flow_core/track2_paper/paper_f.py": (
        "875a4ef0a00238275529e982243caec0f4a77f5123bf52147b6f3a9f31f558fe"
    ),
}


def _state(rho: dict[str, np.ndarray]) -> FlowState:
    return FlowState(
        rho=rho, P_theta=np.zeros(4), mu_curr=np.array([1.0]), tau=0, metadata={"track_id": "T2"}
    )


def _problem(grid: int, alpha: float, gamma: float, delta: float | None = 1e-5) -> T2FreeEnergy:
    """The M5 problem family (same potentials, uniform prior, Turing 0.1 gamma)."""
    species = CanonicalSpecies()
    x = np.linspace(-2, 2, grid)
    potentials = {
        "phono": alpha * (x + 1.5),
        "lex": alpha * (x - 1.5),
        "syntax": alpha * np.abs(x),
        "sem": -alpha * np.abs(x),
    }
    prior = {n: np.full(grid, 1.0 / grid) for n in species.species_names()}
    return T2FreeEnergy(species, potentials, prior, 0.1 * gamma, delta)


def _input(grid: int, seed: int) -> FlowState:
    rng = np.random.default_rng(seed)
    names = CanonicalSpecies().species_names()
    return _state({n: rng.dirichlet(np.ones(grid)) for n in names})


def _support(grid: int, half_width: float = 2.0) -> np.ndarray:
    return np.linspace(-half_width, half_width, grid).reshape(-1, 1)


# ----- composition ---------------------------------------------------------------------


def test_split_sequence_orders_the_sub_steps() -> None:
    assert split_sequence("lie", 0.1) == [("B", 0.1), ("A", 0.1)]
    assert split_sequence("strang", 0.1) == [("A", 0.05), ("B", 0.1), ("A", 0.05)]
    with pytest.raises(ValueError, match="scheme"):
        split_sequence("yoshida", 0.1)  # type: ignore[arg-type]


def _linear_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261030)
    m = rng.normal(size=(3, 3))
    a = -(m @ m.T) - np.eye(3)  # stable, symmetric
    b = rng.normal(size=(3, 3))
    b = b - b.T - 0.5 * np.eye(3)  # rotation plus damping, does not commute with a
    assert np.linalg.norm(a @ b - b @ a) > 1.0
    return a, b, rng.normal(size=3)


def _global_order(scheme: str, step_a, step_b) -> float:  # type: ignore[no-untyped-def]
    a, b, x0 = _linear_pair()
    exact = expm(a + b) @ x0
    errors = []
    for n in ORDER_NS:
        x = x0.copy()
        for _ in range(n):
            x = compose(scheme, step_a, step_b, 1.0 / n, x)  # type: ignore[arg-type]
        errors.append(float(np.linalg.norm(x - exact)))
    hs = np.log(1.0 / np.asarray(ORDER_NS))
    return float(np.polyfit(hs, np.log(errors), 1)[0])


def test_order_with_exact_sub_flows_is_one_for_lie_and_two_for_strang() -> None:
    a, b, _ = _linear_pair()

    def flow_a(x: np.ndarray, s: float) -> np.ndarray:
        return expm(s * a) @ x

    def flow_b(x: np.ndarray, s: float) -> np.ndarray:
        return expm(s * b) @ x

    assert abs(_global_order("lie", flow_a, flow_b) - 1.0) < ORDER_SLOPE_TOL
    assert abs(_global_order("strang", flow_a, flow_b) - 2.0) < ORDER_SLOPE_TOL


def test_strang_of_implicit_euler_sub_steps_is_first_order() -> None:
    """Palindromic composition does not raise the order of first-order sub-steps.

    The JKO sub-steps of the splitting are implicit-Euler-like (first order), so
    order 2 for Strang is a hypothesis of the preregistration, not a given.
    """
    a, b, _ = _linear_pair()
    eye = np.eye(3)

    def euler_a(x: np.ndarray, s: float) -> np.ndarray:
        return np.linalg.solve(eye - s * a, x)

    def euler_b(x: np.ndarray, s: float) -> np.ndarray:
        return np.linalg.solve(eye - s * b, x)

    assert abs(_global_order("lie", euler_a, euler_b) - 1.0) < ORDER_SLOPE_TOL
    assert abs(_global_order("strang", euler_a, euler_b) - 1.0) < ORDER_SLOPE_TOL


# ----- decomposition of the energy -----------------------------------------------------


def test_parts_sum_to_the_smoothed_energy() -> None:
    for gamma in (0.0, 5.0, 100.0):
        f = _problem(8, 5.0, gamma)
        state = _input(8, 3)
        total = T2SeparablePart(f).value(state) + T2NonSeparablePart(f).value(state)
        assert abs(total - f.value(state)) < DECOMPOSITION_TOL * max(1.0, abs(f.value(state)))


def test_composed_part_proxes_equal_the_linearised_full_prox() -> None:
    """prox_A(prox_B(log p, tau), tau) is the closed form of T2FreeEnergy.kl_prox_log."""
    f = _problem(8, 5.0, 30.0)
    state = _input(8, 4)
    log_p = np.log(np.random.default_rng(5).dirichlet(np.ones(8)))
    for name in f.species.species_names():
        full = f.kl_prox_log(state, name)(log_p, 0.7)
        pa = T2SeparablePart(f).kl_prox_log(state, name)
        pb = T2NonSeparablePart(f).kl_prox_log(state, name)
        np.testing.assert_allclose(pa(pb(log_p, 0.7), 0.7), full, rtol=0, atol=PROX_TOL)


def test_parts_refuse_the_historical_energy() -> None:
    f = _problem(8, 5.0, 5.0, delta=None)
    with pytest.raises(ValueError, match="smoothed"):
        T2SeparablePart(f)
    with pytest.raises(ValueError, match="smoothed"):
        T2NonSeparablePart(f)
    with pytest.raises(ValueError, match="smoothed"):
        SplitJKOStep(f, 0.05, _support(8), "lie")


def test_separable_sub_step_is_solved_exactly() -> None:
    """The A sub-step is one certified prox per species: certified at the first check."""
    grid = 8
    f = _problem(grid, 5.0, 30.0)
    state = _input(grid, 6)
    part = T2SeparablePart(f)
    jko = JKOStep(part, 0.2, _support(grid, 0.5), scheme="jko_entropic")
    out = jko.step(state)
    assert jko.last_info["status"] == "converged"
    assert jko.last_info["outer_iterations"] == 0
    assert jko.last_info["kkt_residual"] < SEPARABLE_KKT_TOL
    for name in f.species.species_names():
        q, _ = jko_entropic_prox_certified(
            part.kl_prox_log(state, name),
            state.rho[name],
            0.2,
            _support(grid, 0.5),
            reg=jko.entropic_reg,
        )
        np.testing.assert_allclose(out.rho[name], q, rtol=0, atol=SEPARABLE_KKT_TOL)


# ----- the splitting step --------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["lie", "strang"])
def test_split_step_is_certified_on_a_small_problem(scheme: str) -> None:
    grid = 6
    f = _problem(grid, 5.0, 5.0)
    state = _input(grid, 7)
    step = SplitJKOStep(f, 0.05, _support(grid), scheme)  # type: ignore[arg-type]
    out = step.step(state)
    info = step.last_info
    assert info["status"] == "converged"
    expected = split_sequence(scheme, 0.05)  # type: ignore[arg-type]
    assert [(s["part"], s["h_sub"]) for s in info["substeps"]] == expected
    assert all(s["reg"] == pytest.approx(s["h_sub"] ** 2) for s in info["substeps"])
    assert out.tau == state.tau + 1
    assert_invariants(out)
    for name in f.species.species_names():
        assert abs(float(out.rho[name].sum()) - 1.0) < SIMPLEX_TOL


def test_split_step_failure_raises_or_warns() -> None:
    grid = 6
    f = _problem(grid, 5.0, 30.0)
    state = _input(grid, 8)
    sup = _support(grid)
    with pytest.raises(ImplicitJKOError, match="split JKO step not certified"):
        SplitJKOStep(f, 0.05, sup, "strang", inner_n_iter=1).step(state)
    step = SplitJKOStep(f, 0.05, sup, "strang", on_failure="warn", inner_n_iter=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = step.step(state)
    assert any("not certified" in str(w.message) for w in caught)
    assert step.last_info["status"] == "inner_not_converged"
    assert step.last_info["failed_part"] == "A"
    assert out.tau == state.tau + 1
    for name in f.species.species_names():  # nothing applied before the failing sub-step
        assert np.array_equal(out.rho[name], state.rho[name])


def test_split_step_settings_are_validated() -> None:
    f = _problem(6, 5.0, 5.0)
    sup = _support(6)
    with pytest.raises(ValueError, match="fixed by the splitting"):
        SplitJKOStep(f, 0.05, sup, "lie", reg=0.01)
    with pytest.raises(ValueError, match="scheme"):
        SplitJKOStep(f, 0.05, sup, "yoshida")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="h must be"):
        SplitJKOStep(f, 0.0, sup, "lie")
    with pytest.raises(ValueError, match="on_failure"):
        SplitJKOStep(f, 0.05, sup, "lie", on_failure="ignore")  # type: ignore[arg-type]


# ----- non-regression of the existing schemes ------------------------------------------


def _existing_paths(f: T2FreeEnergy, f_hist: T2FreeEnergy, state: FlowState) -> list[FlowState]:
    sup = _support(6, 0.5)
    return [
        JKOStep(f_hist, 0.2, sup, n_inner=3).step(state),  # fast path
        JKOStep(f_hist, 0.2, sup, n_inner=3, apply_w2_prox=True, scheme="legacy_split").step(state),
        JKOStep(f_hist, 0.2, sup, scheme="jko_entropic", nonseparable="explicit").step(state),
        JKOStep(f, 0.2, sup, scheme="jko_entropic", nonseparable="implicit").step(state),
    ]


def test_existing_schemes_are_bit_identical_after_using_the_splitting() -> None:
    f = _problem(6, 5.0, 5.0)
    f_hist = _problem(6, 5.0, 5.0, delta=None)
    state = _input(6, 9)
    before = _existing_paths(f, f_hist, state)
    snapshot = {n: f.potentials[n].copy() for n in f.potentials}
    j_sym = f._J_sym.copy()
    for scheme in ("lie", "strang"):
        split = SplitJKOStep(f, 0.05, _support(6), scheme, on_failure="warn")  # type: ignore[arg-type]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            split.step(state)
    after = _existing_paths(f, f_hist, state)
    for b, a in zip(before, after, strict=True):
        for name in state.rho:
            assert np.array_equal(b.rho[name], a.rho[name])
    assert all(np.array_equal(snapshot[n], f.potentials[n]) for n in snapshot)
    assert np.array_equal(j_sym, f._J_sym)
    for name in state.rho:  # the input state is not mutated either
        assert np.array_equal(state.rho[name], _input(6, 9).rho[name])


def test_load_bearing_sources_are_those_of_31feab6() -> None:
    root = Path(__file__).resolve().parents[2]
    for rel, digest in SOURCES_31FEAB6.items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, rel
