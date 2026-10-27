"""``JKOStep(nonseparable="implicit")``: implicit treatment of the coupling and Turing terms.

ERRATUM.md section 14: with the non-separable terms linearised at rho_tau
(``"explicit"``, the default) the entropic step is not dissipative when the
Turing term is strong. The implicit mode iterates the linearisation point to
the step's output and halves the step if F would increase; it is the
default since 2026-10-27 (ERRATUM.md section 15).
"""

from __future__ import annotations

import numpy as np
import pytest

from kiki_flow_core.master_equation import NONSEPARABLE_MODES, JKOStep, ZeroF
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState, assert_invariants
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy

GRID = 16
N_STEPS = 3
# Measured on electron-server (results/turing_implicit_check_2026_10_27/): on a
# first step from a random state the two linearisations differ by up to 0.052 in
# TV at gamma <= 5 (the coupling is linearised at different points); the gap
# shrinks to 3e-5 .. 3e-3 after 10 steps.
FIRST_STEP_TV = 0.06
INCREASE_TOL = 1e-12


def _problem(alpha: float, gamma: float) -> T2FreeEnergy:
    species = CanonicalSpecies()
    names = species.species_names()
    x = np.linspace(-2, 2, GRID)
    potentials = {
        "phono": alpha * (x + 1.5),
        "lex": alpha * (x - 1.5),
        "syntax": alpha * np.abs(x),
        "sem": -alpha * np.abs(x),
    }
    prior = {n: np.full(GRID, 1.0 / GRID) for n in names}
    return T2FreeEnergy(
        species=species, potentials=potentials, prior=prior, turing_strength=0.1 * gamma
    )


def _state(f: T2FreeEnergy, seed: int) -> FlowState:
    rng = np.random.default_rng(seed)
    return FlowState(
        rho={n: rng.dirichlet(np.ones(GRID)) for n in f.species.species_names()},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )


def _solver(f: T2FreeEnergy, mode: str) -> FullJKOSolver:
    return FullJKOSolver(
        f_functional=f,
        h=0.05,
        support=np.linspace(-2, 2, GRID).reshape(-1, 1),
        max_iter=300,
        nonseparable=mode,
        prox_tol=1e-12,
    )


def test_mode_validation_and_default_is_implicit() -> None:
    """Default since 2026-10-27 (ERRATUM section 15); explicit stays available."""
    support = np.linspace(0, 1, 4).reshape(-1, 1)
    assert NONSEPARABLE_MODES == ("explicit", "implicit")
    jko = JKOStep(f_functional=ZeroF(), h=0.1, support=support, apply_w2_prox=True)
    assert jko.nonseparable == "implicit"
    assert FullJKOSolver(f_functional=ZeroF(), h=0.1, support=support).nonseparable == "implicit"
    explicit = FullJKOSolver(f_functional=ZeroF(), h=0.1, support=support, nonseparable="explicit")
    assert explicit.nonseparable == "explicit"
    with pytest.raises(ValueError, match="nonseparable"):
        JKOStep(f_functional=ZeroF(), h=0.1, support=support, nonseparable="bogus")  # type: ignore[arg-type]


def test_implicit_matches_explicit_without_turing_term() -> None:
    """gamma = 0: only the J_sym coupling is non-separable; both modes stay close."""
    f = _problem(5.0, 0.0)
    state = _state(f, 0)
    a = _solver(f, "implicit").step(state)
    b = _solver(f, "explicit").step(state)
    assert_invariants(a)
    assert max(0.5 * float(np.abs(a.rho[n] - b.rho[n]).sum()) for n in a.rho) < FIRST_STEP_TV


@pytest.mark.slow
def test_implicit_step_does_not_increase_f_under_strong_turing() -> None:
    f = _problem(5.0, 30.0)
    implicit = _solver(f, "implicit")
    state = _state(f, 1)
    values = [f.value(state)]
    for _ in range(N_STEPS):
        state = implicit.step(state)
        assert_invariants(state)
        assert state.tau >= 1
        values.append(f.value(state))
    assert all(b <= a + INCREASE_TOL for a, b in zip(values, values[1:], strict=False))
