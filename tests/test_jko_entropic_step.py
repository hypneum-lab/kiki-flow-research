"""Tests for ``JKOStep(scheme="jko_entropic")`` and ``wasserstein_ops.jko_entropic_prox``.

The new scheme is the entropic JKO step of F itself (Peyré 2015). The tests
pin: the scaling iteration reproduces the legacy ``prox_w2`` when handed the
KL-prox of ``KL(. || d)``; the default ``FreeEnergy.kl_prox_log`` is the
linearised potential; ``T2FreeEnergy.kl_prox_log`` is the exact minimiser of
``KL(q || p) + tau [<V, q> + KL(q || prior)]``; the step conserves mass,
increments tau, leaves the legacy scheme untouched, dissipates F for a
potential + entropy energy and does not teleport mass across a barrier.
"""

from __future__ import annotations

import numpy as np
import pytest

from kiki_flow_core.master_equation import SCHEMES, FreeEnergy, JKOStep, ZeroF
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState, assert_invariants
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy
from kiki_flow_core.wasserstein_ops import jko_entropic_prox, prox_w2

GRID = 16
H = 0.05
REG = 0.01
N_ITER = 400
LEGACY_MATCH_TOL = 1e-9
MASS_TOL = 1e-10
STATIONARITY_TOL = 1e-8
# Entropic blur (width sqrt(reg) = 0.1) leaks ~6e-6 across the barrier; the
# legacy split moves 4e-2 at the same h (results/jko_step_check_2026_10_25/).
BARRIER_MASS_TOL = 1e-4
DISSIPATION_TOL = 1e-12
IDENTITY_TOL = 1e-3


def _state(rho: np.ndarray, name: str = "phono") -> FlowState:
    return FlowState(
        rho={name: rho},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )


def _support(grid: int = GRID) -> np.ndarray:
    return np.linspace(-1.0, 1.0, grid).reshape(-1, 1)


class PotentialEntropyF(FreeEnergy):
    """F = <V, rho> + theta sum rho log rho with its exact KL-prox."""

    def __init__(self, potential: np.ndarray, theta: float) -> None:
        self.potential = potential
        self.theta = theta

    def value(self, state: FlowState) -> float:
        rho = next(iter(state.rho.values()))
        safe = np.clip(rho, 1e-12, None)
        return float(np.dot(rho, self.potential) + self.theta * np.sum(safe * np.log(safe)))

    def kl_prox_log(self, state: FlowState, species_name: str):  # type: ignore[no-untyped-def]
        potential, theta = self.potential, self.theta

        def prox(log_p: np.ndarray, tau: float) -> np.ndarray:
            return (log_p - tau * potential - tau * theta) / (1.0 + tau * theta)

        return prox


def test_scaling_iteration_reproduces_legacy_prox_w2() -> None:
    rng = np.random.default_rng(0)
    support = _support()
    rho_tau = rng.dirichlet(np.full(GRID, 3.0))
    target = rng.dirichlet(np.full(GRID, 3.0))
    log_target = np.log(target)

    def kl_to_target(log_p: np.ndarray, tau: float) -> np.ndarray:
        return (log_p + tau * log_target) / (1.0 + tau)

    new = jko_entropic_prox(
        kl_to_target, reference=rho_tau, epsilon=H, support=support, n_iter=N_ITER, reg=REG
    )
    legacy = prox_w2(target, reference=rho_tau, epsilon=H, support=support, n_iter=N_ITER, reg=REG)
    np.testing.assert_allclose(new, legacy, atol=LEGACY_MATCH_TOL)


def test_jko_entropic_prox_rejects_bad_parameters() -> None:
    support = _support(4)
    rho = np.full(4, 0.25)

    def identity(log_p: np.ndarray, tau: float) -> np.ndarray:
        return log_p

    with pytest.raises(ValueError, match="regularisation"):
        jko_entropic_prox(identity, rho, epsilon=0.1, support=support, reg=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        jko_entropic_prox(identity, rho, epsilon=-1.0, support=support)
    # tol > 0 stops early and still returns a simplex vector.
    out = jko_entropic_prox(identity, rho, epsilon=0.1, support=support, n_iter=50, tol=1e-6)
    assert abs(out.sum() - 1.0) < MASS_TOL


def test_default_kl_prox_log_is_the_linearised_potential() -> None:
    support = _support(8)
    rng = np.random.default_rng(1)
    rho = rng.dirichlet(np.full(8, 3.0))
    f = PotentialEntropyF(support[:, 0] ** 2, 0.1)
    state = _state(rho)
    grad = f.grad_rho(state, "phono")
    prox = FreeEnergy.kl_prox_log(f, state, "phono")  # the base-class default
    log_p = np.log(rng.dirichlet(np.full(8, 3.0)))
    tau = 3.0
    np.testing.assert_allclose(prox(log_p, tau), log_p - tau * grad)


def test_t2_kl_prox_log_minimises_kl_plus_tau_f() -> None:
    species = CanonicalSpecies()
    names = species.species_names()
    rng = np.random.default_rng(2)
    potentials = {n: rng.standard_normal(GRID) for n in names}
    prior = {n: rng.dirichlet(np.full(GRID, 2.0)) for n in names}
    f = T2FreeEnergy(species=species, potentials=potentials, prior=prior, turing_strength=0.0)
    state = FlowState(
        rho={n: rng.dirichlet(np.full(GRID, 2.0)) for n in names},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )
    name = names[0]
    p = rng.dirichlet(np.full(GRID, 2.0))
    tau = 1.7
    log_q = f.kl_prox_log(state, name)(np.log(p), tau)
    q = np.exp(log_q)
    # Stationarity of KL(q||p) + tau [<V_eff, q> + KL(q||prior)] over R^n_+: the
    # gradient log(q/p) + 1 + tau (V_eff + log(q/prior) + 1) must be constant.
    v_eff = potentials[name] + f.nonseparable_grad(state, name)
    grad = np.log(q / p) + 1.0 + tau * (v_eff + np.log(q / prior[name]) + 1.0)
    assert np.ptp(grad) < STATIONARITY_TOL
    # Coupling gradient: 2 (J_sym rho)_i, Turing term off.
    rhos = [state.rho[n] for n in names]
    expected = 2.0 * sum(f._J_sym[0, k] * rhos[k] for k in range(len(names)))
    np.testing.assert_allclose(f.nonseparable_grad(state, name), expected)


def test_t2_nonseparable_grad_includes_turing_term() -> None:
    species = CanonicalSpecies()
    names = species.species_names()
    rng = np.random.default_rng(3)
    potentials = {n: np.zeros(GRID) for n in names}
    prior = {n: np.full(GRID, 1.0 / GRID) for n in names}
    state = FlowState(
        rho={n: rng.dirichlet(np.full(GRID, 2.0)) for n in names},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )
    off = T2FreeEnergy(species=species, potentials=potentials, prior=prior, turing_strength=0.0)
    on = T2FreeEnergy(species=species, potentials=potentials, prior=prior, turing_strength=0.1)
    g_off = off.nonseparable_grad(state, names[0])
    g_on = on.nonseparable_grad(state, names[0])
    assert np.abs(g_on - g_off).max() > 0.0


def test_step_rejects_unknown_scheme_and_resolves_defaults() -> None:
    """Defaults since 2026-10-25: entropic JKO with a prox (reg = h^2), descent without."""
    support = _support(4)
    with pytest.raises(ValueError, match="scheme"):
        JKOStep(f_functional=ZeroF(), h=0.1, support=support, scheme="bogus")  # type: ignore[arg-type]
    assert SCHEMES == ("legacy_split", "jko_entropic")
    fast = JKOStep(f_functional=ZeroF(), h=0.1, support=support)
    assert fast.scheme == "legacy_split"
    rigorous = JKOStep(f_functional=ZeroF(), h=0.1, support=support, apply_w2_prox=True)
    assert rigorous.scheme == "jko_entropic"
    solver = FullJKOSolver(f_functional=ZeroF(), h=0.1, support=support)
    assert solver.scheme == "jko_entropic"
    assert solver.reg == pytest.approx(0.1**2)
    assert solver.epsilon == pytest.approx(0.1**2)
    legacy = FullJKOSolver(f_functional=ZeroF(), h=0.1, support=support, scheme="legacy_split")
    assert legacy.scheme == "legacy_split"
    assert legacy.reg == 0.01  # noqa: PLR2004  (historical default)
    explicit = FullJKOSolver(f_functional=ZeroF(), h=0.1, support=support, epsilon=0.02)
    assert explicit.reg == 0.02  # noqa: PLR2004


def test_entropic_step_with_zero_f_is_the_entropic_self_prox() -> None:
    """F = 0: the step is argmin_q W_reg(q, rho_tau), a mild blur of rho_tau."""
    support = _support()
    x = support[:, 0]
    rho = np.exp(-0.5 * (x / 0.3) ** 2)
    rho = rho / rho.sum()
    state = _state(rho)
    jko = JKOStep(
        f_functional=ZeroF(),
        h=H,
        support=support,
        reg=1e-3,
        prox_n_iter=N_ITER,
        scheme="jko_entropic",
    )
    out = jko.step(state)
    assert out.tau == state.tau + 1
    assert_invariants(out)
    np.testing.assert_allclose(out.rho["phono"], rho, atol=IDENTITY_TOL)
    # Legacy scheme on the same object parameters is untouched (prox_fn spy).
    legacy = JKOStep(
        f_functional=ZeroF(), h=H, support=support, apply_w2_prox=True, scheme="legacy_split"
    )
    assert legacy.scheme == "legacy_split"
    calls: list[dict] = []
    inner = legacy.prox_fn

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return inner(*args, **kwargs)

    legacy.prox_fn = spy
    legacy.step(state)
    assert len(calls) == 1


def test_entropic_step_dissipates_and_moves_toward_gibbs() -> None:
    support = _support()
    x = support[:, 0]
    f = PotentialEntropyF(4.0 * x**2, 0.1)
    rho = np.exp(-0.5 * ((x + 0.5) / 0.12) ** 2)
    rho = np.clip(rho / rho.sum(), 1e-12, None)
    rho = rho / rho.sum()
    state = _state(rho)
    jko = JKOStep(
        f_functional=f, h=H, support=support, reg=REG, prox_n_iter=N_ITER, scheme="jko_entropic"
    )
    out = jko.step(state).rho["phono"]
    assert abs(out.sum() - 1.0) < MASS_TOL
    assert f.value(_state(out)) < f.value(state) - DISSIPATION_TOL
    # The mean moves toward the Gibbs centre (0) without jumping past it.
    assert float(np.dot(x, rho)) < float(np.dot(x, out)) < 0.0


def test_entropic_step_does_not_teleport_mass_across_a_barrier() -> None:
    support = _support(48)
    x = support[:, 0]
    f = PotentialEntropyF((x**2 - 0.25) ** 2 / 0.0625, 0.05)
    left = np.exp(-0.5 * ((x + 0.5) / 0.1) ** 2)
    right = np.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)
    rho = 0.7 * left / left.sum() + 0.3 * right / right.sum()
    rho = np.clip(rho, 1e-12, None)
    rho = rho / rho.sum()
    jko = JKOStep(
        f_functional=f, h=H, support=support, reg=REG, prox_n_iter=N_ITER, scheme="jko_entropic"
    )
    out = jko.step(_state(rho)).rho["phono"]
    assert abs(float(out[x > 0].sum()) - float(rho[x > 0].sum())) < BARRIER_MASS_TOL


def test_entropic_step_on_t2_energy_keeps_invariants_for_all_species() -> None:
    species = CanonicalSpecies()
    names = species.species_names()
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    potentials = {n: np.zeros(GRID) for n in names}
    prior = {n: np.full(GRID, 1.0 / GRID) for n in names}
    f = T2FreeEnergy(species=species, potentials=potentials, prior=prior, turing_strength=0.1)
    rng = np.random.default_rng(4)
    state = FlowState(
        rho={n: rng.dirichlet(np.full(GRID, 0.5)) for n in names},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )
    solver = FullJKOSolver(
        f_functional=f,
        h=0.05,
        support=support,
        epsilon=0.01,
        max_iter=200,
        scheme="jko_entropic",
        nonseparable="explicit",  # unsmoothed Turing term: explicit path (ERRATUM 16)
    )
    out = solver.step(state)
    assert_invariants(out)
    assert out.tau == 1
    # Entropy toward the uniform prior: every species gets flatter.
    for n in names:
        assert float(np.max(out.rho[n])) < float(np.max(state.rho[n]))
