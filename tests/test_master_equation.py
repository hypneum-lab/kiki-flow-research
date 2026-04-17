import numpy as np

from kiki_flow_core.master_equation import FreeEnergy, JKOStep, ZeroF
from kiki_flow_core.state import FlowState


def make_state(rho_dict: dict[str, np.ndarray]) -> FlowState:
    return FlowState(
        rho=rho_dict,
        P_theta=np.zeros(8),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T1"},
    )


def test_zero_f_returns_zero():
    f = ZeroF()
    assert f.value(make_state({"phono": np.array([0.5, 0.5])})) == 0.0


def test_jko_step_idempotent_under_zero_f():
    """With F=0, JKO step on equilibrium reference should be identity (within tolerance)."""
    rho = np.array([0.25, 0.25, 0.25, 0.25])
    state = make_state({"phono": rho})
    support = np.linspace(0, 1, 4).reshape(-1, 1)
    step = JKOStep(f_functional=ZeroF(), h=0.1, support=support)
    new_state = step.step(state)
    np.testing.assert_allclose(new_state.rho["phono"], rho, atol=1e-2)


def test_jko_step_increments_tau():
    state = make_state({"phono": np.array([0.5, 0.5])})
    support = np.linspace(0, 1, 2).reshape(-1, 1)
    step = JKOStep(f_functional=ZeroF(), h=0.1, support=support)
    new_state = step.step(state)
    assert new_state.tau == state.tau + 1


def test_jko_step_preserves_mass():
    state = make_state({"phono": np.array([0.4, 0.3, 0.2, 0.1])})
    support = np.linspace(0, 1, 4).reshape(-1, 1)
    step = JKOStep(f_functional=ZeroF(), h=0.05, support=support)
    new_state = step.step(state)
    assert abs(new_state.rho["phono"].sum() - 1.0) < 1e-3  # noqa: PLR2004


def test_jko_step_decreases_f_with_quadratic_f():
    """A quadratic F penalizing distance from a target should drive rho toward target."""
    target = np.array([0.7, 0.2, 0.1])
    support = np.linspace(0, 1, 3).reshape(-1, 1)

    class QuadraticF(FreeEnergy):
        def value(self, state: FlowState) -> float:
            r = state.rho["phono"]
            return float(((r - target) ** 2).sum())

    state_init = make_state({"phono": np.array([0.33, 0.33, 0.34])})
    f = QuadraticF()
    f0 = f.value(state_init)
    step = JKOStep(f_functional=f, h=0.5, support=support)
    new_state = step.step(state_init)
    f1 = f.value(new_state)
    assert f1 < f0, f"F should decrease: f0={f0}, f1={f1}"
