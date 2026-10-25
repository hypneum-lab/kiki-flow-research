import numpy as np

from kiki_flow_core.master_equation import ZeroF
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver


def make_state(rho: np.ndarray) -> FlowState:
    return FlowState(
        rho={"phono": rho},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )


def test_full_jko_applies_w2_prox():
    rho = np.array([0.25, 0.25, 0.25, 0.25])
    state = make_state(rho)
    support = np.linspace(0, 1, 4).reshape(-1, 1)
    solver = FullJKOSolver(
        scheme="legacy_split", f_functional=ZeroF(), h=0.1, support=support, epsilon=0.01
    )
    new_state = solver.step(state)
    np.testing.assert_allclose(new_state.rho["phono"], rho, atol=0.05)  # noqa: PLR2004
    assert solver.apply_w2_prox is True


def test_full_jko_tau_increments():
    support = np.linspace(0, 1, 2).reshape(-1, 1)
    state = make_state(np.array([0.5, 0.5]))
    solver = FullJKOSolver(
        scheme="legacy_split", f_functional=ZeroF(), h=0.05, support=support, epsilon=0.01
    )
    new_state = solver.step(state)
    assert new_state.tau == state.tau + 1


def _record_prox_kwargs(solver: FullJKOSolver) -> list[dict]:
    calls: list[dict] = []
    inner = solver.prox_fn

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return inner(*args, **kwargs)

    solver.prox_fn = spy
    return calls


def test_full_jko_passes_epsilon_as_sinkhorn_reg_and_h_as_prox_step():
    support = np.linspace(-2, 2, 16).reshape(-1, 1)
    state = make_state(np.full(16, 1.0 / 16))
    solver = FullJKOSolver(
        scheme="legacy_split", f_functional=ZeroF(), h=0.05, support=support, epsilon=0.01
    )
    calls = _record_prox_kwargs(solver)
    solver.step(state)
    assert calls
    assert calls[0]["epsilon"] == 0.05  # noqa: PLR2004  (JKO step h = prox pull)
    assert calls[0]["reg"] == 0.01  # noqa: PLR2004  (Sinkhorn regularisation)


def test_full_jko_epsilon_changes_the_step():
    support = np.linspace(-2, 2, 16).reshape(-1, 1)
    x = support[:, 0]
    rho = np.exp(-0.5 * ((x - 0.5) / 0.4) ** 2)
    state = make_state(rho / rho.sum())
    out = [
        FullJKOSolver(
            scheme="legacy_split", f_functional=ZeroF(), h=0.05, support=support, epsilon=eps
        )
        .step(state)
        .rho["phono"]
        for eps in (0.01, 0.05)
    ]
    assert np.abs(out[0] - out[1]).sum() > 1e-6  # noqa: PLR2004


def test_plain_jko_with_prox_keeps_default_reg():
    from kiki_flow_core.master_equation import JKOStep  # noqa: PLC0415

    support = np.linspace(-2, 2, 8).reshape(-1, 1)
    jko = JKOStep(
        f_functional=ZeroF(), h=0.05, support=support, apply_w2_prox=True, scheme="legacy_split"
    )
    calls = _record_prox_kwargs(jko)  # type: ignore[arg-type]
    jko.step(make_state(np.full(8, 1.0 / 8)))
    assert "reg" not in calls[0] or calls[0]["reg"] is None


def _peaked_state() -> tuple[np.ndarray, FlowState]:
    support = np.linspace(-2, 2, 16).reshape(-1, 1)
    x = support[:, 0]
    rho = np.exp(-0.5 * ((x - 0.5) / 0.4) ** 2)
    return support, make_state(rho / rho.sum())


def test_full_jko_max_iter_changes_the_step():
    support, state = _peaked_state()
    out = [
        FullJKOSolver(
            scheme="legacy_split",
            f_functional=ZeroF(),
            h=0.05,
            support=support,
            epsilon=0.01,
            max_iter=n,
        )
        .step(state)
        .rho["phono"]
        for n in (5, 100)
    ]
    assert np.abs(out[0] - out[1]).sum() > 1e-6  # noqa: PLR2004


def test_full_jko_forwards_max_iter_as_prox_n_iter():
    support, state = _peaked_state()
    solver = FullJKOSolver(
        scheme="legacy_split",
        f_functional=ZeroF(),
        h=0.05,
        support=support,
        epsilon=0.01,
        max_iter=37,
    )
    calls = _record_prox_kwargs(solver)
    solver.step(state)
    assert calls[0]["n_iter"] == 37  # noqa: PLR2004


def test_full_jko_default_max_iter_reaches_the_prox():
    support, state = _peaked_state()
    solver = FullJKOSolver(
        scheme="legacy_split", f_functional=ZeroF(), h=0.05, support=support, epsilon=0.01
    )
    calls = _record_prox_kwargs(solver)
    solver.step(state)
    assert calls[0]["n_iter"] == solver.max_iter == 1000  # noqa: PLR2004


def test_plain_jko_with_prox_keeps_100_iterations():
    from kiki_flow_core.master_equation import JKOStep  # noqa: PLC0415

    support = np.linspace(-2, 2, 8).reshape(-1, 1)
    jko = JKOStep(
        f_functional=ZeroF(), h=0.05, support=support, apply_w2_prox=True, scheme="legacy_split"
    )
    calls = _record_prox_kwargs(jko)  # type: ignore[arg-type]
    jko.step(make_state(np.full(8, 1.0 / 8)))
    assert calls[0]["n_iter"] == 100  # noqa: PLR2004


def test_default_solver_is_entropic_jko_with_reg_h_squared():
    """Default since 2026-10-25 (ERRATUM section 13): same step as the explicit form."""
    support = np.linspace(-2, 2, 16).reshape(-1, 1)
    x = support[:, 0]
    rho = np.exp(-0.5 * ((x - 0.5) / 0.4) ** 2)
    state = make_state(rho / rho.sum())
    default = FullJKOSolver(f_functional=ZeroF(), h=0.05, support=support, max_iter=200)
    explicit = FullJKOSolver(
        scheme="jko_entropic",
        f_functional=ZeroF(),
        h=0.05,
        support=support,
        epsilon=0.05**2,
        max_iter=200,
    )
    assert default.scheme == "jko_entropic"
    np.testing.assert_array_equal(
        default.step(state).rho["phono"], explicit.step(state).rho["phono"]
    )
