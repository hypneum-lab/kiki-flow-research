import time

import numpy as np
import pytest

from kiki_flow_core.master_equation import ZeroF
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver, MLXFullJKOSolver
from kiki_flow_core.track2_paper.mlx_wasserstein import mlx_prox_w2, mlx_sinkhorn_cost
from kiki_flow_core.wasserstein_ops import prox_w2, sinkhorn_cost

# Parity tolerance between the float32 MLX port and the float64 NumPy
# reference ``wasserstein_ops.prox_w2``. Both run the same Dykstra
# fixed point (MLX in the log domain); the gap is float32 round-off.
# Measured on macM1 (M1 Max, MLX 0.31.1): max-abs gap <= 4e-8 and L1
# gap <= 2.1e-7 on n in {16, 64, 100 (2-D), 256, 1024}; the bounds
# below leave ~25x headroom for other Metal devices.
PARITY_ATOL = 1e-6
PARITY_L1 = 1e-5


def _cost_matrix(support: np.ndarray) -> np.ndarray:
    diff = support[:, None, :] - support[None, :, :]
    return (diff**2).sum(axis=-1)


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    g = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    out: np.ndarray = g / g.sum()
    return out


def test_mlx_sinkhorn_matches_pot_within_tolerance():
    rng = np.random.default_rng(0)
    n = 32
    a = rng.dirichlet(np.ones(n)).astype(np.float32)
    b = rng.dirichlet(np.ones(n)).astype(np.float32)
    support = np.linspace(-2, 2, n).reshape(-1, 1).astype(np.float32)
    cost = _cost_matrix(support)
    mlx_val = mlx_sinkhorn_cost(a, b, cost, epsilon=0.05, n_iter=300)
    pot_val = sinkhorn_cost(a, b, cost, epsilon=0.05, n_iter=300)
    # 5 percent tolerance on the regularized cost
    assert abs(mlx_val - pot_val) / max(abs(pot_val), 1e-9) < 0.05  # noqa: PLR2004


def test_mlx_prox_w2_returns_simplex():
    support = np.linspace(-2, 2, 16).reshape(-1, 1).astype(np.float32)
    x = support[:, 0]
    result = mlx_prox_w2(
        _gauss(x, -0.8, 0.5).astype(np.float32),
        reference=_gauss(x, 0.8, 0.5).astype(np.float32),
        epsilon=0.05,
        support=support,
        n_iter=50,
    )
    assert abs(result.sum() - 1.0) < 1e-4  # noqa: PLR2004
    assert (result >= 0).all()
    assert np.isfinite(result).all()


def test_mlx_prox_w2_depends_on_reference():
    """Changing ``reference`` must change the output.

    Regression for the zero-gradient bug: the old MLX loop used
    ``transport @ diag(cost)`` with a squared self-distance cost whose
    diagonal is zero, so the output never depended on ``reference``
    (it returned ``distribution`` unchanged).
    """
    support = np.linspace(-2, 2, 32).reshape(-1, 1)
    x = support[:, 0]
    dist = _gauss(x, 0.0, 0.5)
    out_left = mlx_prox_w2(dist, reference=_gauss(x, -1.0, 0.4), epsilon=0.1, support=support)
    out_right = mlx_prox_w2(dist, reference=_gauss(x, 1.0, 0.4), epsilon=0.1, support=support)
    assert np.abs(out_left - out_right).sum() > 0.1  # noqa: PLR2004
    assert np.abs(out_left - dist).sum() > 0.05  # noqa: PLR2004
    # The pull is towards the reference: means move in opposite directions.
    assert float(out_left @ x) < float(dist @ x) < float(out_right @ x)


@pytest.mark.parametrize(
    ("n", "dim", "epsilon", "n_iter"),
    [
        (16, 1, 0.05, 200),  # small, 1-D
        (64, 1, 0.1, 400),  # medium, 1-D
        (64, 1, 0.02, 400),  # medium, 1-D, sharper kernel
        (100, 2, 0.1, 300),  # medium, 2-D (10x10 grid)
    ],
)
def test_mlx_prox_w2_matches_numpy_prox_w2(n: int, dim: int, epsilon: float, n_iter: int):
    rng = np.random.default_rng(n + dim)
    if dim == 1:
        support = np.linspace(-2, 2, n).reshape(-1, 1)
    else:
        side = int(round(n**0.5))
        g = np.linspace(-1, 1, side)
        support = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    dist = rng.dirichlet(np.ones(n))
    ref = rng.dirichlet(np.ones(n))
    expected = prox_w2(dist, reference=ref, epsilon=epsilon, support=support, n_iter=n_iter)
    got = mlx_prox_w2(dist, reference=ref, epsilon=epsilon, support=support, n_iter=n_iter)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, expected, rtol=0, atol=PARITY_ATOL)
    assert np.abs(got - expected).sum() < PARITY_L1


def test_mlx_prox_w2_honours_explicit_reg():
    support = np.linspace(-2, 2, 32).reshape(-1, 1)
    rng = np.random.default_rng(7)
    dist = rng.dirichlet(np.ones(32))
    ref = rng.dirichlet(np.ones(32))
    expected = prox_w2(dist, ref, epsilon=0.05, support=support, n_iter=300, reg=0.2)
    got = mlx_prox_w2(dist, ref, epsilon=0.05, support=support, n_iter=300, reg=0.2)
    np.testing.assert_allclose(got, expected, rtol=0, atol=PARITY_ATOL)


def test_mlx_full_jko_step_matches_numpy_full_jko_step():
    support = np.linspace(-2, 2, 32).reshape(-1, 1)
    x = support[:, 0]
    rho = _gauss(x, 0.6, 0.3)
    state = FlowState(
        rho={"phono": rho},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )
    kw = {"f_functional": ZeroF(), "h": 0.05, "support": support, "epsilon": 0.05, "n_inner": 3}
    out_np = FullJKOSolver(**kw).step(state).rho["phono"]
    out_mlx = MLXFullJKOSolver(**kw).step(state).rho["phono"]
    np.testing.assert_allclose(out_mlx, out_np, rtol=0, atol=PARITY_ATOL)


@pytest.mark.slow
def test_mlx_prox_w2_vs_numpy_benchmark():
    """Honest wall-clock comparison of two equivalent computations.

    Both backends run the identical Dykstra fixed point for the same
    number of iterations, and the outputs are checked to agree within
    ``PARITY_ATOL`` first, so the timings compare like with like. No
    speed claim is asserted: the ratio depends on grid size, host and
    Metal warm-up, and is reported in the captured stdout instead.
    """
    timings = {}
    for n in (64, 256, 1024):
        support = np.linspace(-2, 2, n).reshape(-1, 1)
        x = support[:, 0]
        dist = _gauss(x, -1.0, 0.4)
        ref = _gauss(x, 1.0, 0.4)
        kw = {"epsilon": 0.1, "support": support, "n_iter": 200}
        mlx_prox_w2(dist, ref, **kw)  # warm-up (kernel compile)
        prox_w2(dist, ref, **kw)

        t0 = time.perf_counter()
        got = mlx_prox_w2(dist, ref, **kw)
        t_mlx = time.perf_counter() - t0
        t0 = time.perf_counter()
        expected = prox_w2(dist, ref, **kw)
        t_np = time.perf_counter() - t0

        np.testing.assert_allclose(got, expected, rtol=0, atol=PARITY_ATOL)
        timings[n] = (t_np, t_mlx)
        ms_np, ms_mlx = t_np * 1e3, t_mlx * 1e3
        print(f"n={n:5d} numpy={ms_np:8.2f} ms mlx={ms_mlx:8.2f} ms ratio={t_np / t_mlx:.2f}")
    assert all(t > 0 for pair in timings.values() for t in pair)
