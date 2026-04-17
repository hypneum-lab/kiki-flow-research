import numpy as np
import ot

from kiki_flow_core.track2_paper.mlx_wasserstein import mlx_sinkhorn_cost


def test_mlx_sinkhorn_matches_pot_on_small_problem():
    np.random.seed(0)
    n = 8
    a = np.full(n, 1.0 / n, dtype=np.float32)
    b = np.full(n, 1.0 / n, dtype=np.float32)
    xs = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)
    xt = np.linspace(0.5, 1.5, n).reshape(-1, 1).astype(np.float32)
    cost = ot.dist(xs, xt, metric="sqeuclidean").astype(np.float32)

    mlx_cost = mlx_sinkhorn_cost(a, b, cost, epsilon=0.05, n_iter=300)
    pot_cost = float((ot.sinkhorn(a, b, cost, reg=0.05, numItermax=300) * cost).sum())

    # MLX and POT agree within entropic-Sinkhorn tolerance
    assert abs(mlx_cost - pot_cost) < 0.05  # noqa: PLR2004


def test_mlx_sinkhorn_non_negative():
    n = 5
    a = np.full(n, 1.0 / n, dtype=np.float32)
    b = np.full(n, 1.0 / n, dtype=np.float32)
    cost = np.random.default_rng(0).random((n, n)).astype(np.float32)
    mlx_cost = mlx_sinkhorn_cost(a, b, cost, epsilon=0.1, n_iter=100)
    assert mlx_cost >= 0.0
