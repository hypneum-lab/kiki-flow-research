"""MLX-backed log-domain Sinkhorn for Track 2 paper runs (Apple Silicon GPU)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def mlx_sinkhorn_cost(
    a: np.ndarray,
    b: np.ndarray,
    cost_matrix: np.ndarray,
    epsilon: float = 0.01,
    n_iter: int = 200,
) -> float:
    """Numerically-stable log-domain entropic Sinkhorn on Metal.

    Inputs are numpy arrays; output is a Python float. Uses log-domain
    iterations so small epsilon does not underflow the kernel.
    """
    a_mx = mx.array(a, dtype=mx.float32)
    b_mx = mx.array(b, dtype=mx.float32)
    cost_mx = mx.array(cost_matrix, dtype=mx.float32)
    log_k = -cost_mx / epsilon
    log_a = mx.log(a_mx + 1e-30)
    log_b = mx.log(b_mx + 1e-30)
    log_u = mx.zeros(a_mx.shape, dtype=mx.float32)
    log_v = mx.zeros(b_mx.shape, dtype=mx.float32)
    for _ in range(n_iter):
        log_u = log_a - mx.logsumexp(log_k + log_v[None, :], axis=1)
        log_v = log_b - mx.logsumexp(log_k + log_u[:, None], axis=0)
    log_t = log_u[:, None] + log_k + log_v[None, :]
    transport = mx.exp(log_t)
    cost = mx.sum(transport * cost_mx)
    mx.eval(cost)
    return float(cost.item())
