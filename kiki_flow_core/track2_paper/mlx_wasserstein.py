"""MLX-backed log-domain Sinkhorn and W2 prox for Track 2 (Apple Silicon GPU)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def _sinkhorn_log_domain(
    a_mx: mx.array,
    b_mx: mx.array,
    cost_mx: mx.array,
    epsilon: float,
    n_iter: int,
) -> mx.array:
    """Run log-domain Sinkhorn iterations and return the transport plan.

    Kernel behind ``mlx_sinkhorn_cost``.
    """
    log_k = -cost_mx / epsilon
    log_a = mx.log(a_mx + 1e-30)
    log_b = mx.log(b_mx + 1e-30)
    log_u = mx.zeros(a_mx.shape, dtype=mx.float32)
    log_v = mx.zeros(b_mx.shape, dtype=mx.float32)
    for _ in range(n_iter):
        log_u = log_a - mx.logsumexp(log_k + log_v[None, :], axis=1)
        log_v = log_b - mx.logsumexp(log_k + log_u[:, None], axis=0)
    log_t = log_u[:, None] + log_k + log_v[None, :]
    return mx.exp(log_t)


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
    transport = _sinkhorn_log_domain(a_mx, b_mx, cost_mx, epsilon, n_iter)
    cost = mx.sum(transport * cost_mx)
    mx.eval(cost)
    return float(cost.item())


def mlx_prox_w2(
    distribution: np.ndarray,
    reference: np.ndarray,
    epsilon: float,
    support: np.ndarray,
    n_iter: int = 500,
    step_size: float = 0.1,  # retained for API compatibility (unused: Dykstra is step-free)
    sinkhorn_iter: int = 50,  # retained for API compatibility (unused)
    *,
    reg: float | None = None,
) -> np.ndarray:
    """MLX port of the entropic W2 proximal operator ``wasserstein_ops.prox_w2``.

    Solves the same problem with the same Dykstra / iterative Bregman
    scaling (Peyre 2015) on the Gibbs kernel ``K = exp(-cost/reg)``:
        argmin_q  OT_reg(q, reference) + 2 * epsilon * KL(q || distribution).
    The scalings are carried in the log domain (``logsumexp``) so the
    float32 Metal kernel does not underflow where the float64 NumPy
    reference still can represent ``K``; at convergence the iterates are
    the same fixed point. Measured agreement with the NumPy reference is
    within 1e-6 absolute per bin (``tests/track2_paper/test_mlx_prox_w2``).

    The previous MLX implementation ran projected gradient descent with
    ``transport @ diag(cost)`` as the transport gradient. The squared
    self-distance cost has a zero diagonal, so that term vanished and the
    output did not depend on ``reference`` (it returned ``distribution``).
    Returns a float64 numpy array to stay plug-compatible with ``prox_fn``.
    """
    cost_np = _squared_euclidean(np.asarray(support, dtype=np.float64))
    sink_reg = max(reg if reg is not None else epsilon, 1e-3)
    tau = max(epsilon, 0.0)
    lam = sink_reg / (sink_reg + 2.0 * tau)
    # Logs of the clipped marginals are taken in float64 (same clip as the
    # NumPy reference) and only then cast, so zero bins map to -690.8.
    log_dist_np = np.log(np.clip(np.asarray(distribution, dtype=np.float64), 1e-300, None))
    log_ref_np = np.log(np.clip(np.asarray(reference, dtype=np.float64), 1e-300, None))
    log_k = mx.array(-cost_np / sink_reg, dtype=mx.float32)
    log_dist = mx.array(log_dist_np, dtype=mx.float32)
    log_ref = mx.array(log_ref_np, dtype=mx.float32)
    log_b = mx.zeros(log_ref.shape, dtype=mx.float32)
    log_a = mx.zeros(log_dist.shape, dtype=mx.float32)
    for i in range(n_iter):
        log_a, log_b = _dykstra_step(log_k, log_dist, log_ref, log_b, lam)
        if (i + 1) % _EVAL_EVERY == 0:
            mx.eval(log_a, log_b)
    log_q = log_a + mx.logsumexp(log_k + log_b[None, :], axis=1)
    log_q = log_q - mx.logsumexp(log_q)
    q = mx.exp(log_q)
    mx.eval(q)
    out_np: np.ndarray = np.asarray(q, dtype=np.float64)
    s = float(out_np.sum())
    if not np.isfinite(out_np).all() or s <= 0:
        raise FloatingPointError("mlx_prox_w2 Dykstra iteration did not converge (non-finite)")
    return out_np / s


_EVAL_EVERY = 50


@mx.compile
def _dykstra_step(
    log_k: mx.array,
    log_dist: mx.array,
    log_ref: mx.array,
    log_b: mx.array,
    lam: float,
) -> tuple[mx.array, mx.array]:
    """One log-domain Dykstra sweep, mirroring the NumPy ``prox_w2`` loop.

    ``a = dist**(1-lam) * (K b)**(lam-1)`` then ``b = ref / (K^T a)``.
    """
    log_kb = mx.logsumexp(log_k + log_b[None, :], axis=1)
    log_a = (1.0 - lam) * log_dist + (lam - 1.0) * log_kb
    log_ka = mx.logsumexp(log_k + log_a[:, None], axis=0)
    return log_a, log_ref - log_ka


def _squared_euclidean(support: np.ndarray) -> np.ndarray:
    diff = support[:, None, :] - support[None, :, :]
    out: np.ndarray = (diff**2).sum(axis=-1)
    return out
