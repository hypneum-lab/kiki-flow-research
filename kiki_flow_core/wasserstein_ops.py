"""Wasserstein-2 operations via entropic Sinkhorn (POT-backed)."""

from __future__ import annotations

import numpy as np
import ot


def sinkhorn_cost(
    a: np.ndarray,
    b: np.ndarray,
    cost_matrix: np.ndarray,
    epsilon: float = 0.01,
    n_iter: int = 1000,
) -> float:
    """Entropic Sinkhorn approximation of OT cost.

    Uses log-domain Sinkhorn (``method='sinkhorn_log'``) for numerical stability
    when ``epsilon`` is small relative to the cost-matrix scale — the vanilla
    kernel ``exp(-M/epsilon)`` underflows otherwise and the transport plan
    collapses toward zero.
    """
    transport = ot.sinkhorn(
        a,
        b,
        cost_matrix,
        reg=epsilon,
        numItermax=n_iter,
        stopThr=1e-9,
        method="sinkhorn_log",
    )
    return float((transport * cost_matrix).sum())


def w2_distance(
    p: np.ndarray,
    q: np.ndarray,
    support: np.ndarray,
    epsilon: float = 0.005,
    n_iter: int = 2000,
) -> float:
    """Wasserstein-2 distance between p and q on shared support."""
    cost_matrix = ot.dist(support, support, metric="sqeuclidean")
    sq = sinkhorn_cost(p, q, cost_matrix, epsilon=epsilon, n_iter=n_iter)
    return float(np.sqrt(max(sq, 0.0)))


def prox_w2(
    distribution: np.ndarray,
    reference: np.ndarray,
    epsilon: float,
    support: np.ndarray,
    n_iter: int = 500,
    step_size: float = 0.1,  # retained for API compatibility (unused: Dykstra is step-free)
    *,
    reg: float | None = None,
) -> np.ndarray:
    """Entropic Wasserstein-2 proximal operator.

    Solves on the probability simplex
        argmin_q  OT_reg(q, reference) + 2 * epsilon * KL(q || distribution),
    where ``epsilon`` is the proximal pull strength and ``reg`` is the Sinkhorn
    entropic regularization (defaults to ``epsilon``, floored for stability).

    This is the entropic W2 prox of Peyré (2015, "Entropic approximation of
    Wasserstein gradient flows"), solved by Dykstra / iterative Bregman
    projections on the Gibbs kernel ``K = exp(-cost/reg)``:
      * the second marginal is constrained to ``reference``;
      * the first marginal carries the proximal (KL) step toward ``distribution``.

    NOTE: the previous implementation used ``transport @ cost.diagonal()`` as the
    transport gradient. The squared-euclidean self-distance cost has a null
    diagonal, so that term was identically zero and ``prox_w2`` degenerated into a
    pure anti-KL repulsion (no transport). The correct transport gradient is the
    Kantorovich dual potential; here we avoid an explicit descent entirely and use
    the closed-form Dykstra scaling, which is unconditionally stable.
    """
    cost_matrix = ot.dist(support, support, metric="sqeuclidean")
    sink_reg = max(reg if reg is not None else epsilon, 1e-3)
    tau = max(epsilon, 0.0)
    # KL-prox exponent: prox of 2*tau*KL(.||distribution) in the Gibbs geometry.
    lam = sink_reg / (sink_reg + 2.0 * tau)
    gibbs = np.exp(-cost_matrix / sink_reg)
    dist = np.clip(distribution, 1e-300, None)
    ref = np.clip(reference, 1e-300, None)
    b = np.ones_like(ref)
    a = np.ones_like(dist)
    for _ in range(n_iter):
        kb = gibbs @ b
        # First-marginal projection: p = (a*kb) with prox toward `dist`:
        #   p = (a*kb)^lam * dist^(1-lam)  =>  a = dist^(1-lam) * (kb)^(lam-1)
        a = dist ** (1.0 - lam) * np.maximum(kb, 1e-300) ** (lam - 1.0)
        ka = gibbs @ a
        # Second-marginal projection: enforce column marginal == reference.
        b = ref / np.maximum(ka, 1e-300)
    q: np.ndarray = a * (gibbs @ b)
    s = float(q.sum())
    if not np.isfinite(s) or s <= 0:  # Sinkhorn/Dykstra blew up -> report, don't fake.
        raise FloatingPointError("prox_w2 Dykstra iteration did not converge (non-finite marginal)")
    return q / s
