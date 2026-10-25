"""Wasserstein-2 operations via entropic Sinkhorn (POT-backed) and the entropic JKO prox."""

from __future__ import annotations

from collections.abc import Callable

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


KLProxLog = Callable[[np.ndarray, float], np.ndarray]


def _logsumexp_rows(m: np.ndarray) -> np.ndarray:
    """log(sum(exp(m), axis=1)) without scipy, stable for very negative entries."""
    top = m.max(axis=1, keepdims=True)
    out: np.ndarray = np.asarray(
        (top + np.log(np.sum(np.exp(m - top), axis=1, keepdims=True)))[:, 0], dtype=float
    )
    return out


def jko_entropic_prox(
    kl_prox_log: KLProxLog,
    reference: np.ndarray,
    epsilon: float,
    support: np.ndarray,
    n_iter: int = 500,
    *,
    reg: float | None = None,
    tol: float = 0.0,
) -> np.ndarray:
    """Entropic JKO step of a functional F, anchored on ``reference`` (Peyré 2015).

    Solves on the probability simplex
        argmin_q  OT_reg(q, reference) + 2 * epsilon * F(q),
    the entropic form of ``argmin_q F(q) + W2^2(q, reference) / (2 epsilon)``:
    ``epsilon`` is the JKO time step h, ``reg`` the Sinkhorn entropic
    regularisation (defaults to ``epsilon``; no floor, the iteration runs in the
    log domain). F enters only through its KL-proximal map in log form:
    ``kl_prox_log(log_p, tau)`` returns ``log argmin_q KL(q || p) + tau * F(q)``
    over R^n_+, up to an additive constant (see ``FreeEnergy.kl_prox_log``).

    Generalised Sinkhorn / scaling iteration (Chizat, Peyré, Schmitzer, Vialard
    2018) on pi = diag(a) K diag(b), K = exp(-cost / reg):
        log a <- kl_prox_log(log(K b), 2 epsilon / reg) - log(K b)
        log b <- log reference - log(K^T a)
    The second marginal is the reference (the W2 anchor); the first marginal
    carries the proximal step of F. With ``kl_prox_log(log_p, tau) =
    (log_p + tau * log d) / (1 + tau)`` this is exactly ``prox_w2(d, reference,
    ...)`` (the legacy split scheme, see ``tests/test_jko_entropic_step.py``);
    with the KL-prox of F itself it is the entropic JKO step of F.
    ``tol > 0`` stops early once the max change of ``log b`` is below it
    (``tol = 0`` runs all ``n_iter`` iterations, deterministic).
    """
    if epsilon < 0.0:
        raise ValueError("epsilon (JKO step) must be non-negative")
    sink_reg = reg if reg is not None else epsilon
    if sink_reg <= 0.0:
        raise ValueError("Sinkhorn regularisation must be positive")
    cost_matrix = ot.dist(support, support, metric="sqeuclidean")
    tau = 2.0 * epsilon / sink_reg
    log_kernel = -cost_matrix / sink_reg
    log_ref = np.log(np.clip(np.asarray(reference, dtype=float), 1e-300, None))
    log_b = np.zeros_like(log_ref)
    log_a = np.zeros_like(log_ref)
    for _ in range(n_iter):
        log_kb = _logsumexp_rows(log_kernel + log_b[None, :])
        log_a = kl_prox_log(log_kb, tau) - log_kb
        log_a = log_a - log_a.max()  # a global factor is absorbed by the b update
        log_ka = _logsumexp_rows(log_kernel.T + log_a[None, :])
        new_log_b = log_ref - log_ka
        delta = float(np.max(np.abs(new_log_b - log_b)))
        log_b = new_log_b
        if tol > 0.0 and delta < tol:
            break
    log_q = log_a + _logsumexp_rows(log_kernel + log_b[None, :])
    if not np.all(np.isfinite(log_q)):
        raise FloatingPointError(
            "jko_entropic_prox scaling iteration diverged (non-finite marginal)"
        )
    q: np.ndarray = np.exp(log_q - log_q.max())
    out: np.ndarray = q / q.sum()
    return out
