"""Wasserstein-2 operations via entropic Sinkhorn (POT-backed) and the entropic JKO prox."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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


INTERMEDIATE_STAGE_CAP = 2000  # iterations per non-final epsilon-scaling stage


class SinkhornNotConvergedError(RuntimeError):
    """A Sinkhorn-type iteration hit its budget without meeting its tolerance."""


def _entropic_prox_core(
    kl_prox_log: KLProxLog,
    reference: np.ndarray,
    epsilon: float,
    support: np.ndarray,
    n_iter: int,
    reg: float | None,
    tol: float,
    marginal_tol: float,
    init_log_b: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if epsilon < 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon (JKO step) must be finite and non-negative")
    sink_reg = reg if reg is not None else epsilon
    if not (sink_reg > 0.0 and np.isfinite(sink_reg)):
        raise ValueError("Sinkhorn regularisation must be finite and positive")
    ref = np.asarray(reference, dtype=float)
    if not np.all(np.isfinite(ref)) or np.any(ref < 0.0) or ref.sum() <= 0.0:
        raise ValueError("reference must be finite, non-negative and of positive mass")
    cost_matrix = ot.dist(support, support, metric="sqeuclidean")
    tau = 2.0 * epsilon / sink_reg
    log_kernel = -cost_matrix / sink_reg
    log_ref = np.log(np.clip(ref, 1e-300, None))
    log_b = np.zeros_like(log_ref) if init_log_b is None else np.array(init_log_b, dtype=float)
    log_a = np.zeros_like(log_ref)
    delta = np.inf
    iterations = 0
    while iterations < n_iter:
        iterations += 1
        log_kb = _logsumexp_rows(log_kernel + log_b[None, :])
        log_a = kl_prox_log(log_kb, tau) - log_kb
        log_a = log_a - log_a.max()  # a global factor is absorbed by the b update
        log_ka = _logsumexp_rows(log_kernel.T + log_a[None, :])
        new_log_b = log_ref - log_ka
        delta = float(np.max(np.abs(new_log_b - log_b)))
        log_b = new_log_b
        if tol > 0.0 and delta < tol:
            break
    # Certificate: second-marginal violation of the plan built from the a-update
    # that the current b induces (zero exactly at a fixed point).
    log_kb = _logsumexp_rows(log_kernel + log_b[None, :])
    log_a_chk = kl_prox_log(log_kb, tau) - log_kb
    log_a_chk = log_a_chk - log_a_chk.max()
    second = np.exp(log_b + _logsumexp_rows(log_kernel.T + log_a_chk[None, :]))
    residual = float(np.abs(second - ref).sum())
    log_q = log_a + _logsumexp_rows(log_kernel + log_b[None, :])
    if not np.all(np.isfinite(log_q)):
        raise FloatingPointError(
            "jko_entropic_prox scaling iteration diverged (non-finite marginal)"
        )
    q: np.ndarray = np.exp(log_q - log_q.max())
    out: np.ndarray = q / q.sum()
    # Entropic transport value of the output plan pi = diag(a) K diag(b): its
    # marginals are (q, reference), and a Gibbs-form plan with given marginals is
    # the entropic optimum for them, so this is OT_reg(q, reference) (up to the
    # certified marginal residual), with OT_reg = <C, pi> + reg sum pi (log pi - 1).
    log_pi = log_a[:, None] + log_kernel + log_b[None, :]
    pi = np.exp(log_pi)
    transport = float(np.sum(pi * cost_matrix) + sink_reg * np.sum(pi * (log_pi - 1.0)))
    info: dict[str, Any] = {
        "iterations": iterations,
        "last_delta": delta,
        "marginal_residual": residual,
        "converged": bool(residual <= marginal_tol),
        "transport_value": transport,
        "log_b": log_b.copy(),
    }
    return out, info


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
    (``tol = 0`` runs all ``n_iter`` iterations, deterministic). This entry
    point returns no certificate (kept for the explicit scheme and the legacy
    callers); ``jko_entropic_prox_certified`` returns one and raises
    ``SinkhornNotConvergedError`` when the budget is exhausted.
    """
    out, _ = _entropic_prox_core(
        kl_prox_log, reference, epsilon, support, n_iter, reg, tol, marginal_tol=np.inf
    )
    return out


def jko_entropic_prox_certified(
    kl_prox_log: KLProxLog,
    reference: np.ndarray,
    epsilon: float,
    support: np.ndarray,
    *,
    reg: float,
    n_iter: int = 200_000,
    tol: float = 1e-13,
    marginal_tol: float = 1e-10,
    init_log_b: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``jko_entropic_prox`` with a certificate; raise if it is not met.

    ``init_log_b`` warm-starts the scaling (e.g. ``info["log_b"]`` of a previous
    call on a nearby subproblem); the certificate is unchanged.

    Runs the same scaling iteration until the max change of ``log b`` is below
    ``tol`` or ``n_iter`` iterations, then checks the second-marginal violation
    ``|| pi^T 1 - reference ||_1 <= marginal_tol`` of the plan induced by the
    final scaling. Returns ``(q, info)``; raises ``SinkhornNotConvergedError`` (with
    the info in the message) when the certificate fails.
    """
    out, info = _entropic_prox_core(
        kl_prox_log, reference, epsilon, support, n_iter, reg, tol, marginal_tol, init_log_b
    )
    if not info["converged"]:
        raise SinkhornNotConvergedError(f"entropic prox not converged: {info}")
    return out, info


def entropic_ot(
    q: np.ndarray,
    r: np.ndarray,
    cost_matrix: np.ndarray,
    reg: float,
    *,
    n_iter: int = 50_000,
    tol: float = 1e-13,
    marginal_tol: float = 1e-10,
    scaling_factor: float = 0.5,
    warm_start: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Entropic OT value OT_reg(q, r) = min_pi <C, pi> + reg * sum pi (log pi - 1).

    Log-domain Sinkhorn on the dual potentials (pi = exp((f_i + g_j - C_ij) / reg))
    with both marginals fixed, accelerated by epsilon-scaling: the
    regularisation decreases geometrically (``scaling_factor``) from
    max(C) down to ``reg``, each stage warm-started from the previous
    potentials; the last stage runs to ``tol`` / ``marginal_tol``. ``n_iter``
    bounds the total number of iterations. ``warm_start = (f, g)`` (potentials
    returned in ``info["potentials"]`` by a previous call at the same ``reg``)
    skips the scaling stages; the certificate is the same. Returns
    ``(value, f, info)``, ``f``
    the dual potential of the first marginal (the gradient of the value with
    respect to ``q`` up to an additive constant). This is, up to an additive
    constant, the transport part of the objective that ``jko_entropic_prox``
    minimises (reg * KL(pi || K)). Raises ``SinkhornNotConvergedError`` when
    ``|| pi 1 - q ||_1 + || pi^T 1 - r ||_1 > marginal_tol`` at the end.
    """
    qa = np.asarray(q, dtype=float)
    ra = np.asarray(r, dtype=float)
    for name, v in (("q", qa), ("r", ra)):
        if not np.all(np.isfinite(v)) or np.any(v < 0.0) or v.sum() <= 0.0:
            raise ValueError(f"{name} must be finite, non-negative and of positive mass")
    if not (reg > 0.0 and np.isfinite(reg)):
        raise ValueError("reg must be finite and positive")
    if not 0.0 < scaling_factor < 1.0:
        raise ValueError("scaling_factor must be in (0, 1)")
    log_q = np.log(np.clip(qa, 1e-300, None))
    log_r = np.log(np.clip(ra, 1e-300, None))
    stages = [reg]
    if warm_start is None:
        top = max(float(cost_matrix.max()), reg)
        while stages[-1] < top:
            stages.append(stages[-1] / scaling_factor)
        stages = stages[::-1]
        f_pot = np.zeros_like(log_q)
        g_pot = np.zeros_like(log_r)
    else:
        f_pot = np.array(warm_start[0], dtype=float)
        g_pot = np.array(warm_start[1], dtype=float)
    iterations = 0
    delta = np.inf
    for k, eps in enumerate(stages):
        last = k == len(stages) - 1
        stage_tol = tol if last else 1e-6
        stage_cap = n_iter if last else iterations + INTERMEDIATE_STAGE_CAP
        while iterations < min(n_iter, stage_cap):
            iterations += 1
            f_pot = eps * log_q - eps * _logsumexp_rows((g_pot[None, :] - cost_matrix) / eps)
            new_g = eps * log_r - eps * _logsumexp_rows((f_pot[None, :] - cost_matrix.T) / eps)
            delta = float(np.max(np.abs(new_g - g_pot))) / eps
            g_pot = new_g
            if delta < stage_tol:
                break
    log_pi = (f_pot[:, None] + g_pot[None, :] - cost_matrix) / reg
    pi = np.exp(log_pi)
    residual = float(np.abs(pi.sum(axis=1) - qa).sum() + np.abs(pi.sum(axis=0) - ra).sum())
    info: dict[str, Any] = {
        "iterations": iterations,
        "stages": len(stages),
        "last_delta": delta,
        "marginal_residual": residual,
        "converged": bool(residual <= marginal_tol),
    }
    if not info["converged"]:
        raise SinkhornNotConvergedError(f"entropic OT not converged: {info}")
    safe = np.where(pi > 0.0, log_pi, 0.0)
    value = float(np.sum(pi * cost_matrix) + reg * np.sum(pi * (safe - 1.0)))
    info["potentials"] = (f_pot.copy(), g_pot.copy())
    return value, f_pot, info
