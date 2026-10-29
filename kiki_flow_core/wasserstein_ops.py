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
MASS_TOL = 1e-9  # certified entry points: masses must match (and be 1 for the prox)
LOG_FLOOR = 1e-280  # bins below this mass are checked in absolute terms only


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
    marginal_log_tol: float = np.inf,
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
    # Relative (log-domain) residual of the second marginal on every bin that
    # can carry a logarithm: an absolute L1 residual says nothing about bins
    # of tiny mass (review of PR #51, B2).
    live = ref >= LOG_FLOOR
    with np.errstate(divide="ignore"):
        log_second = np.log(second)
    log_residual = float(np.max(np.abs(log_second[live] - log_ref[live]))) if live.any() else 0.0
    if not np.isfinite(log_residual):
        log_residual = np.inf
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
        "marginal_log_residual": log_residual,
        "converged": bool(residual <= marginal_tol and log_residual <= marginal_log_tol),
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
    marginal_log_tol: float = 1e-9,
    init_log_b: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``jko_entropic_prox`` with a certificate; raise if it is not met.

    The reference must be a probability vector (mass 1 within ``MASS_TOL``): the
    output is normalised, so a reference of another mass would be certified
    against a plan it cannot be the marginal of.

    ``init_log_b`` warm-starts the scaling (e.g. ``info["log_b"]`` of a previous
    call on a nearby subproblem); the certificate is unchanged.

    Runs the same scaling iteration until the max change of ``log b`` is below
    ``tol`` or ``n_iter`` iterations, then checks the second-marginal violation
    ``|| pi^T 1 - reference ||_1 <= marginal_tol`` of the plan induced by the
    final scaling, and its relative form ``max_j |log (pi^T 1)_j - log
    reference_j| <= marginal_log_tol`` over the bins of mass >= ``LOG_FLOOR``.
    The output is then the exact prox for a reference perturbed by at most
    ``marginal_log_tol`` in relative terms. Returns ``(q, info)``; raises
    ``SinkhornNotConvergedError`` (with the info in the message) when the
    certificate fails.
    """
    mass = float(np.sum(reference))
    if not abs(mass - 1.0) <= MASS_TOL:
        raise ValueError(f"reference must have mass 1 (got {mass!r})")
    out, info = _entropic_prox_core(
        kl_prox_log,
        reference,
        epsilon,
        support,
        n_iter,
        reg,
        tol,
        marginal_tol,
        init_log_b,
        marginal_log_tol,
    )
    if not info["converged"]:
        raise SinkhornNotConvergedError(f"entropic prox not converged: {info}")
    return out, info


def _entropic_ot_bounds(
    f_pot: np.ndarray,
    g_pot: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    cost_matrix: np.ndarray,
    reg: float,
) -> tuple[float, float, float, float]:
    """Bracket ``lower <= OT_reg(q, r) <= upper``, marginal residual and rounding slack.

    ``lower`` is the dual value <f, q> + <g, r> - reg * sum exp((f + g - C) / reg)
    (weak duality, valid for any potentials); ``upper`` is the primal value of
    the plan built from (f, g) and rounded to the marginals (Altschuler, Weed and
    Rigollet 2017, Algorithm 2), hence feasible. Both need equal masses
    (``entropic_ot`` harmonises them). The potentials are first centred,
    (f + c, g - c) with c = (median g - median f) / 2, which leaves both values
    unchanged in exact arithmetic and keeps the summands small.

    ``slack`` bounds the floating-point error of either value to first order:
    ``ROUNDING_FACTOR * (n m + n + m) * eps * S`` with S the sum of the absolute
    values of the terms that are added and may cancel, S = sum q |f| + sum r |g|
    + sum pi~ C + reg sum pi + reg sum |pi~ (log pi~ - 1)| (pi the Gibbs plan,
    pi~ the rounded one). Each summand carries a relative error of a few eps
    (products, exp and log, the exponent (f + g - C) / reg contributing
    eps (|f| + |g| + C) / reg, i.e. eps pi (|f| + |g| + C) after multiplication by
    reg pi), and a recursive sum of N terms adds at most N eps sum |terms|
    (Higham, Accuracy and Stability of Numerical Algorithms, 2002, section 4.2).
    The deviation of the rounded plan from the exact marginals is of order
    (n + m) eps and moves the primal value by at most that times max |f| + max |g|,
    which the q |f| and r |g| sums cover. It is a declared first-order bound, not
    interval arithmetic.
    """
    shift = 0.5 * (float(np.median(g_pot)) - float(np.median(f_pot)))
    f_c = f_pot + shift
    g_c = g_pot - shift
    log_pi = (f_c[:, None] + g_c[None, :] - cost_matrix) / reg
    pi = np.exp(log_pi)
    lower = float(np.dot(f_c, q) + np.dot(g_c, r) - reg * pi.sum())
    residual = float(np.abs(pi.sum(axis=1) - q).sum() + np.abs(pi.sum(axis=0) - r).sum())
    rows = pi.sum(axis=1)
    x = np.minimum(np.divide(q, rows, out=np.ones_like(q), where=rows > 0), 1.0)
    rounded = pi * x[:, None]
    cols = rounded.sum(axis=0)
    y = np.minimum(np.divide(r, cols, out=np.ones_like(r), where=cols > 0), 1.0)
    rounded = rounded * y[None, :]
    err_r = q - rounded.sum(axis=1)
    err_c = r - rounded.sum(axis=0)
    mass = float(np.abs(err_r).sum())
    if mass > 0.0:
        rounded = rounded + np.outer(err_r, err_c) / mass
    positive = rounded > 0.0
    log_rounded = np.where(positive, np.log(np.where(positive, rounded, 1.0)), 0.0)
    entropy_terms = rounded * (log_rounded - 1.0)
    upper = float(np.sum(rounded * cost_matrix) + reg * np.sum(entropy_terms))
    scale = (
        float(np.dot(np.abs(f_c), q))
        + float(np.dot(np.abs(g_c), r))
        + float(np.sum(rounded * np.abs(cost_matrix)))
        + reg * float(pi.sum())
        + reg * float(np.sum(np.abs(entropy_terms)))
    )
    n, m = cost_matrix.shape
    slack = ROUNDING_FACTOR * (n * m + n + m) * float(np.finfo(float).eps) * scale
    return lower, upper, residual, slack


def _certified_width(lower: float, upper: float, slack: float) -> float:
    """Width of the widened bracket [lower - slack, upper + slack]: the one stopping
    criterion of Sinkhorn, the trigger and the stop of Newton, and the certificate."""
    return (upper + slack) - (lower - slack)


ROUNDING_FACTOR = 4.0  # safety factor of the first-order rounding bound of the OT bracket
NEWTON_MAX_SIZE = 512  # dense Newton polishing of the OT dual up to this many bins
NEWTON_MIN_STEP = 1e-12  # backtracking floor of the Newton line search


def _ot_dual(
    f_pot: np.ndarray, g_pot: np.ndarray, q: np.ndarray, r: np.ndarray, c: np.ndarray, reg: float
) -> float:
    with np.errstate(over="ignore"):
        mass = float(np.exp((f_pot[:, None] + g_pot[None, :] - c) / reg).sum())
    if not np.isfinite(mass):
        return -np.inf
    return float(np.dot(f_pot, q) + np.dot(g_pot, r) - reg * mass)


def _newton_polish(
    f_pot: np.ndarray,
    g_pot: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    reg: float,
    gap_tol: float,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Damped Newton ascent on the (concave) entropic OT dual, last g fixed.

    Sinkhorn converges linearly at a rate that degrades like exp(-C / reg); at
    cost / reg in the thousands it stalls near a gap of 1e-6. Newton steps with
    backtracking usually close it from the Sinkhorn potentials, but no
    convergence is assumed (the Hessian can be numerically singular and the
    ascent can stall): only the bracket of ``_entropic_ot_bounds``, recomputed
    after polishing, certifies the result.
    """
    n = f_pot.size
    steps = 0
    for steps in range(1, max_steps + 1):
        lo, up, _, slack = _entropic_ot_bounds(f_pot, g_pot, q, r, c, reg)
        if _certified_width(lo, up, slack) <= gap_tol:
            return f_pot, g_pot, steps - 1
        pi = np.exp((f_pot[:, None] + g_pot[None, :] - c) / reg)
        grad = np.concatenate([q - pi.sum(axis=1), r - pi.sum(axis=0)])
        hess = np.block([[np.diag(pi.sum(axis=1)), pi], [pi.T, np.diag(pi.sum(axis=0))]])
        direction = np.zeros(2 * n)
        try:
            direction[:-1] = reg * np.linalg.solve(hess[:-1, :-1], grad[:-1])
        except np.linalg.LinAlgError:
            direction[:-1] = reg * np.linalg.lstsq(hess[:-1, :-1], grad[:-1], rcond=None)[0]
        current = _ot_dual(f_pot, g_pot, q, r, c, reg)
        t = 1.0
        while t > NEWTON_MIN_STEP:
            f_new = f_pot + t * direction[:n]
            g_new = g_pot + t * direction[n:]
            if _ot_dual(f_new, g_new, q, r, c, reg) >= current:
                break
            t *= 0.5
        else:
            break  # no ascent: leave the bracket to decide
        f_pot, g_pot = f_new, g_new
    return f_pot, g_pot, steps


def _scaling_stages(
    reg: float,
    top: float,
    scaling_factor: float,
    warm_start: tuple[np.ndarray, np.ndarray] | None,
) -> list[float]:
    """Decreasing regularisations of epsilon-scaling, ending at ``reg``."""
    if warm_start is not None:
        return [reg]
    stages = [reg]
    while stages[-1] < max(top, reg):
        stages.append(stages[-1] / scaling_factor)
    return stages[::-1]


def _check_ot_inputs(
    q: np.ndarray, r: np.ndarray, reg: float, scaling_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    qa = np.asarray(q, dtype=float)
    ra = np.asarray(r, dtype=float)
    for name, v in (("q", qa), ("r", ra)):
        if not np.all(np.isfinite(v)) or np.any(v < 0.0) or v.sum() <= 0.0:
            raise ValueError(f"{name} must be finite, non-negative and of positive mass")
    if not (reg > 0.0 and np.isfinite(reg)):
        raise ValueError("reg must be finite and positive")
    if not abs(float(qa.sum()) - float(ra.sum())) <= MASS_TOL * max(float(qa.sum()), 1.0):
        raise ValueError(f"q and r must have equal masses (got {qa.sum()!r}, {ra.sum()!r})")
    if not 0.0 < scaling_factor < 1.0:
        raise ValueError("scaling_factor must be in (0, 1)")
    return qa, ra


def _sinkhorn_stages(
    qa: np.ndarray,
    ra: np.ndarray,
    cost_matrix: np.ndarray,
    reg: float,
    stages: list[float],
    warm_start: tuple[np.ndarray, np.ndarray] | None,
    *,
    n_iter: int,
    tol: float,
    gap_tol: float,
    check_every: int,
    newton_after: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Log-domain Sinkhorn with epsilon-scaling; stops early on a closed bracket."""
    log_q = np.log(np.clip(qa, 1e-300, None))
    log_r = np.log(np.clip(ra, 1e-300, None))
    if warm_start is None:
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
        last_start = iterations
        stage_cap = n_iter if last else iterations + INTERMEDIATE_STAGE_CAP
        while iterations < min(n_iter, stage_cap):
            iterations += 1
            f_pot = eps * log_q - eps * _logsumexp_rows((g_pot[None, :] - cost_matrix) / eps)
            new_g = eps * log_r - eps * _logsumexp_rows((f_pot[None, :] - cost_matrix.T) / eps)
            delta = float(np.max(np.abs(new_g - g_pot))) / eps
            g_pot = new_g
            if delta < stage_tol:
                break
            if last and iterations % check_every == 0:
                lo, up, _, slack = _entropic_ot_bounds(f_pot, g_pot, qa, ra, cost_matrix, reg)
                if _certified_width(lo, up, slack) <= gap_tol:
                    break
                if iterations - last_start >= newton_after and qa.size <= NEWTON_MAX_SIZE:
                    break
    return f_pot, g_pot, iterations, delta


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
    gap_tol: float = 1e-10,
    check_every: int = 200,
    newton_after: int = 2_000,
    newton_steps: int = 100,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Entropic OT value OT_reg(q, r) = min_pi <C, pi> + reg * sum pi (log pi - 1).

    Log-domain Sinkhorn on the dual potentials (pi = exp((f_i + g_j - C_ij) / reg))
    with both marginals fixed, accelerated by epsilon-scaling: the
    regularisation decreases geometrically (``scaling_factor``) from
    max(C) down to ``reg``, each stage warm-started from the previous
    potentials; the last stage runs to ``tol`` / ``marginal_tol``. ``n_iter``
    bounds the total number of iterations. ``warm_start = (f, g)`` (potentials
    returned in ``info["potentials"]`` by a previous call at the same ``reg``)
    skips the scaling stages; the certificate is the same. Certificate: a
    bracket lower <= OT_reg(q, r) <= upper from the dual value and the rounded
    feasible plan (``_entropic_ot_bounds``). The masses must agree within
    ``MASS_TOL`` (times max(mass, 1)); ``r`` is then rescaled to the mass of
    ``q``, so both bounds are bounds of the same feasible problem
    OT_reg(q, r~), r~ = r |q| / |r| (r~ = r when the masses are equal, as for
    the normalised densities of the JKO step). Both bounds are widened by the
    first-order rounding bound of ``_entropic_ot_bounds`` (scaled by the terms
    that cancel, not by the result); the call is certified when the widened
    width is <= ``gap_tol``. The same widened width is the only criterion used
    to stop Sinkhorn (checked every ``check_every`` iterations of the last
    stage), to trigger Newton polishing and to stop it, so a tolerance below
    the rounding level is refused and a raw bracket reversed beyond the bound
    is an error. The
    value returned is the midpoint, within gap / 2 of OT_reg; ``info["lower"]``
    and ``info["upper"]`` carry the bracket for callers that must compare
    values. ``marginal_tol`` is kept for the call signature and reported, it no
    longer certifies on its own. When the last stage has not certified after
    ``newton_after`` iterations (or at the end of the budget) and the problem has
    at most ``NEWTON_MAX_SIZE`` bins, the dual is polished by damped Newton
    ascent (``newton_steps`` at most): Sinkhorn stalls near a gap of 1e-6 at the
    operating point (cost / reg = 6400). Returns
    ``(value, f, info)``, ``f``
    the dual potential of the first marginal (the gradient of the value with
    respect to ``q`` up to an additive constant). This is, up to an additive
    constant, the transport part of the objective that ``jko_entropic_prox``
    minimises (reg * KL(pi || K)). Raises ``SinkhornNotConvergedError`` when the
    bracket is not certified at the end of the budget.
    """
    qa, ra = _check_ot_inputs(q, r, reg, scaling_factor)
    # Harmonise the masses (they agree within MASS_TOL): both bounds below are
    # then bounds of the same feasible problem OT_reg(q, r~), r~ = r * |q| / |r|.
    ra = ra * (float(qa.sum()) / float(ra.sum()))
    stages = _scaling_stages(reg, float(cost_matrix.max()), scaling_factor, warm_start)
    f_pot, g_pot, iterations, delta = _sinkhorn_stages(
        qa,
        ra,
        cost_matrix,
        reg,
        stages,
        warm_start,
        n_iter=n_iter,
        tol=tol,
        gap_tol=gap_tol,
        check_every=check_every,
        newton_after=newton_after,
    )
    newton = 0
    lower, upper, residual, slack = _entropic_ot_bounds(f_pot, g_pot, qa, ra, cost_matrix, reg)
    width = _certified_width(lower, upper, slack)
    if not width <= gap_tol and qa.size <= NEWTON_MAX_SIZE and newton_steps > 0:
        f_pot, g_pot, newton = _newton_polish(
            f_pot, g_pot, qa, ra, cost_matrix, reg, gap_tol, newton_steps
        )
        lower, upper, residual, slack = _entropic_ot_bounds(f_pot, g_pot, qa, ra, cost_matrix, reg)
    if not (np.isfinite(lower) and np.isfinite(upper) and np.isfinite(slack)):
        raise SinkhornNotConvergedError(f"entropic OT bracket not finite: {lower!r}, {upper!r}")
    if upper - lower < -slack:
        raise SinkhornNotConvergedError(
            f"entropic OT bracket reversed (lower {lower!r} > upper {upper!r}): "
            "infeasible marginals or numerical failure"
        )
    # The rounding bound widens the bracket on both sides, so the certified
    # interval is never reversed and a tolerance below the rounding level of
    # the terms that cancel cannot be met.
    lower_c, upper_c = lower - slack, upper + slack
    info: dict[str, Any] = {
        "iterations": iterations,
        "newton_steps": newton,
        "stages": len(stages),
        "last_delta": delta,
        "marginal_residual": residual,
        "lower": lower_c,
        "upper": upper_c,
        "gap": upper_c - lower_c,
        "lower_raw": lower,
        "upper_raw": upper,
        "rounding_slack": slack,
        "marginal_tol": marginal_tol,
        "converged": bool(upper_c - lower_c <= gap_tol),
    }
    if not info["converged"]:
        raise SinkhornNotConvergedError(f"entropic OT not converged: {info}")
    value = 0.5 * (lower_c + upper_c)
    info["potentials"] = (f_pot.copy(), g_pot.copy())
    return value, f_pot, info
