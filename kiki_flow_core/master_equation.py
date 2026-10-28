"""FreeEnergy abstract base, ZeroF reference, and generic JKOStep."""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import ot

from kiki_flow_core.state import FlowState
from kiki_flow_core.wasserstein_ops import (
    KLProxLog,
    SinkhornNotConvergedError,
    entropic_ot,
    jko_entropic_prox,
    jko_entropic_prox_certified,
    prox_w2,
)

ProxFn = Callable[..., np.ndarray]
Scheme = Literal["legacy_split", "jko_entropic"]
SCHEMES: tuple[Scheme, ...] = ("legacy_split", "jko_entropic")
NonSeparable = Literal["explicit", "implicit"]
NONSEPARABLE_MODES: tuple[NonSeparable, ...] = ("explicit", "implicit")
OnFailure = Literal["raise", "warn"]
_FLOOR = 1e-300


class FreeEnergy(ABC):
    """Abstract free-energy functional on FlowState."""

    @abstractmethod
    def value(self, state: FlowState) -> float:
        """Return the scalar free-energy F[state]."""

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        """Numerical gradient of F w.r.t. rho[species_name].

        Override for analytical gradients when available. Each coordinate is perturbed
        and the distribution is renormalized so the gradient respects the simplex
        constraint (mass conservation of rho).
        """
        rho = state.rho[species_name]
        grad = np.zeros_like(rho)
        f0 = self.value(state)
        for i in range(rho.size):
            perturbed = rho.copy()
            perturbed[i] += eps
            perturbed = perturbed / perturbed.sum()
            new_state = state.model_copy(update={"rho": {**state.rho, species_name: perturbed}})
            grad[i] = (self.value(new_state) - f0) / eps
        return grad

    def kl_prox_log(self, state: FlowState, species_name: str) -> KLProxLog:
        """Log-domain KL-proximal map of F restricted to one species, at ``state``.

        Returns ``prox(log_p, tau) = log argmin_q KL(q || p) + tau * F(q)`` over
        R^n_+ (up to an additive constant), F being restricted to
        ``rho[species_name]`` with the other species frozen at ``state``. This is
        what ``JKOStep`` with ``scheme="jko_entropic"`` hands to
        ``wasserstein_ops.jko_entropic_prox``: the entropic JKO step of F needs
        F only through this map (Peyré 2015).

        Default: F linearised at ``state.rho[species_name]`` (explicit
        potential, first order in the step), ``log_p - tau * grad_rho``. The
        gradient is evaluated once, when the map is built. Override with the
        closed form when F has one (``T2FreeEnergy``: potential + KL to the
        prior exactly, couplings linearised).
        """
        grad = np.asarray(self.grad_rho(state, species_name), dtype=float)

        def prox(log_p: np.ndarray, tau: float) -> np.ndarray:
            return log_p - tau * grad

        return prox


class ZeroF(FreeEnergy):
    """Identically-zero free energy. Used in tests and as no-op placeholder."""

    def value(self, state: FlowState) -> float:
        return 0.0

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        return np.zeros_like(state.rho[species_name])


class ImplicitJKOError(RuntimeError):
    """The certified implicit JKO step failed; ``info`` holds the diagnostics."""

    def __init__(self, message: str, info: dict[str, Any]) -> None:
        super().__init__(f"{message}: {info}")
        self.info = info


class JKOStep:
    """One slow step of the Wasserstein flow of F, in one of two schemes.

    ``scheme=None`` (default) resolves to ``"jko_entropic"`` when
    ``apply_w2_prox`` is set (the "rigorous" path; default since 2026-10-25,
    ERRATUM.md section 13) and to ``"legacy_split"`` otherwise (the fast path:
    Euclidean descent of F without prox, unchanged).

    ``scheme="legacy_split"``: ``n_inner``
    projected Euclidean gradient sub-steps of F on the simplex (explicit Euler
    of the Euclidean gradient flow of F over time h), then, if
    ``apply_w2_prox``, ``prox_w2`` — Peyré's entropic JKO step of
    ``KL(. || d)`` anchored on rho_tau, d being the sub-step output. This is
    **not** the JKO step of F (``docs/notes/2026-10-25-jko-step-identification.md``);
    it is kept, explicitly, for the reproducibility of past results.

    ``scheme="jko_entropic"``: the entropic JKO step of F,
    ``Q_h(q) = F(q) + sum_i OT_reg(q_i, rho_tau_i) / (2h)`` with
    ``OT_reg(q, r) = min_{pi in Pi(q, r)} <C, pi> + reg * sum pi (log pi - 1)``
    (not debiased: OT_reg(r, r) != 0), Sinkhorn reg = ``reg`` (default h^2).
    ``n_inner`` and ``apply_w2_prox`` are ignored. ``nonseparable`` sets how the
    terms of F without a closed-form prox (J_sym coupling, Turing term) enter:

    * ``"explicit"``: linearised once at rho_tau, one generalised-Sinkhorn
      solve per species (``wasserstein_ops.jko_entropic_prox``, ``prox_n_iter``
      iterations, ``prox_tol``). Not a minimiser of Q_h; not dissipative when
      the Turing term is strong (ERRATUM.md section 14). Kept to reproduce
      earlier results.
    * ``"implicit"`` (default): a certified stationary point of Q_h, by
      Bregman (KL) majorisation-minimisation with a line search on Q_h itself.
      With T_L(q) the generalised-Sinkhorn solution of the subproblem in which
      the non-separable part N of F is linearised at q and a proximal term
      L * KL(. || q) is added, the iteration starts from the explicit output
      q_0 = T_0(rho_tau) and accepts q_{k+1} = T_L(q_k) for the smallest L in
      {0, L0, 4 L0, ...} such that Q_h(q_{k+1}) <= Q_h(q_k) - sigma * L *
      KL(q_{k+1} || q_k). It stops when the stationarity residual at the
      returned point, max_i TV(T_0(q)_i, q_i), is below ``kkt_tol``: T_0(q) = q
      is the first-order (KKT) condition of Q_h when N has an exact gradient
      (``F.implicit_ready``). Every inner Sinkhorn solve and every OT_reg
      evaluation carries its own certificate (marginal residual). Exhausting
      ``max_outer`` iterations, a line search beyond ``l_max`` or an inner
      certificate failure raises ``ImplicitJKOError`` (``on_failure="raise"``,
      default); ``on_failure="warn"`` returns the last iterate with a warning
      and ``status != "converged"`` (diagnostics only). ``self.last_info``
      records status, iterations, residuals, Q_h values and costs. Q_h is not
      convex in general (Turing coupling): the certificate is stationarity and
      monotone decrease of Q_h from the explicit output, not global optimality.
      There is no sub-stepping: the step solves Q_h for the requested h.
    """

    def __init__(
        self,
        f_functional: FreeEnergy,
        h: float,
        support: np.ndarray,
        n_inner: int = 50,
        apply_w2_prox: bool = False,
        prox_fn: ProxFn | None = None,
        reg: float | None = None,
        prox_n_iter: int = 100,
        scheme: Scheme | None = None,
        prox_tol: float = 0.0,
        nonseparable: NonSeparable = "implicit",
        kkt_tol: float = 1e-6,
        max_outer: int = 500,
        l0: float = 1.0,
        l_max: float = 1e8,
        sigma: float = 1e-4,
        inner_n_iter: int = 200_000,
        inner_tol: float = 1e-13,
        inner_marginal_tol: float = 1e-10,
        on_failure: OnFailure = "raise",
    ) -> None:
        if not (h > 0 and np.isfinite(h)):
            raise ValueError("h must be finite and positive")
        if scheme is None:
            scheme = "jko_entropic" if apply_w2_prox else "legacy_split"
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
        if nonseparable not in NONSEPARABLE_MODES:
            raise ValueError(
                f"nonseparable must be one of {NONSEPARABLE_MODES}, got {nonseparable!r}"
            )
        if on_failure not in ("raise", "warn"):
            raise ValueError("on_failure must be 'raise' or 'warn'")
        if not (kkt_tol > 0 and max_outer >= 1 and 0 < l0 <= l_max and sigma >= 0):
            raise ValueError("invalid implicit-solver settings")
        if not (inner_n_iter >= 1 and inner_tol > 0 and inner_marginal_tol > 0):
            raise ValueError("invalid inner-solver settings")
        self.f_functional = f_functional
        self.h = h
        self.support = support
        self.n_inner = n_inner
        self.apply_w2_prox = apply_w2_prox
        # Sinkhorn entropic regularisation of the prox. None keeps the prox's
        # own default (reg = its proximal strength, i.e. h) for the legacy split
        # and h^2 for the entropic scheme. The JKO step h is the proximal pull.
        self.reg = reg
        # Dykstra iterations of the prox (100 unless a subclass forwards its own).
        self.prox_n_iter = prox_n_iter
        self.scheme: Scheme = scheme
        self.prox_tol = prox_tol
        self.nonseparable: NonSeparable = nonseparable
        self.kkt_tol = kkt_tol
        self.max_outer = max_outer
        self.l0 = l0
        self.l_max = l_max
        self.sigma = sigma
        self.inner_n_iter = inner_n_iter
        self.inner_tol = inner_tol
        self.inner_marginal_tol = inner_marginal_tol
        self.on_failure: OnFailure = on_failure
        self.last_info: dict[str, Any] = {}
        if (
            self.scheme == "jko_entropic"
            and self.nonseparable == "implicit"
            and not getattr(f_functional, "implicit_ready", True)
        ):
            raise ValueError(
                "the implicit JKO step needs an exact gradient of the non-separable terms; "
                "declare a smoothing of the Turing term (T2FreeEnergy(turing_smoothing=...)) "
                "or use nonseparable='explicit'"
            )
        # Strategy pattern: Track-specific subclasses can inject an MLX-backed
        # prox without forcing a circular import here. Default remains the
        # POT-backed prox_w2. Only the legacy scheme uses it.
        self.prox_fn: ProxFn = prox_fn if prox_fn is not None else cast(ProxFn, prox_w2)

    def step(self, state: FlowState) -> FlowState:
        if self.scheme == "jko_entropic":
            return self._step_jko_entropic(state)
        return self._step_legacy_split(state)

    def _step_legacy_split(self, state: FlowState) -> FlowState:
        new_rho: dict[str, np.ndarray] = {}
        substep = self.h / max(self.n_inner, 1)
        for name, rho_tau in state.rho.items():
            if rho_tau.size != self.support.shape[0]:
                new_rho[name] = rho_tau
                continue
            rho = rho_tau.copy()
            for _ in range(self.n_inner):
                step_state = state.model_copy(update={"rho": {**state.rho, name: rho}})
                grad_f = self.f_functional.grad_rho(step_state, name)
                rho = rho - substep * grad_f
                rho = np.clip(rho, 1e-12, None)
                rho = rho / rho.sum()
            if self.apply_w2_prox:
                prox_kwargs: dict[str, float] = {} if self.reg is None else {"reg": self.reg}
                rho = self.prox_fn(
                    rho,
                    reference=rho_tau,
                    epsilon=self.h,
                    support=self.support,
                    n_iter=self.prox_n_iter,
                    **prox_kwargs,
                )
            new_rho[name] = rho
        return state.model_copy(update={"rho": new_rho, "tau": state.tau + 1})

    @property
    def entropic_reg(self) -> float:
        reg = self.reg if self.reg is not None else self.h * self.h
        if not (reg > 0 and np.isfinite(reg)):
            raise ValueError(f"Sinkhorn regularisation must be finite and positive, got {reg}")
        return float(reg)

    def _step_jko_entropic(self, state: FlowState) -> FlowState:
        if self.nonseparable == "implicit":
            rho = self._implicit_solve(state)
            return state.model_copy(update={"rho": rho, "tau": state.tau + 1})
        new_rho: dict[str, np.ndarray] = {}
        for name, rho_tau in state.rho.items():
            if rho_tau.size != self.support.shape[0]:
                new_rho[name] = rho_tau
                continue
            new_rho[name] = jko_entropic_prox(
                self.f_functional.kl_prox_log(state, name),
                reference=rho_tau,
                epsilon=self.h,
                support=self.support,
                n_iter=self.prox_n_iter,
                reg=self.entropic_reg,
                tol=self.prox_tol,
            )
        return state.model_copy(update={"rho": new_rho, "tau": state.tau + 1})

    # ----- certified implicit step -------------------------------------------------

    def _map(
        self,
        state: FlowState,
        lin: dict[str, np.ndarray],
        reference: dict[str, np.ndarray],
        anchor_weight: float,
        stats: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], float]:
        """T_L(lin) and its Q_h value.

        The non-separable terms are linearised at ``lin`` and L * KL(. || lin) is
        added. Q_h of the output uses the transport value of each prox's own
        plan (the entropic optimum for its marginals, certified by the prox).
        """
        lin_state = state.model_copy(update={"rho": {**state.rho, **lin}})
        out: dict[str, np.ndarray] = {}
        transport = 0.0
        for name, ref in reference.items():
            base = self.f_functional.kl_prox_log(lin_state, name)
            prox = base
            if anchor_weight > 0.0:
                log_anchor = np.log(np.clip(lin[name], _FLOOR, None))
                weight = anchor_weight

                def prox(
                    log_p: np.ndarray,
                    tau: float,
                    base: KLProxLog = base,
                    log_anchor: np.ndarray = log_anchor,
                    weight: float = weight,
                ) -> np.ndarray:
                    scale = 1.0 + tau * weight
                    return base((log_p + tau * weight * log_anchor) / scale, tau / scale)

            warm = stats.setdefault("_warm_b", {})
            q, info = jko_entropic_prox_certified(
                prox,
                ref,
                self.h,
                self.support,
                reg=self.entropic_reg,
                n_iter=self.inner_n_iter,
                tol=self.inner_tol,
                marginal_tol=self.inner_marginal_tol,
                init_log_b=warm.get(name),
            )
            warm[name] = info["log_b"]
            stats["prox_calls"] += 1
            stats["inner_max_iterations"] = max(
                stats["inner_max_iterations"], int(info["iterations"])
            )
            stats["inner_max_residual"] = max(
                stats["inner_max_residual"], float(info["marginal_residual"])
            )
            out[name] = q
            transport += float(info["transport_value"])
        value = self.f_functional.value(state.model_copy(update={"rho": {**state.rho, **out}}))
        return out, value + transport / (2.0 * self.h)

    def objective(
        self,
        state: FlowState,
        q: dict[str, np.ndarray],
        reference: dict[str, np.ndarray],
        stats: dict[str, Any] | None = None,
    ) -> float:
        """Q_h(q) = F(q) + sum_i OT_reg(q_i, rho_tau_i) / (2h) at an arbitrary q.

        Independent evaluation by balanced log-domain Sinkhorn (``entropic_ot``,
        certified); the solver itself uses the values of its prox plans.
        """
        cost = ot.dist(self.support, self.support, metric="sqeuclidean")
        total = self.f_functional.value(state.model_copy(update={"rho": {**state.rho, **q}}))
        warm = stats.setdefault("_warm", {}) if stats is not None else {}
        for name, ref in reference.items():
            try:
                value, _, info = entropic_ot(
                    q[name],
                    ref,
                    cost,
                    self.entropic_reg,
                    n_iter=self.inner_n_iter,
                    tol=self.inner_tol,
                    marginal_tol=self.inner_marginal_tol,
                    warm_start=warm.get(name),
                )
            except SinkhornNotConvergedError:
                if warm.get(name) is None:
                    raise
                # A warm start that does not certify is retried from scratch.
                value, _, info = entropic_ot(
                    q[name],
                    ref,
                    cost,
                    self.entropic_reg,
                    n_iter=self.inner_n_iter,
                    tol=self.inner_tol,
                    marginal_tol=self.inner_marginal_tol,
                )
            warm[name] = info["potentials"]
            total += value / (2.0 * self.h)
            if stats is not None:
                stats["ot_calls"] += 1
                stats["ot_max_residual"] = max(
                    stats["ot_max_residual"], float(info["marginal_residual"])
                )
        return float(total)

    @staticmethod
    def _check_densities(reference: dict[str, np.ndarray]) -> None:
        for name, rho in reference.items():
            arr = np.asarray(rho, dtype=float)
            if not np.all(np.isfinite(arr)) or np.any(arr < 0.0) or arr.sum() <= 0.0:
                raise ValueError(f"density {name!r} must be finite, non-negative, positive mass")

    def _line_search(
        self,
        state: FlowState,
        q: dict[str, np.ndarray],
        q_val: float,
        t0: tuple[dict[str, np.ndarray], float],
        reference: dict[str, np.ndarray],
        stats: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], float, float] | None:
        """Smallest L in {0, l0, 4 l0, ...} with sufficient decrease of Q_h; None if none."""
        weight = 0.0
        cand, c_val = t0
        while True:
            kl = sum(float(np.sum(cand[n] * np.log(cand[n] / q[n]))) for n in reference)
            if c_val <= q_val - self.sigma * weight * kl and c_val <= q_val:
                return cand, c_val, weight
            weight = self.l0 if weight == 0.0 else 4.0 * weight
            if weight > self.l_max:
                return None
            cand, c_val = self._map(state, q, reference, weight, stats)

    def _new_stats(self) -> dict[str, Any]:
        return {
            "status": "running",
            "outer_iterations": 0,
            "prox_calls": 0,
            "ot_calls": 0,
            "inner_max_iterations": 0,
            "inner_max_residual": 0.0,
            "ot_max_residual": 0.0,
            "max_anchor_weight": 0.0,
            "kkt_residual": float("nan"),
            "kkt_tol": self.kkt_tol,
            "h": self.h,
            "reg": self.entropic_reg,
            "Q_final": float("nan"),
        }

    def _implicit_solve(self, state: FlowState) -> dict[str, np.ndarray]:
        tic = time.perf_counter()
        reference = {n: r for n, r in state.rho.items() if r.size == self.support.shape[0]}
        stats = self._new_stats()
        result: dict[str, np.ndarray] = dict(state.rho)
        if not reference:  # nothing on this support: same as the explicit path
            stats.update(status="converged", kkt_residual=0.0, seconds=0.0)
            self.last_info = stats
            return result
        self._check_densities(reference)
        try:
            q, q_val = self._map(state, reference, reference, 0.0, stats)  # explicit output
            stats["Q_explicit"] = q_val
            while True:
                result, stats["Q_final"] = {**state.rho, **q}, q_val
                t0 = self._map(state, q, reference, 0.0, stats)
                stats["kkt_residual"] = max(
                    0.5 * float(np.abs(t0[0][n] - q[n]).sum()) for n in reference
                )
                if stats["kkt_residual"] < self.kkt_tol:
                    stats["status"] = "converged"
                    break
                if stats["outer_iterations"] >= self.max_outer:
                    stats["status"] = "max_outer"
                    break
                found = self._line_search(state, q, q_val, t0, reference, stats)
                if found is None:
                    stats["status"] = "line_search"
                    break
                q, q_val, weight = found
                stats["max_anchor_weight"] = max(stats["max_anchor_weight"], weight)
                stats["outer_iterations"] += 1
        except SinkhornNotConvergedError as exc:
            stats["status"] = "inner_not_converged"
            stats["inner_error"] = str(exc)
        stats["seconds"] = time.perf_counter() - tic
        stats.pop("_warm", None)
        stats.pop("_warm_b", None)
        self.last_info = stats
        if stats["status"] != "converged":
            if self.on_failure == "raise":
                raise ImplicitJKOError("implicit JKO step not certified", stats)
            warnings.warn(f"implicit JKO step not certified: {stats}", stacklevel=3)
        return result
