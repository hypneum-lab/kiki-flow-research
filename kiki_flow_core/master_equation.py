"""FreeEnergy abstract base, ZeroF reference, and generic JKOStep."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal, cast

import numpy as np

from kiki_flow_core.state import FlowState
from kiki_flow_core.wasserstein_ops import KLProxLog, jko_entropic_prox, prox_w2

ProxFn = Callable[..., np.ndarray]
Scheme = Literal["legacy_split", "jko_entropic"]
SCHEMES: tuple[Scheme, ...] = ("legacy_split", "jko_entropic")


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


class JKOStep:
    """One slow step of the Wasserstein flow of F, in one of two schemes.

    ``scheme="legacy_split"`` (default, unchanged behaviour): ``n_inner``
    projected Euclidean gradient sub-steps of F on the simplex (explicit Euler
    of the Euclidean gradient flow of F over time h), then, if
    ``apply_w2_prox``, ``prox_w2`` — Peyré's entropic JKO step of
    ``KL(. || d)`` anchored on rho_tau, d being the sub-step output. This is
    **not** the JKO step of F (``docs/notes/2026-10-25-jko-step-identification.md``);
    it is kept for the reproducibility of past results.

    ``scheme="jko_entropic"``: the entropic JKO step of F itself,
    ``argmin_rho F(rho) + W_reg(rho, rho_tau) / (2h)``, by the generalised
    Sinkhorn iteration on ``F.kl_prox_log`` (``wasserstein_ops.jko_entropic_prox``,
    log domain). ``n_inner`` and ``apply_w2_prox`` are ignored; ``reg`` is the
    Sinkhorn regularisation (default h), ``prox_n_iter`` the iteration count,
    ``prox_tol`` an optional early-stopping tolerance on the scaling vector.
    Species are updated from the same input state (Jacobi), as in the legacy
    scheme.
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
        scheme: Scheme = "legacy_split",
        prox_tol: float = 0.0,
    ) -> None:
        if h <= 0:
            raise ValueError("h must be positive")
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
        self.f_functional = f_functional
        self.h = h
        self.support = support
        self.n_inner = n_inner
        self.apply_w2_prox = apply_w2_prox
        # Sinkhorn entropic regularisation of the prox. None keeps the prox's
        # own default (reg = its proximal strength, i.e. h). The JKO step h is
        # always the proximal pull strength.
        self.reg = reg
        # Dykstra iterations of the prox (100 unless a subclass forwards its own).
        self.prox_n_iter = prox_n_iter
        self.scheme: Scheme = scheme
        self.prox_tol = prox_tol
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

    def _step_jko_entropic(self, state: FlowState) -> FlowState:
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
                reg=self.reg,
                tol=self.prox_tol,
            )
        return state.model_copy(update={"rho": new_rho, "tau": state.tau + 1})
