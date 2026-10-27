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
NonSeparable = Literal["explicit", "implicit"]
NONSEPARABLE_MODES: tuple[NonSeparable, ...] = ("explicit", "implicit")
DISSIPATION_TOL = 1e-12


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

    ``scheme="jko_entropic"``: the entropic JKO step of F itself,
    ``argmin_rho F(rho) + W_reg(rho, rho_tau) / (2h)``, by the generalised
    Sinkhorn iteration on ``F.kl_prox_log`` (``wasserstein_ops.jko_entropic_prox``,
    log domain). ``n_inner`` and ``apply_w2_prox`` are ignored; ``reg`` is the
    Sinkhorn regularisation (default h^2, the step-scaled rule validated on
    2026-10-25; the legacy prox keeps its own default h), ``prox_n_iter`` the
    iteration count,
    ``prox_tol`` an optional early-stopping tolerance on the scaling vector.
    Species are updated from the same input state (Jacobi), as in the legacy
    scheme.

    ``nonseparable`` (entropic scheme only) sets how the terms of F without a
    closed-form prox (J_sym coupling, Turing term) are handled:

    * ``"explicit"`` (default, unchanged): linearised once at rho_tau. Not
      dissipative when the Turing term is strong (gamma >= 10, ERRATUM.md
      section 14).
    * ``"implicit"``: the linearisation point is iterated to the step's own
      output (damped Picard fixed point over all species jointly,
      ``fp_iter`` iterations, damping ``fp_damping``, tolerance ``fp_tol``
      on the max TV change), i.e. a backward-Euler treatment of those
      terms; then a safeguard: if F(output) > F(input) the step is split
      into two half-steps (each implicit, Sinkhorn reg scaled by
      (h_sub / h)^2 so reg / h^2 is kept), recursively up to
      ``max_halvings`` levels. ``self.last_halvings`` records the depth
      used by the last step.
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
        nonseparable: NonSeparable = "explicit",
        fp_iter: int = 30,
        fp_damping: float = 0.5,
        fp_tol: float = 1e-9,
        max_halvings: int = 8,
    ) -> None:
        if h <= 0:
            raise ValueError("h must be positive")
        if scheme is None:
            scheme = "jko_entropic" if apply_w2_prox else "legacy_split"
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
        if nonseparable not in NONSEPARABLE_MODES:
            raise ValueError(
                f"nonseparable must be one of {NONSEPARABLE_MODES}, got {nonseparable!r}"
            )
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
        self.nonseparable: NonSeparable = nonseparable
        self.fp_iter = fp_iter
        self.fp_damping = fp_damping
        self.fp_tol = fp_tol
        self.max_halvings = max_halvings
        self.last_halvings = 0
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

    def _entropic_reg(self, h: float) -> float:
        base = self.reg if self.reg is not None else self.h * self.h
        return base * (h / self.h) ** 2

    def _entropic_map(
        self, lin_state: FlowState, reference: dict[str, np.ndarray], h: float
    ) -> dict[str, np.ndarray]:
        """Entropic JKO step of F anchored on ``reference``, terms linearised at ``lin_state``."""
        return {
            name: jko_entropic_prox(
                self.f_functional.kl_prox_log(lin_state, name),
                reference=ref,
                epsilon=h,
                support=self.support,
                n_iter=self.prox_n_iter,
                reg=self._entropic_reg(h),
                tol=self.prox_tol,
            )
            for name, ref in reference.items()
        }

    def _implicit_step(self, state: FlowState, h: float, depth: int) -> FlowState:
        reference = {n: r for n, r in state.rho.items() if r.size == self.support.shape[0]}
        cur = state
        for _ in range(self.fp_iter):
            out = self._entropic_map(cur, reference, h)
            w = self.fp_damping
            mixed = {n: (1.0 - w) * cur.rho[n] + w * out[n] for n in reference}
            change = max(0.5 * float(np.abs(mixed[n] - cur.rho[n]).sum()) for n in reference)
            cur = state.model_copy(update={"rho": {**state.rho, **mixed}})
            if change < self.fp_tol:
                break
        f_in = self.f_functional.value(state)
        if self.f_functional.value(cur) <= f_in + DISSIPATION_TOL or depth >= self.max_halvings:
            self.last_halvings = max(self.last_halvings, depth)
            return cur
        mid = self._implicit_step(state, 0.5 * h, depth + 1)
        return self._implicit_step(mid, 0.5 * h, depth + 1)

    def _step_jko_entropic(self, state: FlowState) -> FlowState:
        if self.nonseparable == "implicit":
            self.last_halvings = 0
            out = self._implicit_step(state, self.h, 0)
            return state.model_copy(update={"rho": out.rho, "tau": state.tau + 1})
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
                reg=self.reg if self.reg is not None else self.h * self.h,
                tol=self.prox_tol,
            )
        return state.model_copy(update={"rho": new_rho, "tau": state.tau + 1})
