"""Operator splitting of the entropic JKO step of the T2 energy (Lie and Strang).

Preregistration: ``docs/prereg/2026-10-30-jko-splitting-lie-strang.md``.

The smoothed T2 energy is split as F = F_A + F_B:

* F_A (separable): sum_i <rho_i, V_i> + KL(rho_i || prior_i), with the exact
  entropy (0 log 0 = 0) of the smoothed energy. Its KL-prox has a closed form,
  so its entropic JKO step is solved exactly (one certified generalised
  Sinkhorn solve per species; the certified loop stops at its first check).
* F_B (non-separable): the J_sym coupling sum_{i,k} J_sym[i,k] <rho_i, rho_k>
  plus the smoothed Turing term, with exact gradients. Its entropic JKO step is
  solved by the certified implicit solver of ``JKOStep`` (Bregman
  majorisation-minimisation, KKT certificate), applied to F_B alone.

A sub-step of length s is ``JKOStep(part, s, support, scheme="jko_entropic",
nonseparable="implicit")`` with its default Sinkhorn regularisation s^2 and
default tolerances. The schemes compose sub-steps (rightmost first):

* ``"lie"``:    A_h o B_h                 (B over h, then A over h);
* ``"strang"``: A_{h/2} o B_h o A_{h/2}   (A over h/2, B over h, A over h/2).

Each sub-step is a first-order (implicit-Euler-like) approximation of its
sub-flow, so the palindromic composition is not expected to be second order
in general; that is what the preregistration tests. Nothing here changes
``JKOStep`` or ``T2FreeEnergy``: this module only wraps them.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from kiki_flow_core.master_equation import (
    FreeEnergy,
    ImplicitJKOError,
    JKOStep,
    OnFailure,
)
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.paper_f import RHO_FLOOR, T2FreeEnergy, _kl
from kiki_flow_core.wasserstein_ops import KLProxLog

SplitScheme = Literal["lie", "strang"]
SPLIT_SCHEMES: tuple[SplitScheme, ...] = ("lie", "strang")
SPLIT_VERSION = "jko-splitting-1"
Part = Literal["A", "B"]


def split_sequence(scheme: SplitScheme, h: float) -> list[tuple[Part, float]]:
    """Sub-steps in the order they are applied: (part, sub-step length)."""
    if scheme == "lie":
        return [("B", h), ("A", h)]
    if scheme == "strang":
        return [("A", 0.5 * h), ("B", h), ("A", 0.5 * h)]
    raise ValueError(f"scheme must be one of {SPLIT_SCHEMES}, got {scheme!r}")


def compose[X](
    scheme: SplitScheme,
    flow_a: Callable[[X, float], X],
    flow_b: Callable[[X, float], X],
    h: float,
    x: X,
) -> X:
    """One splitting step of length h applied to x, for arbitrary sub-step maps.

    ``flow_a(x, s)`` and ``flow_b(x, s)`` advance x by a time s under each part.
    Generic (used by the order tests on linear flows and by ``SplitJKOStep``).
    """
    flows = {"A": flow_a, "B": flow_b}
    for part, s in split_sequence(scheme, h):
        x = flows[part](x, s)
    return x


def _check_smoothed(f: T2FreeEnergy) -> None:
    if f.turing_smoothing is None:
        raise ValueError(
            "the splitting needs the smoothed T2 energy (turing_smoothing set): "
            "its parts must have exact gradients and the exact entropy"
        )


class T2SeparablePart(FreeEnergy):
    """F_A = sum_i <rho_i, V_i> + KL(rho_i || prior_i) of a smoothed ``T2FreeEnergy``."""

    def __init__(self, f: T2FreeEnergy) -> None:
        _check_smoothed(f)
        self.f = f

    @property
    def implicit_ready(self) -> bool:
        return True

    def value(self, state: FlowState) -> float:
        total = 0.0
        for n in self.f.species.species_names():
            rho = state.rho[n]
            total += float(np.dot(rho, self.f.potentials[n]))
            total += _kl(rho, self.f.prior[n])
        return total

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        """Exact first variation V + log(rho / prior) + 1 (positive bins)."""
        rho = np.clip(np.asarray(state.rho[species_name], dtype=float), 1e-300, None)
        prior = np.clip(self.f.prior[species_name], RHO_FLOOR, None)
        out: np.ndarray = self.f.potentials[species_name] + np.log(rho / prior) + 1.0
        return out

    def kl_prox_log(self, state: FlowState, species_name: str) -> KLProxLog:
        """Exact closed form: log q = (log p + tau log prior - tau V) / (1 + tau)."""
        log_prior = np.log(np.clip(self.f.prior[species_name], RHO_FLOOR, None))
        potential = self.f.potentials[species_name]

        def prox(log_p: np.ndarray, tau: float) -> np.ndarray:
            out: np.ndarray = (log_p + tau * log_prior - tau * potential) / (1.0 + tau)
            return out

        return prox


class T2NonSeparablePart(FreeEnergy):
    """F_B = J_sym coupling + smoothed Turing term of a smoothed ``T2FreeEnergy``."""

    def __init__(self, f: T2FreeEnergy) -> None:
        _check_smoothed(f)
        self.f = f

    @property
    def implicit_ready(self) -> bool:
        return True

    def value(self, state: FlowState) -> float:
        rhos = [state.rho[n] for n in self.f.species.species_names()]
        j = self.f._J_sym
        total = 0.0
        for i, ri in enumerate(rhos):
            for k, rk in enumerate(rhos):
                total += float(j[i, k] * np.dot(ri, rk))
        if self.f.turing_strength > 0.0:
            total += self.f._turing_value(rhos)
        return total

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        """Exact gradient (``T2FreeEnergy.nonseparable_grad``)."""
        return self.f.nonseparable_grad(state, species_name)

    def kl_prox_log(self, state: FlowState, species_name: str) -> KLProxLog:
        """F_B linearised at ``state``: log q = log p - tau grad F_B."""
        grad = self.f.nonseparable_grad(state, species_name)

        def prox(log_p: np.ndarray, tau: float) -> np.ndarray:
            out: np.ndarray = log_p - tau * grad
            return out

        return prox


class SplitJKOStep:
    """One Lie or Strang splitting step of length h of the smoothed T2 energy.

    ``jko_kwargs`` are forwarded to every sub-step ``JKOStep`` (solver
    tolerances and budgets); ``scheme``, ``nonseparable``, ``reg`` and
    ``on_failure`` are fixed by this class (sub-step regularisation = its own
    length squared, the ``JKOStep`` default). A sub-step that is not certified
    ends the step: ``on_failure="raise"`` (default) raises ``ImplicitJKOError``
    with the diagnostics of every sub-step; ``"warn"`` returns the state reached
    before the failing sub-step, with ``last_info["status"]`` set to the failing
    status (diagnostics only, as for ``JKOStep``). ``last_info`` records the
    scheme, the status, the per-sub-step diagnostics and the wall time.
    """

    _FIXED = ("scheme", "nonseparable", "reg", "on_failure", "apply_w2_prox", "prox_fn")

    def __init__(
        self,
        f: T2FreeEnergy,
        h: float,
        support: np.ndarray,
        scheme: SplitScheme,
        on_failure: OnFailure = "raise",
        **jko_kwargs: Any,
    ) -> None:
        if not (h > 0 and np.isfinite(h)):
            raise ValueError("h must be finite and positive")
        if scheme not in SPLIT_SCHEMES:
            raise ValueError(f"scheme must be one of {SPLIT_SCHEMES}, got {scheme!r}")
        if on_failure not in ("raise", "warn"):
            raise ValueError("on_failure must be 'raise' or 'warn'")
        fixed = sorted(set(jko_kwargs) & set(self._FIXED))
        if fixed:
            raise ValueError(f"{fixed} are fixed by the splitting and cannot be passed")
        self.f = f
        self.h = h
        self.support = support
        self.scheme: SplitScheme = scheme
        self.on_failure: OnFailure = on_failure
        self.parts: dict[Part, FreeEnergy] = {
            "A": T2SeparablePart(f),
            "B": T2NonSeparablePart(f),
        }
        self.jko_kwargs = dict(jko_kwargs)
        self.last_info: dict[str, Any] = {}

    def substep(self, part: Part, s: float) -> JKOStep:
        """The certified entropic JKO step of one part over a time s."""
        return JKOStep(
            self.parts[part],
            s,
            self.support,
            scheme="jko_entropic",
            nonseparable="implicit",
            on_failure="raise",
            **self.jko_kwargs,
        )

    def step(self, state: FlowState) -> FlowState:
        tic = time.perf_counter()
        info: dict[str, Any] = {
            "split_version": SPLIT_VERSION,
            "scheme": self.scheme,
            "h": self.h,
            "status": "running",
            "substeps": [],
        }
        current = state
        for part, s in split_sequence(self.scheme, self.h):
            jko = self.substep(part, s)
            try:
                current = jko.step(current)
            except ImplicitJKOError as exc:
                info["substeps"].append({"part": part, "h_sub": s, **exc.info})
                info["status"] = str(exc.info.get("status", "failed"))
                info["failed_part"] = part
                break
            info["substeps"].append({"part": part, "h_sub": s, **jko.last_info})
        else:
            info["status"] = "converged"
        info["seconds"] = time.perf_counter() - tic
        self.last_info = info
        out = current.model_copy(update={"tau": state.tau + 1})
        if info["status"] != "converged":
            if self.on_failure == "raise":
                raise ImplicitJKOError("split JKO step not certified", info)
            warnings.warn(f"split JKO step not certified: {info['status']}", stacklevel=2)
        return out
