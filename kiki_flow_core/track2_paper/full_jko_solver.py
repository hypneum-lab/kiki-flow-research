"""T2 JKO solvers: POT-backed (default) and MLX-backed variants."""

from __future__ import annotations

from typing import Any

import numpy as np

from kiki_flow_core.master_equation import FreeEnergy, JKOStep, NonSeparable, Scheme
from kiki_flow_core.track2_paper.mlx_wasserstein import mlx_prox_w2

LEGACY_EPSILON = 0.01  # FullJKOSolver default before 2026-10-25 (legacy_split)


def resolve_epsilon(epsilon: float | None, h: float, scheme: Scheme) -> float:
    """Sinkhorn regularisation: explicit value, else h^2 (entropic) or 0.01 (legacy)."""
    if epsilon is not None:
        return epsilon
    return h * h if scheme == "jko_entropic" else LEGACY_EPSILON


class FullJKOSolver(JKOStep):
    """JKOStep with apply_w2_prox=True and tighter Sinkhorn params for paper rigor.

    ``h`` is the JKO time step (the proximal pull strength of ``prox_w2``);
    ``epsilon`` is the Sinkhorn entropic regularisation of the prox (``reg``).
    Before 2026-10-24 ``epsilon`` was stored but not passed on, so the prox
    used ``reg = h`` whatever ``epsilon`` was (see ERRATUM.md).
    ``max_iter`` is the number of Dykstra iterations of the prox; before
    2026-10-25 it was stored but not passed on, so the prox ran 100.
    ``scheme`` selects the step: ``"jko_entropic"`` (default since
    2026-10-25, ERRATUM.md section 13: entropic JKO step of F, see
    ``JKOStep``) or ``"legacy_split"`` (the historical "rigorous path":
    Euclidean descent then ``prox_w2``, not a JKO step of F; pass it
    explicitly to reproduce results produced before 2026-10-25).
    ``epsilon=None`` (default) resolves to ``h ** 2`` under ``jko_entropic``
    and to the historical ``0.01`` under ``legacy_split``.
    ``nonseparable`` ("implicit" by default, or "explicit" to reproduce
    earlier results) is forwarded to ``JKOStep`` (entropic scheme only,
    ERRATUM.md sections 14-16), and so are ``prox_tol`` (Sinkhorn early stop
    of the explicit path; 0 runs all ``max_iter`` iterations) and
    ``implicit_options`` (keyword settings of the certified implicit solver:
    ``kkt_tol``, ``max_outer``, ``l0``, ``l_max``, ``sigma``, ``inner_n_iter``,
    ``inner_tol``, ``inner_marginal_tol``, ``on_failure``).
    """

    def __init__(
        self,
        f_functional: FreeEnergy,
        h: float,
        support: np.ndarray,
        epsilon: float | None = None,
        max_iter: int = 1000,
        n_inner: int = 50,
        scheme: Scheme = "jko_entropic",
        nonseparable: NonSeparable = "implicit",
        prox_tol: float = 0.0,
        implicit_options: dict[str, Any] | None = None,
    ) -> None:
        epsilon = resolve_epsilon(epsilon, h, scheme)
        super().__init__(
            f_functional=f_functional,
            h=h,
            support=support,
            n_inner=n_inner,
            apply_w2_prox=True,
            reg=epsilon,
            prox_n_iter=max_iter,
            scheme=scheme,
            nonseparable=nonseparable,
            prox_tol=prox_tol,
            **(implicit_options or {}),
        )
        self.epsilon = epsilon
        self.max_iter = max_iter


class MLXFullJKOSolver(JKOStep):
    """JKOStep with Wasserstein prox routed through the native MLX Sinkhorn.

    Mirrors ``FullJKOSolver`` but uses ``mlx_prox_w2`` (a float32 Metal
    port of ``wasserstein_ops.prox_w2``) for the proximal step. It is not
    faster on small grids: on an M1 Max it is slower than NumPy up to
    n=256 and faster from n~1024 (``test_mlx_prox_w2.py``). MLX is a
    hard import; there is no silent fallback. With ``scheme="jko_entropic"``
    (default since 2026-10-25) the step runs the NumPy log-domain
    ``jko_entropic_prox`` (no MLX port yet); the MLX prox is used only by
    ``scheme="legacy_split"``.
    """

    def __init__(
        self,
        f_functional: FreeEnergy,
        h: float,
        support: np.ndarray,
        epsilon: float | None = None,
        max_iter: int = 1000,
        n_inner: int = 50,
        scheme: Scheme = "jko_entropic",
        nonseparable: NonSeparable = "implicit",
        prox_tol: float = 0.0,
        implicit_options: dict[str, Any] | None = None,
    ) -> None:
        epsilon = resolve_epsilon(epsilon, h, scheme)
        super().__init__(
            f_functional=f_functional,
            h=h,
            support=support,
            n_inner=n_inner,
            apply_w2_prox=True,
            prox_fn=mlx_prox_w2,
            reg=epsilon,
            prox_n_iter=max_iter,
            scheme=scheme,
            nonseparable=nonseparable,
            prox_tol=prox_tol,
            **(implicit_options or {}),
        )
        self.epsilon = epsilon
        self.max_iter = max_iter
