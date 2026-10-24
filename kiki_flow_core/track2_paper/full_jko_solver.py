"""T2 JKO solvers: POT-backed (default) and MLX-backed variants."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.master_equation import FreeEnergy, JKOStep
from kiki_flow_core.track2_paper.mlx_wasserstein import mlx_prox_w2


class FullJKOSolver(JKOStep):
    """JKOStep with apply_w2_prox=True and tighter Sinkhorn params for paper rigor.

    ``h`` is the JKO time step (the proximal pull strength of ``prox_w2``);
    ``epsilon`` is the Sinkhorn entropic regularisation of the prox (``reg``).
    Before 2026-10-24 ``epsilon`` was stored but not passed on, so the prox
    used ``reg = h`` whatever ``epsilon`` was (see ERRATUM.md).
    """

    def __init__(
        self,
        f_functional: FreeEnergy,
        h: float,
        support: np.ndarray,
        epsilon: float = 0.01,
        max_iter: int = 1000,
        n_inner: int = 50,
    ) -> None:
        super().__init__(
            f_functional=f_functional,
            h=h,
            support=support,
            n_inner=n_inner,
            apply_w2_prox=True,
            reg=epsilon,
        )
        self.epsilon = epsilon
        self.max_iter = max_iter


class MLXFullJKOSolver(JKOStep):
    """JKOStep with Wasserstein prox routed through the native MLX Sinkhorn.

    Mirrors ``FullJKOSolver`` but uses ``mlx_prox_w2`` (a float32 Metal
    port of ``wasserstein_ops.prox_w2``) for the proximal step. It is not
    faster on small grids: on an M1 Max it is slower than NumPy up to
    n=256 and faster from n~1024 (``test_mlx_prox_w2.py``). MLX is a
    hard import; there is no silent fallback.
    """

    def __init__(
        self,
        f_functional: FreeEnergy,
        h: float,
        support: np.ndarray,
        epsilon: float = 0.01,
        max_iter: int = 1000,
        n_inner: int = 50,
    ) -> None:
        super().__init__(
            f_functional=f_functional,
            h=h,
            support=support,
            n_inner=n_inner,
            apply_w2_prox=True,
            prox_fn=mlx_prox_w2,
            reg=epsilon,
        )
        self.epsilon = epsilon
        self.max_iter = max_iter
