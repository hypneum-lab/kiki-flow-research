"""Orthophonic correction loop: injects bounded source term into the phonological species."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class PhonologicalLoop:
    """Inject a corrective source term into the phonological species based on output errors.

    The detector callback maps a model output to an error vector (same shape as output).
    The source term is the tanh-compressed error scaled by correction_strength, clipped
    to +/- correction_strength to prevent runaway positive feedback.
    """

    def __init__(
        self,
        detector: Callable[[np.ndarray], np.ndarray],
        correction_strength: float,
    ) -> None:
        if correction_strength < 0:
            raise ValueError("correction_strength must be non-negative")
        self.detector = detector
        self.correction_strength = correction_strength

    def source_term(self, rho_phono: np.ndarray, output: np.ndarray) -> np.ndarray:
        """Compute source S(x, t) for the rho_phono equation."""
        errors = self.detector(output)
        if errors.shape != output.shape:
            raise ValueError(f"detector returned shape {errors.shape}, expected {output.shape}")
        s = np.tanh(errors) * self.correction_strength
        clipped: np.ndarray = np.clip(s, -self.correction_strength, self.correction_strength)
        # Reference rho_phono to signal intent (future versions will modulate by current density).
        _ = rho_phono
        return clipped
