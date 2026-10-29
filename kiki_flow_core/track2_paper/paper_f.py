"""T2 free energy: potential + KL prior + Levelt-Baddeley reaction + Turing cross-diffusion."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.master_equation import FreeEnergy
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.wasserstein_ops import KLProxLog

RHO_FLOOR = 1e-12
TURING_FD_EPS = 1e-4


def gradient_operator(n: int) -> np.ndarray:
    """Matrix D with D @ v == np.gradient(v) (unit spacing, one-sided ends)."""
    return np.stack([np.gradient(e) for e in np.eye(n)], axis=1)


def _kl_historical(rho: np.ndarray, prior: np.ndarray) -> float:
    """Historical entropy: sum rho log(rho / prior) with rho and prior floored at 1e-12.

    Kept bit for bit for the historical energy (``turing_smoothing=None``): its
    value feeds the finite-difference gradient of the legacy, fast and explicit
    paths, so changing it would change committed trajectories.
    """
    rho_safe = np.clip(rho, RHO_FLOOR, None)
    prior_safe = np.clip(prior, RHO_FLOOR, None)
    return float(np.sum(rho_safe * np.log(rho_safe / prior_safe)))


def _kl(rho: np.ndarray, prior: np.ndarray) -> float:
    """sum rho log(rho / prior) with 0 log 0 = 0 (no floor on positive masses)."""
    r = np.maximum(np.asarray(rho, dtype=float), 0.0)
    positive = r > 0.0
    prior_safe = np.clip(prior, RHO_FLOOR, None)
    terms = np.where(positive, r * np.log(np.where(positive, r, 1.0) / prior_safe), 0.0)
    return float(np.sum(terms))


class T2FreeEnergy(FreeEnergy):
    """F_T2 = sum_i <rho_i, V_i> + sum_i KL(rho_i || prior_i) + reaction + turing.

    The Turing term is turing_strength * sum_{i<k} sum_x |g_i(x) g_k(x)| with
    g = np.gradient(rho). It is not smooth. ``turing_smoothing = delta`` (default
    1e-5, PI decision of 2026-10-28, ERRATUM.md section 16) replaces |u| by
    sqrt(u^2 + delta^2) - delta in the value and uses the exact gradient of that
    smoothed term. The bound 0 <= T - T_delta <= n_pairs * n * turing_strength *
    delta (n_pairs = 6 species pairs, n grid points) holds for the Turing term
    T alone: the two energies also differ in their entropy (floored at 1e-12
    for the historical energy, exact for the smoothed one), which adds a small
    term of either sign (about -3e-10 for densities with zero bins on a 4-point
    grid). ``None`` keeps the historical
    energy: exact |.| in the value and a forward-difference gradient, which is
    not a derivative of F across sign changes; drivers behind committed numbers
    pin it, and it also keeps the historical entropy floored at 1e-12, bit for
    bit. The smoothed energy uses the exact entropy convention 0 log 0 = 0 (no
    floor), the function the KL-prox minimises. The certified implicit JKO step
    requires the smoothed energy (``implicit_ready``).
    """

    def __init__(
        self,
        species: CanonicalSpecies,
        potentials: dict[str, np.ndarray],
        prior: dict[str, np.ndarray],
        turing_strength: float = 0.1,
        turing_smoothing: float | None = 1e-5,
    ) -> None:
        if turing_smoothing is not None and not (turing_smoothing > 0.0):
            raise ValueError("turing_smoothing must be positive or None")
        self.species = species
        self.potentials = potentials
        self.prior = prior
        self.turing_strength = turing_strength
        self.turing_smoothing = turing_smoothing
        self._coupling = species.coupling_matrix()
        # Canonical J split: J = J_sym + J_asym.
        # J_sym carries the conservative (gradient-of-scalar-energy) piece;
        # J_asym carries the non-conservative drift. The scalar energy
        # sum_{i,k} J[i,k] <rho_i, rho_k> is unchanged when routed through
        # J_sym because sum_{i,k} J_asym[i,k] <rho_i, rho_k> = 0 identically
        # (antisymmetric tensor contracted with symmetric <rho_i, rho_k>).
        self._J_sym = 0.5 * (self._coupling + self._coupling.T)
        self._J_asym = 0.5 * (self._coupling - self._coupling.T)

    def value(self, state: FlowState) -> float:
        total = 0.0
        names = self.species.species_names()
        rhos = [state.rho[n] for n in names]

        kl = _kl_historical if self.turing_smoothing is None else _kl
        for n, rho in zip(names, rhos, strict=True):
            total += float(np.dot(rho, self.potentials[n]))
            total += kl(rho, self.prior[n])

        # Route the coupling energy through J_sym: mathematically equivalent
        # to using the full J (antisymmetric contribution vanishes), but makes
        # the derivation of the conservative part unambiguous.
        j = self._J_sym
        for i, ri in enumerate(rhos):
            for k, rk in enumerate(rhos):
                total += float(j[i, k] * np.dot(ri, rk))

        if self.turing_strength > 0.0:
            total += self._turing_value(rhos)

        return total

    @property
    def implicit_ready(self) -> bool:
        """True for the smoothed energy only: exact gradients and exact entropy.

        The historical energy (``turing_smoothing=None``) keeps the floored
        entropy, which the closed-form prox does not minimise, so it is refused
        by the certified step even when ``turing_strength == 0``.
        """
        return self.turing_smoothing is not None

    def _abs(self, u: np.ndarray) -> np.ndarray:
        if self.turing_smoothing is None:
            absolute: np.ndarray = np.abs(u)
            return absolute
        d = self.turing_smoothing
        out: np.ndarray = np.sqrt(u * u + d * d) - d
        return out

    def _turing_value(self, rhos: list[np.ndarray]) -> float:
        turing = 0.0
        for i, ri in enumerate(rhos):
            for k in range(i + 1, len(rhos)):
                turing += float(np.sum(self._abs(np.gradient(ri) * np.gradient(rhos[k]))))
        return self.turing_strength * turing

    def _turing_grad_smoothed(self, rhos: list[np.ndarray], idx: int) -> np.ndarray:
        """Exact gradient of the smoothed Turing term w.r.t. rho_idx."""
        assert self.turing_smoothing is not None
        d_op = gradient_operator(rhos[idx].size)
        gs = [d_op @ r for r in rhos]
        acc = np.zeros(rhos[idx].size)
        for k, gk in enumerate(gs):
            if k == idx:
                continue
            u = gs[idx] * gk
            acc += u * gk / np.sqrt(u * u + self.turing_smoothing**2)
        out: np.ndarray = self.turing_strength * (d_op.T @ acc)
        return out

    def nonseparable_grad(self, state: FlowState, species_name: str) -> np.ndarray:
        """Gradient w.r.t. rho[species_name] of the terms of F without a closed-form prox.

        Coupling: d/d rho_i of sum_{i,k} J_sym[i,k] <rho_i, rho_k> = 2 (J_sym rho)_i.
        Turing term: exact gradient of the smoothed term when ``turing_smoothing``
        is set; otherwise the historical forward difference (not a derivative of
        F across sign changes); zero if its strength is 0.
        Evaluated at ``state`` (the other species frozen there); used by
        ``kl_prox_log`` to linearise these terms into an effective potential.
        """
        names = self.species.species_names()
        idx = names.index(species_name)
        rhos = [state.rho[n] for n in names]
        grad: np.ndarray = 2.0 * np.asarray(self._grad_conservative(rhos)[idx], dtype=float)
        if self.turing_strength > 0.0 and self.turing_smoothing is not None:
            grad = grad + self._turing_grad_smoothed(rhos, idx)
        elif self.turing_strength > 0.0:
            base = self._turing_value(rhos)
            rho = rhos[idx]
            fd = np.zeros_like(rho)
            for i in range(rho.size):
                perturbed = rho.copy()
                perturbed[i] += TURING_FD_EPS
                fd[i] = (self._turing_value(rhos[:idx] + [perturbed] + rhos[idx + 1 :]) - base) / (
                    TURING_FD_EPS
                )
            grad = grad + fd
        return np.asarray(grad, dtype=float)

    def kl_prox_log(self, state: FlowState, species_name: str) -> KLProxLog:
        """Closed-form KL-prox of <V, q> + KL(q || prior), couplings linearised.

        argmin_q KL(q || p) + tau [<V_eff, q> + KL(q || prior)] has
        log q = (log p + tau log prior - tau V_eff) / (1 + tau) up to a constant,
        with V_eff = V + ``nonseparable_grad`` at ``state`` (J_sym coupling and
        Turing term frozen at rho_tau: semi-implicit, first order in h like the
        JKO step itself). The entropic part is treated exactly (implicit), so
        the diffusion carries no stability constraint on h.
        """
        log_prior = np.log(np.clip(self.prior[species_name], RHO_FLOOR, None))
        v_eff = self.potentials[species_name] + self.nonseparable_grad(state, species_name)

        def prox(log_p: np.ndarray, tau: float) -> np.ndarray:
            out: np.ndarray = (log_p + tau * log_prior - tau * v_eff) / (1.0 + tau)
            return out

        return prox

    def _grad_conservative(self, rhos: list[np.ndarray]) -> list[np.ndarray]:
        """Per-species conservative coupling drive: (J_sym @ rho)_i.

        This is the J-coupling piece of delta F / delta rho_i coming from
        the symmetric part of J; it is the gradient of the scalar energy
        sum_{i,k} J_sym[i,k] <rho_i, rho_k> when one identifies rhos as
        living on a flat (non-simplex) inner-product space. V_i, KL, and
        Turing contributions are NOT included here: this helper isolates
        the J_sym contribution only so the split invariant
        J rho = J_sym rho + J_asym rho can be checked directly.
        """
        j_sym = self._J_sym
        out: list[np.ndarray] = []
        for i in range(len(rhos)):
            acc = np.zeros_like(rhos[i])
            for k, rk in enumerate(rhos):
                acc = acc + j_sym[i, k] * rk
            out.append(acc)
        return out

    def _drift_nonconservative(self, rhos: list[np.ndarray]) -> list[np.ndarray]:
        """Per-species non-conservative drift: (J_asym @ rho)_i.

        Isolated antisymmetric drift field driven by the asymmetric part
        of the Levelt-Baddeley coupling. Contracted with the symmetric
        inner product <rho_i, rho_k> it integrates to zero against the
        scalar energy, so it cannot arise as the gradient of any scalar
        functional of the rhos — it is a pure non-conservative forcing.
        """
        j_asym = self._J_asym
        out: list[np.ndarray] = []
        for i in range(len(rhos)):
            acc = np.zeros_like(rhos[i])
            for k, rk in enumerate(rhos):
                acc = acc + j_asym[i, k] * rk
            out.append(acc)
        return out

    def coupling_drive(self, rhos: list[np.ndarray]) -> list[np.ndarray]:
        """Legacy full-J coupling drive: (J @ rho)_i per species.

        Backwards-compatible sum of the conservative and non-conservative
        components. By construction equals
        ``[self._coupling[i, k] * rhos[k] for i, k]`` accumulated, which is
        what the pre-split code used as the J-coupling variational drive.
        """
        cons = self._grad_conservative(rhos)
        drift = self._drift_nonconservative(rhos)
        return [c + d for c, d in zip(cons, drift, strict=True)]

    def apply_drift_splitting(self, state: FlowState, h_drift: float = 0.01) -> FlowState:
        """Operator-splitting step for the antisymmetric linear drift b_asym = J_asym @ rho.

        Applies M = exp(h_drift * J_asym) per grid cell to the 4-vector of
        per-species densities. Since J_asym is antisymmetric, M is orthogonal,
        so the total activity sum_i rho_i(x) is preserved at every grid cell x.
        Per-species total mass can drift slightly under the rotation (individual
        species are not preserved axis-wise); we rescale each species back to its
        prior total mass so the simplex constraint (non-negative, sums to 1) still
        holds for every rho_i after the step.

        A small Taylor expansion is used for the matrix exponential so the repo
        does not take a direct dependency on scipy. For ``h_drift`` of order 0.01
        and ``||h_drift * J_asym||_2`` of order 3e-3 the 4-term truncation error
        is below 1e-10 in operator norm.
        """
        names = self.species.species_names()
        # Shape (n_species, grid).
        old_rho = np.stack([np.asarray(state.rho[n]) for n in names], axis=0)
        old_mass = old_rho.sum(axis=1)
        # Matrix exponential via Taylor series: accurate to ~(h*||J||)^4 / 24.
        hj = h_drift * self._J_asym
        eye = np.eye(hj.shape[0])
        hj2 = hj @ hj
        hj3 = hj2 @ hj
        hj4 = hj3 @ hj
        m = eye + hj + hj2 / 2.0 + hj3 / 6.0 + hj4 / 24.0
        new_rho_raw = m @ old_rho
        # Clip tiny numerical negatives that can arise near simplex boundary.
        new_rho_raw = np.maximum(new_rho_raw, 0.0)
        new_mass = new_rho_raw.sum(axis=1)
        scale = old_mass / np.maximum(new_mass, 1e-12)
        new_rho = new_rho_raw * scale[:, None]
        new_rho_dict = {names[i]: new_rho[i] for i in range(len(names))}
        return state.model_copy(update={"rho": new_rho_dict})
