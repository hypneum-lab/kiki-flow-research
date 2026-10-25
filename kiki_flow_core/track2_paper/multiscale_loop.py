"""Fast/slow nested loop orchestration for Track 2 paper runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.particle_simulator import ParticleBatch, ParticleSimulator


def _zero_potential(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)


def resample_particles(
    particles: Any,
    state: FlowState,
    support: np.ndarray,
    species_names: list[str],
    rng: np.random.Generator,
) -> Any:
    """Reinject the JKO output into the particles.

    The 1-D projection (coordinate 0) of each species' particles is redrawn
    from ``state.rho[name]``: a bin of the histogram grid of
    ``_particles_to_flow_state`` is drawn with probability rho, then a uniform
    position inside it. Particle count, species tags and the other latent
    coordinates are kept, so ``_particles_to_flow_state`` of the result has
    ``rho`` as its expectation. Works for numpy and MLX batches (the positions
    are rebuilt with the batch's own array type).
    """
    n_bins = support.shape[0]
    edges = np.linspace(float(support[0, 0]), float(support[-1, 0]), n_bins + 1)
    old_positions = particles["positions"]
    positions = np.array(np.asarray(old_positions), dtype=float, copy=True)
    tags = np.asarray(particles["species_tags"])
    for name in species_names:
        mask = tags == name
        count = int(mask.sum())
        if count == 0:
            continue
        weights = np.clip(np.asarray(state.rho[name], dtype=float), 0.0, None)
        weights = weights / weights.sum()
        bins = rng.choice(n_bins, size=count, p=weights)
        positions[mask, 0] = edges[bins] + rng.random(count) * (edges[bins + 1] - edges[bins])
    new_positions = (
        positions if isinstance(old_positions, np.ndarray) else type(old_positions)(positions)
    )
    return {"positions": new_positions, "species_tags": particles["species_tags"]}


def _particles_to_flow_state(
    particles: ParticleBatch,
    support: np.ndarray,
    species_names: list[str],
) -> FlowState:
    """KDE-free histogramming: bin the 1D projection of positions per species.

    Accepts either numpy or MLX positions; converts to numpy for np.histogram.
    """
    n_bins = support.shape[0]
    lo = float(support[0, 0])
    hi = float(support[-1, 0])
    rho: dict[str, np.ndarray] = {}
    tags = np.asarray(particles["species_tags"])
    positions_np = np.asarray(particles["positions"])
    positions_1d = positions_np[:, 0]
    for name in species_names:
        mask = tags == name
        if not mask.any():
            rho[name] = np.full(n_bins, 1.0 / n_bins)
            continue
        hist, _ = np.histogram(positions_1d[mask], bins=n_bins, range=(lo, hi))
        total = hist.sum()
        rho[name] = (hist / total) if total > 0 else np.full(n_bins, 1.0 / n_bins)
    return FlowState(
        rho=rho,
        P_theta=np.zeros(4),
        mu_curr=np.full(n_bins, 1.0 / n_bins),
        tau=0,
        metadata={"track_id": "T2"},
    )


class MultiscaleLoop:
    """Nest N_fast Langevin substeps inside each slow JKO step.

    Accepts any simulator object with ``initialize``/``evolve``/``species``
    attributes, including the numpy ``ParticleSimulator`` and the MLX
    ``MLXParticleSimulator``. For MLX, ``potential_fn`` is left as None
    (zero drift) since the numpy ``_zero_potential`` would not type-match.

    ``reinject=False`` (default, unchanged behaviour): the JKO output is only
    recorded in the trajectory; the particles keep evolving on their own and
    the next slow step restarts from their histogram, so the recorded slow
    dynamics is not an iteration of the JKO step. ``reinject=True``: after
    each slow step (and the hook) the particles are redrawn from the JKO
    output (``resample_particles``, seeded by ``run(seed)``), so the slow
    steps chain and the trajectory is a JKO sequence driven by the fast
    Langevin layer.
    """

    def __init__(
        self,
        sim: ParticleSimulator | Any,
        jko: JKOStep,
        n_fast: int,
        n_slow: int,
        support: np.ndarray,
        dt_fast: float = 1e-3,
        post_jko_hook: Callable[[FlowState], FlowState] | None = None,
        reinject: bool = False,
    ) -> None:
        self.sim = sim
        self.jko = jko
        self.n_fast = n_fast
        self.n_slow = n_slow
        self.support = support
        self.dt_fast = dt_fast
        self.post_jko_hook = post_jko_hook
        self.reinject = reinject
        self._is_mlx = type(sim).__name__ == "MLXParticleSimulator"

    def run(self, seed: int) -> dict[str, Any]:
        particles = self.sim.initialize()
        names = self.sim.species.species_names()
        trajectory: list[FlowState] = []
        rng = np.random.default_rng(seed)
        for _ in range(self.n_slow):
            if self._is_mlx:
                mlx_sim: Any = self.sim
                particles = mlx_sim.evolve(
                    particles,
                    dt=self.dt_fast,
                    n_steps=self.n_fast,
                    potential_fn=None,
                )
            else:
                particles = self.sim.evolve(
                    particles,
                    dt=self.dt_fast,
                    n_steps=self.n_fast,
                    potential_fn=_zero_potential,
                )
            state = _particles_to_flow_state(particles, self.support, names)
            state = self.jko.step(state)
            if self.post_jko_hook is not None:
                state = self.post_jko_hook(state)
            trajectory.append(state)
            if self.reinject:
                particles = resample_particles(particles, state, self.support, names, rng)
        return {
            "seed": seed,
            "n_slow_completed": len(trajectory),
            "trajectory": trajectory,
            "reinject": self.reinject,
        }
