"""``MultiscaleLoop(reinject=True)``: the JKO output is fed back into the particles."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver
from kiki_flow_core.track2_paper.multiscale_loop import (
    MultiscaleLoop,
    _particles_to_flow_state,
    resample_particles,
)
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy
from kiki_flow_core.track2_paper.particle_simulator import ParticleSimulator

GRID = 8
N_PARTICLES = 4000
HISTOGRAM_TOL = 0.05
N_SLOW = 3


def _problem() -> tuple[CanonicalSpecies, np.ndarray, T2FreeEnergy]:
    species = CanonicalSpecies()
    names = species.species_names()
    potentials = {n: np.zeros(GRID) for n in names}
    prior = {n: np.full(GRID, 1.0 / GRID) for n in names}
    f = T2FreeEnergy(species=species, potentials=potentials, prior=prior, turing_strength=0.0)
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    return species, support, f


def test_resample_particles_matches_target_histogram_and_keeps_tags() -> None:
    species, support, _ = _problem()
    names = species.species_names()
    sim = ParticleSimulator(species=species, n_particles=N_PARTICLES, latent_dim=2, seed=0)
    particles = sim.initialize()
    rng = np.random.default_rng(0)
    target = {n: rng.dirichlet(np.full(GRID, 2.0)) for n in names}
    state = FlowState(
        rho=target, P_theta=np.zeros(4), mu_curr=np.array([1.0]), tau=0, metadata={"track_id": "T2"}
    )
    out = resample_particles(particles, state, support, names, np.random.default_rng(1))
    assert out["species_tags"] is particles["species_tags"]
    assert out["positions"].shape == particles["positions"].shape
    # Coordinate 1 untouched, coordinate 0 redrawn inside the support.
    np.testing.assert_array_equal(out["positions"][:, 1], particles["positions"][:, 1])
    assert out["positions"][:, 0].min() >= support[0, 0]
    assert out["positions"][:, 0].max() <= support[-1, 0]
    hist = _particles_to_flow_state(out, support, names)
    for n in names:
        assert 0.5 * np.abs(hist.rho[n] - target[n]).sum() < HISTOGRAM_TOL
    # Seeded: same rng seed, same positions.
    again = resample_particles(particles, state, support, names, np.random.default_rng(1))
    np.testing.assert_array_equal(again["positions"], out["positions"])


def test_loop_reinject_flag_changes_dynamics_and_default_is_on() -> None:
    species, support, f = _problem()
    jko = FullJKOSolver(
        f_functional=f,
        h=0.05,
        support=support,
        epsilon=0.05,
        max_iter=50,
        n_inner=5,
        scheme="legacy_split",
    )
    runs = {}
    for reinject in (False, True):
        sim = ParticleSimulator(species=species, n_particles=500, latent_dim=2, seed=0)
        loop = MultiscaleLoop(
            sim=sim, jko=jko, n_fast=5, n_slow=N_SLOW, support=support, reinject=reinject
        )
        runs[reinject] = loop.run(seed=0)
        assert runs[reinject]["reinject"] is reinject
        assert runs[reinject]["n_slow_completed"] == N_SLOW
    default_loop = MultiscaleLoop(
        sim=ParticleSimulator(species=species, n_particles=500, latent_dim=2, seed=0),
        jko=jko,
        n_fast=5,
        n_slow=N_SLOW,
        support=support,
    )
    assert default_loop.reinject is True  # default since 2026-10-25
    # First slow step identical (reinjection acts after it), later ones differ.
    first_off = runs[False]["trajectory"][0].rho
    first_on = runs[True]["trajectory"][0].rho
    for n in species.species_names():
        np.testing.assert_array_equal(first_off[n], first_on[n])
    last_off = runs[False]["trajectory"][-1].rho
    last_on = runs[True]["trajectory"][-1].rho
    assert any(np.abs(last_off[n] - last_on[n]).sum() > 0.0 for n in species.species_names())
