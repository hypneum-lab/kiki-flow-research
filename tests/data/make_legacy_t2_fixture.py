"""Generate the non-regression fixture of the historical (unsmoothed) T2 paths.

Run from the root of a checkout of the reference commit (49480e0, main before
PR #51) with ``PYTHONPATH=. python tests/data/make_legacy_t2_fixture.py OUT.npz``.
The same function is imported by ``tests/test_legacy_t2_reproducible.py`` and
run on the current code, which must reproduce the stored densities.
"""

from __future__ import annotations

import inspect
import sys

import numpy as np

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy

GRID = 16
LEGACY_STEPS = 20
EXPLICIT_STEPS = 5
CONCENTRATION = 0.2  # sparse Dirichlet draws: entries far below 1e-12


def historical_energy() -> T2FreeEnergy:
    species = CanonicalSpecies()
    names = species.species_names()
    x = np.linspace(-2, 2, GRID)
    pots = dict(
        zip(names, [8 * (x + 1.5), 8 * (x - 1.5), 8 * np.abs(x), -8 * np.abs(x)], strict=True)
    )
    prior = {n: np.full(GRID, 1 / GRID) for n in names}
    params = inspect.signature(T2FreeEnergy).parameters
    kw = {"turing_smoothing": None} if "turing_smoothing" in params else {}
    return T2FreeEnergy(species, pots, prior, 0.5, **kw)


def initial_state(names: list[str]) -> FlowState:
    rng = np.random.default_rng(0)
    rho = {n: rng.dirichlet(np.full(GRID, CONCENTRATION)) for n in names}
    return FlowState(
        rho=rho, P_theta=np.zeros(4), mu_curr=np.ones(1), tau=0, metadata={"track_id": "T2"}
    )


def trajectories() -> dict[str, np.ndarray]:
    f = historical_energy()
    names = f.species.species_names()
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    out: dict[str, np.ndarray] = {}
    legacy = JKOStep(f, 0.05, support, n_inner=10, apply_w2_prox=False)
    s = initial_state(names)
    for _ in range(LEGACY_STEPS):
        s = legacy.step(s)
    out["legacy_split"] = np.concatenate([s.rho[n] for n in names])
    out["legacy_split_F"] = np.array([f.value(s)])
    explicit = JKOStep(f, 0.05, support, scheme="jko_entropic", nonseparable="explicit")
    s = initial_state(names)
    for _ in range(EXPLICIT_STEPS):
        s = explicit.step(s)
    out["explicit"] = np.concatenate([s.rho[n] for n in names])
    out["explicit_F"] = np.array([f.value(s)])
    return out


if __name__ == "__main__":
    np.savez(sys.argv[1], **trajectories())
