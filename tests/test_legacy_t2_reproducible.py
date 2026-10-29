"""Historical T2 paths reproduce main (49480e0) after PR #51 (second review, item 1).

The unsmoothed energy (``turing_smoothing=None``) keeps the floored entropy, so
the finite-difference gradient of the legacy split and the explicit entropic
step are unchanged. The reference is main's ``paper_f.py``, stored verbatim in
``tests/data/paper_f_main_49480e0.py`` and run in the same process: the legacy
path is sensitive to the platform (a stored trajectory differs by up to 5e-3
between machines), so the comparison is made in one environment, bit for bit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy

MAIN_PAPER_F = Path(__file__).parent / "data" / "paper_f_main_49480e0.py"
GRID = 16
LEGACY_STEPS = 20
EXPLICIT_STEPS = 5
CONCENTRATION = 0.2  # sparse Dirichlet draws: entries far below 1e-12


def _main_energy_class() -> Any:
    spec = importlib.util.spec_from_file_location("paper_f_main_49480e0", MAIN_PAPER_F)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.T2FreeEnergy


def _energy(cls: Any, **kwargs: Any) -> Any:
    species = CanonicalSpecies()
    names = species.species_names()
    x = np.linspace(-2, 2, GRID)
    pots = dict(
        zip(names, [8 * (x + 1.5), 8 * (x - 1.5), 8 * np.abs(x), -8 * np.abs(x)], strict=True)
    )
    prior = {n: np.full(GRID, 1 / GRID) for n in names}
    return cls(species, pots, prior, 0.5, **kwargs)


def _state(names: list[str]) -> FlowState:
    rng = np.random.default_rng(0)
    rho = {n: rng.dirichlet(np.full(GRID, CONCENTRATION)) for n in names}
    return FlowState(
        rho=rho, P_theta=np.zeros(4), mu_curr=np.ones(1), tau=0, metadata={"track_id": "T2"}
    )


def _trajectories(f: Any) -> dict[str, np.ndarray]:
    names = f.species.species_names()
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    out: dict[str, np.ndarray] = {}
    for key, jko, steps in (
        ("legacy_split", JKOStep(f, 0.05, support, n_inner=10, apply_w2_prox=False), LEGACY_STEPS),
        (
            "explicit",
            JKOStep(f, 0.05, support, scheme="jko_entropic", nonseparable="explicit"),
            EXPLICIT_STEPS,
        ),
    ):
        s = _state(names)
        for _ in range(steps):
            s = jko.step(s)
        out[key] = np.concatenate([s.rho[n] for n in names])
        out[key + "_F"] = np.array([f.value(s)])
    return out


def test_historical_t2_paths_reproduce_main() -> None:
    expected = _trajectories(_energy(_main_energy_class()))
    got = _trajectories(_energy(T2FreeEnergy, turing_smoothing=None))
    assert float(expected["legacy_split"].min()) < 1e-12  # noqa: PLR2004  the floor is exercised
    for key, value in expected.items():
        np.testing.assert_array_equal(got[key], value, err_msg=key)
