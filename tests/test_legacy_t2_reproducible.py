"""Historical T2 paths reproduce main (49480e0) after PR #51 (review v2, item 1).

The unsmoothed energy (``turing_smoothing=None``) keeps the floored entropy, so
the finite-difference gradient of the legacy split and the explicit entropic
step are unchanged. Fixture: ``tests/data/make_legacy_t2_fixture.py`` run on
main (20 legacy steps from sparse Dirichlet draws with entries below 1e-12).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
REPRO_TOL = 1e-12


def _fixture_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "legacy_fixture", DATA / "make_legacy_t2_fixture.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_historical_t2_paths_reproduce_main() -> None:
    expected = np.load(DATA / "legacy_t2_main_49480e0.npz")
    got = _fixture_module().trajectories()
    for key in ("legacy_split", "legacy_split_F", "explicit", "explicit_F"):
        np.testing.assert_allclose(got[key], expected[key], rtol=0.0, atol=REPRO_TOL, err_msg=key)
