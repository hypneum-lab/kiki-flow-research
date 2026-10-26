"""Light tests for ``experiments/excess_over_gibbs_2026_10_25.py``.

Pins the Gibbs reference (rho_i ∝ prior_i exp(-V_i), zero gap at alpha = 0,
growing with alpha), the per-seed bootstrap and its verdicts, and the output
paths. No scientific value of the sweep is asserted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "experiments" / "excess_over_gibbs_2026_10_25.py"
GAP_TOL = 1e-12


def _load() -> object:
    name = "exp_excess_over_gibbs_2026_10_25"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_gibbs_reference_is_boltzmann_of_the_potentials() -> None:
    mod = _load()
    assert mod.gap_of(mod.gibbs_potentials(mod.problem(0.0, 0.0))) == pytest.approx(
        0.0, abs=GAP_TOL
    )
    gaps = [mod.gap_of(mod.gibbs_potentials(mod.problem(a, 0.0))) for a in (1.0, 5.0, 30.0)]
    assert gaps == sorted(gaps)
    assert gaps[-1] < mod.MAX_ENTROPY_BITS
    f = mod.problem(3.0, 0.0)
    rho = mod.gibbs_potentials(f)["phono"]
    expected = np.exp(-f.potentials["phono"])
    np.testing.assert_allclose(rho, expected / expected.sum())


def test_bootstrap_and_verdicts() -> None:
    mod = _load()
    rng = np.random.default_rng(0)
    pos = mod.bootstrap_ci([0.5, 0.6, 0.55, 0.52, 0.58], rng)
    assert pos["lo"] > 0
    assert mod.verdict(pos) == "excess > 0"
    assert mod.verdict(mod.bootstrap_ci([-0.5, -0.6, -0.55], rng)) == "deficit (< 0)"
    assert mod.verdict(mod.bootstrap_ci([-0.5, 0.6, 0.0, -0.2, 0.3], rng)) == "CI includes 0"
    assert mod.verdict(mod.bootstrap_ci([], rng)) == "n/a"


def test_analyse_pairs_controls_by_seed() -> None:
    mod = _load()
    refs = {
        (0.0, 0.0): {"gap_gibbs": 0.0, "gap_fmin": 0.0},
        (5.0, 0.0): {"gap_gibbs": 2.0, "gap_fmin": 1.9},
    }
    recs = [
        {"alpha": a, "gamma": 0.0, "seed": s, "gap_bits": g}
        for s in range(3)
        for a, g in ((0.0, 0.2), (5.0, 2.5))
    ]
    rows = {r["alpha"]: r for r in mod.analyse(recs, refs, 0)}
    assert rows[5.0]["excess"]["mean"] == pytest.approx(0.5)
    assert rows[5.0]["excess_fmin"]["mean"] == pytest.approx(0.6)
    assert rows[5.0]["excess_ctrl"]["mean"] == pytest.approx(0.3)


def test_outputs_are_new_files_under_results() -> None:
    mod = _load()
    for path in mod.output_paths().values():
        assert path.parent == mod.OUT_DIR
        assert mod.OUT_DIR.parts[0] == "results"
