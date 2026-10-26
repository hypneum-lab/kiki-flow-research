"""Light tests for the preregistered strong-Turing driver.

Pins the preregistered grid, the new seeds (disjoint from seeds 0-9), the
slow-step configuration, the Holm step-down, the bootstrap p-value and the
primary statistic's pairing with the alpha = 0 control. No scientific value
is asserted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "experiments" / "turing_strong_excess_2026_10_26.py"
)


def _load() -> object:
    name = "exp_turing_strong_excess_2026_10_26"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_preregistered_grid_and_seeds() -> None:
    mod = _load()
    assert mod.GAMMAS == [20.0, 30.0, 50.0, 70.0, 100.0]
    assert mod.ALPHAS == [2.0, 5.0, 10.0]
    assert list(range(100, 110)) == mod.SEEDS
    assert not set(mod.SEEDS) & mod.FORBIDDEN_SEEDS
    assert len(mod.cells()) == 15  # noqa: PLR2004
    assert len(mod.couples()) == 20  # noqa: PLR2004
    assert len(mod.jobs()) == 200  # noqa: PLR2004
    assert pytest.approx(0.05**2) == mod.REG
    assert mod.FAMILY_LEVEL == 0.05  # noqa: PLR2004


def test_holm_step_down() -> None:
    mod = _load()
    adj, rej = mod.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert rej == [True, False, False, True]
    adj, rej = mod.holm([0.2, 0.9])
    assert rej == [False, False]


def test_bootstrap_p_value() -> None:
    mod = _load()
    rng = np.random.default_rng(0)
    pos = mod.bootstrap([0.3, 0.25, 0.35, 0.28, 0.31, 0.27, 0.33, 0.29, 0.3, 0.26], rng)
    assert pos["p_one_sided"] < 1e-4  # noqa: PLR2004
    assert pos["ci95_lo"] > 0
    null = mod.bootstrap([-0.1, 0.1, -0.05, 0.05, 0.0, -0.02, 0.02, 0.01, -0.01, 0.0], rng)
    assert null["p_one_sided"] > 0.1  # noqa: PLR2004


def test_primary_statistic_pairs_alpha0_control(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load()
    monkeypatch.setattr(mod, "N_BOOT", 1000)
    monkeypatch.setattr(
        mod.excess,
        "references",
        lambda cs: {
            c: {"gap_gibbs": 1.0, "gap_fmin": 1.0, "fmin_delta": 0.0, "fmin_iter": 1} for c in cs
        },
    )
    recs = []
    for s in mod.SEEDS:
        for a, g in mod.couples():
            recs.append({"alpha": a, "gamma": g, "seed": s, "gap_bits": 1.5 if a > 0 else 0.2})
    res = mod.analyse(recs)
    assert len(res["rows"]) == 15  # noqa: PLR2004
    for r in res["rows"]:
        assert r["primary"]["mean"] == pytest.approx(0.3)
        assert r["excess_no_control"]["mean"] == pytest.approx(0.5)
        assert r["verdict"] == "excess > 0"


def test_outputs_are_new_files_under_results() -> None:
    mod = _load()
    for path in mod.output_paths().values():
        assert path.parent == mod.OUT_DIR
        assert mod.OUT_DIR.parts[0] == "results"
