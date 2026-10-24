"""Runner tests for ``experiments/rigorous_specialisation_wide_2026_10_24.py``.

Checks the widened grid (original 72 points kept as a subset, no two beta
values with the same ``n_inner``, an alpha = 0 control for every point),
the reuse of the seed 0-4 records of the first sweep, the runtime estimate
and the output paths. No scientific values are asserted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "experiments" / "rigorous_specialisation_wide_2026_10_24.py"
N_CORE = 72
N_SEEDS = 10
BUDGET_S = 3 * 3600


def _load() -> object:
    name = "exp_rigorous_spec_wide"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_core_grid_is_a_subset() -> None:
    mod = _load()
    pts = set(mod.points())
    core = set(mod.base.configs())
    assert len(core) == N_CORE
    assert core <= pts
    assert len(mod.SEEDS) == N_SEEDS


def test_beta_values_have_distinct_n_inner() -> None:
    mod = _load()
    betas = sorted({b for _, b, _ in mod.points()})
    n_inner = [mod.n_inner(b) for b in betas]
    assert len(set(n_inner)) == len(n_inner)


def test_every_point_has_an_alpha0_control() -> None:
    mod = _load()
    pts = set(mod.points())
    assert all((0.0, b, g) in pts for _, b, g in pts)


def test_reuse_takes_only_core_seeds_0_to_4(tmp_path: Path) -> None:
    mod = _load()
    recs = [
        {"path": "fast", "seed": 0, "alpha": 10.0, "beta": 1.0, "gamma": 0.0},
        {"path": "rigorous", "seed": 4, "alpha": 0.0, "beta": 0.0, "gamma": 5.0},
    ]
    reused = mod.reuse_records(recs)
    assert {r["source"] for r in reused} == {mod.REUSE_SOURCE}
    assert len(reused) == len(recs)
    keys = {mod.key(r["path"], r["seed"], (r["alpha"], r["beta"], r["gamma"])) for r in reused}
    todo = mod.todo(done=keys)
    assert ("fast", 0, (10.0, 1.0, 0.0)) not in todo
    assert ("fast", 5, (10.0, 1.0, 0.0)) in todo


def test_estimate_fits_budget() -> None:
    mod = _load()
    est = mod.estimate_seconds(mod.todo(done=mod.reused_keys()))
    assert 0 < est < BUDGET_S


def test_outputs_are_new_files() -> None:
    mod = _load()
    first = set(mod.base.output_paths().values())
    wide = set(mod.output_paths().values())
    assert not first & wide
    assert Path("paper/hyperparam_sweep_dense.json") not in wide
