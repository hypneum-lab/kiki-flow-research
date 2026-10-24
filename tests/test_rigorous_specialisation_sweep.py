"""Runner tests for ``experiments/rigorous_specialisation_2026_10_24.py``.

Checks that the rigorous sweep uses the same grid and beta -> n_inner map
as ``scripts/hyperparam_sweep_dense.py``, switches on the W2 prox, and never
writes to the historical fast-path file. No scientific values are asserted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "experiments" / "rigorous_specialisation_2026_10_24.py"
_FAST_SCRIPT = _REPO_ROOT / "scripts" / "hyperparam_sweep_dense.py"
N_CONFIGS = 72
HIGH_BETA = 2.0
HIGH_BETA_N_INNER = 20


def _load(path: Path, name: str) -> object:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_grid_matches_fast_sweep() -> None:
    mod = _load(_SCRIPT, "exp_rigorous_spec")
    fast = _load(_FAST_SCRIPT, "scripts_hyperparam_sweep_dense")
    assert mod.ALPHAS == fast.ALPHAS
    assert mod.BETAS == fast.BETAS
    assert mod.GAMMAS == fast.GAMMAS
    assert len(mod.configs()) == N_CONFIGS
    assert mod.SEEDS[0] == fast.SEED


def test_make_jko_rigorous_and_fast() -> None:
    mod = _load(_SCRIPT, "exp_rigorous_spec")
    _, support, f_func = mod.problem(alpha=10.0, gamma=0.0)
    rig = mod.make_jko(f_func, support, beta=HIGH_BETA, rigorous=True)
    fast = mod.make_jko(f_func, support, beta=HIGH_BETA, rigorous=False)
    assert isinstance(rig, FullJKOSolver)
    assert rig.apply_w2_prox
    assert not fast.apply_w2_prox
    assert rig.n_inner == fast.n_inner == HIGH_BETA_N_INNER
    assert mod.make_jko(f_func, support, beta=0.0, rigorous=True).n_inner == 1


def test_outputs_never_overwrite_fast_results() -> None:
    mod = _load(_SCRIPT, "exp_rigorous_spec")
    historical = Path("paper/hyperparam_sweep_dense.json")
    assert historical not in mod.output_paths().values()
    assert all(p.parent == mod.OUT_DIR for p in mod.output_paths().values())


def test_run_one_returns_bits_per_species() -> None:
    mod = _load(_SCRIPT, "exp_rigorous_spec")
    ent = mod.run_one(alpha=10.0, beta=0.5, gamma=0.0, rigorous=True, seed=0, n_slow=2)
    assert set(ent) == {"phono", "lex", "syntax", "sem"}
    assert all(0.0 <= v <= np.log2(mod.GRID) + 1e-9 for v in ent.values())
