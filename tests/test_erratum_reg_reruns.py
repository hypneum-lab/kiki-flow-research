"""Runner tests for ``experiments/erratum_reg_reruns_2026_10_24.py``.

The reruns use the fixed ``FullJKOSolver`` (epsilon forwarded as the Sinkhorn
``reg``). Checks the solver configuration, the decoupled epsilon sweep, the
grid reuse and that no historical output path is reused. No scientific
values are asserted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "experiments" / "erratum_reg_reruns_2026_10_24.py"
)
RIG_EPS = 0.01
H = 0.05
N_WIDE_POINTS = 138


def _load() -> object:
    name = "exp_erratum_reg"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_specialisation_solver_uses_configured_reg() -> None:
    mod = _load()
    _, support, f_func = mod.wide.base.problem(alpha=10.0, gamma=0.0)
    jko = mod.wide.base.make_jko(f_func, support, beta=1.0, rigorous=True)
    assert jko.reg == RIG_EPS
    assert jko.h == H
    assert len(mod.wide.points()) == N_WIDE_POINTS


def test_eps_sweep_varies_reg_only() -> None:
    mod = _load()
    outs = mod.eps_outputs([0.01, 0.1])
    assert len(outs) == 2  # noqa: PLR2004
    assert all(abs(o.sum() - 1.0) < 1e-9 for o in outs)  # noqa: PLR2004
    assert np.abs(outs[0] - outs[1]).sum() > 0


def test_fast_records_are_copied_not_rerun() -> None:
    mod = _load()
    recs = [
        {"path": "fast", "seed": 0, "alpha": 1.0, "beta": 0.0, "gamma": 0.0},
        {"path": "rigorous", "seed": 0, "alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    ]
    kept = mod.fast_records(recs)
    assert [r["path"] for r in kept] == ["fast"]


def test_outputs_are_new() -> None:
    mod = _load()
    new = {p.resolve() for p in mod.output_paths().values()}
    old = {
        Path("paper_rigorous_fixedprox").resolve(),
        Path("paper/erratum_2026_10_24").resolve(),
        mod.wide.OUT_DIR.resolve(),
        mod.wide.base.OUT_DIR.resolve(),
    }
    assert not any(any(o == p or o in p.parents for o in old) for p in new)
