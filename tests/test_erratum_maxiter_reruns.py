"""Runner tests for ``experiments/erratum_maxiter_reruns_2026_10_25.py``.

The reruns use the fixed ``FullJKOSolver`` (``max_iter`` forwarded to the
prox). Checks the solver configuration of each rerun path and that no
historical output path is reused. No scientific values are asserted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "experiments" / "erratum_maxiter_reruns_2026_10_25.py"
)
MAX_ITER = 1000
RIG_EPS = 0.01


def _load() -> object:
    name = "exp_erratum_maxiter"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_specialisation_solver_runs_configured_iterations() -> None:
    mod = _load()
    _, support, f_func = mod.wide.base.problem(alpha=10.0, gamma=0.0)
    jko = mod.wide.base.make_jko(f_func, support, beta=1.0, rigorous=True)
    assert jko.prox_n_iter == MAX_ITER
    assert jko.reg == RIG_EPS


def test_outputs_are_new() -> None:
    mod = _load()
    new = {p.resolve() for p in mod.output_paths().values()}
    old = {
        Path("paper_rigorous_fixedprox").resolve(),
        Path("paper_rigorous_fixedeps").resolve(),
        Path("paper/erratum_2026_10_24").resolve(),
        Path("paper/erratum_2026_10_24_reg").resolve(),
        mod.reg.SPEC_DIR.resolve(),
        mod.wide.OUT_DIR.resolve(),
        mod.wide.base.OUT_DIR.resolve(),
    }
    assert not any(any(o == p or o in p.parents for o in old) for p in new)
