"""Light tests for ``experiments/jko_entropic_sweep_2026_10_25.py``.

Pins the configuration (entropic scheme, reg = h^2, reinjection), the job
list (distinct (alpha, gamma) couples x seeds plus the sensitivity runs),
the beta invariance that justifies running one couple for every beta, the
per-seed pairing of the summary and the output paths. No scientific value
is asserted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from kiki_flow_core.state import FlowState

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "experiments" / "jko_entropic_sweep_2026_10_25.py"
)
GRID = 16
N_COUPLES = 33
N_SENS_COUPLES = 4
N_SLOW_SMOKE = 2


def _load() -> object:
    name = "exp_jko_entropic_sweep_2026_10_25"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_solver_is_entropic_with_reg_h_squared() -> None:
    mod = _load()
    _, _, jko = mod.make_solver(10.0, 0.0, mod.MAIN_REG)
    assert jko.scheme == "jko_entropic"
    assert jko.reg == pytest.approx(mod.H**2)
    assert jko.prox_n_iter == mod.MAX_ITER


def test_jobs_cover_distinct_couples_and_sensitivity() -> None:
    mod = _load()
    pts = mod.points()
    assert len(pts) == 138  # noqa: PLR2004  (widened sweep)
    assert len(mod.couples(pts)) == N_COUPLES
    jobs = mod.jobs(mod.SEEDS)
    main = [j for j in jobs if j[3] == mod.MAIN_REG]
    sens = [j for j in jobs if j[3] == mod.SENS_REG]
    assert len(main) == N_COUPLES * len(mod.SEEDS)
    assert len(sens) == N_SENS_COUPLES * len(mod.SENS_SEEDS)


def test_entropic_step_ignores_beta() -> None:
    """beta only set n_inner of the legacy split: the entropic step does not use it."""
    mod = _load()
    species, support, f_func = mod.base.problem(8.0, 5.0)
    rng = np.random.default_rng(0)
    state = FlowState(
        rho={n: rng.dirichlet(np.full(GRID, 2.0)) for n in species.species_names()},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )
    outs = []
    for beta in (0.0, 0.5, 2.0):
        jko = mod.base.make_jko(f_func, support, beta, rigorous=True, epsilon=mod.MAIN_REG)
        jko.scheme = "jko_entropic"
        jko.prox_n_iter = 50
        outs.append(jko.step(state).rho)
    for other in outs[1:]:
        for n in outs[0]:
            np.testing.assert_array_equal(outs[0][n], other[n])


def test_summary_pairs_controls_by_seed() -> None:
    mod = _load()
    recs = []
    for s in (0, 1):
        for a, g in mod.couples(mod.points()):
            h = 3.5 if a > 0 else 3.9
            recs.append(
                {
                    "alpha": a,
                    "gamma": g,
                    "seed": s,
                    "reg": mod.MAIN_REG,
                    "mean_entropy_bits": h,
                    "gap_bits": 4.0 - h,
                }
            )
    rows = mod.summarise(recs, [], mod.MAIN_REG, [0, 1])
    assert len(rows) == len(mod.points())
    row = next(r for r in rows if (r["alpha"], r["beta"], r["gamma"]) == (10.0, 1.0, 0.0))
    assert row["entropic_gap"]["mean"] == pytest.approx(0.5)
    assert row["entropic_S"]["mean"] == pytest.approx(0.4)
    assert row["legacy_rigorous_gap"]["n"] == 0
    # A missing control drops that seed.
    rows = mod.summarise(
        [r for r in recs if not (r["alpha"] == 0.0 and r["seed"] == 1)], [], mod.MAIN_REG, [0, 1]
    )
    row = next(r for r in rows if (r["alpha"], r["beta"], r["gamma"]) == (10.0, 1.0, 0.0))
    assert row["seeds"] == [0]


@pytest.mark.slow
def test_run_couple_smoke() -> None:
    mod = _load()
    r = mod.run_couple(10.0, 0.0, 0, mod.MAIN_REG, n_slow=N_SLOW_SMOKE)
    assert set(r["per_species"]) == {"phono", "lex", "syntax", "sem"}
    assert 0.0 <= r["gap_bits"] <= 4.0  # noqa: PLR2004


def test_outputs_are_new_files_under_results() -> None:
    mod = _load()
    for path in mod.output_paths().values():
        assert path.parent == mod.OUT_DIR
        assert mod.OUT_DIR.parts[0] == "results"
