"""Runner tests for ``experiments/eps_sweep_2026_10_25.py``.

Checks that the sweep's solver receives the swept epsilon as the Sinkhorn
``reg`` (and the JKO step h as the proximal pull), that the epsilon flag of
``make_jko`` leaves the default untouched, that ``run_point`` reproduces
``base.run_one``, that the summary is computed per seed, and that all
outputs are new files. No scientific values are asserted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kiki_flow_core.state import FlowState

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "experiments" / "eps_sweep_2026_10_25.py"
H = 0.05
DEFAULT_EPS = 0.01
SWEPT_EPS = 0.02
N_SLOW_SMOKE = 2
ENTROPY_TOL = 1e-9
GRID = 16


def _load() -> object:
    name = "exp_eps_sweep_2026_10_25"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_make_jko_epsilon_flag_sets_reg_and_keeps_default() -> None:
    base = _load().base
    _, support, f_func = base.problem(alpha=10.0, gamma=0.0)
    swept = base.make_jko(f_func, support, beta=1.0, rigorous=True, epsilon=SWEPT_EPS)
    assert swept.reg == SWEPT_EPS
    assert swept.epsilon == SWEPT_EPS
    assert swept.h == H
    default = base.make_jko(f_func, support, beta=1.0, rigorous=True)
    assert default.reg == DEFAULT_EPS
    fast = base.make_jko(f_func, support, beta=1.0, rigorous=False, epsilon=SWEPT_EPS)
    assert not fast.apply_w2_prox
    assert fast.reg is None


def test_sweep_prox_receives_swept_reg_not_h() -> None:
    base = _load().base
    _, support, f_func = base.problem(alpha=0.0, gamma=0.0)
    jko = base.make_jko(f_func, support, beta=0.0, rigorous=True, epsilon=SWEPT_EPS)
    calls: list[dict] = []
    inner = jko.prox_fn

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return inner(*args, **kwargs)

    jko.prox_fn = spy
    state = FlowState(
        rho={n: np.full(GRID, 1.0 / GRID) for n in ("phono", "lex", "syntax", "sem")},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )
    jko.step(state)
    assert calls
    assert all(c["reg"] == SWEPT_EPS for c in calls)
    assert all(c["epsilon"] == H for c in calls)


def test_points_are_key_points_then_alpha0_controls() -> None:
    mod = _load()
    assert mod.points() == [(10.0, 1.0, 0.0), (8.0, 0.5, 5.0), (0.0, 1.0, 0.0), (0.0, 0.5, 5.0)]
    assert mod.EPSILONS == [0.01, 0.02, 0.05]
    assert mod.SEEDS == [0, 1, 2, 3, 4]
    n_fast = len(mod.points())
    assert len(mod.jobs(mod.SEEDS, mod.EPSILONS)) == len(mod.SEEDS) * n_fast * (
        1 + len(mod.EPSILONS)
    )


@pytest.mark.slow
def test_run_point_matches_base_run_one() -> None:
    mod = _load()
    p = (10.0, 0.5, 0.0)
    with mx.stream(mx.cpu):
        rho = mod.run_point(p, rigorous=True, seed=0, epsilon=None, n_slow=N_SLOW_SMOKE)
        ent = mod.base.run_one(*p, rigorous=True, seed=0, n_slow=N_SLOW_SMOKE)
    assert set(rho) == {"phono", "lex", "syntax", "sem"}
    for name, h in ent.items():
        assert abs(mod.entropy_bits(rho[name]) - h) < ENTROPY_TOL


def _uniform_record(mod: object, path: str, eps: float | None, seed: int, p: tuple) -> dict:
    rho = {n: [1.0 / GRID] * GRID for n in ("phono", "lex", "syntax", "sem")}
    return {
        "path": path,
        "epsilon": eps,
        "seed": seed,
        "alpha": p[0],
        "beta": p[1],
        "gamma": p[2],
        "mean_entropy_bits": float(np.log2(GRID)),
        "gap_bits": 0.0,
        "per_species": {n: float(np.log2(GRID)) for n in rho},
        "rho": rho,
        "runtime_s": 0.0,
    }


def test_summarise_pairs_by_seed_and_is_zero_on_uniform() -> None:
    mod = _load()
    seeds = [0, 1]
    recs = []
    for s in seeds:
        for p in mod.points():
            recs.append(_uniform_record(mod, "fast", None, s, p))
            recs.append(_uniform_record(mod, "rigorous", 0.05, s, p))
    rows = mod.summarise(recs, seeds, [0.05])
    assert [(r["alpha"], r["beta"], r["gamma"]) for r in rows] == mod.KEY_POINTS
    for r in rows:
        assert r["seeds"] == seeds
        assert r["rigorous"]["gap"]["mean"] == 0.0
        assert r["specialisation_vs_alpha0"]["mean"] == 0.0
        assert r["kl_rigorous_vs_fast_nats_sum4"]["mean"] == 0.0
        assert r["kl_rigorous_vs_fast_nats_sum4"]["n"] == len(seeds)
    # A missing control drops that seed from the row instead of crashing.
    rows = mod.summarise(recs[:-1], seeds, [0.05])
    assert rows[-1]["seeds"] == [0]


def test_outputs_are_new_files_under_results() -> None:
    mod = _load()
    for path in mod.output_paths().values():
        assert path.parent == mod.OUT_DIR
        assert mod.OUT_DIR.parts[0] == "results"
    assert not any(
        part.startswith(("paper", "paper_rigorous")) for part in mod.OUT_DIR.resolve().parts
    )
