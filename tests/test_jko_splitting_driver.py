"""Driver of the Lie / Strang splitting preregistration (2026-10-30): plan and analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

DRIVER = Path(__file__).resolve().parents[1] / "experiments" / "jko_splitting_2026_10_30.py"
N_JOBS = 8 * 3 * 2 * 3 * 4  # gammas x alphas x seeds x schemes x ladder
EXPECTED_LENGTHS = {  # trajectory length per (scheme, h), prereg section 4
    ("implicit", 0.1): 1,
    ("implicit", 0.05): 2,
    ("implicit", 0.025): 2,
    ("implicit", 0.0125): 2,
    ("lie", 0.1): 1,
    ("lie", 0.05): 3,
    ("lie", 0.025): 3,
    ("lie", 0.0125): 2,
    ("strang", 0.1): 1,
    ("strang", 0.05): 3,
    ("strang", 0.025): 3,
    ("strang", 0.0125): 2,
}
TINY_GRID = 4
LIE_SUBSTEPS = 2
PRIMARY_CONFIGS = 60  # gamma <= 30: 5 x 3 alphas x 2 seeds x 2 h
PRIMARY_CELLS = 30  # gamma <= 30: 5 x 3 alphas x 2 seeds


@pytest.fixture(scope="module")
def driver() -> Any:
    spec = importlib.util.spec_from_file_location("jko_splitting_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["jko_splitting_driver"] = module  # picklable for the process pool
    spec.loader.exec_module(module)
    return module


def test_inputs_are_those_of_the_m5_certify_check(driver: Any) -> None:
    m5 = driver.m5
    f, state = driver.initial_state(5.0, 30.0, 500)
    rng = np.random.default_rng(500)
    for n in f.species.species_names():
        assert np.array_equal(state.rho[n], rng.dirichlet(np.ones(16)))
    ref = m5.problem(16, 5.0, 30.0)
    assert f.turing_strength == ref.turing_strength
    assert f.turing_smoothing == m5.DELTA
    assert all(np.array_equal(f.potentials[n], ref.potentials[n]) for n in ref.potentials)
    assert (driver.GAMMAS, driver.ALPHAS, driver.SEEDS) == (m5.GAMMAS, m5.ALPHAS, m5.INIT_SEEDS)
    assert set(driver.M5_HS) == set(m5.HS)
    assert driver.PILOT_SEED not in driver.SEEDS


def test_plan_of_the_confirmatory_run(driver: Any) -> None:
    jobs = driver.confirm_jobs("GO test", set())
    assert len(jobs) == N_JOBS
    assert len({(j.alpha, j.gamma, j.seed, j.scheme, j.h) for j in jobs}) == N_JOBS
    assert all(j.store and j.cap == driver.STEP_CAP_S and j.go == "GO test" for j in jobs)
    lengths = {(j.scheme, j.h): j.steps for j in jobs}
    assert lengths == EXPECTED_LENGTHS
    assert jobs[0].gamma == max(driver.GAMMAS)  # longest first
    done = {(j.alpha, j.gamma, j.seed, j.scheme, j.h) for j in jobs[:10]}
    assert len(driver.confirm_jobs("GO test", done)) == N_JOBS - 10
    pilot = driver.pilot_jobs()
    assert all(not j.store and j.seed == driver.PILOT_SEED and j.steps == 1 for j in pilot)


def test_confirm_needs_a_go_and_workers_are_capped(driver: Any) -> None:
    with pytest.raises(SystemExit, match="GO"):
        driver.main(["confirm"])
    with pytest.raises(SystemExit, match="at most"):
        driver.main(["pilot", "--workers", "5"])


def test_wall_cap_interrupts_a_step(driver: Any) -> None:
    with pytest.raises(driver.StepTimeoutError), driver.wall_cap(1):
        time.sleep(3)


def test_trajectory_job_records_densities_and_brackets(driver: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(driver, "GRID", TINY_GRID)
    rec = driver.trajectory_job(driver.Job(5.0, 0.0, 500, "lie", 0.05, 2, 600, True, "test"))
    assert [s["status"] for s in rec["steps"]] == ["converged", "converged"]
    first = rec["steps"][0]
    assert len(first["rho_out"]["phono"]) == TINY_GRID
    assert first["dissipation"]["bracket"] == "ok"
    assert len(first["substeps"]) == LIE_SUBSTEPS
    pilot = driver.trajectory_job(driver.Job(5.0, 0.0, 999, "strang", 0.05, 1, 600, False, None))
    kept = pilot["steps"][0]
    assert "rho_out" not in kept and "F_out" not in kept and "dissipation" not in kept
    assert kept["bracket_wall"] > 0


# ----- analysis on synthetic records ---------------------------------------------------


def _shift(eps: float) -> dict[str, list[float]]:
    """Densities at TV distance eps (every species) from the base density."""
    base = np.full(16, 1.0 / 16)
    moved = base.copy()
    moved[0] += eps
    moved[1] -= eps
    return {n: moved.tolist() for n in ("phono", "lex", "syntax", "sem")}


def _synthetic(
    driver: Any, error: dict[str, float], order: dict[str, float]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Records with e_X = error[X] and s_Y(h) = h ** order[Y] (s_S is also the margin)."""
    records = {}
    ok = {"bracket": "ok", "Q_out": [0.0, 0.0], "Q_in": [1.0, 1.0]}
    for g in driver.GAMMAS:
        for a in driver.ALPHAS:
            for s in driver.SEEDS:
                for y in driver.SCHEMES:
                    for h in driver.LADDER:
                        n = driver.n_steps(y, h)
                        step0 = 0.0 if y == "implicit" else error[y]
                        # step 1 of the h job is compared with step 0 of the 2h job
                        level = 2 * h
                        second = step0 + level ** order[y] if level in driver.ORDER_HS else step0
                        steps = []
                        for k in range(n):
                            eps = step0 if k == 0 else second
                            steps.append(
                                {
                                    "status": "converged",
                                    "rho_out": _shift(eps),
                                    "F_in": 1.0,
                                    "F_out": 0.5,
                                    "dissipation": ok,
                                    "wall": 1.0,
                                }
                            )
                        records[(a, g, s, y, h)] = {"steps": steps, **driver.RUN_KEY}
    return records


def test_analysis_verdicts_follow_the_rule(driver: Any) -> None:
    # margin s_S(h) = h^2 (2.5e-3, 6.3e-4): Lie inside, Strang outside; slopes S 2, Lie 2, Strang 3.
    records = _synthetic(
        driver, {"lie": 1e-4, "strang": 1e-2}, {"implicit": 2.0, "lie": 2.0, "strang": 3.0}
    )
    out = driver.analyse(records)
    assert out["verdicts"]["lie"]["substitution"] == "substitutable"
    assert out["verdicts"]["strang"]["substitution"] == "not_substitutable"
    assert len(out["verdicts"]["strang"]["c2_not_accurate"]) == PRIMARY_CONFIGS
    assert out["order"]["gated"] == PRIMARY_CELLS
    assert out["order"]["lie"]["verdict"] == "first_order_like"
    assert out["order"]["strang"]["verdict"] == "second_order_like"


def test_order_is_not_identifiable_when_the_control_fails(driver: Any) -> None:
    records = _synthetic(
        driver, {"lie": 1e-4, "strang": 1e-4}, {"implicit": 0.5, "lie": 2.0, "strang": 3.0}
    )
    out = driver.analyse(records)
    assert out["order"]["gated"] == 0
    assert out["order"]["strang"]["verdict"] == "not_identifiable"


def test_failures_and_violations_block_substitution(driver: Any) -> None:
    records = _synthetic(
        driver, {"lie": 1e-4, "strang": 1e-4}, {"implicit": 2.0, "lie": 2.0, "strang": 2.0}
    )
    records[(2.0, 0.0, 500, "lie", 0.05)]["steps"][2]["status"] = "max_outer"
    bad = {"bracket": "ok", "Q_out": [2.0, 2.1], "Q_in": [1.0, 1.0]}
    records[(2.0, 5.0, 501, "strang", 0.025)]["steps"][1]["dissipation"] = bad
    out = driver.analyse(records)
    assert out["verdicts"]["lie"]["c1_not_certified"] == [[2.0, 0.0, 500, 0.05]]
    assert out["verdicts"]["strang"]["c3_violations"] == [[2.0, 5.0, 501, 0.025]]
    assert out["verdicts"]["lie"]["substitution"] == "not_substitutable"
    assert out["verdicts"]["strang"]["substitution"] == "not_substitutable"
    # no reference at all: accuracy undecidable
    for key in list(records):
        if key[3] == "implicit":
            records[key]["steps"][0]["status"] = "timeout"
    assert driver.analyse(records)["verdicts"]["lie"]["substitution"] == "undecided"


def test_records_of_another_run_are_ignored(driver: Any, tmp_path: Path) -> None:
    path = tmp_path / "confirm.jsonl"
    good = {"alpha": 2.0, "gamma": 0.0, "seed": 500, "scheme": "lie", "h": 0.05}
    rows = [
        {**good, "steps": [], **driver.RUN_KEY},
        {**good, "seed": 501, "steps": [], **driver.RUN_KEY, "git_commit": "other"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert set(driver.load_records(path)) == {(2.0, 0.0, 500, "lie", 0.05)}
