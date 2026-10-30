"""Driver of the Lie / Strang splitting preregistration (2026-10-30): plan and analysis."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
HALF_CELLS = 15
FORBIDDEN_PILOT_KEYS = {"Q_final", "Q_explicit", "kkt_residual", "log_residual", "q_min"}
FORBIDDEN_PILOT_KEYS |= {"rho_out", "F_in", "F_out", "dissipation"}


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
    assert len({j.key() for j in jobs}) == N_JOBS
    assert all(j.store and j.cap == driver.STEP_CAP_S and j.go == "GO test" for j in jobs)
    assert {(j.scheme, j.h): j.steps for j in jobs} == EXPECTED_LENGTHS
    assert jobs[0].gamma == max(driver.GAMMAS)  # longest first
    done = {j.key() for j in jobs[:10]}
    assert len(driver.confirm_jobs("GO test", done)) == N_JOBS - 10
    pilot = driver.pilot_jobs()
    assert all(not j.store and j.seed == driver.PILOT_SEED and j.steps == 1 for j in pilot)


def test_confirm_needs_a_go_and_workers_are_capped(driver: Any) -> None:
    with pytest.raises(SystemExit, match="GO"):
        driver.main(["confirm"])
    with pytest.raises(SystemExit, match="at most"):
        driver.main(["pilot", "--workers", "5"])
    with pytest.raises(SystemExit, match="between 1"):
        driver.main(["pilot", "--workers", "0"])


def test_wall_cap_interrupts_in_a_worker_and_the_next_job_runs(driver: Any) -> None:
    with pytest.raises(driver.StepTimeoutError), driver.wall_cap(1):
        time.sleep(3)
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        first = pool.submit(driver.probe_cap, 3.0, 1)
        second = pool.submit(driver.probe_cap, 0.1, 5)
        assert (first.result(), second.result()) == ("timeout", "done")


def _keys(obj: Any) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | set().union(*(_keys(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(_keys(v) for v in obj)) if obj else set()
    return set()


def test_trajectory_job_records_densities_and_brackets(driver: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(driver, "GRID", TINY_GRID)
    rec = driver.trajectory_job(driver.Job(5.0, 0.0, 500, "lie", 0.05, 2, 600, True, "test"))
    assert [s["status"] for s in rec["steps"]] == ["converged", "converged"]
    first = rec["steps"][0]
    assert len(first["rho_out"]["phono"]) == TINY_GRID
    assert first["dissipation"]["bracket"] == "ok"
    assert driver.dissipation_state(first) in ("proven", "indeterminate", "violated")
    assert len(first["substeps"]) == LIE_SUBSTEPS
    assert rec["kind"] == "trajectory" and "node" in rec["host"]
    pilot = driver.trajectory_job(driver.Job(5.0, 0.0, 999, "strang", 0.05, 1, 600, False, None))
    assert not _keys(pilot) & FORBIDDEN_PILOT_KEYS  # recursively, sub-steps included
    assert pilot["steps"][0]["bracket_wall"] > 0


def test_timeout_keeps_the_finished_sub_steps(driver: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(driver, "GRID", TINY_GRID)
    f, state = driver.initial_state(5.0, 0.0, 500)
    split = driver.make_step(f, "strang", 0.05)
    real = split.substep

    def slow_b(part: str, s: float) -> Any:
        jko = real(part, s)
        if part == "B":
            jko.step = lambda st: time.sleep(5)  # type: ignore[method-assign]
        return jko

    monkeypatch.setattr(split, "substep", slow_b)
    new, info = driver._solve(split, state, 1, pilot=False)
    assert new is None and info["status"] == "timeout"
    assert [s["part"] for s in info["substeps"]] == ["A"] and info["active_part"] == "B"


# ----- analysis on synthetic records ---------------------------------------------------


def _shift(eps: float) -> dict[str, list[float]]:
    """Densities at TV distance eps (every species) from the uniform density."""
    base = np.full(16, 1.0 / 16)
    moved = base.copy()
    moved[0] += eps
    moved[1] -= eps
    return {n: moved.tolist() for n in ("phono", "lex", "syntax", "sem")}


PROVEN = {"bracket": "ok", "Q_out": [0.0, 0.0], "Q_in": [1.0, 1.0]}


def _synthetic(
    driver: Any, error: dict[str, float], order: dict[str, float]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Records with e_X = error[X] and s_Y(h) = h ** order[Y] (s_S is also the margin)."""
    records = {}
    for g in driver.GAMMAS:
        for a in driver.ALPHAS:
            for s in driver.SEEDS:
                for y in driver.SCHEMES:
                    for h in driver.LADDER:
                        step0 = 0.0 if y == "implicit" else error[y]
                        level = 2 * h  # step 1 of the h job vs step 0 of the 2h job
                        second = step0 + level ** order[y] if level in driver.ORDER_HS else step0
                        steps = [
                            {
                                "status": "converged",
                                "rho_out": _shift(step0 if k == 0 else second),
                                "F_in": 1.0,
                                "F_out": 0.5,
                                "dissipation": PROVEN,
                                "solver_wall": 1.0,
                            }
                            for k in range(driver.n_steps(y, h))
                        ]
                        records[(a, g, s, y, h)] = {"steps": steps, **driver.RUN_KEY}
    return records


def _regular(driver: Any) -> dict[tuple[Any, ...], dict[str, Any]]:
    # margin s_S(h) = h^2 (2.5e-3, 6.3e-4): Lie inside, Strang outside; S 2, Lie 2, Strang 3.
    return _synthetic(
        driver, {"lie": 1e-4, "strang": 1e-2}, {"implicit": 2.0, "lie": 2.0, "strang": 3.0}
    )


def test_analysis_verdicts_follow_the_rule(driver: Any) -> None:
    out = driver.analyse(_regular(driver))
    v = out["verdicts"]
    assert v["usable_margins"] == PRIMARY_CONFIGS
    assert v["lie"]["substitution"] == "substitutable"
    assert v["strang"]["substitution"] == "not_substitutable"
    assert len(v["strang"]["c2_fail"]) == PRIMARY_CONFIGS
    assert out["order"]["gated"] == PRIMARY_CELLS
    assert out["order"]["lie"]["verdict"] == "first_order_like"
    assert out["order"]["strang"]["verdict"] == "second_order_like"
    assert v["rates"]["gamma"]["0.0"]["S_step1"] == len(driver.ALPHAS) * 2 * 2


def test_order_is_not_identifiable_when_the_control_fails(driver: Any) -> None:
    records = _synthetic(
        driver, {"lie": 1e-4, "strang": 1e-4}, {"implicit": 0.5, "lie": 2.0, "strang": 3.0}
    )
    out = driver.analyse(records)
    assert out["order"]["gated"] == 0
    assert out["order"]["strang"]["verdict"] == "not_identifiable"


def test_missing_slopes_count_against_the_majority(driver: Any) -> None:
    records = _regular(driver)
    cells = [(a, g, s) for g in driver.PRIMARY_GAMMAS for a in driver.ALPHAS for s in driver.SEEDS]
    for a, g, s in cells[:HALF_CELLS]:  # 15 of 30 gated cells lose the Strang h = 0.0125 job
        records[(a, g, s, "strang", 0.0125)]["steps"][1]["status"] = "max_outer"
    out = driver.analyse(records)
    assert out["order"]["gated"] == PRIMARY_CELLS
    assert out["order"]["strang"]["classes"] == {"second": HALF_CELLS, "unresolved": HALF_CELLS}
    assert out["order"]["strang"]["verdict"] == "not_identifiable"  # 15 is not > 15


def test_inconsistent_adjacent_exponents_are_not_classed(driver: Any) -> None:
    records = _regular(driver)
    key = (2.0, 0.0, 500, "strang", 0.05)  # s(0.1) is compared with step 1 of the 0.05 job
    records[key]["steps"][1]["rho_out"] = _shift(1e-2 + 0.1**2)  # exponent 2 then 3
    cell = next(
        c for c in driver.order_cells(records) if (c["alpha"], c["gamma"], c["seed"]) == key[:3]
    )
    assert cell["strang"]["class"] == "neither"


def test_margin_coverage_does_not_depend_on_the_split(driver: Any) -> None:
    records = _regular(driver)
    primary = [
        (a, g, s, h)
        for g in driver.PRIMARY_GAMMAS
        for a in driver.ALPHAS
        for s in driver.SEEDS
        for h in driver.M5_HS
    ]
    for a, g, s, h in primary[:31]:  # Lie fails its first step in 31 of 60 configurations
        records[(a, g, s, "lie", h)]["steps"][0]["status"] = "line_search"
    v = driver.analyse(records)["verdicts"]
    assert v["usable_margins"] == PRIMARY_CONFIGS
    assert v["lie"]["substitution"] == "not_substitutable"
    assert len(v["lie"]["c1_not_certified"]) == len(v["lie"]["c2_fail"]) == 31  # noqa: PLR2004


def test_degenerate_margin_is_not_usable(driver: Any) -> None:
    records = _regular(driver)
    for key, rec in records.items():  # S at h/2 does not move: R = rho0 far from S_h
        if key[3] == "implicit":
            _, state0 = driver.initial_state(key[0], key[1], key[2])
            for st in rec["steps"][1:]:
                st["rho_out"] = {n: state0.rho[n].tolist() for n in state0.rho}
    v = driver.analyse(records)["verdicts"]
    assert v["usable_margins"] == 0
    assert v["lie"]["substitution"] == "undecided"


def test_dissipation_states(driver: Any) -> None:
    def st(d: Any) -> dict[str, Any]:
        return {} if d is None else {"dissipation": d}

    ok = {"bracket": "ok"}
    assert driver.dissipation_state(st({**ok, "Q_out": [0, 1], "Q_in": [1, 2]})) == "proven"
    assert driver.dissipation_state(st({**ok, "Q_out": [3, 4], "Q_in": [1, 2]})) == "violated"
    assert driver.dissipation_state(st({**ok, "Q_out": [1, 3], "Q_in": [2, 4]})) == "indeterminate"
    assert driver.dissipation_state(st(None)) == "missing"
    assert driver.dissipation_state(st({"bracket": "timeout"})) == "missing"
    assert driver.dissipation_state(st({**ok, "Q_out": [2, 1], "Q_in": [0, 0]})) == "missing"


def test_failures_and_missing_brackets_block_substitution(driver: Any) -> None:
    records = _synthetic(
        driver, {"lie": 1e-4, "strang": 1e-4}, {"implicit": 2.0, "lie": 2.0, "strang": 2.0}
    )
    records[(2.0, 0.0, 500, "lie", 0.05)]["steps"][2]["status"] = "max_outer"
    del records[(2.0, 5.0, 501, "strang", 0.025)]["steps"][1]["dissipation"]
    v = driver.analyse(records)["verdicts"]
    assert v["lie"]["c1_not_certified"] == [[2.0, 0.0, 500, 0.05]]
    assert v["strang"]["c3_fail"] == [[2.0, 5.0, 501, 0.025]]
    assert v["lie"]["substitution"] == v["strang"]["substitution"] == "not_substitutable"


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_attempts_errors_duplicates_and_completeness(driver: Any, tmp_path: Path) -> None:
    path = tmp_path / "confirm.jsonl"
    ident = {"alpha": 2.0, "gamma": 0.0, "seed": 500, "scheme": "lie", "h": 0.05}
    key = (2.0, 0.0, 500, "lie", 0.05)
    other = {**ident, "seed": 501}
    rows = [
        {**ident, "kind": "code", "status": "job_error", "steps": [], **driver.RUN_KEY},
        {**ident, "kind": "trajectory", "steps": [{"status": "converged"}], **driver.RUN_KEY},
        {**other, "kind": "infrastructure", "status": "job_error", "steps": [], **driver.RUN_KEY},
        {**other, "kind": "trajectory", "steps": [], **driver.RUN_KEY, "git_commit": "other"},
    ]
    _write(path, rows)
    attempts = driver.load_attempts(path)
    primary = driver.primary_records(attempts)
    assert primary[key]["status"] == "job_error"  # the first observation stays
    assert (2.0, 0.0, 501, "lie", 0.05) not in primary  # infrastructure only, other run ignored
    assert driver.done_keys(path, retry_infrastructure=False) == set(attempts)
    assert driver.done_keys(path, retry_infrastructure=True) == {key}
    summary = driver.analyse_file(path)
    prov = summary["provenance"]
    assert not prov["complete"] and len(prov["missing_jobs"]) == N_JOBS - 1
    assert prov["jobs_with_several_records"] == [key]
    assert "verdicts" not in summary and "provisional_verdicts" in summary
