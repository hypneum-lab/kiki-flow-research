"""Robustness of the M5 validation driver (review v2 of PR #51, item 2)."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from kiki_flow_core.master_equation import JKOStep

DRIVER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "implicit_jko_certified_validation_2026_10_27.py"
)


@pytest.fixture(scope="module")
def driver() -> Any:
    spec = importlib.util.spec_from_file_location("validation_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["validation_driver"] = module  # picklable for the process pool
    spec.loader.exec_module(module)
    return module


def _boom(x: float) -> dict[str, Any]:
    raise RuntimeError(f"boom {x}")


def _ok(x: float) -> dict[str, Any]:
    return {"x": x, "status": "converged"}


def test_a_failing_job_is_recorded_and_the_batch_goes_on(driver: Any, tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    driver.run_pool(_boom, [(1.0,)], 1, path)
    driver.run_pool(_ok, [(2.0,)], 1, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["status"] == "job_error" and "boom 1.0" in rows[0]["error"]
    assert rows[0]["solver_version"] == driver.RUN_KEY["solver_version"]
    assert rows[1] == {"x": 2.0, "status": "converged"}


def test_operating_records_an_uncertified_step_instead_of_crashing(
    driver: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def short(*args: Any, **kwargs: Any) -> JKOStep:
        return JKOStep(*args, **kwargs, inner_n_iter=1)

    monkeypatch.setattr(driver, "JKOStep", short)
    with pytest.warns(UserWarning, match="not certified"):
        row = driver.operating_job(10.0, 0.0)
    assert row["status"] == "inner_not_converged"
    assert row["check"].startswith("not_run")


def test_reference_fails_explicitly_when_every_start_fails(
    driver: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(fun: Any, z0: np.ndarray, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(x=z0, fun=float(fun(z0)[0]), success=False)

    monkeypatch.setattr(driver, "minimize", failed)
    f = driver.problem(4, 2.0, 0.0)
    sup = np.linspace(-0.5, 0.5, 4).reshape(-1, 1)
    rho = {n: np.full(4, 0.25) for n in f.species.species_names()}
    with pytest.raises(RuntimeError, match="every start"):
        driver.reference_minimiser(f, rho, sup, 0.2, 0.04, [rho])


def test_resume_key_covers_solver_and_settings(driver: Any) -> None:
    assert {"solver_version", "delta", "n_steps"} <= set(driver.RUN_KEY)
    record = dict(driver.RUN_KEY, steps=[])
    assert driver._same_run(record)
    assert not driver._same_run({**record, "delta": 1e-4})


def _counts(**kw: int) -> dict[str, int]:
    base = {"certified_decrease": 0, "no_decrease": 0, "indeterminate": 0, "not_certified": 0}
    return {**base, **kw}


def test_operating_verdict_has_three_states(driver: Any) -> None:
    assert driver.operating_verdict(_counts(no_decrease=120)) == "passed"
    assert driver.operating_verdict(_counts(certified_decrease=1, no_decrease=119)) == "failed"
    assert driver.operating_verdict(_counts(no_decrease=119, indeterminate=1)) == "inconclusive"
    assert driver.operating_verdict(_counts(not_certified=120)) == "inconclusive"
    assert driver.operating_verdict(_counts()) == "inconclusive"


def test_operating_with_no_certified_comparison_is_inconclusive(
    driver: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex fault injection: certified step and base bracket, every transfer fails."""

    class Injected(JKOStep):
        calls = 0

        def step(self, state: Any) -> Any:
            self.last_info = {
                "status": "converged",
                "kkt_residual": 0.0,
                "log_residual": 0.0,
                "q_min": float(min(r.min() for r in state.rho.values())),
                "seconds": 0.0,
            }
            return state

        def objective_bounds(self, *args: Any, **kwargs: Any) -> tuple[float, float]:
            Injected.calls += 1
            if Injected.calls == 1:
                return (0.0, 1e-12)
            raise driver.SinkhornNotConvergedError("injected")

    monkeypatch.setattr(driver, "JKOStep", Injected)
    row = driver.operating_job(10.0, 0.0)
    assert row["transfers"]["not_certified"] == 120  # noqa: PLR2004
    assert row["check"] == "inconclusive"


def test_run_key_holds_effective_defaults_and_code_identity(driver: Any) -> None:
    key = driver.RUN_KEY
    defaults = key["jko_defaults"]
    for name in ("kkt_tol", "log_tol", "max_outer", "inner_n_iter", "inner_tol"):
        assert name in defaults
    signature = inspect.signature(JKOStep.__init__).parameters
    assert defaults["kkt_tol"] == signature["kkt_tol"].default
    assert defaults["log_tol"] == signature["log_tol"].default
    assert {"git_commit", "git_dirty", "git_diff_sha256"} <= set(key)
    changed = {**key, "jko_defaults": {**defaults, "kkt_tol": defaults["kkt_tol"] * 10}}
    assert driver._same_run(key)
    assert not driver._same_run(changed)
