"""Robustness of the M5 validation driver (review v2 of PR #51, item 2)."""

from __future__ import annotations

import importlib.util
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
