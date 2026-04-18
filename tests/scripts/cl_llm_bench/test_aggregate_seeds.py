# tests/scripts/cl_llm_bench/test_aggregate_seeds.py
"""Tests for the multi-seed aggregator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.cl_llm_bench.aggregate_seeds import aggregate, load_results

TASKS = ["phono_sst2", "lex_cola", "syn_boolq"]
TOL = 1e-6


def _make_result(
    seed: int, accs: tuple[float, float, float], forgetting: tuple[float, float, float]
) -> dict:
    im = dict(zip(TASKS, accs, strict=True))
    fi = {TASKS[0]: accs[0] - forgetting[0], TASKS[1]: accs[1] - forgetting[1], TASKS[2]: accs[2]}
    fo = dict(zip(TASKS, forgetting, strict=True))
    return {
        "status": "ok",
        "seed": seed,
        "tasks": TASKS,
        "immediate_accuracy": im,
        "final_accuracy": fi,
        "forgetting": fo,
        "wall_time_s": 260.0,
    }


def test_load_results_validates_schema(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_make_result(0, (0.9, 0.8, 0.6), (0.1, 0.0, 0.0))))
    out = load_results([good])
    assert len(out) == 1

    # missing key
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"status": "ok", "seed": 1, "tasks": TASKS}))
    with pytest.raises(ValueError, match="missing"):
        load_results([bad])

    # status != ok
    failed = tmp_path / "failed.json"
    payload = _make_result(2, (0.9, 0.8, 0.6), (0.1, 0.0, 0.0))
    payload["status"] = "failed"
    failed.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="status="):
        load_results([failed])


def test_aggregate_computes_mean_and_std() -> None:
    results = [
        _make_result(0, (0.90, 0.80, 0.60), (0.10, 0.00, 0.00)),
        _make_result(1, (0.94, 0.76, 0.58), (0.06, 0.00, 0.00)),
        _make_result(2, (0.92, 0.82, 0.56), (0.08, 0.00, 0.00)),
    ]
    summary = aggregate(results)
    assert summary["n_seeds"] == 3  # noqa: PLR2004
    assert summary["seeds"] == [0, 1, 2]
    assert summary["tasks"] == TASKS

    im_phono = summary["stats"]["immediate"]["phono_sst2"]
    assert abs(im_phono["mean"] - 0.92) < TOL
    assert abs(im_phono["std"] - 0.016329931618554534) < TOL

    fo_phono = summary["stats"]["forgetting"]["phono_sst2"]
    assert abs(fo_phono["mean"] - 0.08) < TOL
    # std of [0.10, 0.06, 0.08] is 0.01632...
    assert abs(fo_phono["std"] - 0.016329931618554534) < TOL

    # last task has zero forgetting by construction
    fo_syn = summary["stats"]["forgetting"]["syn_boolq"]
    assert fo_syn["mean"] == 0.0
    assert fo_syn["std"] == 0.0


def test_aggregate_rejects_mismatched_task_lists() -> None:
    ok = _make_result(0, (0.9, 0.8, 0.6), (0.1, 0.0, 0.0))
    shuffled = _make_result(1, (0.9, 0.8, 0.6), (0.1, 0.0, 0.0))
    shuffled["tasks"] = ["lex_cola", "phono_sst2", "syn_boolq"]
    with pytest.raises(ValueError, match="task list mismatch"):
        aggregate([ok, shuffled])


def test_aggregate_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no results"):
        aggregate([])
