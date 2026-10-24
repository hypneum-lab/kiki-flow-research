import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from kiki_flow_core.hooks import AeonAdapter
from kiki_flow_core.track1_perf.offline_consolidator import run_once
from kiki_flow_core.track1_perf.trigger_heuristic import DriftTrigger


def test_run_once_smoke(tmp_path: Path):
    def aeon_fetcher(h: int) -> list[dict]:
        return [{"id": 1, "concepts": ["phono"]}]

    def moe_snapshotter() -> dict[str, np.ndarray]:
        return {"code": np.zeros(4), "math": np.zeros(4)}

    routing = MagicMock()
    cfg = {
        "stack_names": ["code", "math"],
        "n_grid": 16,
        "checkpoint_dir": tmp_path,
    }
    manifest_out = run_once(
        config=cfg,
        aeon_fetcher=aeon_fetcher,
        moe_snapshotter=moe_snapshotter,
        advisory_publisher=routing.publish_advisory,
    )
    assert manifest_out["status"] == "ok"
    assert routing.publish_advisory.called
    assert (tmp_path / "latest.safetensors").exists()


def _run(tmp_path: Path, concepts: list[str]) -> dict:
    run_once(
        config={"stack_names": ["code", "math"], "n_grid": 16, "checkpoint_dir": tmp_path},
        aeon_fetcher=lambda h: [{"concepts": concepts}],
        moe_snapshotter=lambda: {"code": np.zeros(4), "math": np.zeros(4)},
        advisory_publisher=MagicMock(),
    )
    return json.loads((tmp_path / "latest.json").read_text())


def test_latest_manifest_drift_rate_counts_known_concepts(tmp_path: Path):
    """Regression: latest.json had no known_concepts, so the rate was always 1.0."""
    manifest = _run(tmp_path, ["a", "b", "c", "d"])
    aeon = AeonAdapter(fetcher=lambda h: [{"concepts": ["a", "b", "c", "d"]}])
    trigger = DriftTrigger(state_path=tmp_path / "drift_state.json")
    assert trigger.drift_rate(aeon, manifest) == 0.0
    assert trigger.should_fire(aeon, manifest) is False


def test_known_concepts_accumulate_across_runs(tmp_path: Path):
    _run(tmp_path, ["a", "b"])
    manifest = _run(tmp_path, ["b", "c"])
    assert manifest["known_concepts"] == ["a", "b", "c"]


def test_legacy_manifest_without_known_concepts_is_bootstrapped(tmp_path: Path):
    manifest = _run(tmp_path, ["a"])
    del manifest["known_concepts"]
    (tmp_path / "latest.json").write_text(json.dumps(manifest))
    aeon = AeonAdapter(fetcher=lambda h: [{"concepts": ["a"]}])
    # Pre-fix manifest: every concept counts as new (documented behaviour).
    assert DriftTrigger(threshold_off=None).drift_rate(aeon, manifest) == 1.0
    # The next run starts the field from the current window.
    assert _run(tmp_path, ["b"])["known_concepts"] == ["b"]
