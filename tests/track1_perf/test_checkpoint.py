import json
from pathlib import Path

import numpy as np

from kiki_flow_core.state import FlowState
from kiki_flow_core.track1_perf.checkpoint import (
    load_checkpoint,
    load_known_concepts,
    save_checkpoint,
)


def make_state() -> FlowState:
    return FlowState(
        rho={"phono:code": np.array([0.5, 0.5])},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=3,
        metadata={"track_id": "T1", "step_id": "abc"},
    )


def test_save_roundtrip(tmp_path: Path):
    state = make_state()
    path = tmp_path / "ckpt"
    save_checkpoint(state, path)
    assert (path.with_suffix(".safetensors")).exists()
    assert (path.with_suffix(".json")).exists()
    loaded = load_checkpoint(path)
    assert loaded.tau == 3  # noqa: PLR2004
    np.testing.assert_array_equal(loaded.rho["phono:code"], state.rho["phono:code"])


def test_manifest_contains_git_sha(tmp_path: Path):
    state = make_state()
    path = tmp_path / "ckpt"
    save_checkpoint(state, path)
    manifest = json.loads((path.with_suffix(".json")).read_text())
    assert "git_sha" in manifest
    assert "timestamp" in manifest


def test_manifest_writes_sorted_known_concepts(tmp_path: Path):
    path = tmp_path / "ckpt"
    save_checkpoint(make_state(), path, known_concepts={"b", "a", "c"})
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert manifest["known_concepts"] == ["a", "b", "c"]


def test_manifest_known_concepts_defaults_to_empty(tmp_path: Path):
    path = tmp_path / "ckpt"
    save_checkpoint(make_state(), path)
    assert json.loads(path.with_suffix(".json").read_text())["known_concepts"] == []


def test_load_known_concepts_tolerates_legacy_manifest(tmp_path: Path):
    path = tmp_path / "ckpt"
    assert load_known_concepts(path) == set()  # no manifest yet
    save_checkpoint(make_state(), path, known_concepts=["x"])
    assert load_known_concepts(path) == {"x"}
    manifest = json.loads(path.with_suffix(".json").read_text())
    del manifest["known_concepts"]  # pre-fix manifest
    path.with_suffix(".json").write_text(json.dumps(manifest))
    assert load_known_concepts(path) == set()
