# tests/scripts/cl_llm_bench/test_bridge_integration.py
from pathlib import Path

import pytest

from scripts.cl_llm_bench.bridge_integration import AdvisoryRecorder

EXPECTED_LINES = 2


def test_advisory_recorder_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIKI_FLOW_ENABLED", "1")
    out = tmp_path / "advisory.jsonl"
    weights_path = Path("kiki_flow_core/track3_deploy/weights/v0.2-d128.safetensors")
    recorder = AdvisoryRecorder(weights_path=weights_path, out_path=out)
    recorder.record("train sample 1", task_name="phono_sst2", step=0)
    recorder.record("train sample 2", task_name="phono_sst2", step=1)
    recorder.flush()
    lines = out.read_text().splitlines()
    assert len(lines) == EXPECTED_LINES
