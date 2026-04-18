"""Integrate KikiFlowBridge into the LoRA training loop.

Records per-sample advisory weights so that downstream analysis can
correlate routing choices with catastrophic-forgetting events. Does not
modify the base trainer; it is a read-only observer plus an optional
mixing function that the trainer calls if wired in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kiki_flow_core.track3_deploy.kiki_flow_bridge import KikiFlowBridge


class AdvisoryRecorder:
    """Queries the bridge for each training sample and appends a JSONL line."""

    def __init__(self, weights_path: Path, out_path: Path) -> None:
        self.bridge = KikiFlowBridge(weights_path=weights_path)
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[str] = []

    def record(self, text: str, task_name: str, step: int) -> np.ndarray | None:
        advisory = self.bridge.route_advisory(text)
        self._buffer.append(
            json.dumps(
                {
                    "task": task_name,
                    "step": step,
                    "advisory": advisory.tolist() if advisory is not None else None,
                }
            )
        )
        return advisory

    def flush(self) -> None:
        if self._buffer:
            with self.out_path.open("a") as f:
                for line in self._buffer:
                    f.write(line + "\n")
            self._buffer.clear()
