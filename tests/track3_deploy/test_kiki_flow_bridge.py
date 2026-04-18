"""Tests for the KikiFlowBridge, exercising the full integration path
with a mock router. This is the in-repo validation that the bridge
design satisfies the micro-kiki integration requirements (issue #4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kiki_flow_core.track3_deploy.kiki_flow_bridge import KikiFlowBridge

WEIGHTS_V02 = Path("kiki_flow_core/track3_deploy/weights/v0.2-d128.safetensors")


def test_bridge_disabled_by_default(monkeypatch):
    """When KIKI_FLOW_ENABLED is unset, bridge is off and returns None."""
    monkeypatch.delenv("KIKI_FLOW_ENABLED", raising=False)
    bridge = KikiFlowBridge(weights_path=WEIGHTS_V02)
    assert bridge.enabled is False
    assert bridge.route_advisory("any query") is None


def test_bridge_explicit_off(monkeypatch):
    """Explicit KIKI_FLOW_ENABLED=0 also disables the bridge."""
    monkeypatch.setenv("KIKI_FLOW_ENABLED", "0")
    bridge = KikiFlowBridge(weights_path=WEIGHTS_V02)
    assert bridge.enabled is False
    assert bridge.route_advisory("any query") is None


def test_bridge_enabled_returns_32_weights(monkeypatch):
    """When enabled with real v0.2 weights, advisory is a (32,) float array."""
    monkeypatch.setenv("KIKI_FLOW_ENABLED", "1")
    bridge = KikiFlowBridge(weights_path=WEIGHTS_V02)
    assert bridge.enabled is True
    weights = bridge.route_advisory("integration test query")
    assert weights is not None
    assert weights.shape == (32,)
    assert np.isfinite(weights).all()


def test_bridge_missing_weights_gracefully_disables(monkeypatch, tmp_path):
    """Weights file absent -> bridge disables itself silently."""
    monkeypatch.setenv("KIKI_FLOW_ENABLED", "1")
    bogus = tmp_path / "does_not_exist.safetensors"
    bridge = KikiFlowBridge(weights_path=bogus)
    assert bridge.enabled is False
    assert bridge.route_advisory("any query") is None


def test_bridge_survives_arbitrary_queries(monkeypatch):
    """Bridge should not crash on short / empty / unicode queries."""
    monkeypatch.setenv("KIKI_FLOW_ENABLED", "1")
    bridge = KikiFlowBridge(weights_path=WEIGHTS_V02)
    assert bridge.enabled is True
    for query in ["", "a", "très long query avec accents àéîôü", "?"]:
        weights = bridge.route_advisory(query)
        assert weights is not None
        assert weights.shape == (32,)


def test_mock_router_blend_integration(monkeypatch):
    """End-to-end: a mock MetaRouter that blends 10 percent bridge advisory.

    Mirrors the wire-up sketched in docs/superpowers/patches/
    02-micro-kiki-dep3-runner-factory.md.
    """
    monkeypatch.setenv("KIKI_FLOW_ENABLED", "1")
    bridge = KikiFlowBridge(weights_path=WEIGHTS_V02)

    class MockMetaRouter:
        def __init__(self, kiki_flow_bridge: KikiFlowBridge | None = None) -> None:
            self.kiki_flow_bridge = kiki_flow_bridge
            self._base_scores = np.array([1.0 / 32] * 32, dtype=np.float32)

        def route_with_advisory(self, query: str, prior: float = 0.1) -> np.ndarray:
            scores = self._base_scores.copy()
            if self.kiki_flow_bridge is not None:
                advisory = self.kiki_flow_bridge.route_advisory(query)
                if advisory is not None:
                    scores = (1 - prior) * scores + prior * advisory
            return scores

    router = MockMetaRouter(kiki_flow_bridge=bridge)
    final_scores = router.route_with_advisory("hello")
    assert final_scores.shape == (32,)
    assert np.isfinite(final_scores).all()


def test_integration_path_is_non_destructive_on_bridge_crash(monkeypatch):
    """If the bridge raises internally, router falls back to native scores."""
    monkeypatch.setenv("KIKI_FLOW_ENABLED", "1")
    bridge = KikiFlowBridge(weights_path=WEIGHTS_V02)

    class BrokenRunner:
        def on_query(self, _query: str) -> dict:
            raise RuntimeError("simulated runner crash")

    bridge._runner = BrokenRunner()  # type: ignore[assignment]

    out = bridge.route_advisory("query")
    assert out is None  # swallowed exception, advisory dropped
