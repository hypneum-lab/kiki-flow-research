# Dep 3 — Query-string plumbing resolution (option b)

**Status**: design + reference implementation, not an auto-apply patch.
**Decision**: option (b) from `integration-notes.md` — give the runner its
own tokenizer so it does not depend on changes to `MetaRouter.forward()`.

## Why option (b)

- **Option (a)** (thread `query` through `forward()`): requires changing
  the router's interface, which cascades through every call site (MoE
  layer, batch inference, ANE compilation). High integration cost.
- **Option (b)** (runner owns tokenizer): the runner is constructed
  once at router init with a tokenizer handle, and re-extracts the
  query embedding from `input_ids` that flow into the router. Self-
  contained, zero interface change, costs one extra MiniLM forward
  pass per query (~3 ms, well within the 10 ms SLO).

## Reference implementation (apply manually to micro-kiki)

Add this adapter class on the micro-kiki side:

```python
# In ~/KIKI-Mac_tunner/src/serving/kiki_flow_bridge.py (NEW FILE)
"""Bridge between micro-kiki's routing hot path and kiki-flow's StreamingRunner.

Owns the tokenizer handle so the runner does not depend on MetaRouter.forward
receiving a query-string parameter. The bridge is constructed once at router
init and threaded through as an optional attribute.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class KikiFlowBridge:
    """Lazy-loaded StreamingRunner wrapper with its own MiniLM tokenizer.

    Returns None from route_advisory() if disabled, NaN, dim mismatch, or any
    other failure; callers must treat the result as purely advisory.
    """

    def __init__(
        self,
        weights_path: Path,
        state_dim: int = 128,
        embed_dim: int = 384,
        hidden: int = 256,
        tokenizer: Any | None = None,
        prior_weight: float = 0.1,
    ) -> None:
        self.enabled = bool(int(os.environ.get("KIKI_FLOW_ENABLED", "0")))
        self.prior_weight = prior_weight
        self._runner = None
        if not self.enabled:
            return
        try:
            from kiki_flow_core.hooks import RoutingAdapter
            from kiki_flow_core.state import FlowState
            from kiki_flow_core.track3_deploy.neural_surrogate import NeuralSurrogate
            from kiki_flow_core.track3_deploy.query_encoder import QueryEncoder
            from kiki_flow_core.track3_deploy.streaming_runner import StreamingRunner

            surrogate = NeuralSurrogate.load(
                weights_path, state_dim=state_dim, embed_dim=embed_dim, hidden=hidden
            )
            # QueryEncoder will use the provided tokenizer if MiniLM is
            # unavailable, else its own MiniLM via sentence-transformers.
            encoder = QueryEncoder(use_stub=(tokenizer is None))
            # Initial state: uniform over all species
            stacks = [f"stack_{i:02d}" for i in range(32)]
            orthos = ["phono", "lex", "syntax", "sem"]
            initial = FlowState(
                rho={f"{o}:{s}": np.array([1.0 / 32]) for o in orthos for s in stacks},
                P_theta=np.zeros(8),
                mu_curr=np.array([1.0]),
                tau=0,
                metadata={"track_id": "T3"},
            )
            self._runner = StreamingRunner(
                surrogate=surrogate,
                encoder=encoder,
                routing_adapter=RoutingAdapter(publisher=lambda _adv: None),
                initial_state=initial,
            )
            logger.info("kiki-flow bridge ready, state_dim=%d", state_dim)
        except Exception as e:  # noqa: BLE001
            logger.warning("kiki-flow bridge init failed, disabling: %s", e)
            self.enabled = False

    def route_advisory(self, query: str) -> np.ndarray | None:
        """Return a (32,)-shaped array of stack-level advisory weights, or None."""
        if not self.enabled or self._runner is None:
            return None
        try:
            advisory = self._runner.on_query(query)
            summary = advisory.get("state_summary", {})
            if not summary:
                return None
            # Aggregate over the 4 ortho species per stack
            weights = np.zeros(32, dtype=np.float32)
            for key, val in summary.items():
                if ":" not in key:
                    continue
                _ortho, stack_name = key.split(":", 1)
                if not stack_name.startswith("stack_"):
                    continue
                idx = int(stack_name.split("_", 1)[1])
                if 0 <= idx < 32:
                    weights[idx] += float(val) * 0.25  # /4 ortho contributions
            return weights
        except Exception as e:  # noqa: BLE001
            logger.warning("kiki-flow advisory failed, passthrough: %s", e)
            return None
```

Then, in `MetaRouter.__init__`:

```python
from serving.kiki_flow_bridge import KikiFlowBridge

self.kiki_flow_bridge = KikiFlowBridge(
    weights_path=Path("/path/to/v0.2-d128.safetensors"),
    state_dim=128,
)
```

And inside whatever inference pathway has the query string (the
FastAPI endpoint or the LLM wrapper, NOT the `forward()` call itself):

```python
def inference_with_advisory(query: str, hidden_state):
    base_scores = router.forward(hidden_state)            # 32 logits
    advisory = router.kiki_flow_bridge.route_advisory(query)  # (32,) or None
    if advisory is not None:
        blended = (1 - self.prior_weight) * base_scores + self.prior_weight * advisory
    else:
        blended = base_scores
    return top_k(blended)
```

## Risk / blast radius

- If `KIKI_FLOW_ENABLED=0` (default) the bridge is a no-op; zero risk.
- If enabled and weights missing → bridge disables itself with a WARNING.
- If enabled and any step raises → route_advisory returns None, router
  falls back to native scores. No silent corruption.
- The blending is linear at the logit level with a tunable
  `prior_weight` (default 10%) so the maximum effect on top-1 selection
  is bounded.

## Testing plan (côté micro-kiki)

1. Add `tests/serving/test_kiki_flow_bridge.py` with:
   - Bridge disabled (`KIKI_FLOW_ENABLED` unset) → route_advisory returns None
   - Bridge enabled + stub weights → returns (32,) float array
   - Bridge enabled + weights file missing → graceful disable
   - Bridge enabled + malformed query → returns None without raising
2. Run existing routing tests with `KIKI_FLOW_ENABLED=0` (no behavior change)
3. Run an A/B benchmark with `=0` vs `=1` on a fixed query set to
   quantify the effect of the advisory at the current 10% blend weight.

## What this still does not provide

- **Real query-conditioned routing**: MiniLM encoding of the query is
  the full picture; we do not use the hidden state semantically. If
  `hidden_state` is expected to carry semantic intent that MiniLM does
  not, the advisory is weaker.
- **Per-user personalization**: the state is shared across all
  sessions. Per-user or per-session trajectories would need a
  session-keyed StreamingRunner pool.
