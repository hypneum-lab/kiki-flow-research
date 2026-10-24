"""Drift heuristic: fires when new-concept rate in recent Aeon exceeds threshold.

Hysteresis (``threshold_off``, on by default) and an optional refractory
period (``cooldown_s``) suppress chattering when the drift rate hovers
around ``threshold``. Defaults (0.07 / 0.05, no cooldown) come from
``scripts/drift_trigger_sweep.py`` on synthetic traces; recalibrate them on a
recorded Aeon trace.
"""

from __future__ import annotations

import enum
import json
import os
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiki_flow_core.hooks import AeonAdapter

_STATE_VERSION = 1
DEFAULT_THRESHOLD_ON = 0.07
DEFAULT_THRESHOLD_OFF = 0.05


class _Default(enum.Enum):
    TOKEN = 0


class DriftTrigger:
    """Fires when ``rate = |new_concepts| / |concepts_in_window| > threshold``.

    Defaults: ``threshold=0.07``, ``threshold_off=0.05``, no cooldown. Legacy
    mode (``threshold_off=None``, ``cooldown_s=None``) is memoryless: every
    tick with ``rate > threshold`` fires. Passing ``threshold <= 0.05`` without
    an explicit ``threshold_off`` raises, since the default ``threshold_off``
    would not be below it.

    Hysteresis mode (``threshold_off`` set, ``0 <= threshold_off < threshold``):
    after a fire the trigger is *disarmed* and cannot fire again until a tick
    observes ``rate < threshold_off``, which re-arms it. Both comparisons are
    strict, so ``rate == threshold`` never fires and ``rate == threshold_off``
    never re-arms.

    Cooldown (``cooldown_s`` set): a fire also requires
    ``clock() - last_fire >= cooldown_s``. A tick blocked by the cooldown does
    not fire and does not disarm.

    State (``armed``, ``last_fire_ts``) lives in memory by default, which only
    helps a long-lived scheduler process; ``should_fire`` then emits a
    ``RuntimeWarning`` whenever hysteresis or cooldown is active. A cron job
    that builds a new ``DriftTrigger`` per tick must pass ``state_path``: the state is then read
    at the start of every ``should_fire`` call and atomically rewritten when it
    changes. Ticks with an empty window (no episodes or no concepts) return
    ``False`` and leave the state untouched.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD_ON,
        window_h: int = 24,
        threshold_off: float | None | _Default = _Default.TOKEN,
        cooldown_s: float | None = None,
        state_path: Path | str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(threshold_off, _Default):
            if threshold <= DEFAULT_THRESHOLD_OFF:
                raise ValueError(
                    f"threshold={threshold} is not above the default threshold_off="
                    f"{DEFAULT_THRESHOLD_OFF}; pass threshold_off explicitly "
                    "(None restores the legacy memoryless trigger)"
                )
            threshold_off = DEFAULT_THRESHOLD_OFF
        if threshold_off is not None and not 0.0 <= threshold_off < threshold:
            raise ValueError(
                f"threshold_off must satisfy 0 <= threshold_off < threshold "
                f"(got threshold_off={threshold_off}, threshold={threshold})"
            )
        if cooldown_s is not None and cooldown_s < 0:
            raise ValueError(f"cooldown_s must be >= 0 (got {cooldown_s})")
        self.threshold = threshold
        self.window_h = window_h
        self.threshold_off = threshold_off
        self.cooldown_s = cooldown_s
        self.state_path = Path(state_path) if state_path is not None else None
        self.clock = clock
        self.armed = True
        self.last_fire_ts: float | None = None

    @property
    def threshold_on(self) -> float:
        """Alias of ``threshold`` (the fire threshold) for hysteresis readers."""
        return self.threshold

    def drift_rate(self, aeon: AeonAdapter, last_snapshot_manifest: dict[str, Any]) -> float | None:
        """New-concept rate over the window, or ``None`` if the window is empty."""
        episodes = aeon.fetch_recent_episodes(window_h=self.window_h)
        if not episodes:
            return None
        all_concepts: set[str] = set()
        for ep in episodes:
            all_concepts.update(ep.get("concepts", []))
        if not all_concepts:
            return None
        known = set(last_snapshot_manifest.get("known_concepts", []))
        return len(all_concepts - known) / len(all_concepts)

    def update(self, rate: float) -> bool:
        """Advance the in-memory state machine by one tick; return whether to fire."""
        if self.threshold_off is not None and not self.armed and rate < self.threshold_off:
            self.armed = True
        if not self.armed or rate <= self.threshold:
            return False
        if (
            self.cooldown_s is not None
            and self.last_fire_ts is not None
            and self.clock() - self.last_fire_ts < self.cooldown_s
        ):
            return False
        if self.threshold_off is not None:
            self.armed = False
        self.last_fire_ts = float(self.clock())
        return True

    def should_fire(self, aeon: AeonAdapter, last_snapshot_manifest: dict[str, Any]) -> bool:
        if self.state_path is None and (
            self.threshold_off is not None or self.cooldown_s is not None
        ):
            warnings.warn(
                "DriftTrigger hysteresis/cooldown state is held in memory only "
                "(state_path=None); a cron job that builds a new trigger per tick "
                "re-arms it on every tick. Pass state_path to persist it.",
                RuntimeWarning,
                stacklevel=2,
            )
        rate = self.drift_rate(aeon, last_snapshot_manifest)
        if rate is None:
            return False
        self._load_state()
        before = (self.armed, self.last_fire_ts)
        fire = self.update(rate)
        if (self.armed, self.last_fire_ts) != before:
            self._save_state()
        return fire

    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text())
            armed = payload["armed"]
            last = payload["last_fire_ts"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"corrupt DriftTrigger state file {self.state_path}") from exc
        if not isinstance(armed, bool) or not (last is None or isinstance(last, int | float)):
            raise ValueError(f"corrupt DriftTrigger state file {self.state_path}")
        self.armed = armed
        self.last_fire_ts = None if last is None else float(last)

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        payload = {
            "version": _STATE_VERSION,
            "armed": self.armed,
            "last_fire_ts": self.last_fire_ts,
        }
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.state_path)
