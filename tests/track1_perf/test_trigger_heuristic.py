from kiki_flow_core.hooks import AeonAdapter
from kiki_flow_core.track1_perf.trigger_heuristic import DriftTrigger


def test_drift_trigger_fires_when_new_concepts_above_threshold():
    aeon = AeonAdapter(fetcher=lambda h: [{"concepts": ["a", "b", "c", "d", "e"]}])
    manifest = {"known_concepts": ["a", "b"]}  # 3 new of 5 => 60% drift
    trigger = DriftTrigger(threshold=0.2)
    assert trigger.should_fire(aeon, manifest) is True


def test_drift_trigger_holds_when_low_drift():
    aeon = AeonAdapter(fetcher=lambda h: [{"concepts": ["a", "b", "c"]}])
    manifest = {"known_concepts": ["a", "b", "c"]}  # 0% drift
    trigger = DriftTrigger(threshold=0.2)
    assert trigger.should_fire(aeon, manifest) is False


def test_drift_trigger_handles_empty_aeon():
    aeon = AeonAdapter(fetcher=lambda h: [])
    trigger = DriftTrigger(threshold=0.05)
    assert trigger.should_fire(aeon, {"known_concepts": []}) is False


# --- Hysteresis (threshold_on / threshold_off), cooldown, persistence ---

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

N_CONCEPTS = 100


def _aeon() -> AeonAdapter:
    """Aeon window with N_CONCEPTS concepts; novelty is set by the manifest."""
    concepts = [f"c{i}" for i in range(N_CONCEPTS)]
    return AeonAdapter(fetcher=lambda h: [{"concepts": concepts}])


def _manifest_at(rate_pct: int) -> dict[str, list[str]]:
    return {"known_concepts": [f"c{i}" for i in range(N_CONCEPTS - rate_pct)]}


def _run(trigger: DriftTrigger, rates_pct: list[int]) -> list[bool]:
    return [trigger.should_fire(_aeon(), _manifest_at(r)) for r in rates_pct]


def test_drift_rate_exposed():
    trigger = DriftTrigger()
    assert trigger.drift_rate(_aeon(), _manifest_at(7)) == pytest.approx(0.07)
    assert trigger.drift_rate(AeonAdapter(fetcher=lambda h: []), {}) is None


def test_legacy_default_chatters_like_before():
    """threshold_off=None keeps the memoryless behaviour (fires every tick)."""
    trigger = DriftTrigger(threshold=0.05)
    assert trigger.threshold_off is None
    assert _run(trigger, [6, 4, 6, 6, 5, 6]) == [True, False, True, True, False, True]


def test_hysteresis_chattering_fires_once():
    trigger = DriftTrigger(threshold=0.05, threshold_off=0.03)
    # Rate hovers around threshold_on but never drops below threshold_off.
    assert _run(trigger, [6, 4, 6, 4, 6, 5, 6, 3]) == [True] + [False] * 7


def test_hysteresis_rearms_below_off():
    trigger = DriftTrigger(threshold=0.05, threshold_off=0.03)
    assert _run(trigger, [6, 6, 2, 6, 6]) == [True, False, False, True, False]


def test_hysteresis_boundaries_are_strict():
    trigger = DriftTrigger(threshold=0.05, threshold_off=0.03)
    # rate == threshold_on does not fire (strict >, as legacy).
    assert _run(trigger, [5]) == [False]
    assert trigger.armed is True
    assert _run(trigger, [6]) == [True]
    # rate == threshold_off does not re-arm (strict <).
    assert _run(trigger, [3, 6]) == [False, False]
    assert trigger.armed is False
    assert _run(trigger, [2, 6]) == [False, True]


def test_threshold_off_must_be_below_threshold_on():
    with pytest.raises(ValueError, match="threshold_off"):
        DriftTrigger(threshold=0.05, threshold_off=0.05)
    with pytest.raises(ValueError, match="threshold_off"):
        DriftTrigger(threshold=0.05, threshold_off=-0.01)
    with pytest.raises(ValueError, match="cooldown"):
        DriftTrigger(threshold=0.05, cooldown_s=-1.0)


def test_empty_window_leaves_state_untouched():
    trigger = DriftTrigger(threshold=0.05, threshold_off=0.03)
    assert _run(trigger, [6]) == [True]
    empty = AeonAdapter(fetcher=lambda h: [])
    assert trigger.should_fire(empty, {}) is False
    assert trigger.armed is False


def test_cooldown_blocks_refire_until_elapsed():
    now = [0.0]
    trigger = DriftTrigger(threshold=0.05, cooldown_s=100.0, clock=lambda: now[0])
    assert _run(trigger, [6]) == [True]
    now[0] = 99.0
    assert _run(trigger, [6]) == [False]
    now[0] = 100.0  # elapsed == cooldown -> allowed
    assert _run(trigger, [6]) == [True]


def test_cooldown_does_not_disarm_when_blocked():
    now = [0.0]
    trigger = DriftTrigger(
        threshold=0.05, threshold_off=0.03, cooldown_s=100.0, clock=lambda: now[0]
    )
    assert _run(trigger, [6, 2]) == [True, False]  # fired, then re-armed
    now[0] = 50.0
    assert _run(trigger, [6]) == [False]  # blocked by cooldown, stays armed
    assert trigger.armed is True
    now[0] = 150.0
    assert _run(trigger, [6]) == [True]


def test_state_persists_across_instances(tmp_path: Path):
    state = tmp_path / "drift_trigger_state.json"
    first = DriftTrigger(threshold=0.05, threshold_off=0.03, state_path=state)
    assert _run(first, [6]) == [True]
    # New instance per cron tick: must stay disarmed.
    assert _run(DriftTrigger(threshold=0.05, threshold_off=0.03, state_path=state), [6]) == [False]
    assert _run(DriftTrigger(threshold=0.05, threshold_off=0.03, state_path=state), [2]) == [False]
    assert _run(DriftTrigger(threshold=0.05, threshold_off=0.03, state_path=state), [6]) == [True]
    payload = json.loads(state.read_text())
    assert payload["armed"] is False
    assert isinstance(payload["last_fire_ts"], float)


def test_cooldown_persists_across_instances(tmp_path: Path):
    state = tmp_path / "s.json"
    now = [1000.0]
    kw = {"threshold": 0.05, "cooldown_s": 60.0, "state_path": state, "clock": lambda: now[0]}
    assert _run(DriftTrigger(**kw), [6]) == [True]
    now[0] = 1030.0
    assert _run(DriftTrigger(**kw), [6]) == [False]
    now[0] = 1060.0
    assert _run(DriftTrigger(**kw), [6]) == [True]


def test_corrupt_state_file_raises(tmp_path: Path):
    state = tmp_path / "s.json"
    state.write_text("{not json")
    trigger = DriftTrigger(threshold=0.05, threshold_off=0.03, state_path=state)
    with pytest.raises(ValueError, match="state"):
        trigger.should_fire(_aeon(), _manifest_at(6))
