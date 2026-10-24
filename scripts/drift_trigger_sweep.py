"""Sweep DriftTrigger (threshold_on, threshold_off, cooldown) on synthetic drift traces.

Trace model (one tick per nightly cron run): the background new-concept
rate is an AR(1) process hovering just under the legacy threshold
(mean 0.04, stationary std ~0.012), so it crosses 0.05 often. Drift events
are injected every ~45-75 ticks: a 3-tick linear ramp up to a plateau in
[0.08, 0.15] held for 7-14 ticks.

Two feedback modes:
- open-loop: a fire does not lower the rate (worst case for chattering);
- closed-loop: a fire consolidates the event, i.e. its excess novelty is
  absorbed and the rate falls back to the background for the rest of it.

Metrics per config, averaged over seeds: fires per year, spurious fires
(outside [onset, end + 3]), in-event fires per event, event recall, and
detection delay in ticks (first fire minus onset, detected events only).

Usage: PYTHONPATH=. uv run python scripts/drift_trigger_sweep.py [--seeds 20]
Prints Markdown tables on stdout; writes nothing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from kiki_flow_core.track1_perf.trigger_heuristic import DriftTrigger

N_TICKS = 365
TICK_S = 86_400.0
BASE_MEAN = 0.04
AR_PHI = 0.6
AR_SIGMA = 0.0096  # stationary std = sigma / sqrt(1 - phi^2) ~ 0.012
RAMP = 3
GRACE = 3


@dataclass(frozen=True)
class Event:
    onset: int
    end: int


@dataclass(frozen=True)
class Trace:
    background: np.ndarray
    excess: np.ndarray
    events: list[Event]


def make_trace(seed: int) -> Trace:
    rng = np.random.default_rng(seed)
    noise = np.zeros(N_TICKS)
    for t in range(1, N_TICKS):
        noise[t] = AR_PHI * noise[t - 1] + rng.normal(0.0, AR_SIGMA)
    background = np.clip(BASE_MEAN + noise, 0.0, 1.0)
    excess = np.zeros(N_TICKS)
    events: list[Event] = []
    t = int(rng.integers(20, 40))
    while t + RAMP + 14 < N_TICKS:
        level = float(rng.uniform(0.08, 0.15))
        hold = int(rng.integers(7, 15))
        for k in range(RAMP):
            excess[t + k] = (level - BASE_MEAN) * (k + 1) / RAMP
        excess[t + RAMP : t + RAMP + hold] = level - BASE_MEAN
        events.append(Event(onset=t, end=t + RAMP + hold - 1))
        t += RAMP + hold + int(rng.integers(45, 76))
    return Trace(background, excess, events)


def simulate(trace: Trace, on: float, off: float | None, cd_ticks: int, closed: bool) -> list[int]:
    now = [0.0]
    trigger = DriftTrigger(
        threshold=on,
        threshold_off=off,
        cooldown_s=cd_ticks * TICK_S if cd_ticks else None,
        clock=lambda: now[0],
    )
    excess = trace.excess.copy()
    fires = []
    for t in range(N_TICKS):
        now[0] = t * TICK_S
        rate = min(1.0, float(trace.background[t] + excess[t]))
        if trigger.update(rate):
            fires.append(t)
            if closed:
                for ev in trace.events:
                    if ev.onset <= t <= ev.end:
                        excess[t + 1 : ev.end + 1] = 0.0
    return fires


def score(fires: list[int], events: list[Event]) -> tuple[int, int, int, list[int]]:
    in_event: set[int] = set()
    detected = 0
    delays = []
    for ev in events:
        hits = [f for f in fires if ev.onset <= f <= ev.end + GRACE]
        in_event.update(hits)
        if hits:
            detected += 1
            delays.append(hits[0] - ev.onset)
    return len(fires), len(fires) - len(in_event), detected, delays


CONFIGS: list[tuple[float, float | None, int]] = [
    (0.05, None, 0),  # pre-2026-10-24 default (memoryless)
    (0.05, None, 3),
    (0.05, None, 7),
    (0.05, 0.045, 0),
    (0.05, 0.04, 0),
    (0.05, 0.03, 0),
    (0.06, None, 0),
    (0.06, 0.05, 0),
    (0.06, 0.04, 0),
    (0.06, 0.03, 0),
    (0.07, None, 0),
    (0.07, 0.05, 0),  # current default
    (0.07, 0.04, 0),
    (0.08, None, 0),
    (0.08, 0.05, 0),
    (0.08, 0.04, 0),
    (0.06, 0.05, 3),
    (0.07, 0.05, 3),
]

HEADER = (
    "| on | off | cooldown (ticks) | fires/yr | spurious/yr | in-event fires/event "
    "| recall | delay mean | delay max |"
)


def table(traces: list[Trace], closed: bool) -> None:
    n = len(traces)
    n_events = sum(len(tr.events) for tr in traces)
    print(f"\n### {'closed' if closed else 'open'}-loop ({n} seeds, {n_events} events)\n")
    print(HEADER)
    print("|---|---|---|---|---|---|---|---|---|")
    for on, off, cd in CONFIGS:
        tot = spur = det = 0
        delays: list[int] = []
        for tr in traces:
            f, s, d, dl = score(simulate(tr, on, off, cd, closed), tr.events)
            tot += f
            spur += s
            det += d
            delays += dl
        mean_d = f"{np.mean(delays):.2f}" if delays else "n/a"
        max_d = str(max(delays)) if delays else "n/a"
        off_s = "None" if off is None else f"{off:.3f}"
        print(
            f"| {on:.2f} | {off_s} | {cd} | {tot / n:.1f} | {spur / n:.1f} | "
            f"{(tot - spur) / n_events:.2f} | {det / n_events:.2f} | {mean_d} | {max_d} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DriftTrigger hysteresis sweep")
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args(argv)
    traces = [make_trace(s) for s in range(args.seeds)]
    print(f"{args.seeds} seeds x {N_TICKS} daily ticks, background mean {BASE_MEAN}")
    table(traces, closed=False)
    table(traces, closed=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
