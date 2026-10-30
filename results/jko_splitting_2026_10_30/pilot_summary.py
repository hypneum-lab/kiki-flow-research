"""Timing summary of the splitting pilot and cost estimate of the confirmatory run.

Reads only ``status``, the sub-step statuses, ``wall`` and ``bracket_wall`` of
each record. Usage: ``python pilot_summary.py <pilot.jsonl> [cap_s] [mode]``
with mode ``central`` (a pilot timeout becomes one capped step, then the
trajectory stops) or ``upper`` (it converges at the cap on every step).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

CLASS = {
    0.0: (2.0, 0.0),
    5.0: (10.0, 5.0),
    10.0: (10.0, 5.0),
    20.0: (5.0, 30.0),
    30.0: (5.0, 30.0),
    50.0: (2.0, 70.0),
    70.0: (2.0, 70.0),
    100.0: (10.0, 100.0),
}
LADDER = (0.1, 0.05, 0.025, 0.0125)
M5_HS = (0.05, 0.025)
PILOT_CAP = 240.0
N_CELLS_PER_GAMMA = 6  # 3 alphas x 2 seeds


def n_steps(scheme: str, h: float) -> int:
    if h == LADDER[0]:
        return 1
    if h == LADDER[-1]:
        return 2
    return 2 if scheme == "implicit" else 3


def main() -> None:
    path = Path(sys.argv[1])
    cap = float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0  # noqa: PLR2004
    mode = sys.argv[3] if len(sys.argv) > 3 else "central"  # noqa: PLR2004
    with path.open() as fh:
        rows = [json.loads(line) for line in fh]
    t = {}
    print("| (alpha, gamma) | scheme | h | status | step wall (s) | bracket wall (s) | sub-steps |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["gamma"], r["alpha"], r["scheme"], -r["h"])):
        s = r["steps"][0]
        subs = ",".join(f"{x['part']}:{x['status']}" for x in s.get("substeps", []))
        bw = s.get("bracket_wall", 0.0)
        t[(r["alpha"], r["gamma"], r["scheme"], r["h"])] = (s["status"], s["wall"], bw)
        cell = f"({r['alpha']:g}, {r['gamma']:g})"
        print(
            f"| {cell} | {r['scheme']} | {r['h']:g} | {s['status']} "
            f"| {s['wall']:.1f} | {bw:.1f} | {subs} |"
        )
    low: dict[str, float] = defaultdict(float)
    high: dict[str, float] = defaultdict(float)
    for cls in CLASS.values():
        for scheme in ("implicit", "lie", "strang"):
            for h in LADDER:
                n = n_steps(scheme, h)
                status, wall, bw = t[(*cls, scheme, h)]
                br = bw if h in M5_HS else 0.0
                if status == "timeout":
                    lo, hi = PILOT_CAP, (n * cap if mode == "upper" else cap)
                elif status != "converged":
                    lo = hi = wall
                else:
                    lo = hi = n * (wall + br)
                low[scheme] += N_CELLS_PER_GAMMA * lo
                high[scheme] += N_CELLS_PER_GAMMA * hi
    print()
    total_lo, total_hi = sum(low.values()) / 3600, sum(high.values()) / 3600
    print(f"cap {cap:g} s, mode {mode}: {total_lo:.1f}-{total_hi:.1f} CPU-h")
    print(f"wall at 4 workers: {total_lo / 4:.1f}-{total_hi / 4:.1f} h")


if __name__ == "__main__":
    main()
