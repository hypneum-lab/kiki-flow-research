"""Run a tail shard of the M5 ``certify`` grid on macM3 (outside the repository).

Imports the unchanged driver ``experiments/implicit_jko_certified_validation_2026_10_27``
and calls its own ``_guarded(certify_job, ...)`` on a filtered list of the same
(alpha, gamma, h, seed) tuples: no parameter, tolerance or budget is changed here.

Order: the driver's ``run_certify`` builds the grid as
``for g in reversed(GAMMAS) for a in ALPHAS for h in HS for s in INIT_SEEDS`` and
submits it in that order to a FIFO pool, so electron-server processes index 0
first and index 95 last. This shard takes indices ``split .. 95`` and walks them
in reverse (95 first). Jobs are submitted lazily (at most ``workers`` in flight)
and ``split`` is re-read from ``split.txt`` before each submission, so the split
can be moved up without a restart (already-running jobs always finish).

Records are appended to ``<out>/certify.jsonl`` exactly as the driver writes them
(same record, plus a ``platform`` provenance field and ``grid_index``); resume skips tuples already present with the same
RUN_KEY, as the driver does.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import implicit_jko_certified_validation_2026_10_27 as drv

EXPECTED_COMMIT = "31feab68cff3af26f7334b2e44d44abc7018bf61"


def grid() -> list[tuple[float, float, float, int]]:
    """The driver's certify grid, in the driver's submission order."""
    return [
        (a, g, h, s)
        for g in reversed(drv.GAMMAS)
        for a in drv.ALPHAS
        for h in drv.HS
        for s in drv.INIT_SEEDS
    ]


def read_split(path: Path, default: int) -> int:
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True, help="lowest grid index taken here")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split-file", type=Path, required=True)
    ap.add_argument("--rule-sha", required=True, help="SHA of the pre-registered rule tag")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exclude-jsonl", type=Path, action="append", default=[],
                    help="records computed elsewhere (e.g. electron-server): skip their configs")
    args = ap.parse_args()

    key = drv.RUN_KEY
    if key["git_commit"] != EXPECTED_COMMIT or key["git_dirty"] is not False:
        print(f"refusing: code identity {key['git_commit']} dirty={key['git_dirty']}")
        return 2
    jobs = grid()
    assert len(jobs) == 96 and len(set(jobs)) == 96
    path = args.out / "certify.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if drv._same_run(r) and "steps" in r:
                done.add((r["alpha"], r["gamma"], r["h"], r["seed"]))
    for ex in args.exclude_jsonl:
        for line in ex.read_text().splitlines():
            r = json.loads(line)
            if drv._same_run(r) and "steps" in r:
                k = (r["alpha"], r["gamma"], r["h"], r["seed"])
                if k in done:
                    print(f"refusing: {k} present both here and in {ex}")
                    return 3
                done.add(k)
    order = [i for i in range(95, -1, -1) if jobs[i] not in done]
    if args.dry_run:
        for i in order:
            if i >= args.split:
                print(i, jobs[i])
        return 0
    if not args.split_file.exists():
        args.split_file.write_text(f"{args.split}\n")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"rule_sha={args.rule_sha} split={args.split} RUN_KEY={json.dumps(key)}", flush=True)
    plat = {
        "host": "macM3",
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "rule_tag": "prereg-m5-certify-20261030",
        "rule_tag_object": args.rule_sha,
    }
    queue = list(order)
    n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool, path.open("a") as fh:
        running: dict = {}
        while True:
            split = read_split(args.split_file, args.split)
            while queue and len(running) < args.workers and queue[0] >= split:
                i = queue.pop(0)
                running[pool.submit(drv._guarded, drv.certify_job, *jobs[i])] = i
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                i = running.pop(fut)
                r = fut.result()
                r["platform"] = plat  # provenance only, outside RUN_KEY
                r["grid_index"] = i
                fh.write(json.dumps(r, default=float) + "\n")
                fh.flush()
                n += 1
                st = [s["status"] for s in r.get("steps", [])] or r.get("status")
                print(f"[{n}] idx={i} {jobs[i]} {st}", flush=True)
    print("shard done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
