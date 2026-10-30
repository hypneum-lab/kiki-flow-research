"""Pool the M5 certify records of electron-server and macM3 (run only at the end).

Usage (stdlib only, run with any Python >= 3.10):
    python3 -I merge_certify.py ES_certify.jsonl MACM3_certify.jsonl OUT.jsonl
        [--drop-duplicates-from es|macm3]

Keeps only records of code 31feab6 (clean) with the M5 RUN_KEY fields, tags every
record with its host platform (ES records carry none; macM3 records already carry
``platform`` from certify_shard.py), checks that each of the 96 configurations of
the driver grid appears exactly once, and writes the pooled set. Exits non-zero on
a missing or duplicated configuration (a duplicate is only resolved by the explicit
``--drop-duplicates-from`` choice). Prints certification counts per stratum
(certified = exactly 3 steps, all ``converged``), descriptively.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

COMMIT = "31feab68cff3af26f7334b2e44d44abc7018bf61"
GAMMAS = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0]
ALPHAS = [2.0, 5.0, 10.0]
HS = [0.05, 0.025]
SEEDS = [500, 501]
GRID = [(a, g, h, s) for g in reversed(GAMMAS) for a in ALPHAS for h in HS for s in SEEDS]
ES_PLATFORM = {"host": "electron-server", "system": "Linux", "machine": "x86_64"}


def key_of(r: dict) -> tuple:
    if "steps" in r:
        return (float(r["alpha"]), float(r["gamma"]), float(r["h"]), int(r["seed"]))
    a, g, h, s = r["job"]  # job_error record
    return (float(a), float(g), float(h), int(s))


def load(path: str, host: str) -> list[dict]:
    out = []
    for n, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("git_commit") != COMMIT or r.get("git_dirty") is not False:
            print(f"skip {path}:{n} (code {r.get('git_commit')}, dirty={r.get('git_dirty')})")
            continue
        if r.get("n_steps") != 3 or r.get("delta") != 1e-5:
            print(f"skip {path}:{n} (RUN_KEY mismatch)")
            continue
        if host == "es":
            if "platform" in r:
                sys.exit(f"{path}:{n}: ES record unexpectedly carries a platform")
            r["platform"] = ES_PLATFORM
        elif r.get("platform", {}).get("host") != "macM3":
            sys.exit(f"{path}:{n}: macM3 record without macM3 platform")
        r["_src"] = host
        out.append(r)
    return out


def certified(r: dict) -> bool:
    st = r.get("steps") or []
    return len(st) == 3 and all(s.get("status") == "converged" for s in st)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("es")
    ap.add_argument("macm3")
    ap.add_argument("out")
    ap.add_argument("--drop-duplicates-from", choices=["es", "macm3"])
    a = ap.parse_args()
    recs = load(a.es, "es") + load(a.macm3, "macm3")
    by = defaultdict(list)
    for r in recs:
        by[key_of(r)].append(r)
    unknown = [k for k in by if k not in GRID]
    missing = [k for k in GRID if k not in by]
    dups = {k: [r["_src"] for r in v] for k, v in by.items() if len(v) > 1}
    if a.drop_duplicates_from and dups:
        for k in dups:
            kept = [r for r in by[k] if r["_src"] != a.drop_duplicates_from]
            by[k] = kept or by[k]
        print(f"dropped duplicates from {a.drop_duplicates_from}: {sorted(dups)}")
        dups = {k: [r["_src"] for r in v] for k, v in by.items() if len(v) > 1}
    ok = True
    for name, items in (("unknown", unknown), ("missing", missing), ("duplicated", dups)):
        if items:
            ok = False
            print(f"{name} ({len(items)}):", *(sorted(items) if not isinstance(items, dict)
                                             else sorted(items.items())), sep="\n  ")
    hosts = defaultdict(int)
    with open(a.out, "w") as fh:
        for i, k in enumerate(GRID):
            for r in by.get(k, []):
                hosts[r["_src"]] += 1
                r = {kk: v for kk, v in r.items() if kk != "_src"}
                r["grid_index"] = i
                fh.write(json.dumps(r) + "\n")
    print(f"written {a.out}: {sum(hosts.values())} records, per host {dict(hosts)}")
    for label, test in (("gamma<=30", lambda g: g <= 30), ("gamma>=50", lambda g: g >= 50)):
        ks = [k for k in GRID if test(k[1])]
        c = sum(1 for k in ks if by.get(k) and certified(by[k][0]))
        print(f"{label}: {c}/{len(ks)} certified (descriptive; missing counts as not)")
    print("CHECK", "OK: 96 configurations, each exactly once" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
