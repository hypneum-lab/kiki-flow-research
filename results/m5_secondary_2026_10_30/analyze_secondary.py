"""Table of the M5 secondary analysis (descriptive; never replaces the primary verdict).

Usage: analyze_secondary.py secondary.jsonl primary1.jsonl [primary2.jsonl ...]

Remaining-failure kinds (last step of a non-certified configuration):
* budget      -- the budget was hit while the residual was still decreasing
                 (inner: a prox call used all inner_n_iter iterations; outer:
                 max_outer with the outer KKT/log residual still falling, i.e.
                 last value < STALL_RATIO x the value at 75 % of the history);
* stagnation  -- outer residual flat over the last quarter of the history
                 (ratio >= STALL_RATIO), or line search exhausted (no L <= l_max
                 gives sufficient decrease) with Q decreases above rounding;
* precision_floor -- an inner prox stopped on delta < inner_tol (1e-13) before its
                 budget with its certificate above tolerance, or a line search
                 exhausted while the median of the last 5 accepted decreases of Q
                 is below the accuracy of Q itself: max(ROUND_FACTOR eps |Q|,
                 inner_max_residual * MAX_COST / (2h)) (the transport values in Q
                 are certified only up to the marginal residual times the largest
                 cost, MAX_COST = 16 on the support [-2, 2]).
"same path as primary": every recorded step of the re-run has the same status,
outer_iterations and Q_final (bit for bit) as the primary record, i.e. the larger
budget was never binding (meaningful when the arithmetic matches: ES records).
The classification uses only recorded quantities (status, inner failure info,
outer history); it is descriptive.
"""
import json
import sys

EPS = 2.220446049250313e-16
STALL_RATIO = 0.5
ROUND_FACTOR = 1e3
MAX_COST = 16.0


def primary_index(paths):
    idx = {}
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if "steps" in r:
                idx[(r["alpha"], r["gamma"], r["h"], r["seed"])] = (
                    r["steps"], (r.get("platform") or {}).get("host", "electron-server"))
            else:
                idx[tuple(r["job"])] = ([{"status": "job_error"}], "?")
    return idx


def outer_trend(hist):
    oh = hist["outer_history"]
    if len(oh) < 8:
        return None
    k = int(0.75 * len(oh))
    a = max(oh[k]["kkt"] / 1e-6, oh[k]["log"] / 1e-5)
    b = max(oh[-1]["kkt"] / 1e-6, oh[-1]["log"] / 1e-5)
    return b / a if a > 0 else None


def classify(step, hist):
    st = step["status"]
    if st == "converged":
        return "certified", ""
    if st == "inner_not_converged":
        f = hist.get("inner_failure") or {}
        if not f:
            return "unknown", "no inner failure recorded"
        detail = (f"prox call at outer {f['outer_iteration']}: {f['iterations']}/{f['n_iter_budget']} it, "
                  f"L1 {f['marginal_residual']:.2e} (tol 1e-10), log {f['marginal_log_residual']:.2e} "
                  f"(tol 1e-9), last_delta {f['last_delta']:.2e}")
        if f["iterations"] >= f["n_iter_budget"]:
            return "budget (inner)", detail
        return "precision_floor (inner)", detail
    if st == "max_outer":
        r = outer_trend(hist)
        oh = hist["outer_history"][-1]
        detail = (f"kkt {oh['kkt']:.2e}/1e-6, log {oh['log']:.2e}/1e-5, "
                  f"last-quarter residual ratio {r:.2f}" if r is not None else "short history")
        if r is not None and r >= STALL_RATIO:
            return "stagnation (outer)", detail
        return "budget (outer)", detail
    if st == "line_search":
        oh = [e for e in hist["outer_history"] if e.get("Q_next") is not None]
        last = hist["outer_history"][-1]
        dq = sorted(e["Q"] - e["Q_next"] for e in oh[-5:])
        floor = max(ROUND_FACTOR * EPS * abs(last["Q"]),
                    step["inner_max_residual"] * MAX_COST / (2.0 * step["h"]))
        med = dq[len(dq) // 2] if dq else float("nan")
        detail = (f"outer {len(hist['outer_history'])}, kkt {last['kkt']:.2e}/1e-6, log "
                  f"{last['log']:.2e}/1e-5, median last-5 dQ {med:.1e} vs Q accuracy {floor:.1e}")
        if dq and med <= floor:
            return "precision_floor (line search)", detail
        return "stagnation (line search)", detail
    return st, ""


PENDING = []


def main():
    args = sys.argv[1:]
    while "--pending" in args:
        i = args.index("--pending")
        a, g, h, s_ = args[i + 1].split(",")
        PENDING.append((float(a), float(g), float(h), int(s_)))
        del args[i:i + 2]
    rows = [json.loads(l) for l in open(args[0]) if l.strip()]
    prim = primary_index(args[1:])
    counts = {}
    rows.sort(key=lambda r: (-r["config"]["gamma"], r["config"]["alpha"], -r["config"]["h"],
                             r["config"]["seed"]))
    print("| alpha | gamma | h | seed | primary | secondary steps | certified (larger budget) | "
          "remaining failure | detail | same path as primary | wall (s) | host |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["config"]
        key = (c["alpha"], c["gamma"], c["h"], c["seed"])
        steps = r["record"].get("steps", [])
        if steps:
            kind, detail = classify(steps[-1], r["histories"][len(steps) - 1])
        else:
            kind, detail = "job_error", r["record"].get("error", "")[:120]
        psteps, phost = prim.get(key, ([{"status": "?"}], "?"))
        same = (len(psteps) == len(steps) and all(
            a["status"] == b["status"] and a.get("outer_iterations") == b.get("outer_iterations")
            and a.get("Q_final") == b.get("Q_final") for a, b in zip(psteps, steps)))
        rhost = r["platform"]["host"]
        mac_pair = phost == "macM3" and r["platform"]["system"] == "Darwin"
        es_pair = phost != "macM3" and (r["platform"].get("arith_env") or {}).get("OPENBLAS_CORETYPE") == "NEHALEM"
        same_s = ("yes" if same else "no") if (mac_pair or es_pair) else "n/a (other arithmetic)"
        stratum = "criterion" if c["gamma"] <= 30 else "stress"
        counts.setdefault((stratum, "certified" if r["certified"] else kind.split(" (")[0]), []).append(key)
        print(f"| {c['alpha']:g} | {c['gamma']:g} | {c['h']:g} | {c['seed']} | "
              f"{','.join(s['status'] for s in psteps)} | {','.join(r['statuses']) or '-'} | "
              f"{'yes' if r['certified'] else 'no'} | {'' if r['certified'] else kind} | "
              f"{'' if r['certified'] else detail} | {same_s} | {r['wall_seconds']:.0f} | "
              f"{r['platform']['host']} |")
    for p in PENDING:
        print(f"| {p[0]:g} | {p[1]:g} | {p[2]:g} | {p[3]} | "
              f"{','.join(s['status'] for s in prim.get(p, ([{'status': '?'}], ''))[0])} | "
              "in progress | in progress | | | | | macM3 |")
    n = len(rows)
    k = sum(r["certified"] for r in rows)
    print(f"\n{k}/{n} completed re-runs certified with the larger budget"
          f" ({len(PENDING)} still running).\n")
    print("| stratum | outcome | n |")
    print("|---|---|---|")
    for (st, kd), v in sorted(counts.items()):
        print(f"| {st} | {kd} | {len(v)} |")


if __name__ == "__main__":
    main()
