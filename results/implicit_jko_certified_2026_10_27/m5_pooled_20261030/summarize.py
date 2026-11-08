import json
import sys
from collections import defaultdict

R = [json.loads(line) for line in open(sys.argv[1])]


def cert(r):
    st = r.get("steps") or []
    return len(st) == 3 and all(s["status"] == "converged" for s in st)


def fail(r):
    st = r.get("steps") or []
    if not st:
        return str(r.get("status"))
    bad = [s for s in st if s["status"] != "converged"]
    return f"{bad[0]['status']}@step{len(st)}" if bad else f"short({len(st)})"


print("PRIMARY failures:")
for r in R:
    if r["gamma"] <= 30 and not cert(r):
        s = r["steps"][-1]
        print(r["grid_index"], (r["alpha"], r["gamma"], r["h"], r["seed"]), fail(r),
              "kkt/tol=%.3g log/tol=%.3g outer=%d innermax=%d" % (
                  s["kkt_residual"] / s["kkt_tol"], s["log_residual"] / s["log_tol"],
                  s["outer_iterations"], s["inner_max_iterations"]))
for lab, test in (("primary", lambda g: g <= 30), ("stress", lambda g: g >= 50)):
    c = defaultdict(int)
    for r in R:
        if test(r["gamma"]) and not cert(r):
            c[fail(r).split("@")[0]] += 1
    print(lab, "statuses", dict(c))
for key in ("alpha", "h", "seed"):
    for lab, test in (("primary", lambda g: g <= 30), ("stress", lambda g: g >= 50)):
        d = defaultdict(lambda: [0, 0])
        for r in R:
            if test(r["gamma"]):
                d[r[key]][0] += cert(r)
                d[r[key]][1] += 1
        print(lab, key, {k: f"{v[0]}/{v[1]}" for k, v in sorted(d.items())})
fac = []
for r in R:
    if r["gamma"] >= 50 and not cert(r):
        s = r["steps"][-1]
        if s["status"] in ("max_outer", "line_search"):
            fac.append((s["kkt_residual"] / s["kkt_tol"], s["log_residual"] / s["log_tol"]))
fac.sort()
if fac:
    print("stress max_outer/line_search kkt/tol min %.3g median %.3g max %.3g" % (
        fac[0][0], fac[len(fac) // 2][0], fac[-1][0]))
steps = [s for r in R for s in r["steps"] if "F_out" in s]
print("F_out<F_in on computed steps:", sum(s["F_out"] < s["F_in"] for s in steps), "/", len(steps))
print("non-decrease:", [(r["grid_index"], r["gamma"], i) for r in R
                        for i, s in enumerate(r["steps"]) if "F_out" in s and not s["F_out"] < s["F_in"]])
qs = [s for r in R for s in r["steps"] if s.get("Q_explicit") is not None and s.get("Q_final") is not None]
print("Q_final<=Q_explicit:", sum(s["Q_final"] <= s["Q_explicit"] for s in qs), "/", len(qs))
w = defaultdict(list)
for r in R:
    w[(r["platform"]["host"], r["gamma"])].append(sum(s.get("seconds") or 0 for s in r["steps"]))
for k in sorted(w):
    v = sorted(w[k])
    print("wall", k, "n=%d median %.0f s max %.0f s" % (len(v), v[len(v) // 2], v[-1]))
an = [r for r in R if (r["alpha"], r["gamma"], r["h"], r["seed"]) == (10.0, 10.0, 0.05, 500)][0]
print("anomaly", an["grid_index"], [(s["status"], s["inner_max_iterations"], s["inner_max_residual"]) for s in an["steps"]])
inc = [(r["grid_index"], r["gamma"], s["inner_max_iterations"]) for r in R for s in r["steps"]
       if s["status"] == "inner_not_converged"]
print("all inner_not_converged (idx, gamma, inner_max_iterations of successful calls):", inc)
