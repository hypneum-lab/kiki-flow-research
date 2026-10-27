"""Timed probe for the recomputations under the implicit default (2026-10-27).

Times a few slow steps of each recomputation on electron-server and
extrapolates a wall-clock ETA per stage, before anything is launched:

* (a) sweep of PR #46: one run of ``N_PROBE_SLOW`` slow steps per probed
  (alpha, gamma) couple, scaled to 30 slow steps, summed over the 33 couples x
  10 seeds + 20 sensitivity runs (each couple timed at its nearest probed gamma);
* (b) paper scale: ``N_PROBE_PAPER`` slow steps of seed 0, scaled to 100;
* (c) strong-Turing grid: alpha in {2, 5, 10} x gamma in {20, 30, 50, 70, 100}
  plus alpha = 0 controls, 10 seeds, from the probed gammas.

Prints one JSON line per measurement and a final ETA table; wall time assumes
``WORKERS`` parallel processes. Writes ``results/implicit_eta_probe_2026_10_27/
probe.json``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType


def _load(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    path = Path("experiments") / filename
    if not path.exists():
        path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


imp = _load("experiments_implicit_sweep_2026_10_27", "implicit_sweep_2026_10_27.py")
paper = _load("experiments_paper_scale_implicit_2026_10_27", "paper_scale_implicit_2026_10_27.py")

N_PROBE_SLOW = 3
N_PROBE_PAPER = 2
WORKERS = 6
PROBE_GAMMAS = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 70.0, 100.0]
PROBE_ALPHA = 10.0
OUT = Path("results/implicit_eta_probe_2026_10_27/probe.json")


def nearest(g: float) -> float:
    return min(PROBE_GAMMAS, key=lambda p: abs(p - g))


def main() -> int:
    per_run = {}
    for g in PROBE_GAMMAS:
        t = time.perf_counter()
        imp.run_couple(PROBE_ALPHA, g, 0, imp.MAIN_REG, n_slow=N_PROBE_SLOW)
        dt = time.perf_counter() - t
        per_run[g] = dt * imp.sweep.base.N_SLOW / N_PROBE_SLOW
        print(
            json.dumps({"stage": "a/c", "gamma": g, "probe_s": dt, "run_s_est": per_run[g]}),
            flush=True,
        )
    jobs_a = imp.sweep.jobs(imp.SEEDS)
    cpu_a = sum(per_run[nearest(g)] for _, g, _, _ in jobs_a)
    tmp = paper.OUT_DIR
    paper.OUT_DIR = Path("results/implicit_eta_probe_2026_10_27/paper_tmp")
    t = time.perf_counter()
    paper.run(0, N_PROBE_PAPER)
    dt_paper = time.perf_counter() - t
    paper.OUT_DIR = tmp
    seed_s = dt_paper * paper.PARAMS["n_slow"] / N_PROBE_PAPER
    print(json.dumps({"stage": "b", "probe_s": dt_paper, "seed_s_est": seed_s}), flush=True)
    cells_c = [(a, g) for g in (20.0, 30.0, 50.0, 70.0, 100.0) for a in (0.0, 2.0, 5.0, 10.0)]
    cpu_c = sum(per_run[nearest(g)] for _, g in cells_c) * 10
    eta = {
        "a_runs": len(jobs_a),
        "a_cpu_h": cpu_a / 3600,
        "a_wall_h": cpu_a / 3600 / WORKERS,
        "b_seed_h": seed_s / 3600,
        "b_wall_h": seed_s / 3600,  # 5 seeds in parallel on 5 workers
        "c_runs": len(cells_c) * 10,
        "c_cpu_h": cpu_c / 3600,
        "c_wall_h": cpu_c / 3600 / WORKERS,
        "per_run_s": per_run,
        "note": "single-process probe on an idle host; parallel runs share the CPU",
    }
    eta["total_wall_h"] = eta["a_wall_h"] + eta["b_wall_h"] + eta["c_wall_h"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(eta, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in eta.items()
                if k != "per_run_s"
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
