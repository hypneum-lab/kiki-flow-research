"""Is the default slow step dissipative in the strong-Turing regime? (2026-10-27)

Two checks behind ``docs/notes/2026-10-27-full-energy-reference-pilot.md``:

* ``step``: iterate the default ``FullJKOSolver`` (entropic JKO step of F,
  h = 0.05, reg = h^2) on the grid, without particles, from the Gibbs state of
  the potentials at alpha = 5 and gamma in {0, 5, 10, 20, 30}, for 30 steps,
  and count the steps where F increases;
* ``final``: F at the final states of the preregistered strong-Turing runs
  (``results/turing_strong_excess_2026_10_26/partial.jsonl``) next to the
  lowest F reached by the pilot reference
  (``results/full_energy_excess_2026_10_26/pilot.json``) and F at the Gibbs
  state of the potentials.

Outputs: ``results/full_energy_excess_2026_10_26/{dissipation_check.log,
final_state_energy.log}`` (stdout). Deterministic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver


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


ref = _load("experiments_full_energy_excess_2026_10_26", "full_energy_excess_2026_10_26.py")
STEP_CELLS = [(5.0, 0.0), (5.0, 5.0), (5.0, 10.0), (5.0, 20.0), (5.0, 30.0)]
FINAL_CELLS = [(2.0, 20.0), (5.0, 20.0), (5.0, 30.0), (5.0, 50.0), (10.0, 30.0), (5.0, 100.0)]
N_STEPS = 30
INCREASE_TOL = 1e-9
PREREG_RECORDS = Path("results/turing_strong_excess_2026_10_26/partial.jsonl")
PILOT = Path("results/full_energy_excess_2026_10_26/pilot.json")


def check_step() -> None:
    for cell in STEP_CELLS:
        f = ref.excess.problem(*cell)
        jko = FullJKOSolver(f_functional=f, h=0.05, support=ref._support())
        state = ref._state(ref.excess.gibbs_potentials(f))
        values = [f.value(state)]
        for _ in range(N_STEPS):
            state = jko.step(state)
            values.append(f.value(state))
        increases = sum(1 for a, b in zip(values, values[1:], strict=False) if b > a + INCREASE_TOL)
        print(
            f"{cell} F: start {values[0]:.3f} end {values[-1]:.3f} min {min(values):.3f} "
            f"max {max(values):.3f}; steps with F increase: {increases}/{N_STEPS}",
            flush=True,
        )


def check_final() -> None:
    records = [json.loads(x) for x in PREREG_RECORDS.read_text().splitlines()]
    pilot = json.loads(PILOT.read_text())["details"]["pilot_a"]
    f_low = {(r["alpha"], r["gamma"]): min(v["F"] for v in r["runs"].values()) for r in pilot}
    for cell in FINAL_CELLS:
        f = ref.excess.problem(*cell)
        sel = [r for r in records if (r["alpha"], r["gamma"]) == cell]
        values = [f.value(ref._state({n: np.asarray(v) for n, v in r["rho"].items()})) for r in sel]
        f_gibbs = f.value(ref._state(ref.excess.gibbs_potentials(f)))
        print(
            f"{cell} F(final sim) mean {np.mean(values):.3f} "
            f"[{min(values):.3f}..{max(values):.3f}] F_min approx {f_low[cell]:.3f} "
            f"F(Gibbs pot.) {f_gibbs:.3f} gap sim {np.mean([r['gap_bits'] for r in sel]):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    {"step": check_step, "final": check_final}[sys.argv[1]]()
