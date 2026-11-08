"""Paper-scale T2 run under the implicit default (2026-10-27).

Recomputation of ``experiments/paper_scale_jko_entropic_2026_10_25.py``
(grid 64, 10 000 particles, 1000 fast x 100 slow steps, h = 0.05, seeds 0-4,
MLX simulator on the CPU, reinjection) with the default slow step since
2026-10-27 (ERRATUM.md section 15): entropic JKO step of F, Sinkhorn reg = h^2,
coupling and Turing terms treated implicitly.

``run SEED`` writes ``results/paper_scale_implicit_2026_10_27/seed{SEED}/``:
``steps.jsonl`` (one line per slow step, appended as it completes: entropies,
F, halvings) and ``result.json`` at the end. A run interrupted mid-seed keeps
its completed steps; rerunning a seed whose ``result.json`` exists is a no-op.
``aggregate`` writes ``summary.json`` / ``summary.md`` next to the
entropic-explicit run of 2026-10-25 and the legacy run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

OUT_DIR = Path("results/paper_scale_implicit_2026_10_27")
EXPLICIT = Path("results/paper_scale_jko_entropic_2026_10_25/summary.json")
LEGACY_STATS = Path("paper_rigorous_maxiter/stats.json")
SEEDS = [0, 1, 2, 3, 4]
PARAMS = {"grid_size": 64, "n_particles": 10_000, "n_fast": 1000, "n_slow": 100}


def seed_dir(seed: int) -> Path:
    return OUT_DIR / f"seed{seed}"


def entropy_bits(rho: np.ndarray) -> float:
    r = np.clip(np.asarray(rho, dtype=float), 1e-12, None)
    return float(-(r * np.log2(r)).sum())


def run(seed: int, n_slow: int) -> None:
    import mlx.core as mx  # noqa: PLC0415

    from kiki_flow_core.species import CanonicalSpecies  # noqa: PLC0415
    from kiki_flow_core.state import FlowState  # noqa: PLC0415
    from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver  # noqa: PLC0415
    from kiki_flow_core.track2_paper.mlx_particle_simulator import (  # noqa: PLC0415
        MLXParticleSimulator,
    )
    from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop  # noqa: PLC0415
    from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy  # noqa: PLC0415

    out = seed_dir(seed)
    if (out / "result.json").exists():
        print(f"seed {seed}: already done", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    steps_path = out / "steps.jsonl"
    steps_path.write_text("")  # a seed restarts from scratch (particles are not checkpointed)
    grid = PARAMS["grid_size"]
    t0 = time.perf_counter()
    with mx.stream(mx.cpu):
        species = CanonicalSpecies()
        names = species.species_names()
        support = np.linspace(-2, 2, grid).reshape(-1, 1)
        f_func = T2FreeEnergy(
            species=species,
            potentials={n: np.zeros(grid) for n in names},
            prior={n: np.full(grid, 1.0 / grid) for n in names},
            turing_strength=0.1,
        )
        jko = FullJKOSolver(f_functional=f_func, h=0.05, support=support, nonseparable="implicit")
        counter = {"k": 0}

        def log_step(state: FlowState) -> FlowState:
            counter["k"] += 1
            with steps_path.open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "step": counter["k"],
                            "elapsed_s": time.perf_counter() - t0,
                            "F": f_func.value(state),
                            "halvings": jko.last_halvings,
                            "entropy_bits": {n: entropy_bits(state.rho[n]) for n in names},
                        }
                    )
                    + "\n"
                )
            return state

        sim = MLXParticleSimulator(
            species=species, n_particles=PARAMS["n_particles"], latent_dim=2, seed=seed
        )
        loop = MultiscaleLoop(
            sim=sim,
            jko=jko,
            n_fast=PARAMS["n_fast"],
            n_slow=n_slow,
            support=support,
            post_jko_hook=log_step,
        )
        manifest = loop.run(seed=seed)
    traj = manifest["trajectory"]
    result = {
        "seed": seed,
        "runtime_s": time.perf_counter() - t0,
        "n_slow_completed": manifest["n_slow_completed"],
        "reinject": manifest["reinject"],
        "scheme": jko.scheme,
        "nonseparable": jko.nonseparable,
        "reg": jko.reg,
        "entropy_bits_per_step": {n: [entropy_bits(st.rho[n]) for st in traj] for n in names},
        "final_rho": {n: [float(v) for v in traj[-1].rho[n]] for n in names},
    }
    (out / "result.json").write_text(json.dumps(result), encoding="utf-8")
    print(f"seed {seed}: {result['runtime_s']:.0f} s", flush=True)


def aggregate() -> dict[str, Any]:
    per_seed = {s: json.loads((seed_dir(s) / "result.json").read_text()) for s in SEEDS}
    names = list(per_seed[SEEDS[0]]["final_rho"])
    new = {n: [per_seed[s]["entropy_bits_per_step"][n][-1] for s in SEEDS] for n in names}
    max_bits = float(np.log2(PARAMS["grid_size"]))
    gaps = [max_bits - float(np.mean([new[n][i] for n in names])) for i in range(len(SEEDS))]
    explicit = json.loads(EXPLICIT.read_text()) if EXPLICIT.exists() else None
    legacy = json.loads(LEGACY_STATS.read_text()) if LEGACY_STATS.exists() else None
    summary: dict[str, Any] = {
        "params": PARAMS,
        "seeds": SEEDS,
        "scheme": "jko_entropic, nonseparable implicit (default), reg = h^2, reinject=True",
        "runtime_s": {s: per_seed[s]["runtime_s"] for s in SEEDS},
        "entropy_bits": {
            n: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for n, v in new.items()
        },
        "gap_bits": {"mean": float(np.mean(gaps)), "sd": float(np.std(gaps, ddof=1))},
        "entropic_explicit_2026_10_25": explicit["gap_bits"] if explicit else None,
        "legacy_gap_bits": float(
            max_bits
            - np.mean([legacy["aggregate"][n]["entropy_bits_mean"] for n in legacy["aggregate"]])
        )
        if legacy
        else None,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = [
        "# Paper-scale run under the implicit default (2026-10-27)",
        "",
        "Grid 64, 10 000 particles, 1000 fast x 100 slow steps, h = 0.05, seeds 0-4, MLX "
        "on the CPU; entropic JKO step of F with implicit coupling and Turing terms "
        "(ERRATUM §15), reg = h², reinjection. Flat potentials: the Gibbs state is uniform "
        "(gap 0).",
        "",
        "| species | final entropy (bits, mean ± sd) |",
        "|---|---|",
    ]
    for n, st in summary["entropy_bits"].items():
        lines.append(f"| {n} | {st['mean']:.4f} ± {st['sd']:.4f} |")
    exp_gap = summary["entropic_explicit_2026_10_25"]
    lines += [
        "",
        f"Gap log2(64) − mean H: {summary['gap_bits']['mean']:.4f} ± "
        f"{summary['gap_bits']['sd']:.4f} bits (entropic explicit 2026-10-25: "
        f"{exp_gap['mean']:.4f}; legacy: {summary['legacy_gap_bits']:.4f}). Runtime per seed: "
        + ", ".join(f"{v:.0f} s" for v in summary["runtime_s"].values())
        + ".",
        "",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("seed", type=int)
    r.add_argument("--n-slow", type=int, default=PARAMS["n_slow"])
    sub.add_parser("aggregate")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args.seed, args.n_slow)
    else:
        print(json.dumps(aggregate()["gap_bits"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
