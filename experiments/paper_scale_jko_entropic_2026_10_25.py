"""Paper-scale T2 run under the new default slow step (2026-10-25).

Same setup as the paper-scale rigorous runs (``paper_rigorous_maxiter/``:
grid 64, 10 000 particles, 1000 fast x 100 slow steps, h = 0.05, seeds 0-4,
MLX simulator on the CPU) but with the defaults switched on 2026-10-25
(ERRATUM.md section 13): ``FullJKOSolver`` entropic JKO step of F with
Sinkhorn reg = h^2, ``MultiscaleLoop(reinject=True)``.

``run SEED`` runs one seed with the exact construction of ``run_paper``
(``use_w2_prox=True``, defaults) and writes
``results/paper_scale_jko_entropic_2026_10_25/seed{SEED}/result.json`` (final
rho and per-species entropy at every slow step, runtime); ``aggregate``
reads the five seeds and writes ``summary.json`` and
``summary.md`` with the per-species final entropies next to the legacy run
(``paper_rigorous_maxiter/stats.json``). One process per seed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

OUT_DIR = Path("results/paper_scale_jko_entropic_2026_10_25")
LEGACY_STATS = Path("paper_rigorous_maxiter/stats.json")
SEEDS = [0, 1, 2, 3, 4]
PARAMS = {"grid_size": 64, "n_particles": 10_000, "n_fast": 1000, "n_slow": 100}


def seed_dir(seed: int) -> Path:
    return OUT_DIR / f"seed{seed}"


def run(seed: int, n_slow: int) -> None:
    import mlx.core as mx  # noqa: PLC0415

    from kiki_flow_core.species import CanonicalSpecies  # noqa: PLC0415
    from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver  # noqa: PLC0415
    from kiki_flow_core.track2_paper.mlx_particle_simulator import (  # noqa: PLC0415
        MLXParticleSimulator,
    )
    from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop  # noqa: PLC0415
    from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy  # noqa: PLC0415

    grid = PARAMS["grid_size"]
    t0 = time.perf_counter()
    with mx.stream(mx.cpu):
        # Same construction as run_paper(use_w2_prox=True) with its defaults.
        species = CanonicalSpecies()
        names = species.species_names()
        support = np.linspace(-2, 2, grid).reshape(-1, 1)
        f_func = T2FreeEnergy(
            species=species,
            potentials={n: np.zeros(grid) for n in names},
            prior={n: np.full(grid, 1.0 / grid) for n in names},
            turing_strength=0.1,
            turing_smoothing=None,
        )
        # As run on 2026-10-25: explicit non-separable terms (ERRATUM section 15).
        jko = FullJKOSolver(f_functional=f_func, h=0.05, support=support, nonseparable="explicit")
        sim = MLXParticleSimulator(
            species=species, n_particles=PARAMS["n_particles"], latent_dim=2, seed=seed
        )
        loop = MultiscaleLoop(
            sim=sim, jko=jko, n_fast=PARAMS["n_fast"], n_slow=n_slow, support=support
        )
        manifest = loop.run(seed=seed)
    elapsed = time.perf_counter() - t0
    traj = manifest["trajectory"]
    out = {
        "seed": seed,
        "runtime_s": elapsed,
        "n_slow_completed": manifest["n_slow_completed"],
        "reinject": manifest["reinject"],
        "scheme": jko.scheme,
        "reg": jko.reg,
        "entropy_bits_per_step": {n: [entropy_bits(st.rho[n]) for st in traj] for n in names},
        "final_rho": {n: [float(v) for v in traj[-1].rho[n]] for n in names},
    }
    seed_dir(seed).mkdir(parents=True, exist_ok=True)
    (seed_dir(seed) / "result.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"seed {seed}: {elapsed:.0f} s", flush=True)


def entropy_bits(rho: np.ndarray) -> float:
    r = np.clip(np.asarray(rho, dtype=float), 1e-12, None)
    return float(-(r * np.log2(r)).sum())


def aggregate() -> dict[str, Any]:
    per_seed = {s: json.loads((seed_dir(s) / "result.json").read_text()) for s in SEEDS}
    names = list(per_seed[SEEDS[0]]["final_rho"])
    new = {n: [per_seed[s]["entropy_bits_per_step"][n][-1] for s in SEEDS] for n in names}
    legacy = json.loads(LEGACY_STATS.read_text()) if LEGACY_STATS.exists() else None
    max_bits = float(np.log2(PARAMS["grid_size"]))
    summary: dict[str, Any] = {
        "params": PARAMS,
        "seeds": SEEDS,
        "scheme": "jko_entropic (default), reg = h^2, reinject=True",
        "runtime_s": {s: per_seed[s]["runtime_s"] for s in SEEDS},
        "first_step_entropy_bits": {
            n: float(np.mean([per_seed[s]["entropy_bits_per_step"][n][0] for s in SEEDS]))
            for n in names
        },
        "entropy_bits": {
            n: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for n, v in new.items()
        },
        "gap_bits": {
            "mean": float(max_bits - np.mean([np.mean(v) for v in new.values()])),
        },
        "legacy": None,
    }
    if legacy is not None:
        summary["legacy"] = {
            "source": str(LEGACY_STATS),
            "entropy_bits": {
                n: legacy["aggregate"][n]["entropy_bits_mean"] for n in legacy["aggregate"]
            },
            "gap_bits": float(
                max_bits
                - np.mean(
                    [legacy["aggregate"][n]["entropy_bits_mean"] for n in legacy["aggregate"]]
                )
            ),
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = [
        "# Paper-scale run under the entropic JKO default (2026-10-25)",
        "",
        "Grid 64, 10 000 particles, 1000 fast x 100 slow steps, h = 0.05, seeds 0-4, MLX "
        "simulator on the CPU, `FullJKOSolver` default (entropic JKO step of F, reg = h^2), "
        "`MultiscaleLoop(reinject=True)`. Final entropy per species (bits, mean ± sd over "
        "seeds) next to the legacy rigorous run `paper_rigorous_maxiter/` (legacy split, "
        "epsilon 0.01, no reinjection; seed-mean only).",
        "",
        "| species | entropic default | legacy rigorous |",
        "|---|---|---|",
    ]
    for n, st in summary["entropy_bits"].items():
        old = summary["legacy"]["entropy_bits"].get(n) if summary["legacy"] else None
        lines.append(
            f"| {n} | {st['mean']:.4f} ± {st['sd']:.4f} | {old:.4f} |"
            if old is not None
            else f"| {n} | {st['mean']:.4f} ± {st['sd']:.4f} | — |"
        )
    gap_old = summary["legacy"]["gap_bits"] if summary["legacy"] else float("nan")
    lines += [
        "",
        f"Gap log2(64) − mean H: {summary['gap_bits']['mean']:.4f} bits (legacy "
        f"{gap_old:.4f}). Runtime per seed: "
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
