"""Sinkhorn-regularisation sensitivity of the rigorous specialisation gap.

Follow-up to ERRATUM.md section 9: until 2026-10-24 ``FullJKOSolver.epsilon``
was not forwarded, so the rigorous specialisation sweeps ran with Sinkhorn
reg = h = 0.05 instead of the configured 0.01, and the rerun at the forwarded
0.01 moved the gap by at most 0.06 bits. This driver measures the gap at the
two key points of the 72-point grid, (10, 1, 0) and (8, 0.5, 5), for
epsilon in {0.01, 0.02, 0.05} on the rigorous path, with the alpha = 0
controls at the same (beta, gamma) and the fast path (no prox, independent
of epsilon) as references. Same problem, grid, particle count and step
counts as ``experiments/rigorous_specialisation_2026_10_24.py`` (``base``);
the solver is built by ``base.make_jko(..., epsilon=eps)``.

Per (epsilon, point, seed) it records the final per-species entropies
(bits), the gap ``log2(16) - mean H``, the paired specialisation
``S = H(alpha = 0) - H`` and ``KL(rigorous || fast)`` summed over the four
species (nats) at the final slow step. That KL is on the sweep's reduced
setup (500 particles, 30 slow steps, grid 16), not the paper-scale run of
``paper_rigorous_fixedeps/``.

Outputs (new files): ``results/eps_sweep_2026_10_25/sweep.json`` and
``summary.md``. MLX is pinned to the CPU device in ``main``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import mlx.core as mx
import numpy as np

from kiki_flow_core.track2_paper.mlx_particle_simulator import MLXParticleSimulator
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop


def _load_base() -> ModuleType:
    name = "experiments_rigorous_specialisation_2026_10_24"
    if name in sys.modules:
        return sys.modules[name]
    path = Path("experiments/rigorous_specialisation_2026_10_24.py")
    if not path.exists():  # imported from the tests, cwd may differ
        path = Path(__file__).with_name("rigorous_specialisation_2026_10_24.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()

Point = tuple[float, float, float]

EPSILONS = [0.01, 0.02, 0.05]
SEEDS = [0, 1, 2, 3, 4]
KEY_POINTS: list[Point] = [(10.0, 1.0, 0.0), (8.0, 0.5, 5.0)]
OUT_DIR = Path("results/eps_sweep_2026_10_25")
# Committed macM1 sweeps, read (if present) for the cross-host reference rows.
REF_PREFIX = Path("paper/rigorous_specialisation_2026_10_24/summary.json")
REF_FIXED = Path("paper/rigorous_specialisation_wide_2026_10_24_reg/sweep_wide_records.json")
RHO_FLOOR = 1e-12


def output_paths() -> dict[str, Path]:
    return {
        "sweep": OUT_DIR / "sweep.json",
        "summary": OUT_DIR / "summary.md",
        "partial": OUT_DIR / "partial.jsonl",
    }


def points() -> list[Point]:
    """Key points, then their alpha = 0 controls at the same (beta, gamma)."""
    controls = [(0.0, b, g) for _, b, g in KEY_POINTS]
    return KEY_POINTS + [c for c in controls if c not in KEY_POINTS]


def run_point(
    p: Point, rigorous: bool, seed: int, epsilon: float | None, n_slow: int = base.N_SLOW
) -> dict[str, np.ndarray]:
    """``base.run_one`` with the epsilon flag, returning the final rho per species."""
    species, support, f_func = base.problem(p[0], p[2])
    jko = base.make_jko(f_func, support, p[1], rigorous, epsilon=epsilon)
    sim = MLXParticleSimulator(
        species=species, n_particles=base.N_PARTICLES, latent_dim=2, seed=seed
    )
    loop = MultiscaleLoop(sim=sim, jko=jko, n_fast=base.N_FAST, n_slow=n_slow, support=support)
    final = loop.run(seed=seed)["trajectory"][-1]
    return {
        n: np.clip(np.asarray(rho, dtype=np.float64), RHO_FLOOR, None)
        for n, rho in final.rho.items()
    }


def entropy_bits(rho: np.ndarray) -> float:
    return float(-(rho * np.log2(rho)).sum())


def kl_nats(p: np.ndarray, q: np.ndarray) -> float:
    return float((p * np.log(p / q)).sum())


def key(path: str, epsilon: float | None, seed: int, p: Point) -> str:
    return f"{path}|{epsilon}|{seed}|{p[0]}|{p[1]}|{p[2]}"


def jobs(seeds: list[int], epsilons: list[float]) -> list[tuple[str, float | None, int, Point]]:
    out: list[tuple[str, float | None, int, Point]] = []
    for seed in seeds:
        for p in points():
            out.append(("fast", None, seed, p))
            out.extend(("rigorous", eps, seed, p) for eps in epsilons)
    return out


def _record(path: str, epsilon: float | None, seed: int, p: Point, t0: float) -> dict[str, Any]:
    rho = run_point(p, path == "rigorous", seed, epsilon)
    ent = {n: entropy_bits(r) for n, r in rho.items()}
    h_mean = float(np.mean(list(ent.values())))
    return {
        "path": path,
        "epsilon": epsilon,
        "seed": seed,
        "alpha": p[0],
        "beta": p[1],
        "gamma": p[2],
        "mean_entropy_bits": h_mean,
        "gap_bits": base.MAX_ENTROPY_BITS - h_mean,
        "per_species": ent,
        "rho": {n: r.tolist() for n, r in rho.items()},
        "runtime_s": time.perf_counter() - t0,
    }


def run_all(seeds: list[int], epsilons: list[float]) -> None:
    paths = output_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    if paths["partial"].exists():
        for line in paths["partial"].read_text().splitlines():
            r = json.loads(line)
            done[key(r["path"], r["epsilon"], r["seed"], (r["alpha"], r["beta"], r["gamma"]))] = r
    todo = [j for j in jobs(seeds, epsilons) if key(*j) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already in partial.jsonl)", flush=True)
    with paths["partial"].open("a") as fh:
        for path, eps, seed, p in todo:
            rec = _record(path, eps, seed, p, time.perf_counter())
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[key(path, eps, seed, p)] = rec
            print(
                f"{path} eps={eps} seed={seed} p={p} H={rec['mean_entropy_bits']:.4f} "
                f"({rec['runtime_s']:.1f} s)",
                flush=True,
            )
    write_outputs(list(done.values()), seeds, epsilons)


def _stats(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def summarise(
    records: list[dict[str, Any]], seeds: list[int], epsilons: list[float]
) -> list[dict[str, Any]]:
    by = {
        key(r["path"], r["epsilon"], r["seed"], (r["alpha"], r["beta"], r["gamma"])): r
        for r in records
    }
    rows = []
    for eps in epsilons:
        for p in KEY_POINTS:
            ctrl: Point = (0.0, p[1], p[2])
            ok = [
                s
                for s in seeds
                if all(
                    k in by
                    for k in (
                        key("rigorous", eps, s, p),
                        key("rigorous", eps, s, ctrl),
                        key("fast", None, s, p),
                    )
                )
            ]
            if not ok:
                continue
            rig = [by[key("rigorous", eps, s, p)] for s in ok]
            rig0 = [by[key("rigorous", eps, s, ctrl)] for s in ok]
            fast = [by[key("fast", None, s, p)] for s in ok]
            h = np.array([r["mean_entropy_bits"] for r in rig])
            h0 = np.array([r["mean_entropy_bits"] for r in rig0])
            hf = np.array([r["mean_entropy_bits"] for r in fast])
            names = list(rig[0]["per_species"])
            kl = [
                sum(kl_nats(np.asarray(r["rho"][n]), np.asarray(f["rho"][n])) for n in names)
                for r, f in zip(rig, fast, strict=True)
            ]
            rows.append(
                {
                    "epsilon": eps,
                    "alpha": p[0],
                    "beta": p[1],
                    "gamma": p[2],
                    "n_inner": max(1, int(10 * p[1])) if p[1] > 0 else 1,
                    "seeds": ok,
                    "rigorous": {
                        "H": _stats(h.tolist()),
                        "gap": _stats((base.MAX_ENTROPY_BITS - h).tolist()),
                        "per_species_H": {
                            n: _stats([r["per_species"][n] for r in rig]) for n in names
                        },
                    },
                    "control_alpha0": {
                        "H": _stats(h0.tolist()),
                        "gap": _stats((base.MAX_ENTROPY_BITS - h0).tolist()),
                    },
                    "specialisation_vs_alpha0": _stats((h0 - h).tolist()),
                    "fast": {
                        "H": _stats(hf.tolist()),
                        "gap": _stats((base.MAX_ENTROPY_BITS - hf).tolist()),
                    },
                    "fast_minus_rigorous_gap": _stats((h - hf).tolist()),
                    "kl_rigorous_vs_fast_nats_sum4": _stats(kl),
                    "runtime_s_rigorous": float(sum(r["runtime_s"] for r in rig)),
                }
            )
    return rows


def reference_macm1(seeds: list[int]) -> dict[str, Any]:
    """Committed macM1 numbers at the key points (seeds restricted to ``seeds``)."""
    out: dict[str, Any] = {}
    if REF_PREFIX.exists():
        rows = json.loads(REF_PREFIX.read_text())["rows"]
        out["reg_0.05_prefix"] = {
            "source": f"{REF_PREFIX} (before the fix: reg = h = 0.05), seeds 0-4",
            "points": {
                f"{r['alpha']},{r['beta']},{r['gamma']}": {
                    "rigorous_gap": {
                        "mean": base.MAX_ENTROPY_BITS - r["rigorous_H"]["mean"],
                        "sd": r["rigorous_H"]["sd"],
                    },
                    "fast_gap": {
                        "mean": base.MAX_ENTROPY_BITS - r["fast_H"]["mean"],
                        "sd": r["fast_H"]["sd"],
                    },
                }
                for r in rows
                if (r["alpha"], r["beta"], r["gamma"]) in KEY_POINTS
            },
        }
    if REF_FIXED.exists():
        recs = json.loads(REF_FIXED.read_text())["results"]
        pts: dict[str, Any] = {}
        for p in KEY_POINTS:
            h = [
                r["mean_entropy_bits"]
                for r in recs
                if r["path"] == "rigorous"
                and (r["alpha"], r["beta"], r["gamma"]) == p
                and r["seed"] in seeds
            ]
            if h:
                pts[f"{p[0]},{p[1]},{p[2]}"] = {
                    "rigorous_gap": _stats((base.MAX_ENTROPY_BITS - np.array(h)).tolist())
                }
        out["eps_0.01_fixed"] = {
            "source": f"{REF_FIXED} (fix forwarded, epsilon = 0.01), seeds {seeds}",
            "points": pts,
        }
    return out


def provenance() -> dict[str, Any]:
    prov = base.provenance()
    prov["driver"] = "experiments/eps_sweep_2026_10_25.py"
    prov["mlx_version"] = mx.__version__
    prov["prox_params"] = {
        "epsilon": "h = 0.05 (JKO step, proximal pull)",
        "reg": "swept: FullJKOSolver.epsilon (Sinkhorn entropic regularisation)",
        "n_iter": 100,
    }
    return prov


def _fmt(s: dict[str, float], nd: int = 3) -> str:
    return f"{s['mean']:.{nd}f} ± {s['sd']:.{nd}f}"


def render_summary(rows: list[dict[str, Any]], ref: dict[str, Any], prov: dict[str, Any]) -> str:
    lines = [
        "# Sinkhorn regularisation sweep, rigorous specialisation gap (2026-10-25)",
        "",
        f"Driver `{prov['driver']}` at `{prov['git_commit'][:8]}`, host `{prov['host']}` "
        f"({prov['machine']}), Python {prov['python']}, MLX {prov['mlx_version']} on "
        f"`{prov['mlx_default_device']}`, {prov['date_utc']}.",
        "",
        "Problem of `experiments/rigorous_specialisation_2026_10_24.py` (grid 16, 500 "
        "particles, 200 fast x 30 slow steps, h = 0.05). epsilon is the Sinkhorn "
        "regularisation of the rigorous prox (`FullJKOSolver.epsilon`, forwarded as "
        "`reg`); the proximal pull stays at h. Gap = log2(16) - mean H over the four "
        "species, in bits; S = H(alpha = 0, same beta, gamma, seed) - H; KL in nats, "
        "summed over species at the final slow step, on this reduced setup. "
        "Mean ± sd (ddof = 1) over seeds.",
        "",
        "| eps | (alpha, beta, gamma) | n_inner | rigorous gap | S vs alpha=0 | fast gap "
        "| fast - rigorous gap | KL(rig‖fast) | seeds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['epsilon']} | ({r['alpha']:g}, {r['beta']:g}, {r['gamma']:g}) | "
            f"{r['n_inner']} | {_fmt(r['rigorous']['gap'])} | "
            f"{_fmt(r['specialisation_vs_alpha0'])} | {_fmt(r['fast']['gap'])} | "
            f"{_fmt(r['fast_minus_rigorous_gap'])} | "
            f"{_fmt(r['kl_rigorous_vs_fast_nats_sum4'])} | {len(r['seeds'])} |"
        )
    for p in KEY_POINTS:
        gaps = [
            r["rigorous"]["gap"]["mean"] for r in rows if (r["alpha"], r["beta"], r["gamma"]) == p
        ]
        sds = [r["rigorous"]["gap"]["sd"] for r in rows if (r["alpha"], r["beta"], r["gamma"]) == p]
        if len(gaps) > 1:
            lines.append("")
            lines.append(
                f"At ({p[0]:g}, {p[1]:g}, {p[2]:g}) the rigorous gap spans "
                f"{max(gaps) - min(gaps):.3f} bits across epsilon (seed sd {min(sds):.3f}-"
                f"{max(sds):.3f} bits)."
            )
    if ref:
        lines += ["", "## Committed macM1 references at the same points", ""]
        lines.append("| source | (alpha, beta, gamma) | rigorous gap | fast gap |")
        lines.append("|---|---|---|---|")
        for label, block in ref.items():
            for pt, v in block["points"].items():
                fast = v.get("fast_gap")
                lines.append(
                    f"| {label} | ({pt}) | {_fmt(v['rigorous_gap'])} | "
                    f"{_fmt(fast) if fast else '—'} |"
                )
        lines.append("")
        for label, block in ref.items():
            lines.append(f"- `{label}`: {block['source']}")
    return "\n".join(lines) + "\n"


def write_outputs(records: list[dict[str, Any]], seeds: list[int], epsilons: list[float]) -> None:
    paths = output_paths()
    records = sorted(
        records,
        key=lambda r: (
            r["path"],
            r["epsilon"] or 0.0,
            r["seed"],
            r["alpha"],
            r["beta"],
            r["gamma"],
        ),
    )
    rows = summarise(records, seeds, epsilons)
    ref = reference_macm1(seeds)
    prov = provenance()
    paths["sweep"].write_text(
        json.dumps(
            {
                "metric": "H = mean over 4 species of final entropy (bits); gap = log2(16) - H; "
                "S = H(alpha=0, same beta, gamma, seed) - H; KL(rigorous||fast) in nats "
                "summed over species at the final slow step",
                "epsilons": epsilons,
                "seeds": seeds,
                "key_points": KEY_POINTS,
                "params": {
                    "grid": base.GRID,
                    "n_particles": base.N_PARTICLES,
                    "n_fast": base.N_FAST,
                    "n_slow": base.N_SLOW,
                    "h": base.H,
                },
                "rows": rows,
                "reference_macm1": ref,
                "n_records": len(records),
                "runtime_s": float(sum(r["runtime_s"] for r in records)),
                "results": records,
                "provenance": prov,
            },
            indent=1,
        )
    )
    paths["summary"].write_text(render_summary(rows, ref, prov))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epsilons", type=float, nargs="+", default=EPSILONS)
    ap.add_argument("--estimate", action="store_true", help="print the run count and exit")
    args = ap.parse_args()
    if args.estimate:
        print(f"{len(jobs(args.seeds, args.epsilons))} runs")
        return
    # Set here, not at import, so importing this module (tests) leaves the
    # process-wide MLX device untouched.
    mx.set_default_device(mx.cpu)
    run_all(args.seeds, args.epsilons)


if __name__ == "__main__":
    main()
