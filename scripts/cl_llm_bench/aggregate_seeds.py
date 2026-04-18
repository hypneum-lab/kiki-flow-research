"""Aggregate a multi-seed real-CL sweep into mean±std stats + fig7.

Input:  N ``result.json`` manifests emitted by ``_run_real_sequence`` —
        each contains ``immediate_accuracy``, ``final_accuracy``,
        ``forgetting``, ``tasks``, ``seed``, ``wall_time_s``.
Output: one ``summary.json`` with per-task mean±std for every metric
        and a regenerated ``fig7_cl_forgetting.{png,pdf}`` plotted
        from the mean forgetting across seeds.

Typical use — after a 5-seed sweep has finished:

    PYTHONPATH=. uv run python -m scripts.cl_llm_bench.aggregate_seeds \\
      --glob "bench/cl_llm/runs/e3-5k-seed*/result.json" \\
      --out  bench/cl_llm/runs/e3_5seeds_summary.json \\
      --fig-dir paper/figures
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from kiki_flow_core.track2_paper.figures.fig7_cl_forgetting import make_cl_forgetting


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and validate N result.json manifests. Raises on schema drift."""
    out: list[dict[str, Any]] = []
    required = {"tasks", "immediate_accuracy", "final_accuracy", "forgetting", "seed"}
    for p in paths:
        data = json.loads(p.read_text())
        missing = required - data.keys()
        if missing:
            msg = f"{p}: missing keys {missing}"
            raise ValueError(msg)
        if data.get("status") != "ok":
            msg = f"{p}: status={data.get('status')!r}, expected 'ok'"
            raise ValueError(msg)
        out.append(data)
    return out


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean±std per (task, metric) across seeds.

    Raises if the per-seed task lists disagree — mixing two sweeps
    into a single aggregate would break provenance.
    """
    if not results:
        msg = "no results to aggregate"
        raise ValueError(msg)
    tasks = results[0]["tasks"]
    for r in results[1:]:
        if r["tasks"] != tasks:
            msg = f"task list mismatch: {tasks} vs {r['tasks']} (seed {r['seed']})"
            raise ValueError(msg)

    stats: dict[str, dict[str, dict[str, Any]]] = {
        "immediate": {},
        "final": {},
        "forgetting": {},
    }
    metric_key = {
        "immediate": "immediate_accuracy",
        "final": "final_accuracy",
        "forgetting": "forgetting",
    }
    for metric_short, metric_full in metric_key.items():
        for t in tasks:
            vals = np.array([r[metric_full][t] for r in results])
            stats[metric_short][t] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "per_seed": vals.tolist(),
            }

    return {
        "n_seeds": len(results),
        "seeds": sorted(r["seed"] for r in results),
        "tasks": tasks,
        "stats": stats,
        "wall_time_s_per_seed": [r.get("wall_time_s") for r in results],
        "wall_time_s_total": sum(r.get("wall_time_s") or 0.0 for r in results),
    }


def render_fig7(summary: dict[str, Any], fig_dir: Path) -> Path:
    """Regenerate fig7_cl_forgetting from the aggregated mean forgetting.

    The 'with bridge' arm is left at zero because the bridge is
    architecturally inactive in single-dense-LoRA runs; the paper
    explains this.
    """
    tasks = summary["tasks"]
    forgetting_without = {t: summary["stats"]["forgetting"][t]["mean"] for t in tasks}
    forgetting_with = {t: 0.0 for t in tasks}
    return make_cl_forgetting(forgetting_without, forgetting_with, out_dir=fig_dir)


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--glob",
        type=str,
        required=True,
        help='Glob pattern, e.g. "bench/cl_llm/runs/e3-5k-seed*/result.json"',
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the aggregated summary JSON.",
    )
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Directory for regenerated fig7 (default: paper/figures).",
    )
    args = ap.parse_args()

    paths = sorted(Path(p) for p in _glob.glob(args.glob))
    if not paths:
        print(f"error: no files match {args.glob!r}", file=sys.stderr)
        return 1

    print(f"aggregating {len(paths)} files:")
    for p in paths:
        print(f"  {p}")

    results = load_results(paths)
    summary = aggregate(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote summary: {args.out}")

    fig_path = render_fig7(summary, args.fig_dir)
    print(f"wrote fig7: {fig_path}")

    print("\n=== per-task stats ===")
    print(f"{'Task':<14} {'Immediate':<22} {'Final':<22} {'Forgetting':<22}")
    for t in summary["tasks"]:
        im = summary["stats"]["immediate"][t]
        fi = summary["stats"]["final"][t]
        fo = summary["stats"]["forgetting"][t]
        print(
            f"{t:<14} "
            f"{im['mean']:.3f} ± {im['std']:.3f}        "
            f"{fi['mean']:.3f} ± {fi['std']:.3f}        "
            f"{fo['mean']:.3f} ± {fo['std']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
