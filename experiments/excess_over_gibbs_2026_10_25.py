"""Specialisation as excess over the Gibbs equilibrium of the potentials.

Primary statistic (PI decision of 2026-10-25): for each sweep point and seed,
``excess = gap - gap_Gibbs``, where ``gap = log2(16) - mean_i H(rho_i)`` is the
specialisation gap of the final state and ``gap_Gibbs`` the same quantity at
the equilibrium imposed by the potentials, rho_i ∝ prior_i exp(-V_i) (theta = 1,
the KL term of ``T2FreeEnergy``). It measures what the dynamics adds to the
equilibrium fixed by the potentials; a claim of emergent specialisation needs
an excess > 0 whose 95 % bootstrap CI (over seeds) excludes 0.

Secondary references, descriptive:

* ``gap_Fmin``: the minimiser of the full ``T2FreeEnergy`` (J_sym coupling and
  Turing term included), by a damped fixed point rho_i ∝ prior_i
  exp(-V_i - dE_nonsep/d rho_i), and ``excess_Fmin = gap - gap_Fmin``;
* ``excess_ctrl = (gap - gap_Gibbs) - gap(alpha = 0, same gamma, seed)``,
  paired by seed: the alpha = 0 control has a uniform equilibrium (gap 0) but
  a positive measured gap (finite-particle anchor of the slow step), which
  this subtracts.

Inputs: the entropic sweep ``results/jko_entropic_sweep_2026_10_25/records.json``
(PR #46; beta-invariant, one record per (alpha, gamma, seed)) and, for
comparison only, the legacy rigorous and fast records of
``paper/rigorous_specialisation_wide_2026_10_24_reg/``. Output (new files):
``results/excess_over_gibbs_2026_10_25/{excess.json, summary.md}``.
Pure NumPy, deterministic (bootstrap seeded).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy

GRID = 16
MAX_ENTROPY_BITS = float(np.log2(GRID))
RHO_FLOOR = 1e-12
N_BOOT = 10_000
BOOT_SEED = 0
FMIN_ITER = 300
FMIN_DAMPING = 0.5
FMIN_TOL = 1e-10
ENTROPIC = Path("results/jko_entropic_sweep_2026_10_25/records.json")
LEGACY = Path("paper/rigorous_specialisation_wide_2026_10_24_reg/sweep_wide_records.json")
OUT_DIR = Path("results/excess_over_gibbs_2026_10_25")
KEY_POINTS = [(10.0, 0.0), (8.0, 5.0), (15.0, 0.0), (30.0, 0.0)]


def output_paths() -> dict[str, Path]:
    return {"excess": OUT_DIR / "excess.json", "summary": OUT_DIR / "summary.md"}


def problem(alpha: float, gamma: float) -> T2FreeEnergy:
    """Potentials of ``experiments/rigorous_specialisation_2026_10_24.problem``."""
    species = CanonicalSpecies()
    names = species.species_names()
    x = np.linspace(-2, 2, GRID)
    potentials = {
        "phono": alpha * (x + 1.5),
        "lex": alpha * (x - 1.5),
        "syntax": alpha * np.abs(x),
        "sem": -alpha * np.abs(x),
    }
    prior = {n: np.full(GRID, 1.0 / GRID) for n in names}
    return T2FreeEnergy(
        species=species,
        potentials=potentials,
        prior=prior,
        turing_strength=gamma * 0.1,
        turing_smoothing=None,
    )


def entropy_bits(rho: np.ndarray) -> float:
    r = np.clip(np.asarray(rho, dtype=float), RHO_FLOOR, None)
    r = r / r.sum()
    return float(-(r * np.log2(r)).sum())


def gap_of(rhos: dict[str, np.ndarray]) -> float:
    return MAX_ENTROPY_BITS - float(np.mean([entropy_bits(r) for r in rhos.values()]))


def _boltzmann(log_w: np.ndarray) -> np.ndarray:
    w = np.exp(log_w - log_w.max())
    return w / w.sum()


def gibbs_potentials(f: T2FreeEnergy) -> dict[str, np.ndarray]:
    return {n: _boltzmann(np.log(f.prior[n]) - f.potentials[n]) for n in f.species.species_names()}


def gibbs_full(f: T2FreeEnergy) -> tuple[dict[str, np.ndarray], float, int]:
    """Damped fixed point of rho_i ∝ prior_i exp(-V_i - dE_nonsep/drho_i)."""
    names = f.species.species_names()
    rho = gibbs_potentials(f)
    delta = np.inf
    it = 0
    while it < FMIN_ITER and delta >= FMIN_TOL:
        it += 1
        state = FlowState(
            rho=rho,
            P_theta=np.zeros(4),
            mu_curr=np.array([1.0]),
            tau=0,
            metadata={"track_id": "T2"},
        )
        target = {
            n: _boltzmann(np.log(f.prior[n]) - f.potentials[n] - f.nonseparable_grad(state, n))
            for n in names
        }
        new = {n: (1 - FMIN_DAMPING) * rho[n] + FMIN_DAMPING * target[n] for n in names}
        delta = max(float(np.abs(new[n] - rho[n]).max()) for n in names)
        rho = new
    return rho, delta, it


def bootstrap_ci(values: list[float], rng: np.random.Generator) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
        "n": int(n),
    }


def verdict(ci: dict[str, float]) -> str:
    if ci["n"] == 0:
        return "n/a"
    if ci["lo"] > 0:
        return "excess > 0"
    if ci["hi"] < 0:
        return "deficit (< 0)"
    return "CI includes 0"


def references(couples: list[tuple[float, float]]) -> dict[tuple[float, float], dict[str, Any]]:
    out = {}
    for a, g in couples:
        f = problem(a, g)
        rho_full, delta, it = gibbs_full(f)
        out[(a, g)] = {
            "gap_gibbs": gap_of(gibbs_potentials(f)),
            "gap_fmin": gap_of(rho_full),
            "fmin_delta": delta,
            "fmin_iter": it,
        }
    return out


def analyse(
    records: list[dict[str, Any]], refs: dict[tuple[float, float], dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """records: one per (alpha, gamma, seed) with gap_bits."""
    rng = np.random.default_rng(seed)
    by = {(r["alpha"], r["gamma"], r["seed"]): r["gap_bits"] for r in records}
    rows = []
    for a, g in sorted(refs):
        seeds = sorted(s for (aa, gg, s) in by if (aa, gg) == (a, g))
        ref = refs[(a, g)]
        ex = [by[(a, g, s)] - ref["gap_gibbs"] for s in seeds]
        exf = [by[(a, g, s)] - ref["gap_fmin"] for s in seeds]
        exc = [
            by[(a, g, s)] - ref["gap_gibbs"] - by[(0.0, g, s)] for s in seeds if (0.0, g, s) in by
        ]
        primary = bootstrap_ci(ex, rng)
        rows.append(
            {
                "alpha": a,
                "gamma": g,
                "seeds": seeds,
                "gap": bootstrap_ci([by[(a, g, s)] for s in seeds], rng),
                "gap_gibbs": ref["gap_gibbs"],
                "gap_fmin": ref["gap_fmin"],
                "excess": primary,
                "verdict": verdict(primary),
                "excess_fmin": bootstrap_ci(exf, rng),
                "excess_ctrl": bootstrap_ci(exc, rng),
                "verdict_ctrl": verdict(bootstrap_ci(exc, np.random.default_rng(seed + 1))),
            }
        )
    return rows


def legacy_by_couple(path: str) -> list[dict[str, Any]]:
    """Legacy records depend on beta: keep beta = 1 (the paper's key beta) per (alpha, gamma)."""
    if not LEGACY.exists():
        return []
    recs = json.loads(LEGACY.read_text())["results"]
    best: dict[tuple[float, float, int], dict[str, Any]] = {}
    for r in recs:
        if r["path"] != path:
            continue
        k = (r["alpha"], r["gamma"], r["seed"])
        # Prefer beta = 1, else the beta closest to 1.
        if k not in best or abs(r["beta"] - 1.0) < abs(best[k]["beta"] - 1.0):
            best[k] = r
    return [{**r, "beta_used": r["beta"]} for r in best.values()]


def _fmt(ci: dict[str, float]) -> str:
    if not ci["n"]:
        return "—"
    return f"{ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]"


def write_summary(res: dict[str, Any], path: Path) -> None:
    lines = [
        "# Specialisation as excess over the Gibbs equilibrium (2026-10-25)",
        "",
        "Excess = gap − gap_Gibbs (bits), gap_Gibbs = gap of rho_i ∝ prior_i exp(−V_i). "
        "Mean over seeds [95 % bootstrap CI over seeds, 10 000 resamples]. "
        "`Fmin` = minimiser of the full T2 energy (coupling + Turing). `ctrl` = excess "
        "minus the paired alpha = 0 gap (finite-particle bias of the anchor).",
        "",
        "## Entropic JKO step (PR #46: reg h², reinjection; beta-invariant)",
        "",
        "| alpha | gamma | gap | gap_Gibbs | excess | verdict | gap_Fmin | excess vs Fmin "
        "| excess − alpha0 control | verdict (ctrl) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["entropic"]:
        if r["alpha"] == 0.0:
            continue
        lines.append(
            f"| {r['alpha']:g} | {r['gamma']:g} | {r['gap']['mean']:.3f} | {r['gap_gibbs']:.3f} "
            f"| {_fmt(r['excess'])} | {r['verdict']} | {r['gap_fmin']:.3f} "
            f"| {_fmt(r['excess_fmin'])} | {_fmt(r['excess_ctrl'])} | {r['verdict_ctrl']} |"
        )
    for label, key in (
        ("Legacy rigorous path (beta = 1 or closest), descriptive", "legacy_rigorous"),
        ("Fast path (beta = 1 or closest), descriptive", "legacy_fast"),
    ):
        lines += [
            "",
            f"## {label}",
            "",
            "| alpha | gamma | gap | gap_Gibbs | excess |",
            "|---|---|---|---|---|",
        ]
        for r in res[key]:
            if r["alpha"] == 0.0:
                continue
            lines.append(
                f"| {r['alpha']:g} | {r['gamma']:g} | {r['gap']['mean']:.3f} "
                f"| {r['gap_gibbs']:.3f} | {_fmt(r['excess'])} |"
            )
    n_pos = sum(1 for r in res["entropic"] if r["alpha"] > 0 and r["verdict"] == "excess > 0")
    n_ctrl = sum(1 for r in res["entropic"] if r["alpha"] > 0 and r["verdict_ctrl"] == "excess > 0")
    n_all = sum(1 for r in res["entropic"] if r["alpha"] > 0)
    lines += [
        "",
        f"Entropic scheme: {n_pos} / {n_all} (alpha > 0) couples with excess > 0 (CI excludes 0); "
        f"{n_ctrl} / {n_all} after subtracting the alpha = 0 control.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ent = json.loads(ENTROPIC.read_text())["results"]
    reg_main = min(r["reg"] for r in ent)
    ent = [r for r in ent if r["reg"] == reg_main]
    couples = sorted({(r["alpha"], r["gamma"]) for r in ent})
    refs = references(couples)
    leg_rig = legacy_by_couple("rigorous")
    leg_fast = legacy_by_couple("fast")
    res = {
        "meta": {
            "script": "experiments/excess_over_gibbs_2026_10_25.py",
            "entropic_source": str(ENTROPIC),
            "legacy_source": str(LEGACY),
            "reg_main": reg_main,
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
        },
        "references": [{"alpha": a, "gamma": g, **v} for (a, g), v in sorted(refs.items())],
        "entropic": analyse(ent, refs, BOOT_SEED),
        "legacy_rigorous": analyse(
            leg_rig,
            {k: v for k, v in refs.items() if any((r["alpha"], r["gamma"]) == k for r in leg_rig)},
            BOOT_SEED,
        ),
        "legacy_fast": analyse(
            leg_fast,
            {k: v for k, v in refs.items() if any((r["alpha"], r["gamma"]) == k for r in leg_fast)},
            BOOT_SEED,
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths()["excess"].write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_summary(res, output_paths()["summary"])
    print(output_paths()["summary"].read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
