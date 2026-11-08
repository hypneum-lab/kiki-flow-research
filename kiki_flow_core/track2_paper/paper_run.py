"""Top-level driver: run 5 seeds, produce figures, aggregate stats."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np
from safetensors.numpy import save_file

from kiki_flow_core.master_equation import SOLVER_VERSION, JKOStep, NonSeparable, Scheme
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.figures.continual_learning_gap import (
    make_continual_learning_gap,
)
from kiki_flow_core.track2_paper.figures.f_decay_curves import make_f_decay_curves
from kiki_flow_core.track2_paper.figures.kl_vs_epsilon import make_kl_vs_epsilon
from kiki_flow_core.track2_paper.figures.phase_portrait import make_phase_portrait
from kiki_flow_core.track2_paper.figures.turing_patterns import make_turing_patterns
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver, MLXFullJKOSolver
from kiki_flow_core.track2_paper.mlx_particle_simulator import MLXParticleSimulator
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy
from kiki_flow_core.track2_paper.particle_simulator import ParticleSimulator

SinkhornBackend = Literal["pot", "mlx"]


def _persist_trajectory(trajectory: list[FlowState], out_path: Path) -> None:
    """Save a trajectory list of FlowStates as safetensors (rho arrays only)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for tau_idx, state in enumerate(trajectory):
        for name, rho in state.rho.items():
            tensors[f"tau{tau_idx:03d}::{name}"] = np.asarray(rho, dtype=np.float32)
    save_file(tensors, str(out_path))


def _aggregate_rho_stats(trajectories: list[list[FlowState]]) -> dict[str, Any]:
    """Aggregate informative stats per species at the final tau: peak amplitude
    (max of rho), Shannon entropy (bits, lower = more concentrated), and the
    location of the peak on the grid index."""
    names = list(trajectories[0][-1].rho.keys())
    peak: dict[str, list[float]] = {n: [] for n in names}
    entropy_bits: dict[str, list[float]] = {n: [] for n in names}
    peak_idx: dict[str, list[int]] = {n: [] for n in names}
    for traj in trajectories:
        final = traj[-1]
        for n in names:
            rho = np.asarray(final.rho[n])
            rho_safe = np.clip(rho, 1e-12, None)
            peak[n].append(float(rho.max()))
            entropy_bits[n].append(float(-(rho_safe * np.log2(rho_safe)).sum()))
            peak_idx[n].append(int(np.argmax(rho)))
    return {
        n: {
            "peak_mean": float(np.mean(peak[n])),
            "peak_std": float(np.std(peak[n])),
            "entropy_bits_mean": float(np.mean(entropy_bits[n])),
            "entropy_bits_std": float(np.std(entropy_bits[n])),
            "peak_idx_mean": float(np.mean(peak_idx[n])),
            "peak_idx_std": float(np.std(peak_idx[n])),
        }
        for n in names
    }


def _make_jko(
    f_func: T2FreeEnergy,
    support: np.ndarray,
    use_w2_prox: bool,
    sinkhorn_backend: SinkhornBackend,
    scheme: Scheme | None,
    nonseparable: NonSeparable,
    implicit_options: dict[str, Any] | None,
) -> JKOStep:
    """Slow-step solver of ``run_paper``; ``nonseparable`` reaches all three branches."""
    if use_w2_prox:
        if sinkhorn_backend == "mlx":
            jko: JKOStep = MLXFullJKOSolver(
                f_functional=f_func,
                h=0.05,
                support=support,
                epsilon=0.01 if scheme == "legacy_split" else None,
                scheme=scheme or "jko_entropic",
                nonseparable=nonseparable,
                implicit_options=implicit_options,
            )
        else:
            jko = FullJKOSolver(
                f_functional=f_func,
                h=0.05,
                support=support,
                epsilon=0.01 if scheme == "legacy_split" else None,
                scheme=scheme or "jko_entropic",
                nonseparable=nonseparable,
                implicit_options=implicit_options,
            )
    else:
        jko = JKOStep(
            f_functional=f_func,
            h=0.05,
            support=support,
            n_inner=10,
            apply_w2_prox=False,
            scheme=scheme,
            nonseparable=nonseparable,
            **(implicit_options or {}),
        )

    return jko


# Provenance is read from the repository that holds this module, not from the
# current working directory (a driver may run from elsewhere).
_REPO_DIR = Path(__file__).resolve().parents[2]


def _git(*args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            cwd=_REPO_DIR,
        )
    except OSError, subprocess.SubprocessError:
        return None


def _git_commit() -> str:
    out = _git("rev-parse", "HEAD")
    return (out.stdout.strip() or "unknown") if out is not None else "unknown"


def _git_dirty() -> bool | None:
    """True when tracked files differ from the commit (results not reproducible as is)."""
    out = _git("status", "--porcelain", "--untracked-files=no")
    if out is None or out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def _git_diff_sha256() -> str | None:
    """SHA-256 of ``git diff HEAD`` (tracked changes), identifying a dirty tree."""
    out = _git("diff", "HEAD", "--no-color")
    if out is None or out.returncode != 0:
        return None
    return hashlib.sha256(out.stdout.encode()).hexdigest()


def run_paper(
    seeds: list[int],
    n_particles: int = 10_000,
    n_fast: int = 1000,
    n_slow: int = 100,
    grid_size: int = 64,
    out_dir: Path = Path("paper"),
    use_mlx: bool = True,
    use_w2_prox: bool = False,
    sinkhorn_backend: SinkhornBackend = "pot",
    save_trajectories: bool = False,
    make_all_figures: bool = False,
    scheme: Scheme | None = None,
    reinject: bool = True,
    nonseparable: NonSeparable = "implicit",
    turing_smoothing: float | None = 1e-5,
    implicit_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run N seeds of the T2 multiscale loop, producing figures and stats.

    Parameters
    ----------
    use_mlx : bool (default True)
        Use MLXParticleSimulator (Apple Silicon Metal) instead of numpy.
    use_w2_prox : bool (default False)
        Use FullJKOSolver with Wasserstein proximal step (rigorous but ~55x slower).
        Default False uses plain JKOStep (projected gradient descent only).
    save_trajectories : bool (default False)
        Persist each seed's trajectory as safetensors for post-hoc figure
        generation or ablations.
    make_all_figures : bool (default False)
        Produce fig1-5 using the first seed's trajectory (requires
        save_trajectories implicitly). fig4/fig5 use synthetic input
        unless a real KL-sweep or CL benchmark is plumbed in.
    scheme : None (default), "jko_entropic" or "legacy_split"
        Slow-step scheme. None follows the solver defaults: the entropic JKO
        step of F (reg = h^2) with ``use_w2_prox``, the Euclidean descent
        without it. ``"legacy_split"`` reproduces the runs made before
        2026-10-25 (with ``reinject=False``).
    reinject : bool (default True)
        Feed the JKO output back into the particles after each slow step.
        The runs made before 2026-10-25 never reinjected.
    nonseparable : "implicit" (default) or "explicit"
        Treatment of the coupling and Turing terms in the entropic step
        (ERRATUM.md sections 14-16); "explicit" reproduces earlier runs.
        Forwarded in all three solver branches.
    turing_smoothing : float or None (default 1e-5)
        Declared smoothing of the Turing term (``T2FreeEnergy``), required by
        the certified implicit step; ignored (None) for the explicit and legacy
        schemes, which keep the historical energy.
    implicit_options : dict or None
        Settings of the certified implicit solver (see ``JKOStep``).

    ``stats.json`` records the resolved solver configuration (scheme,
    nonseparable, reg, tolerances, smoothing, reinjection, git commit) and,
    for the implicit step, a summary of the per-step certificates.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    species = CanonicalSpecies()
    names = species.species_names()
    support = np.linspace(-2, 2, grid_size).reshape(-1, 1)
    potentials = {n: np.zeros(grid_size) for n in names}
    prior = {n: np.full(grid_size, 1.0 / grid_size) for n in names}
    resolved_scheme = scheme or ("jko_entropic" if use_w2_prox else "legacy_split")
    smoothing = (
        turing_smoothing
        if resolved_scheme == "jko_entropic" and nonseparable == "implicit"
        else None
    )
    f_func = T2FreeEnergy(
        species=species,
        potentials=potentials,
        prior=prior,
        turing_strength=0.1,
        turing_smoothing=smoothing,
    )

    jko = _make_jko(
        f_func, support, use_w2_prox, sinkhorn_backend, scheme, nonseparable, implicit_options
    )

    per_seed: list[dict[str, Any]] = []
    trajectories: list[list[FlowState]] = []
    for seed in seeds:
        if use_mlx:
            sim: Any = MLXParticleSimulator(
                species=species, n_particles=n_particles, latent_dim=2, seed=seed
            )
        else:
            sim = ParticleSimulator(
                species=species, n_particles=n_particles, latent_dim=2, seed=seed
            )
        loop = MultiscaleLoop(
            sim=sim,
            jko=jko,
            n_fast=n_fast,
            n_slow=n_slow,
            support=support,
            reinject=reinject,
        )
        certificates: list[dict[str, Any]] = []
        if jko.scheme == "jko_entropic" and jko.nonseparable == "implicit":
            base_step = jko.step

            def certified_step(
                state: FlowState, base_step: Any = base_step, certificates: Any = certificates
            ) -> FlowState:
                out: FlowState = base_step(state)
                certificates.append(dict(jko.last_info))
                return out

            jko.step = certified_step  # type: ignore[method-assign]
        manifest = loop.run(seed=seed)
        if certificates:
            jko.step = base_step  # type: ignore[method-assign]
        make_phase_portrait(
            manifest["trajectory"],
            out_dir / "figures",
            filename=f"fig1_phase_seed{seed}",
        )
        if save_trajectories or make_all_figures:
            _persist_trajectory(
                manifest["trajectory"], out_dir / "trajectories" / f"seed{seed}.safetensors"
            )
            trajectories.append(manifest["trajectory"])
        entry: dict[str, Any] = {"seed": seed, "n_slow_completed": manifest["n_slow_completed"]}
        if certificates:
            entry["implicit_certificates"] = {
                "steps": len(certificates),
                "all_converged": all(c["status"] == "converged" for c in certificates),
                "max_kkt_residual": max(c["kkt_residual"] for c in certificates),
                "max_outer_iterations": max(c["outer_iterations"] for c in certificates),
                "max_inner_residual": max(c["inner_max_residual"] for c in certificates),
                "seconds": float(sum(c["seconds"] for c in certificates)),
            }
        per_seed.append(entry)

    stats: dict[str, Any] = {
        "n_seeds": len(seeds),
        "per_seed": per_seed,
        "backend": {
            "simulator": "mlx" if use_mlx else "numpy",
            "w2_prox": use_w2_prox,
            "sinkhorn_backend": sinkhorn_backend if use_w2_prox else None,
            # The entropic scheme runs the NumPy log-domain solver whatever the
            # backend; the MLX prox is used by the legacy split only.
            "prox_backend_effective": (
                "numpy (entropic scheme)"
                if jko.scheme == "jko_entropic"
                else (sinkhorn_backend if use_w2_prox else None)
            ),
        },
        "solver": {
            "scheme": jko.scheme,
            "nonseparable": jko.nonseparable,
            "h": jko.h,
            "reg": jko.entropic_reg if jko.scheme == "jko_entropic" else jko.reg,
            "prox_n_iter": jko.prox_n_iter,
            "prox_tol": jko.prox_tol,
            "kkt_tol": jko.kkt_tol,
            "log_tol": jko.log_tol,
            "max_outer": jko.max_outer,
            "l0": jko.l0,
            "l_max": jko.l_max,
            "sigma": jko.sigma,
            "inner_n_iter": jko.inner_n_iter,
            "inner_tol": jko.inner_tol,
            "inner_marginal_tol": jko.inner_marginal_tol,
            "inner_log_tol": jko.inner_log_tol,
            "ot_gap_tol": jko.ot_gap_tol,
            "on_failure": jko.on_failure,
            "solver_version": SOLVER_VERSION,
            "turing_smoothing": smoothing,
            "reinject": reinject,
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "git_diff_sha256": _git_diff_sha256(),
        },
    }

    if make_all_figures and trajectories:
        traj0 = trajectories[0]
        make_f_decay_curves(traj0, f_functional=f_func, out_dir=out_dir / "figures")
        make_turing_patterns(traj0, out_dir=out_dir / "figures")
        # KL-vs-eps: synthetic placeholder (real sweep elsewhere)
        epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]
        kl_values = [1.2, 0.45, 0.12, 0.018, 0.003]
        make_kl_vs_epsilon(epsilons, kl_values, out_dir=out_dir / "figures")
        # CL gap: synthetic placeholder
        make_continual_learning_gap(
            tasks=["phonology", "lexicon", "syntax"],
            with_consolidation=[0.87, 0.81, 0.78],
            without_consolidation=[0.72, 0.64, 0.61],
            out_dir=out_dir / "figures",
        )
        stats["aggregate"] = _aggregate_rho_stats(trajectories)

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
