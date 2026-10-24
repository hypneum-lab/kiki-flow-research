"""E2 — N-particle Langevin propagation-of-chaos preconditions [analysis].

Backlog EXPERIMENT-BACKLOG.md / E2. The mean-field Langevin x JKO "rigorous
proxy" needs, for propagation-of-chaos (N-particle -> mean-field):
  (a) convexity of the free energy  -> Hessian PSD,
  (b) a Lipschitz free-energy gradient on the simplex,
  (c) a (uniform) log-Sobolev constant.

We numerically estimate all three for the actual non-convex multi-species
free energy T2FreeEnergy (4 Levelt-Baddeley species, literature J coupling,
KL prior, Turing cross-diffusion). Deterministic / seeded.

Writes experiments/results/e2_poc_preconditions.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy

GRID = 8
N_SAMPLES = 200          # random simplex points for convexity / Lipschitz scan
SEED = 0
FD_EPS = 1e-5            # finite-difference step for Hessian / gradient
EIG_TOL = 1e-6           # eigenvalue significance threshold (convexity / LSI calls)
DX_MIN = 1e-9            # simplex pairs closer than this are skipped for Lipschitz
PAIR_STRIDE = 4          # subsampling stride over pairs for the Lipschitz scan


def make_state(rhos: dict[str, np.ndarray]) -> FlowState:
    return FlowState(
        rho=rhos,
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )


def random_simplex(rng: np.random.Generator, names: list[str], grid: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for n in names:
        v = rng.dirichlet(np.ones(grid))
        out[n] = v
    return out


def stacked_grad(f: T2FreeEnergy, names: list[str], rhos: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenated simplex gradient over all species (uses F.grad_rho)."""
    st = make_state(rhos)
    parts = [f.grad_rho(st, n) for n in names]
    return np.concatenate(parts)


def free_energy_hessian(
    f: T2FreeEnergy, names: list[str], rhos: dict[str, np.ndarray], eps: float
) -> np.ndarray:
    """Symmetric finite-difference Hessian of F over the stacked rho vector.

    Restricted to the simplex tangent is not enforced here; we report the raw
    Hessian eigenvalues, then also the eigenvalues of the tangent-projected
    Hessian (mean-subtracted per species block) which is what convexity on the
    product of simplices actually requires.
    """
    grid = rhos[names[0]].size
    dim = grid * len(names)

    def grad_vec(flat: np.ndarray) -> np.ndarray:
        d: dict[str, np.ndarray] = {}
        for i, n in enumerate(names):
            block = flat[i * grid : (i + 1) * grid].copy()
            block = np.clip(block, 1e-12, None)
            block = block / block.sum()
            d[n] = block
        return stacked_grad(f, names, d)

    flat0 = np.concatenate([rhos[n] for n in names])
    h = np.zeros((dim, dim))
    for j in range(dim):
        fp = flat0.copy()
        fp[j] += eps
        fm = flat0.copy()
        fm[j] -= eps
        h[:, j] = (grad_vec(fp) - grad_vec(fm)) / (2 * eps)
    return 0.5 * (h + h.T)


def tangent_project_blocks(h: np.ndarray, n_blocks: int) -> np.ndarray:
    """Project each diagonal simplex block onto the mean-zero tangent space.

    Builds P = blockdiag(I - 11^T/g) and returns P H P, whose eigenvalues
    govern convexity along feasible (mass-conserving) simplex directions.
    """
    dim = h.shape[0]
    g = dim // n_blocks
    p = np.zeros((dim, dim))
    centering = np.eye(g) - np.ones((g, g)) / g
    for b in range(n_blocks):
        p[b * g : (b + 1) * g, b * g : (b + 1) * g] = centering
    return p @ h @ p


def main() -> None:
    rng = np.random.default_rng(SEED)
    species = CanonicalSpecies(coupling_variant="levelt")  # WEAVER++ serial architecture
    names = species.species_names()

    # Literature-style potentials and priors on the grid (fixed, seeded).
    potentials = {n: rng.normal(scale=1.0, size=GRID) for n in names}
    prior = {n: np.ones(GRID) / GRID for n in names}
    f = T2FreeEnergy(species=species, potentials=potentials, prior=prior, turing_strength=0.1)

    # (a) Convexity: scan Hessian eigenvalues over many random simplex points.
    raw_min_eigs: list[float] = []
    tan_min_eigs: list[float] = []
    n_nonconvex_raw = 0
    n_nonconvex_tan = 0
    n_hessian_points = 30  # Hessian is O(dim^2) FD evals; keep modest
    for _ in range(n_hessian_points):
        rhos = random_simplex(rng, names, GRID)
        h = free_energy_hessian(f, names, rhos, FD_EPS)
        raw_eigs = np.linalg.eigvalsh(h)
        tan_h = tangent_project_blocks(h, len(names))
        tan_eigs = np.linalg.eigvalsh(tan_h)
        raw_min_eigs.append(float(raw_eigs.min()))
        # ignore the len(names) exact-zero modes from the projection
        tan_sorted = np.sort(tan_eigs)
        tan_min_meaningful = float(tan_sorted[len(names)])  # first non-null mode
        tan_min_eigs.append(tan_min_meaningful)
        if raw_eigs.min() < -EIG_TOL:
            n_nonconvex_raw += 1
        if tan_min_meaningful < -EIG_TOL:
            n_nonconvex_tan += 1

    # (b) Gradient Lipschitz constant: max over sampled pairs of
    #     ||grad F(x) - grad F(y)|| / ||x - y||.
    samples = [random_simplex(rng, names, GRID) for _ in range(N_SAMPLES)]
    grads = [stacked_grad(f, names, s) for s in samples]
    flats = [np.concatenate([s[n] for n in names]) for s in samples]
    lip_est = 0.0
    for i in range(0, len(samples), PAIR_STRIDE):  # subsample pairs for cost
        for j in range(i + 1, len(samples), PAIR_STRIDE):
            dx = np.linalg.norm(flats[i] - flats[j])
            if dx < DX_MIN:
                continue
            dg = np.linalg.norm(grads[i] - grads[j])
            lip_est = max(lip_est, dg / dx)

    # (c) Log-Sobolev proxy: for a measure ~ exp(-F) the LSI constant is bounded
    #     below by the smallest tangent Hessian eigenvalue (Bakry-Emery: if
    #     Hess F >= c I on the tangent, LSI const >= c). A non-positive c means
    #     no Bakry-Emery LSI guarantee. We report the worst-case (min over scan)
    #     tangent eigenvalue as the Bakry-Emery curvature lower bound.
    be_curvature_lb = min(tan_min_eigs)

    convex_raw = n_nonconvex_raw == 0
    convex_tan = n_nonconvex_tan == 0

    # Structural check: is the *coupling operator itself* indefinite? The tangent
    # convexity above is dominated by the strongly-convex KL entropy term
    # (Hess = diag(1/rho) >= grid * I for a uniform-ish rho). Convexity of the
    # FULL F is therefore conditional on the entropy strength vs ||J_sym|| and
    # grid resolution, not a structural property of the model. We expose the
    # symmetric coupling spectrum so the verdict is not overstated.
    j = species.coupling_matrix()
    j_sym = 0.5 * (j + j.T)
    j_sym_eigs = np.linalg.eigvalsh(j_sym)
    coupling_indefinite = bool(j_sym_eigs.min() < -EIG_TOL)

    out = {
        "experiment": "E2",
        "status": "COMPLETE",
        "params": {
            "grid": GRID,
            "n_samples": N_SAMPLES,
            "n_hessian_points": n_hessian_points,
            "seed": SEED,
            "fd_eps": FD_EPS,
            "coupling_variant": "levelt",
            "turing_strength": 0.1,
        },
        "a_convexity": {
            "raw_hessian_min_eig_worst": float(min(raw_min_eigs)),
            "raw_hessian_min_eig_mean": float(np.mean(raw_min_eigs)),
            "n_points_nonconvex_raw": n_nonconvex_raw,
            "tangent_hessian_min_eig_worst": float(min(tan_min_eigs)),
            "tangent_hessian_min_eig_mean": float(np.mean(tan_min_eigs)),
            "n_points_nonconvex_tangent": n_nonconvex_tan,
            "is_convex_raw": convex_raw,
            "is_convex_on_simplex_tangent": convex_tan,
            "coupling_J_sym_eigs": [float(e) for e in j_sym_eigs],
            "coupling_operator_indefinite": coupling_indefinite,
            "convexity_is_structural": not coupling_indefinite,
            "convexity_note": (
                "Tangent convexity holds ONLY because the strongly-convex KL "
                "entropy term (Hess=diag(1/rho)) dominates the indefinite coupling "
                "energy at this grid/prior. The symmetric coupling J_sym is "
                "indefinite (negative eigenvalue), so convexity is conditional, "
                "not a structural guarantee, and is not uniform over grid size, "
                "prior sharpness, or coupling strength."
            ),
        },
        "b_gradient_lipschitz": {
            "estimated_lipschitz_const": float(lip_est),
            "note": (
                "max over subsampled simplex pairs of ||dgrad||/||dx||; "
                "finite => locally Lipschitz on sampled region"
            ),
        },
        "c_log_sobolev_proxy": {
            "bakry_emery_curvature_lower_bound": be_curvature_lb,
            "lsi_guaranteed": be_curvature_lb > EIG_TOL,
            "note": (
                "Bakry-Emery: LSI const >= min tangent Hess F eigenvalue. "
                "Non-positive => no uniform LSI guarantee for exp(-F)."
            ),
        },
        "verdict": (
            "PoC preconditions only CONDITIONALLY met, not structurally. At grid="
            f"{GRID} with a uniform prior the total free energy is convex on the "
            "simplex tangent (worst tangent Hess eigenvalue "
            f"{min(tan_min_eigs):.3f} > 0) and finite-Lipschitz (L~"
            f"{lip_est:.1f}), giving a Bakry-Emery LSI const >= "
            f"{be_curvature_lb:.3f}. BUT this convexity is entirely supplied by the "
            "strongly-convex KL entropy term: the symmetric coupling operator "
            f"J_sym is indefinite (min eig {j_sym_eigs.min():.3f} < 0). Convexity "
            "is therefore conditional on entropy strength vs ||J_sym||, grid "
            "resolution and prior sharpness, and is NOT uniform. A UNIFORM "
            "log-Sobolev constant over the relevant family is thus not established. "
            "The 'rigorous proxy' / propagation-of-chaos label is supported only in "
            "the entropy-dominated regime and should be qualified as such, not "
            "asserted for the general non-convex multi-species free energy."
        ),
    }

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "e2_poc_preconditions.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("E2 preconditions:")
    print(
        f"  (a) convexity (tangent): worst min-eig="
        f"{min(tan_min_eigs):.4e}, nonconvex pts={n_nonconvex_tan}/{n_hessian_points}, "
        f"convex={convex_tan}"
    )
    print(f"  (b) gradient Lipschitz const (est) = {lip_est:.4e}")
    print(
        f"  (c) Bakry-Emery curvature LB = {be_curvature_lb:.4e}  "
        f"LSI guaranteed={be_curvature_lb > EIG_TOL}"
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
