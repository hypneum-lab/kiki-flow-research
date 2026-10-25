"""Light tests for ``experiments/jko_step_check_2026_10_25.py``.

They pin what the check relies on, not its scientific outcome: the test
energy's analytic gradients agree with the base-class finite difference, the
step wrappers call ``JKOStep``/``FullJKOSolver`` with the intended h, reg and
Dykstra count, the exact 1-D W2 agrees between POT and the quantile formula,
the generalised Sinkhorn reference reproduces ``prox_w2`` for the functional
``prox_w2`` actually minimises, and the exact JKO reference solver returns a
certified, dissipative minimiser on a tiny grid. All outputs are new files.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from kiki_flow_core.wasserstein_ops import prox_w2

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "experiments" / "jko_step_check_2026_10_25.py"

GRID = 12
H = 0.05
REG = 0.01
N_INNER = 3
FD_EPS = 1e-7
GRAD_TOL = 1e-4
W2_TOL = 1e-10
PROX_MATCH_TOL = 1e-9
GAP_TOL = 1e-7
DISSIPATION_TOL = 1e-10
SIMPLEX_TOL = 1e-10
LARGE_H = 1e4
LARGE_H_TV_TOL = 1e-3
SMALL_H = 1e-5
SMALL_H_TV_TOL = 2e-2
ANCHOR_H = 0.02
ANCHOR_GAMMA = 0.005
FLOW_T = 0.05
PEYRE_MATCH_TOL = 1e-6


def _load() -> object:
    name = "exp_jko_step_check_2026_10_25"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dirichlet(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.dirichlet(np.full(n, 5.0))


def test_analytic_gradients_match_base_class_finite_difference() -> None:
    mod = _load()
    x = mod.support(GRID)
    spec = mod.energies(x)[0]
    rng = np.random.default_rng(0)
    rho = _dirichlet(rng, GRID)
    state = mod.make_state(rho)
    numerical = mod.PotentialEntropyF(spec, "numerical").grad_rho(state, mod.SPECIES, eps=FD_EPS)
    tangential = mod.PotentialEntropyF(spec, "tangential").grad_rho(state, mod.SPECIES)
    raw = mod.PotentialEntropyF(spec, "raw").grad_rho(state, mod.SPECIES)
    np.testing.assert_allclose(numerical, tangential, atol=GRAD_TOL)
    # Base-class convention: the renormalised perturbation is the tangent e_i - rho.
    np.testing.assert_allclose(tangential, raw - float(np.dot(raw, rho)), atol=1e-12)
    with pytest.raises(ValueError, match="grad_mode"):
        mod.PotentialEntropyF(spec, "bogus")


def test_kiki_step_forwards_h_reg_and_dykstra_count() -> None:
    mod = _load()
    x = mod.support(GRID)
    spec = mod.energies(x)[1]
    f = mod.PotentialEntropyF(spec, "tangential")
    calls: list[dict] = []
    original = mod.FullJKOSolver.step

    def spy_step(self, state):  # type: ignore[no-untyped-def]
        inner = self.prox_fn

        def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs)
            return inner(*args, **kwargs)

        self.prox_fn = spy
        return original(self, state)

    mod.FullJKOSolver.step = spy_step
    try:
        rig = mod.kiki_step(f, spec.rho_tau, x, H, N_INNER, rigorous=True, reg=REG)
    finally:
        mod.FullJKOSolver.step = original
    assert len(calls) == 1
    assert calls[0]["reg"] == REG
    assert calls[0]["epsilon"] == H
    assert calls[0]["n_iter"] == mod.PROX_N_ITER
    np.testing.assert_allclose(calls[0]["reference"], spec.rho_tau)
    # Same output as building FullJKOSolver by hand.
    solver = mod.FullJKOSolver(
        f_functional=f,
        h=H,
        support=x.reshape(-1, 1),
        epsilon=REG,
        max_iter=mod.PROX_N_ITER,
        n_inner=N_INNER,
    )
    expected = solver.step(mod.make_state(spec.rho_tau)).rho[mod.SPECIES]
    np.testing.assert_allclose(rig, expected)
    fast = mod.kiki_step(f, spec.rho_tau, x, H, N_INNER, rigorous=False)
    assert abs(fast.sum() - 1.0) < SIMPLEX_TOL
    assert not np.allclose(fast, rig)


def test_exact_w2_matches_quantile_formula() -> None:
    mod = _load()
    x = mod.support(GRID)
    cost = mod.cost_matrix(x)
    rng = np.random.default_rng(1)
    for _ in range(5):
        p, q = _dirichlet(rng, GRID), _dirichlet(rng, GRID)
        assert abs(mod.w2sq_exact(p, q, cost) - mod.w2sq_quantile(p, q, x)) < W2_TOL
    assert mod.w2_exact(p, p, cost) < W2_TOL
    assert mod.tv(p, p) == 0.0


def test_generalised_sinkhorn_reproduces_prox_w2_for_kl_to_target() -> None:
    """The repo prox is Peyre's entropic JKO step of KL(. || distribution)."""
    mod = _load()
    x = mod.support(GRID)
    cost = mod.cost_matrix(x)
    rng = np.random.default_rng(2)
    rho_tau = _dirichlet(rng, GRID)
    target = _dirichlet(rng, GRID)
    ours, info = mod.jko_entropic(
        mod.kl_prox_to_target(target), rho_tau, cost, ANCHOR_H, ANCHOR_GAMMA
    )
    repo = prox_w2(
        target,
        reference=rho_tau,
        epsilon=ANCHOR_H,
        support=x.reshape(-1, 1),
        n_iter=5000,
        reg=ANCHOR_GAMMA,
    )
    np.testing.assert_allclose(ours, repo, atol=PROX_MATCH_TOL)
    assert info["marginal_residual"] < PROX_MATCH_TOL


def test_exact_jko_reference_is_certified_and_dissipative() -> None:
    mod = _load()
    x = mod.support(GRID)
    cost = mod.cost_matrix(x)
    spec = mod.energies(x)[0]
    f = mod.PotentialEntropyF(spec)
    ref = mod.jko_exact(f.energy, f.first_variation, spec.rho_tau, cost, H)
    assert ref.converged
    assert ref.gap < GAP_TOL
    assert ref.dual_violation < W2_TOL
    assert abs(ref.rho.sum() - 1.0) < SIMPLEX_TOL
    assert ref.rho.min() > 0.0
    lhs = f.energy(ref.rho) + mod.w2sq_exact(ref.rho, spec.rho_tau, cost) / (2.0 * H)
    assert lhs <= f.energy(spec.rho_tau) + DISSIPATION_TOL
    assert abs(lhs - ref.objective) < GAP_TOL
    # h -> infinity: the Gibbs state; h -> 0: rho_tau.
    gibbs = mod.simplex(np.exp(-spec.potential / spec.theta))
    far = mod.jko_exact(f.energy, f.first_variation, spec.rho_tau, cost, LARGE_H)
    assert mod.tv(far.rho, gibbs) < LARGE_H_TV_TOL
    near = mod.jko_exact(f.energy, f.first_variation, spec.rho_tau, cost, SMALL_H)
    assert mod.tv(near.rho, spec.rho_tau) < SMALL_H_TV_TOL


def test_fp_generator_conserves_mass_and_fixes_gibbs() -> None:
    mod = _load()
    x = mod.support(GRID)
    for spec in mod.energies(x):
        gen = mod.fp_generator(spec, x)
        np.testing.assert_allclose(gen.sum(axis=0), 0.0, atol=1e-9)
        gibbs = mod.simplex(np.exp(-spec.potential / spec.theta))
        np.testing.assert_allclose(gen @ gibbs, 0.0, atol=1e-9)
        flowed = mod.fp_flow(gen, spec.rho_tau, FLOW_T)
        assert abs(flowed.sum() - 1.0) < SIMPLEX_TOL
        assert flowed.min() > 0.0
    # V = 0: symmetric diffusion with coefficient theta / dx^2.
    ent = mod.energies(x)[2]
    gen = mod.fp_generator(ent, x)
    dx = x[1] - x[0]
    np.testing.assert_allclose(gen, gen.T, atol=1e-12)
    np.testing.assert_allclose(gen[0, 1], ent.theta / dx**2)


def test_kiki_step_scheme_flag_and_reg_specs() -> None:
    mod = _load()
    x = mod.support(GRID)
    spec = mod.energies(x)[0]
    f = mod.PotentialEntropyF(spec)
    legacy = mod.kiki_step(f, spec.rho_tau, x, H, N_INNER, rigorous=True, reg=REG)
    fixed = mod.kiki_step(
        f, spec.rho_tau, x, H, N_INNER, rigorous=True, reg=REG, scheme="jko_entropic"
    )
    assert not np.allclose(legacy, fixed)
    # The entropic step equals the script's own Peyre reference (same problem).
    cost = mod.cost_matrix(x)
    peyre, _ = mod.jko_entropic(f.kl_prox_fn(), spec.rho_tau, cost, H, REG)
    assert mod.tv(fixed, peyre) < PEYRE_MATCH_TOL
    with pytest.raises(ValueError, match="scheme"):
        mod.kiki_step(f, spec.rho_tau, x, H, N_INNER, rigorous=True, reg=REG, scheme="bogus")
    assert mod.parse_regs("0.05, h, h^2") == [0.05, None, "h^2"]
    assert mod.resolve_reg("h^2", H) == H * H
    assert mod.resolve_reg(None, H) is None
    assert mod.resolve_reg(0.02, H) == 0.02  # noqa: PLR2004
    assert mod.reg_key("h^2") == "h^2"
    assert mod.reg_key(None) == "default(h)"


def test_outputs_are_new_files_under_results() -> None:
    mod = _load()
    for path in mod.output_paths().values():
        assert path.parent == mod.OUT_DIR
        assert mod.OUT_DIR.parts[0] == "results"
    assert not any(part.startswith("paper") for part in mod.OUT_DIR.resolve().parts)
