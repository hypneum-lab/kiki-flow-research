import numpy as np
import pytest

from kiki_flow_core.wasserstein_ops import prox_w2, sinkhorn_cost, w2_distance

# Tolerances used across tests (names help ruff PLR2004 and document intent).
SELF_DISTANCE_TOL = 0.1  # relaxed: pure Sinkhorn has epsilon-dependent bias
SYMMETRY_TOL = 1e-5
GOLDEN_TOL = 1e-5
GAUSSIAN_DISCRETIZATION_TOL = 0.05
WEAK_PULL_TOL = 0.1  # weak proximal pull lands within this of the source variance


def test_w2_distance_to_self_is_small():
    """Sinkhorn-divergence implementation gives ~0 for self-distance; pure Sinkhorn gives an
    epsilon-dependent bias. We accept a relaxed tolerance here and rely on the
    Gaussian-Gaussian closed-form test below for metric correctness."""
    p = np.array([0.4, 0.6])
    d = w2_distance(p, p, support=np.array([[0.0], [1.0]]))
    assert d < SELF_DISTANCE_TOL


def test_w2_distance_symmetric():
    np.random.seed(1)
    p = np.random.dirichlet(np.ones(4))
    q = np.random.dirichlet(np.ones(4))
    support = np.linspace(0, 1, 4).reshape(-1, 1)
    d_pq = w2_distance(p, q, support=support)
    d_qp = w2_distance(q, p, support=support)
    assert abs(d_pq - d_qp) < SYMMETRY_TOL


def test_prox_w2_idempotent_on_reference():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    support = np.linspace(0, 1, 4).reshape(-1, 1)
    out = prox_w2(p, reference=p, epsilon=0.01, support=support, n_iter=200)
    np.testing.assert_allclose(out, p, atol=1e-3)


@pytest.mark.golden
def test_sinkhorn_5x5_matches_golden():
    data = np.load("tests/golden/sinkhorn_5x5.npz")
    cost = sinkhorn_cost(data["a"], data["b"], data["M"], epsilon=0.01, n_iter=100)
    assert abs(cost - float(data["expected_cost"])) < GOLDEN_TOL


def test_w2_gaussian_gaussian_closed_form():
    """W2^2(N(m1,s1), N(m2,s2)) = (m1-m2)^2 + (s1-s2)^2 in 1D."""
    n = 200
    support = np.linspace(-5, 5, n).reshape(-1, 1)
    m1, s1 = -1.0, 0.5
    m2, s2 = 1.0, 1.0
    p = np.exp(-0.5 * ((support[:, 0] - m1) / s1) ** 2)
    p /= p.sum()
    q = np.exp(-0.5 * ((support[:, 0] - m2) / s2) ** 2)
    q /= q.sum()
    expected_w2_sq = (m1 - m2) ** 2 + (s1 - s2) ** 2
    computed = w2_distance(p, q, support=support) ** 2
    assert abs(computed - expected_w2_sq) < GAUSSIAN_DISCRETIZATION_TOL  # discretization error


def test_prox_w2_jko_contracts_gaussian_variance():
    """A correct entropic-W2 prox is a contraction toward `reference` in W2 geometry.

    Anchor at a unit Gaussian (reference) and start from a *wider* Gaussian
    (distribution). The prox must (a) be idempotent on the reference and (b)
    interpolate monotonically: a strong proximal pull (small `epsilon`) lands near
    `reference`, a weak pull (large `epsilon`) stays near `distribution`. This is
    the contraction the broken `transport @ diag(cost)` gradient could never
    produce (that term is identically zero on a sqeuclidean self-cost)."""
    x = np.linspace(-5.0, 5.0, 121)
    support = x.reshape(-1, 1)

    def variance(rho: np.ndarray) -> float:
        m = float(np.dot(rho, x))
        return float(np.dot(rho, (x - m) ** 2))

    ref = np.exp(-0.5 * x**2)
    ref /= ref.sum()
    dist = np.exp(-0.5 * (x / 1.4) ** 2)
    dist /= dist.sum()
    var_ref, var_dist = variance(ref), variance(dist)
    assert var_dist > var_ref  # sanity: distribution is wider

    # Idempotency: prox(reference, reference) == reference.
    out_self = prox_w2(ref, reference=ref, epsilon=0.05, support=support, n_iter=400)
    np.testing.assert_allclose(out_self, ref, atol=1e-3)

    # Monotone interpolation in epsilon (proximal pull strength).
    v_strong = variance(prox_w2(dist, ref, epsilon=0.02, support=support, n_iter=600))
    v_mid = variance(prox_w2(dist, ref, epsilon=0.2, support=support, n_iter=600))
    v_weak = variance(prox_w2(dist, ref, epsilon=2.0, support=support, n_iter=600))

    # Strong pull contracts toward the (narrower) reference; weak pull stays wide.
    # (At weak pull the entropic prox adds a little diffusion, so v_weak may sit a
    # hair above var_dist -- allow that small overshoot.)
    assert var_ref <= v_strong < v_mid < v_weak
    assert abs(v_weak - var_dist) < WEAK_PULL_TOL  # weak pull ~ distribution (plus diffusion)
    assert v_strong < var_dist  # genuine contraction occurred
