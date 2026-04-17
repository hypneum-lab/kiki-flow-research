import numpy as np

from kiki_flow_core.track2_paper.pca_projection import PCAProjection


def test_pca_fits_and_projects_to_2d():
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((100, 384))
    pca = PCAProjection(n_components=2).fit(embeddings)
    projected = pca.project(embeddings)
    assert projected.shape == (100, 2)


def test_pca_inverse_approximates_original():
    rng = np.random.default_rng(1)
    u = rng.standard_normal((50, 2))
    basis = rng.standard_normal((2, 10))
    embeddings = u @ basis + 0.01 * rng.standard_normal((50, 10))
    pca = PCAProjection(n_components=2).fit(embeddings)
    recon = pca.inverse(pca.project(embeddings))
    err = float(np.linalg.norm(recon - embeddings) / np.linalg.norm(embeddings))
    assert err < 0.1  # noqa: PLR2004


def test_pca_deterministic_with_seed():
    rng1 = np.random.default_rng(0)
    emb1 = rng1.standard_normal((30, 8))
    rng2 = np.random.default_rng(0)
    emb2 = rng2.standard_normal((30, 8))
    p1 = PCAProjection(n_components=2, seed=7).fit(emb1).project(emb1)
    p2 = PCAProjection(n_components=2, seed=7).fit(emb2).project(emb2)
    np.testing.assert_allclose(p1, p2, atol=1e-10)
