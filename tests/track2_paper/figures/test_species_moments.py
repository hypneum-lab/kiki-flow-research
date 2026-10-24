from pathlib import Path

import numpy as np

from kiki_flow_core.track2_paper.figures import make_species_moments
from kiki_flow_core.track2_paper.figures.species_moments import species_moments


def test_moments_of_point_mass_and_uniform() -> None:
    x = np.linspace(-2.0, 2.0, 5)
    point = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    uniform = np.full(5, 0.2)
    mean, entropy = species_moments(np.stack([point, uniform]), x)
    np.testing.assert_allclose(mean, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(entropy, [0.0, np.log2(5)], atol=1e-9)


def test_moments_distinguish_what_rho_mean_cannot() -> None:
    # The retired phase portrait plotted rho.mean(), constant for any histogram.
    x = np.linspace(-2.0, 2.0, 8)
    left = np.zeros(8)
    left[0] = 1.0
    right = np.zeros(8)
    right[-1] = 1.0
    assert left.mean() == right.mean()
    mean, _ = species_moments(np.stack([left, right]), x)
    assert mean[0] < 0 < mean[1]


def test_make_species_moments_writes_png_and_pdf(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    arr = rng.dirichlet(np.ones(8), size=(2, 3, 4))  # seeds, steps, species, grid
    x = np.linspace(-2.0, 2.0, 8)
    out = make_species_moments({"fast": arr, "rigorous": arr}, x, out_dir=tmp_path)
    assert out.exists()
    assert (tmp_path / "fig1_species_moments.pdf").exists()
