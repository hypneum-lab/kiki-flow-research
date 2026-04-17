import numpy as np

from kiki_flow_core.modules.phonological_loop import PhonologicalLoop


def test_zero_error_produces_zero_source():
    loop = PhonologicalLoop(detector=lambda out: np.zeros(out.size), correction_strength=1.0)
    output = np.array([1.0, 0.5, 0.2])
    s = loop.source_term(rho_phono=np.array([0.3, 0.4, 0.3]), output=output)
    assert (s == 0).all()


def test_uniform_error_produces_uniform_source():
    loop = PhonologicalLoop(detector=lambda out: np.full(out.size, 0.5), correction_strength=1.0)
    output = np.array([0.3, 0.3, 0.4])
    s = loop.source_term(rho_phono=np.array([0.3, 0.4, 0.3]), output=output)
    assert np.allclose(s, s.mean())


def test_correction_gain_bounded():
    """Source magnitude should not exceed correction_strength regardless of error magnitude."""
    loop = PhonologicalLoop(
        detector=lambda out: np.full(out.size, 1e6),  # absurdly high error
        correction_strength=0.5,
    )
    s = loop.source_term(rho_phono=np.array([0.5, 0.5]), output=np.array([1.0, 1.0]))
    assert np.abs(s).max() <= 0.5 + 1e-6  # noqa: PLR2004


def test_balanced_signed_errors_sum_to_zero():
    """A detector producing balanced positive/negative errors yields a zero-sum source."""
    loop = PhonologicalLoop(
        detector=lambda out: np.array([0.3, -0.3]),
        correction_strength=1.0,
    )
    s = loop.source_term(rho_phono=np.array([0.5, 0.5]), output=np.array([1.0, 1.0]))
    assert abs(s.sum()) < 1e-6  # noqa: PLR2004


def test_detector_is_called_with_output():
    """Ensure the detector callback receives the model output."""
    received: list[np.ndarray] = []

    def recording_detector(out: np.ndarray) -> np.ndarray:
        received.append(out.copy())
        return np.zeros_like(out)

    loop = PhonologicalLoop(detector=recording_detector, correction_strength=1.0)
    out = np.array([0.1, 0.2, 0.7])
    loop.source_term(rho_phono=np.array([0.3, 0.3, 0.4]), output=out)
    assert len(received) == 1
    np.testing.assert_array_equal(received[0], out)
