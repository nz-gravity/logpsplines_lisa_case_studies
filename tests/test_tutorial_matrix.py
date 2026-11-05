"""Independent Gaussian operator controls for the channel-level tutorials."""

import numpy as np

from lisa_case_studies.tutorial_data import (
    MatrixSimulation,
    matrix_root,
    residual_matrix,
)


def test_tutorial_joint_covariance_matches_independent_time_domain_columns():
    simulation = MatrixSimulation(
        24,
        5.0,
        np.array([0.0, 115.0]),
        np.array(
            [[8.29, 8.31, 8.35, 8.34, 8.30, 8.32], [8.35, 8.27, 8.32, 8.31, 8.34, 8.28]]
        ),
    )
    window = np.hanning(simulation.n)
    window[7:11] = 0
    bins = np.array([2, 3, 7])
    correction = residual_matrix(simulation.frequency[1:])
    actual = simulation.feature_covariance(window, bins, correction=correction)
    transfer = simulation.roots @ matrix_root(correction)[None]
    transfer[:, -1] = transfer[:, -1].real
    expected = np.zeros_like(actual)
    for j in range(1, simulation.n // 2 + 1):
        for source in range(3):
            for quad in range(1 if j == simulation.n // 2 else 2):
                amplitude = np.sqrt(simulation.n / (2 * simulation.dt))
                if j != simulation.n // 2:
                    amplitude /= np.sqrt(2)
                amplitude *= 1 if quad == 0 else 1j
                series = np.zeros((simulation.n, 3))
                for e, weight in enumerate(simulation.weights):
                    coefficients = np.zeros((simulation.n // 2 + 1, 3), complex)
                    coefficients[j] = transfer[e, j - 1, :, source] * amplitude
                    series += weight[:, None] * np.fft.irfft(
                        coefficients, n=simulation.n, axis=0
                    )
                measured = simulation.features(series, window, bins)
                expected += np.outer(measured, measured)
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(actual / scale, expected / scale, rtol=1e-10, atol=1e-12)
    nominal = simulation.feature_covariance(window, bins)
    reference = simulation.window_references({"gap": window})["gap"]
    half = len(nominal) // 2
    complex_cov = (
        nominal[:half, :half]
        + nominal[half:, half:]
        + 1j * (nominal[half:, :half] - nominal[:half, half:])
    )
    for i, k in enumerate(bins):
        np.testing.assert_allclose(
            reference[k] / scale,
            complex_cov[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] / scale,
            rtol=1e-10,
            atol=1e-12,
        )


def test_tutorial_frozen_epoch_reduces_to_one_stationary_filter():
    delays = np.array([[8.29, 8.31, 8.35, 8.34, 8.30, 8.32]])
    one = MatrixSimulation(32, 5.0, np.array([0.0]), delays)
    repeated = MatrixSimulation(
        32, 5.0, np.array([0.0, 155.0]), np.repeat(delays, 2, axis=0)
    )
    np.testing.assert_allclose(one.draw(17), repeated.draw(17), rtol=1e-12, atol=1e-32)
