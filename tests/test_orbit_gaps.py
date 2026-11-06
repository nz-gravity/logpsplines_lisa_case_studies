"""Response identities and a dense, independent gap-covariance calculation."""

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from lisa_case_studies.noise import ARM_LENGTH, C, xyz_components
from lisa_case_studies.orbit_reference import (
    reference_components,
    two_sided_matrix,
    windowed_matrix,
)


def test_equal_arm_limit_matches_first_generation_times_tdi2_factor():
    f = np.geomspace(0.0004, 0.099, 83)
    actual = reference_components(f, np.full(6, ARM_LENGTH / C))
    factor = 4 * np.sin(4 * np.pi * f * ARM_LENGTH / C) ** 2
    for value, old in zip(actual, xyz_components(f), strict=True):
        np.testing.assert_allclose(
            value, old * factor[:, None, None], rtol=2e-11, atol=1e-54
        )


def test_epoch_average_is_average_of_covariances_and_keeps_complex_cross_terms():
    f = np.linspace(0.029, 0.031, 51)
    epochs = np.array(
        [[8.29, 8.31, 8.35, 8.34, 8.30, 8.32], [8.30, 8.32, 8.34, 8.33, 8.31, 8.33]]
    )
    average = reference_components(f, epochs)
    individual = [reference_components(f, epoch) for epoch in epochs]
    for k, value in enumerate(average):
        np.testing.assert_allclose(
            value, (individual[0][k] + individual[1][k]) / 2, rtol=1e-12, atol=1e-54
        )
        np.testing.assert_allclose(
            value, value.conj().swapaxes(-1, -2), rtol=1e-12, atol=1e-54
        )
    assert np.max(np.abs(average[1][:, 0, 1].imag)) > 0
    eigen = np.linalg.eigvalsh(sum(average))
    assert eigen.min() > 0


def test_matrix_window_matches_full_time_covariance_not_only_real_diagonals():
    n, dt = 24, 5.0
    frequency = np.fft.rfftfreq(n, dt)
    matrix = np.zeros((len(frequency), 3, 3), complex)
    # Frequency-dependent complex cross spectra with real-series symmetry.
    for k, f in enumerate(frequency):
        root = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.35 * np.exp(2j * np.pi * f * 4), 1.0, 0.0],
                [0.2, 0.3 * np.exp(-2j * np.pi * f * 7), 0.8],
            ]
        )
        matrix[k] = root @ root.conj().T
    full = two_sided_matrix(matrix, n)
    gamma = np.fft.ifft(full, axis=0).real / (2 * dt)
    lag = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
    time_cov = gamma[lag].transpose(0, 2, 1, 3).reshape(n * 3, n * 3)
    window = np.ones(n)
    window[5:9] = 0
    window[:3] *= np.array([0.2, 0.6, 0.9])
    selected = np.array([2, 5, 9])
    transform = (
        np.sqrt(2 * dt / n)
        * np.exp(-2j * np.pi * selected[:, None] * np.arange(n)[None, :] / n)
        * window
    )
    operator = np.kron(transform, np.eye(3))
    dense = (operator @ time_cov @ operator.conj().T).reshape(
        len(selected), 3, len(selected), 3
    )
    actual = windowed_matrix(matrix, window)
    for i, k in enumerate(selected):
        np.testing.assert_allclose(actual[k], dense[i, :, i, :], rtol=1e-12, atol=1e-12)
    # Gaps really do couple distinct frequency bins; the point-fit prototype
    # explicitly omits this block rather than claiming an exact likelihood.
    assert np.max(np.abs(dense[0, :, 1, :])) > 1e-3


def test_gaplike_exact_stationary_control_matches_dense_likelihood_differences():
    gl = pytest.importorskip("gaplike")
    n, dt = 64, 5.0
    frequency = np.fft.fftfreq(n, dt)
    components = [1 / (1 + (frequency / 0.015) ** 2), 0.4 * np.ones(n)]
    mask = gl.gaps.mask_from_intervals(n, dt, [(60.0, 100.0), (220.0, 245.0)])
    exact = gl.TimeDomainExact(mask, components, dt)
    data = np.random.default_rng(4).normal(size=(2, n))
    observed = exact.transform(data)
    matrices = [
        gl.covariance.restricted_autocov(c, np.flatnonzero(mask), dt)
        for c in components
    ]

    def dense(lam):
        cov = sum(10**x * c for x, c in zip(lam, matrices, strict=True))
        factor = cho_factor(cov, lower=True)
        logdet = 2 * np.log(np.diag(factor[0])).sum()
        quadratic = np.sum(observed * cho_solve(factor, observed.T).T)
        return -0.5 * (2 * logdet + quadratic)

    a, b = np.array([0.13, -0.2]), np.array([-0.1, 0.25])
    actual = exact.loglike(observed, a) - exact.loglike(observed, b)
    expected = dense(a) - dense(b)
    assert np.isfinite(actual) and np.isfinite(expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, equal_nan=False)
