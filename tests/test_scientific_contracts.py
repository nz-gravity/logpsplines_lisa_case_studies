"""Independent formula, simulation-normalization and matrix-target checks."""

import jax
import jax.numpy as jnp
import numpy as np
from log_psplines import WishartGridData

from lisa_case_studies.noise import (
    ARM_LENGTH,
    XYZ_TO_AET,
    C,
    aet_components,
    coherence,
    sample_complex,
    sample_real_powers,
    xyz_components,
)
from lisa_case_studies.xyz import matrix_log_likelihood


def test_xyz_templates_match_independent_michelson_formulas():
    f = np.geomspace(0.4e-3, 25e-3, 40)
    x = 2 * np.pi * f * ARM_LENGTH / C
    pm = (
        (3e-15) ** 2
        * (1 + (0.4e-3 / f) ** 2)
        * (1 + (f / 8e-3) ** 4)
        / (2 * np.pi * f * C) ** 2
    )
    op = (15e-12) ** 2 * (1 + (2e-3 / f) ** 4) * (2 * np.pi * f / C) ** 2
    expected_x = 16 * np.sin(x) ** 2 * (op + 2 * (1 + np.cos(x) ** 2) * pm)
    expected_xy = -8 * np.sin(x) ** 2 * np.cos(x) * (op + 4 * pm)
    acc, oms = xyz_components(f)
    matrix = acc + oms
    np.testing.assert_allclose(matrix[:, 0, 0], expected_x, rtol=1e-12)
    np.testing.assert_allclose(matrix[:, 0, 1], expected_xy, rtol=1e-12)
    assert np.linalg.eigvalsh(matrix).min() > 0
    np.testing.assert_allclose(XYZ_TO_AET @ XYZ_TO_AET.T, np.eye(3), atol=1e-15)
    diagonal = sum(aet_components(f))
    rotated = XYZ_TO_AET @ matrix @ XYZ_TO_AET.T
    np.testing.assert_allclose(
        np.diagonal(rotated, axis1=-2, axis2=-1), diagonal, rtol=1e-8
    )
    assert np.all(coherence(matrix) <= 1 + 1e-12)


def test_gaussian_simulators_have_the_stated_psd_normalization():
    rng = np.random.default_rng(841)
    real = sample_real_powers(np.ones(100_000), 4, rng)
    np.testing.assert_allclose(real.mean(), 4, rtol=0.02)
    np.testing.assert_allclose(real.var(), 8, rtol=0.03)
    matrix = np.array([[2.0, 0.7, 0.2], [0.7, 1.5, 0.1], [0.2, 0.1, 1.0]])
    z = sample_complex(matrix, 100_000, rng)
    empirical = z @ z.conj().T / z.shape[-1]
    np.testing.assert_allclose(empirical, matrix, rtol=0.04, atol=0.02)
    # Properness is a separate requirement from covariance normalization.
    assert np.max(np.abs(z @ z.T / z.shape[-1])) < 0.03


def test_matrix_likelihood_and_gradient_match_direct_logdet_solve():
    rng = np.random.default_rng(5)
    f = np.geomspace(0.001, 0.02, 8)
    acc, oms = xyz_components(f)
    whitening = np.linalg.inv(np.linalg.cholesky(acc + oms))

    def transform(m):
        return whitening @ m @ np.swapaxes(whitening, -1, -2)

    a, b = transform(acc), transform(oms)
    truth = 1.4 * a + 0.8 * b
    data = WishartGridData.from_coefficients(
        sample_complex(truth, 7, rng)[None], np.array([0.0]), f
    )

    def target(values):
        matrix = jnp.exp(values[0]) * a + jnp.exp(values[1]) * b
        return matrix_log_likelihood(matrix[None], data)

    def direct(values):
        matrix = jnp.exp(values[0]) * a + jnp.exp(values[1]) * b
        return -jnp.sum(
            data.counts * jnp.linalg.slogdet(matrix)[1]
            + jnp.trace(
                jnp.linalg.solve(matrix, jnp.asarray(data.Y[0])), axis1=-2, axis2=-1
            ).real
        )

    point = jnp.array([0.1, -0.2])
    np.testing.assert_allclose(target(point), direct(point), rtol=1e-12)
    np.testing.assert_allclose(
        jax.grad(target)(point), jax.grad(direct)(point), rtol=1e-12, atol=1e-12
    )


def test_full_covariance_information_is_not_replaced_by_diagonal_powers():
    rng = np.random.default_rng(4)
    matrix = np.broadcast_to(
        np.array([[2.0, 0.8, 0.3], [0.8, 1.5, 0.2], [0.3, 0.2, 1.0]]), (5, 3, 3)
    )
    z = sample_complex(matrix, 8, rng)
    data = WishartGridData.from_coefficients(
        z[None], np.array([0.0]), np.arange(1.0, 6.0)
    )
    diagonal = np.eye(3) * np.diagonal(matrix, axis1=-2, axis2=-1)[..., None, :]
    assert not np.isclose(
        matrix_log_likelihood(matrix[None], data),
        matrix_log_likelihood(diagonal[None], data),
    )
