"""Reference congruence, real adaptive pooling and library reconstruction."""

import numpy as np
import xarray as xr
from log_psplines import StationaryConfig
from log_psplines.inference.model import prepare_channel_model, prepare_model
from log_psplines.models.reconstruction import reconstruct_stationary_spectrum
from numpyro import handlers

from lisa_case_studies.reference_matrix import (
    adaptive_partition,
    coefficient_data,
    matrix_mean,
    pool_log_frequency,
    pooling_diagnostic,
)


def test_adaptive_pooling_preserves_unsmoothed_complex_products_and_counts():
    rng = np.random.default_rng(49)
    frequency = np.linspace(0.0297, 0.0303, 121)
    coefficients = (
        rng.normal(size=(121, 3)) + 1j * rng.normal(size=(121, 3))
    ) / np.sqrt(2)
    data = coefficient_data(coefficients, frequency)
    partition = adaptive_partition(data, tolerance=10, max_width=12, pilot_window=9)
    pooled = pool_log_frequency(data, partition)
    expected = np.add.reduceat(data.Y, partition.J_start, axis=0)
    np.testing.assert_allclose(pooled.Y, expected, atol=1e-12)
    np.testing.assert_array_equal(pooled.counts, partition.J_stop - partition.J_start)
    assert pooled.counts.sum() == len(frequency)
    assert np.any(pooled.counts > 1)
    for start, stop in zip(partition.J_start, partition.J_stop, strict=True):
        if np.any(np.abs(frequency[start:stop] - 0.03) < 1e-4):
            assert stop - start == 1
    assert partition.J_stop[0] == 1
    assert partition.J_start[-1] == len(frequency) - 1
    np.testing.assert_allclose(
        pooled.freq, 1 + np.log(partition.f_coarse / frequency[0])
    )
    assert np.max(np.abs(pooled.Y[:, 0, 1].imag)) > 0.1
    correction = np.repeat(
        (1 + np.arange(partition.Nc) / partition.Nc)[:, None, None] * np.eye(3),
        partition.freq_bin_counts,
        axis=0,
    )
    check = pooling_diagnostic(data, partition, correction)
    assert abs(check["difference_per_native_bin"]) < 1e-12


def test_native_reference_whitening_preserves_matrix_likelihood_differences():
    rng = np.random.default_rng(17)
    z = rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3))
    baseline = rng.normal(size=(9, 3, 3)) + 1j * rng.normal(size=(9, 3, 3))
    baseline = baseline @ baseline.conj().swapaxes(-1, -2) + np.eye(3)
    root = np.linalg.cholesky(baseline)
    white = np.linalg.solve(root, z[..., None])[..., 0]
    residuals = []
    for scale in (0.6, 1.4):
        a = np.array([[1, 0, 0], [0.3 + 0.2j, scale, 0], [-0.1j, 0.2, 1.1]])
        correction = np.broadcast_to(a @ a.conj().T, baseline.shape)
        physical = root @ correction @ root.conj().swapaxes(-1, -2)

        def nll(c, y):
            return (
                np.linalg.slogdet(c)[1].sum()
                + np.einsum(
                    "fi,fi->", y.conj(), np.linalg.solve(c, y[..., None])[..., 0]
                ).real
            )

        residuals.append(nll(physical, z) - nll(correction, white))
    np.testing.assert_allclose(
        residuals, np.linalg.slogdet(baseline)[1].sum(), rtol=1e-12
    )


def test_case_reconstruction_matches_library_and_uses_collapsed_theta():
    rng = np.random.default_rng(19)
    z = rng.normal(size=(40, 3)) + 1j * rng.normal(size=(40, 3))
    data = coefficient_data(z, np.linspace(0.001, 0.099, 40))
    config = StationaryConfig(
        n_knots=5, knot_kwargs={"method": "uniform"}, verbose=False
    )
    kwargs, components = prepare_model(data, config)
    posterior = {}
    for key in components.expected_component_order:
        k = components.component(key).n_basis
        posterior[f"weights_{key.name}"] = (
            ("chain", "draw", f"{key.name}_weights"),
            rng.normal(0, 0.1, (1, 4, k)),
        )
    posterior = xr.Dataset(posterior)
    native = np.asarray(
        reconstruct_stationary_spectrum(posterior, components, data)
    ).mean(axis=(0, 1))
    actual = matrix_mean(
        posterior, components, components.diagonal_models[0].frequency.grid
    )
    np.testing.assert_allclose(actual, native, rtol=1e-11, atol=1e-12)
    assert np.linalg.eigvalsh(actual).min() > 0
    model, row_kwargs, initial = prepare_channel_model(kwargs, 2)
    trace = handlers.trace(
        handlers.seed(handlers.substitute(model, data=initial), rng_seed=3)
    ).get_trace(**row_kwargs)
    sampled = [name for name, site in trace.items() if site["type"] == "sample"]
    assert any(name.startswith("sigma_theta_") for name in sampled)
    assert not any(name.startswith("weights_theta_") for name in sampled)
