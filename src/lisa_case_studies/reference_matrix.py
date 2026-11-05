"""Orbit reference + full Cholesky spline correction with collapsed VI.

The independent-frequency residual model is a Whittle approximation. The
nominal reference propagates the real orbit and gap/window operators; the
exact joint Gaussian in joint_gaussian remains a separate benchmark.
"""

import hashlib
import json
from dataclasses import replace

import numpy as np
from log_psplines import StationaryConfig, WishartData, fit
from log_psplines.inference.model import prepare_model
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainSpec,
    apply_coarse_grain_multivar_fft,
    compute_adaptive_binning_structure,
)
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from scipy.interpolate import BSpline


def coefficient_data(coefficients, frequency, *, enbw=1.0):
    """Encode rank-one complex XYZ products without adding artificial noise."""
    u = np.zeros((len(frequency), 3, 3), complex)
    u[:, :, 0] = coefficients
    return WishartData(
        u_re=u.real,
        u_im=u.imag,
        freq=frequency,
        N=len(frequency),
        p=3,
        Nb=1,
        duration=1.0,
        enbw=enbw,
        raw_psd=coefficients[:, :, None] * coefficients[:, None, :].conj(),
        preprocessing={
            "mean_removal": "latent DC zero; no data-dependent centering",
            "window": "GapLike gate/taper; native nominal-matrix whitening",
        },
    )


def adaptive_partition(pilot, *, tolerance=0.35, max_width=256, pilot_window=301):
    """Gapped training-only matrix pilot; same frozen bins for both observations.

    Pool unsmoothed statistics, not the pilot. Prespecified null neighbourhoods
    retain native bins. Endpoint singleton bins keep the reconstruction domain
    identical to the original Fourier grid. No validation/truth enters here.
    """
    adaptive = compute_adaptive_binning_structure(
        pilot, tolerance=tolerance, max_bin_width=max_width, pilot_window=pilot_window
    )
    f = pilot.freq
    protect = np.any(np.abs(f[:, None] - np.array([0.03, 0.06, 0.09])) < 1e-4, axis=1)
    indices = np.flatnonzero(protect)
    edges = np.unique(
        np.r_[
            adaptive.J_start,
            adaptive.J_stop,
            0,
            1,
            len(f) - 1,
            len(f),
            indices,
            indices + 1,
        ]
    )
    starts, stops = edges[:-1], edges[1:]
    mids = starts + (stops - starts - 1) // 2
    return CoarseGrainSpec(
        f_coarse=f[mids], Nh=1, J_start=starts, J_stop=stops, J_mid=mids, Nc=len(starts)
    )


def pool_log_frequency(data, partition):
    """Pool in physical frequency, then use log-frequency spline coordinates."""
    pooled = apply_coarse_grain_multivar_fft(data, partition)
    coordinate = 1 + np.log(pooled.freq / data.freq[0])
    return replace(pooled, freq=coordinate, raw_freq=coordinate)


def basis_at(basis, coordinate):
    """Restore the stationary library basis's endpoint multiplicities.

    The pinned design_at handles clamped vectors only, while stationary
    models store breakpoints. Our reconstruction test checks exact agreement
    with the library's fitting-grid evaluation, including its penalty model.
    """
    knots = np.asarray(basis.knots)
    if basis.knot_convention == "breakpoints":
        knots = np.r_[
            np.repeat(knots[0], basis.degree), knots, np.repeat(knots[-1], basis.degree)
        ]
    design = BSpline(
        knots, np.eye(basis.basis.shape[1]), basis.degree, extrapolate=False
    )(coordinate)
    if not np.isfinite(design).all():
        raise ValueError("Reconstruction outside the fitted spline domain")
    return design


def matrix_mean(posterior, components, coordinate, *, chunk=256):
    """Mean of positive-definite reconstructed draws, including restored theta.

    A matrix mean remains positive definite. Elementwise medians need not.
    Coordinates are in the library's normalized [0,1] spline domain.
    """
    coordinate = np.asarray(coordinate)
    result = np.empty((len(coordinate), 3, 3), complex)
    for start in range(0, len(coordinate), chunk):
        x = coordinate[start : start + chunk]
        h, lower = [], None
        for row in range(3):
            basis = basis_at(components.diagonal_models[row].frequency, x)
            weights = np.asarray(posterior[f"weights_delta_{row}"]).reshape(
                -1, basis.shape[1]
            )
            h.append(weights @ basis.T)
        h = np.stack(h, axis=-1)
        lower = np.zeros((*h.shape[:2], 3, 3), complex)
        for row, col in components.theta_pairs:
            for part, multiplier in (("re", 1), ("im", 1j)):
                basis = basis_at(
                    components.get_theta_model(part, row, col).frequency, x
                )
                weights = np.asarray(
                    posterior[f"weights_theta_{part}_{row}_{col}"]
                ).reshape(-1, basis.shape[1])
                lower[..., row, col] += multiplier * (weights @ basis.T)
        # (I-theta)^-1 = I + theta + theta^2 for a 3x3 strict lower triangle.
        inverse = np.eye(3) + lower + lower @ lower
        covariance = (inverse * np.exp(h)[..., None, :]) @ inverse.conj().swapaxes(
            -1, -2
        )
        result[start : start + len(x)] = covariance.mean(axis=0)
    if not np.isfinite(result).all() or np.any(np.linalg.eigvalsh(result) <= 0):
        raise RuntimeError("Non-finite or non-positive matrix reconstruction")
    return result


def elbo_diagnostics(losses, represented_bins):
    """Prespecified numerical plateau check; not posterior convergence."""
    losses = np.asarray(losses, float)
    if not np.isfinite(losses).all():
        raise RuntimeError("Non-finite VI losses")
    width = max(2, len(losses) // 5)
    tail, previous = losses[-width:], losses[-2 * width : -width]
    if len(previous) == 0:
        previous = losses[:width]
    drift = abs(float(tail.mean() - previous.mean())) / represented_bins
    return dict(
        tail_mean=float(tail.mean()),
        tail_std=float(tail.std()),
        tail_mean_change_per_native_bin=drift,
        plateau=bool(len(losses) >= 5 and drift < 1e-3),
        interpretation="Stochastic ELBO plateau diagnostic only; no NUTS-equivalent convergence or uncertainty claim",
    )


def fit_reference_correction(
    data, partition, output, name, *, seed, quick=False, steps=None, restarts=3
):
    """Use the library's marginal model and restore exact conditional theta."""
    pooled = pool_log_frequency(data, partition)
    steps = steps or (1500 if quick else 6000)
    n_draws = 64 if quick else 256
    config = StationaryConfig(
        method="vi",
        n_knots=10,
        degree=3,
        knot_kwargs={"method": "uniform"},
        roughness_scale=1.28,
        vi_steps=steps,
        vi_lr=0.01,
        vi_guide="mvn",
        vi_posterior_draws=n_draws,
        vi_early_stopping=False,
        vi_progress_bar=False,
        verbose=False,
    )
    _, components = prepare_model(pooled, config)
    basis_payload = {}
    basis_hash = hashlib.sha256()
    for key in components.expected_component_order:
        field = components.component(key)
        for kind, value in (
            ("knots", field.knots),
            ("penalty", field.penalty_matrix),
            ("basis", field.basis),
        ):
            array = np.asarray(value, dtype=np.float64)
            basis_payload[f"{kind}_{key.name}"] = array
            basis_hash.update(array.tobytes())
    np.savez_compressed(output / f"{name}_basis.npz", **basis_payload)
    # Singleton products are rank one: their empirical conditional variances
    # are zero and make poor VI starting values. Start at the known reference
    # R=I and sigma=0.3. These are guide initial values only: all library
    # likelihoods, Gaussian penalties and hyperpriors remain unchanged.
    initial = {}
    for key in components.expected_component_order:
        initial[f"sigma_{key.name}"] = 0.3
        if key.family == "delta":
            initial[f"weights_{key.name}_raw"] = np.zeros(
                components.component(key).n_basis
            )

    def reference_guide(model):
        return AutoMultivariateNormal(model, init_loc_fn=init_to_value(values=initial))

    config = replace(config, vi_guide=reference_guide)
    diagnostic_x = np.linspace(0, 1, 192)
    candidates, diagnostics, traces, summaries = [], [], {}, []
    for restart in range(restarts):
        run_seed = seed + 200 + restart
        print(
            f"{name}: collapsed VI restart {restart + 1}/{restarts}, {steps} updates per row",
            flush=True,
        )
        candidate = fit(pooled, replace(config, rng_key=run_seed))
        if candidate.metadata["theta_parameterization"] != "marginalized":
            raise RuntimeError("Library did not use marginalised theta")
        blocks = candidate.vi.losses_per_block
        block_checks = [
            elbo_diagnostics(loss, int(data.counts.sum())) for loss in blocks
        ]
        score = float(sum(check["tail_mean"] for check in block_checks))
        for row, loss in enumerate(blocks):
            traces[f"restart_{restart}_row_{row}"] = np.asarray(loss)
        diagnostics.append(
            dict(
                seed=run_seed,
                selection_score=score,
                rows=block_checks,
                theta_parameterization=candidate.metadata["theta_parameterization"],
                restored_theta_fields=[
                    field
                    for field in candidate.posterior
                    if field.startswith("weights_theta_")
                ],
                timings=candidate.vi.timings,
            )
        )
        summaries.append(matrix_mean(candidate.posterior, components, diagnostic_x))
        candidates.append(candidate)
    chosen = int(np.argmin([d["selection_score"] for d in diagnostics]))
    comparison = []
    for i in range(restarts):
        for j in range(i):
            a = np.diagonal(summaries[i], axis1=-2, axis2=-1).real
            b = np.diagonal(summaries[j], axis1=-2, axis2=-1).real
            comparison.append(float(np.sqrt(np.mean(np.log10(a / b) ** 2))))
    selected = candidates[chosen]
    selected.to_netcdf(output / f"{name}_correction.nc")
    np.savez_compressed(output / f"{name}_elbo.npz", **traces)
    report = dict(
        method="Library collapsed multivariate VI; frequency-only correction to evolving reference",
        selected_restart=chosen,
        selection="Minimum mean trailing negative ELBO; no truth/validation used",
        restarts=diagnostics,
        steps_per_row=steps,
        vi_draws=n_draws,
        guide="mvn",
        guide_initialization="Known reference R=I, all smoothing scales start at 0.3; prior unchanged",
        basis_penalty_sha256=basis_hash.hexdigest(),
        all_rows_plateau=bool(
            all(check["plateau"] for d in diagnostics for check in d["rows"])
        ),
        max_restart_auto_log10_rmse=max(comparison, default=0),
        basis_functions_per_field=components.n_basis,
        real_spline_fields=9,
        theta_integrated_fields=6,
        sampled_delta_fields=3,
        prior="Library stationary max-normalized derivative penalty, ridge 1e-6; sigma ~ HalfNormal(1.28); same priors for both observations",
    )
    (output / f"{name}_vi.json").write_text(json.dumps(report, indent=2))
    return selected, components, report


def marginal_predictive_score(coefficients, reference, correction):
    """Independent-draw per-bin complex Gaussian score; explicitly approximate."""
    root = np.linalg.cholesky(reference)
    whitened = np.linalg.solve(root[None], coefficients[..., None])[..., 0]
    solved = np.linalg.solve(correction[None], whitened[..., None])[..., 0]
    q = np.einsum("rfi,rfi->rf", whitened.conj(), solved).real
    logdet = np.linalg.slogdet(correction)[1]
    return dict(
        mean_q_per_dimension=float(q.mean() / 3),
        mean_nll_per_dimension=float((q + logdet).mean() / 3),
        individual_q_per_dimension=(q.mean(axis=1) / 3).tolist(),
        target="Marginal complex Whittle score in fixed reference coordinates; drops frequency/pseudo-covariance; not comparable to full-joint log likelihood",
    )


def pooling_diagnostic(data, partition, correction):
    """Compare native and pooled point likelihoods at the fitted curve."""
    pooled = apply_coarse_grain_multivar_fft(data, partition)
    center = correction[partition.J_mid]
    native = (
        np.linalg.slogdet(correction)[1]
        + np.einsum("fij,fji->f", np.linalg.inv(correction), data.Y).real
    ).sum() / data.enbw
    coarse = (
        pooled.counts * np.linalg.slogdet(center)[1]
        + np.einsum("fij,fji->f", np.linalg.inv(center), pooled.Y).real
    ).sum() / data.enbw
    return dict(
        native_point_nll=float(native),
        pooled_point_nll=float(coarse),
        difference_per_native_bin=float((coarse - native) / data.counts.sum()),
        interpretation="One fitted-point diagnostic, not a posterior error bound or a fine-grid inference comparison",
    )
