"""Seven-day orbit / gap-window point-fit prototype, not an exact TV likelihood."""

import importlib.metadata
import json

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from log_psplines import WishartGridData
from scipy.optimize import minimize

from .common import create_output, parser, receipt, save_figure
from .orbit_reference import (
    DAY,
    LINKS,
    orbit_light_times,
    reference_components,
    simulate_circulant_xyz,
    two_sided_matrix,
    windowed_matrix,
)
from .xyz import matrix_log_likelihood


def point_fit(components, coefficients, whitening):
    """Full per-bin XYZ Whittle MLE; no interval calibration is claimed."""
    adjoint = whitening.conj().swapaxes(-1, -2)
    templates = jnp.asarray([whitening @ c @ adjoint for c in components])
    data = WishartGridData.from_coefficients(
        (whitening @ coefficients[..., None])[None],
        np.array([0.0]),
        np.arange(1.0, len(coefficients) + 1),
    )

    @jax.jit
    def objective(log_scales):
        covariance = jnp.einsum("k,kfij->fij", jnp.exp(log_scales), templates)
        return -matrix_log_likelihood(covariance[None], data)

    value_gradient = jax.jit(jax.value_and_grad(objective))
    fitted = minimize(
        lambda x: tuple(np.asarray(v) for v in value_gradient(x)),
        np.zeros(2),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-3, 3)] * 2,
        options={"ftol": 1e-12, "gtol": 1e-6, "maxiter": 200},
    )
    if not fitted.success:
        raise RuntimeError(f"point optimization failed: {fitted.message}")
    scales = np.exp(fitted.x)
    model = sum(scale * c for scale, c in zip(scales, components, strict=True))
    normalized = whitening @ model @ adjoint
    observed = whitening @ coefficients[..., None]
    q = np.einsum(
        "fi,fij,fj->f",
        observed[..., 0].conj(),
        np.linalg.inv(normalized),
        observed[..., 0],
    ).real
    return scales, dict(
        negative_log_likelihood=float(fitted.fun),
        mean_whitened_q=float(q.mean()),
        iterations=int(fitted.nit),
        success=bool(fitted.success),
        at_parameter_bound=bool(np.any(np.abs(fitted.x) > 2.999)),
        q_interpretation="In-sample MLE amplitude score enforces mean q near 3; not an independent goodness-of-fit test",
    )


def main():
    try:
        import gaplike as gl
    except ImportError as error:
        raise SystemExit(
            "Install optional dependencies: uv sync --extra gapped"
        ) from error
    cli = parser(__doc__, "outputs/03_orbit_response_and_gaps")
    cli.add_argument(
        "--orbit",
        choices=["keplerian", "esa"],
        default="keplerian",
        help="physical Keplerian approximation (offline), or ESA trailing ephemerides (download)",
    )
    args = cli.parse_args()
    create_output(args.output)
    duration, dt = 7 * DAY, 5.0
    n = int(duration / dt)
    time = dt * np.arange(n)
    frequency = np.fft.rfftfreq(n, dt)
    epochs = (np.arange(8) + 0.5) * duration / 8
    light_times, origin = orbit_light_times(
        np.r_[epochs, duration / 2], model=args.orbit
    )
    averaged = [np.zeros((len(frequency), 3, 3), complex) for _ in range(2)]
    frozen = [v.copy() for v in averaged]
    for dst, value in zip(
        averaged, reference_components(frequency[1:], light_times[:-1]), strict=True
    ):
        dst[1:] = value
    for dst, value in zip(
        frozen, reference_components(frequency[1:], light_times[-1]), strict=True
    ):
        dst[1:] = value
    injected = np.array([1.4, 0.8])
    truth = sum(s * c for s, c in zip(injected, averaged, strict=True))
    series = simulate_circulant_xyz(truth, dt, np.random.default_rng(args.seed))
    intervals = [(1.5 * DAY, 1.5 * DAY + 4 * 3600), (4.25 * DAY, 4.25 * DAY + 6 * 3600)]
    mask = gl.gaps.mask_from_intervals(n, dt, intervals)
    window = gl.gaps.effective_window(
        gl.gaps.gate_from_mask(mask, dt, taper_s=1800), gl.gaps.segment_window(n, 0.05)
    )
    corrected = [windowed_matrix(c, window) for c in averaged]
    frozen_corrected = [windowed_matrix(c, window) for c in frozen]
    # Cross-check each auto spectrum against GapLike's scalar implementation.
    for raw, conditioned in zip(averaged, corrected, strict=True):
        full = two_sided_matrix(raw, n)
        for channel in range(3):
            np.testing.assert_allclose(
                conditioned[:, channel, channel].real,
                gl.covariance.convolved_diag(window, full[:, channel, channel].real),
                rtol=1e-9,
                atol=1e-54,
            )
    # Prespecified sparse bins keep the prototype small while using the true
    # seven-day FFT and retaining the full XYZ covariance at every chosen bin.
    band = np.flatnonzero((frequency >= 0.4e-3) & (frequency <= 0.099))
    selected = (
        band[np.linspace(0, len(band) - 1, 1024, dtype=int)] if args.quick else band
    )
    mean_light_times = light_times[:-1].mean(axis=0)
    x_sum = mean_light_times[[0, 5, 3, 2]].sum()
    y_sum = mean_light_times[[0, 5, 1, 4]].sum()
    z_sum = mean_light_times[[1, 4, 3, 2]].sum()
    nulls = np.array([[k / s for k in (1, 2, 3)] for s in (x_sum, y_sum, z_sum)])
    coefficients = gl.covariance.transform(series.T, window, selected, dt).T
    baseline = corrected[0][selected] + corrected[1][selected]
    whitening = np.linalg.inv(np.linalg.cholesky(baseline))
    candidates = {
        "frozen_orbit_window_corrected": frozen_corrected,
        "epoch_average_window_power_only": [
            gl.gaps.window_power(window) * c for c in averaged
        ],
        "epoch_average_window_corrected": corrected,
    }
    fits = {}
    for label, components in candidates.items():
        print(f"Point fit: {label}", flush=True)
        scales, diagnostic = point_fit(
            [c[selected] for c in components], coefficients, whitening
        )
        fits[label] = dict(psd_scales=scales.tolist(), **diagnostic)
    np.savez_compressed(
        args.output / "reference.npz",
        frequency=frequency,
        epoch_seconds=epochs,
        light_times=light_times,
        selected_bins=selected,
        mask=mask,
        window=window,
        observed=coefficients,
        averaged=sum(averaged),
        frozen=sum(frozen),
        window_corrected=sum(corrected),
        injected_scales=injected,
    )
    (args.output / "point_fits.json").write_text(json.dumps(fits, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for column, link in enumerate(LINKS):
        axes[0, 0].plot(
            epochs / DAY,
            1e3 * (light_times[:-1, column] - light_times[0, column]),
            label=str(link),
        )
    axes[0, 0].set(
        xlabel="Days into segment",
        ylabel="Change in light travel time (ms)",
        title="Orbit-derived directed light travel times",
    )
    axes[0, 0].legend(ncol=3, fontsize=8, frameon=False)
    f0 = nulls[0, 0]
    close = np.abs(frequency - f0) < 60 / duration
    axes[0, 1].semilogy(
        frequency[close] * 1000,
        sum(frozen)[close, 0, 0].real,
        label="Frozen at midpoint",
        ls="--",
    )
    axes[0, 1].semilogy(
        frequency[close] * 1000,
        sum(averaged)[close, 0, 0].real,
        label="Average of 8 epochs",
    )
    axes[0, 1].semilogy(
        frequency[close] * 1000,
        sum(corrected)[close, 0, 0].real,
        label="Average + gap/window convolution",
    )
    axes[0, 1].set(
        xlabel="Frequency (mHz)",
        ylabel="Nominal X PSD (1/Hz)",
        title="First X response null · seven-day frequency grid",
    )
    axes[0, 1].legend(fontsize=8, frameon=False)
    axes[1, 0].plot(time / DAY, window, lw=1)
    axes[1, 0].set(
        xlabel="Days into segment",
        ylabel="Effective gate × taper",
        title=f"GapLike masks/windows · {mask.mean():.1%} duty cycle",
    )
    for i, label in enumerate(candidates):
        axes[1, 1].plot(
            np.arange(2) + 0.1 * (i - 1),
            fits[label]["psd_scales"],
            "o",
            label=label.replace("_", " "),
        )
    axes[1, 1].plot(np.arange(2), injected, "kx", ms=9, label="Injected")
    axes[1, 1].set(
        xticks=[0, 1],
        xticklabels=["TM PSD scale", "OMS PSD scale"],
        ylabel="Point estimate",
        title="Full per-bin XYZ likelihood · no interval claim",
    )
    axes[1, 1].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Prototype: orbit-averaged stationary surrogate with real time-domain gaps"
    )
    save_figure(fig, args.output / "comparison.png")
    receipt(
        args.output,
        case="orbit_reference_and_gaps_prototype",
        seed=args.seed,
        quick=args.quick,
        duration_days=7,
        dt_seconds=dt,
        n_samples=n,
        orbit_model=args.orbit,
        orbit_origin_seconds=origin,
        epochs=8,
        directed_link_order=LINKS.tolist(),
        gaplike_version=gl.__version__,
        gaplike_commit="99c61931f3f47997d589f255191052ccdbb79f97",
        lisaorbits_version=importlib.metadata.version("lisaorbits"),
        duty_cycle=float(mask.mean()),
        gap_intervals_seconds=intervals,
        selected_bins=len(selected),
        injected_psd_scales=injected.tolist(),
        point_fits=fits,
        simulation="Stationary circulant surrogate with orbit-averaged covariance, NOT an end-to-end nonstationary TDI simulation",
        limitations=[
            "Per-bin Whittle target drops gap-induced cross-frequency correlations",
            "No filtering/decimation chain or aliasing is simulated",
            "No calibrated uncertainty or completed time-varying spline recovery claim",
        ],
        source_paper="https://arxiv.org/html/2610.28726v1",
    )
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
