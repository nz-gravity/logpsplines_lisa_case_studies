"""Recover a nonstationary A-channel PSD from local Gaussian powers."""

import json

import matplotlib.pyplot as plt
import numpy as np
from log_psplines import (
    LogPSpline,
    PowerConfig,
    PowerData,
    SplineBasis,
    fit,
    mask_power,
)

from .common import create_output, log_rmse, parser, receipt, save_figure
from .noise import aet_components, sample_real_powers, time_scales


def main():
    args = parser(__doc__, "outputs/01_time_varying_noise").parse_args()
    create_output(args.output)
    rng = np.random.default_rng(args.seed)
    nt, nf, replicates = (32, 64, 8) if args.quick else (64, 128, 16)
    time, frequency = np.linspace(0, 1, nt), np.geomspace(0.4e-3, 25e-3, nf)
    acc, oms = aet_components(frequency)
    acc_scale, oms_scale = time_scales(time)
    truth = acc_scale[:, None] * acc[None, :, 0] + oms_scale[:, None] * oms[None, :, 0]
    powers = sample_real_powers(truth, replicates, rng)
    training = np.arange(nt) % 4 != 2
    # Numerical scaling uses only observed training coefficients, not truth.
    scale = float(np.median(powers[training] / replicates))
    data = mask_power(
        PowerData(powers / scale, replicates, frequency, time),
        np.broadcast_to(training[:, None], powers.shape),
    )
    kf = 12 if args.quick else 20
    bf = SplineBasis.from_grid(
        frequency,
        interior_knots=np.geomspace(frequency[0], frequency[-1], kf + 2)[1:-1],
    )
    bt = SplineBasis.from_grid(time, 3 if args.quick else 5)
    # A constant time basis gives the stationary member of the same scalar
    # likelihood family, with no time roughness penalty.
    constant = SplineBasis(
        time,
        np.array([0.0, 1.0]),
        np.ones((nt, 1)),
        np.zeros((1, 1)),
        degree=0,
        penalty_order=0,
        penalty_normalization="trace",
        penalty_ridge=0,
        knot_convention="clamped",
    )
    config = PowerConfig(
        method="vi",
        vi_steps=3000 if args.quick else 8000,
        vi_lr=0.05,
        vi_early_stopping=False,
        vi_posterior_draws=64,
        spectrum_draws=1,
        spectrum_chunk_size=32,
        roughness_scale=10,
        seed=args.seed,
        progress_bar=False,
    )
    estimates, losses = {}, {}
    for label, basis_time in (("time_varying", bt), ("stationary", constant)):
        print(f"Fitting {label} ...", flush=True)
        result = fit(data, config, model=LogPSpline(bf, time=basis_time))
        result.metadata.update(physical_psd_scale=scale, physical_psd_units="1/Hz")
        # The median PSD is an exploratory VI point summary, not a calibrated
        # uncertainty interval. Truth never enters fit or basis construction.
        estimates[label] = np.asarray(result.quantiles()).real[1, ..., 0, 0] * scale
        losses[label] = np.asarray(result.vi.losses)
        result.to_netcdf(args.output / f"{label}.nc")
    scores = {
        label: log_rmse(value[~training], truth[~training])
        for label, value in estimates.items()
    }
    np.savez_compressed(
        args.output / "data.npz",
        time=time,
        frequency=frequency,
        power=powers,
        counts=replicates,
        training=training,
        truth=truth,
        scale=scale,
        **estimates,
    )
    np.savez_compressed(args.output / "elbo.npz", **losses)
    (args.output / "errors.json").write_text(
        json.dumps(
            dict(
                heldout_log10_rmse=scores,
                heldout_cells=int((~training).sum() * nf),
                definition="RMSE(log10(estimate/truth)) on fixed held-out time rows",
            ),
            indent=2,
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), layout="constrained")
    vmin, vmax = np.log10(truth).min(), np.log10(truth).max()
    for ax, values, title in (
        (axes[0, 0], truth, "Injected nonstationary A noise"),
        (axes[0, 1], estimates["time_varying"], "Time-varying log-P-spline · VI"),
    ):
        mesh = ax.pcolormesh(
            time,
            frequency * 1000,
            np.log10(values).T,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set(
            yscale="log",
            ylabel="Frequency (mHz)",
            xlabel="Fraction of demo duration",
            title=title,
        )
        fig.colorbar(mesh, ax=ax, label=r"$\log_{10}$ PSD (1/Hz)")
    residual = np.log10(estimates["time_varying"] / truth)
    limit = max(0.05, float(np.max(np.abs(residual))))
    mesh = axes[1, 0].pcolormesh(
        time,
        frequency * 1000,
        residual.T,
        shading="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axes[1, 0].set(
        yscale="log",
        ylabel="Frequency (mHz)",
        xlabel="Fraction of demo duration",
        title="Signed reconstruction residual",
    )
    fig.colorbar(mesh, ax=axes[1, 0], label=r"$\log_{10}(\widehat S/S)$")
    for index, color in ((nt // 4, "#2878ad"), (3 * nt // 4, "#a43d55")):
        axes[1, 1].loglog(
            frequency * 1000,
            truth[index],
            color=color,
            ls="--",
            label=f"Truth u={time[index]:.2f}",
        )
        axes[1, 1].loglog(
            frequency * 1000,
            estimates["time_varying"][index],
            color=color,
            label=f"VI u={time[index]:.2f}",
        )
    axes[1, 1].loglog(
        frequency * 1000,
        estimates["stationary"][nt // 2],
        color="0.25",
        ls=":",
        label="Stationary spline",
    )
    axes[1, 1].set(
        xlabel="Frequency (mHz)",
        ylabel="PSD (1/Hz)",
        title="A stationary spectrum averages over the drift",
    )
    axes[1, 1].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Illustrative local-coefficient simulation"
        + (" · quick execution check" if args.quick else "")
    )
    save_figure(fig, args.output / "comparison.png")
    receipt(
        args.output,
        case="time_varying_A",
        seed=args.seed,
        quick=args.quick,
        simulation="Independent real Gaussian local coefficients; not a WDM transform or year-long TDI time series",
        psd_units="fractional_frequency_squared_per_hz",
        replicates=replicates,
        numerical_scale=scale,
        vi_steps=config.vi_steps,
        vi_lr=config.vi_lr,
        basis_functions={
            "frequency": int(bf.basis.shape[1]),
            "time_varying_time": int(bt.basis.shape[1]),
            "stationary_time": 1,
        },
        vi_point_estimates_only=True,
        heldout_log10_rmse=scores,
        output_scaling="Stored .nc PSDs are normalized; multiply spectra by data.npz scale for physical 1/Hz units",
    )
    print(json.dumps(scores, indent=2))
    print(f"Saved {args.output / 'comparison.png'}")


if __name__ == "__main__":
    main()
