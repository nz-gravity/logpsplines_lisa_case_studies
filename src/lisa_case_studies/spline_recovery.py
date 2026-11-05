"""Seven-day evolving XYZ: smooth link spectra and a joint gap-aware likelihood."""

import json
import time as timer

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .common import create_output, parser, receipt, save_figure
from .joint_gaussian import (
    TVSimulator,
    covariance_bank,
    feature_vector,
    source_bands,
    whiten_bank,
)
from .orbit_reference import DAY, LINKS, orbit_light_times


def fit_splines(bank, observed, basis, penalty, *, seed=7, restarts=3):
    """MAP with the library's cubic basis / derivative penalty, fixed prior.

    A proper N(0, Q^-1) prior has Q=10 P + I/0.8^2, independently for
    the TM and OMS natural-log PSD corrections. P is trace-normalized.
    This is point optimization, not VI or posterior sampling.
    """
    templates, whitening, condition = whiten_bank(bank)
    y = jnp.asarray(whitening @ observed)
    b = jnp.asarray(basis)
    templates = jnp.asarray(templates)
    precision = 10 * jnp.asarray(penalty) + jnp.eye(b.shape[1]) / 0.8**2

    @jax.jit
    def objective(flat):
        weights = flat.reshape(2, b.shape[1])
        covariance = jnp.einsum("ck,ckij->ij", jnp.exp(weights @ b.T), templates)
        root = jnp.linalg.cholesky(covariance)
        solved = jax.scipy.linalg.solve_triangular(root, y, lower=True)
        nll = jnp.log(jnp.diag(root)).sum() + 0.5 * jnp.sum(solved**2)
        return nll + 0.5 * jnp.einsum("ci,ij,cj->", weights, precision, weights)

    value_gradient = jax.jit(jax.value_and_grad(objective))
    rng = np.random.default_rng(seed)
    starts = [np.zeros(2 * b.shape[1])] + [
        rng.normal(0, 0.2, 2 * b.shape[1]) for _ in range(restarts - 1)
    ]
    fits = []
    for start in starts:
        fit = minimize(
            lambda x: tuple(np.asarray(v) for v in value_gradient(x)),
            start,
            jac=True,
            method="BFGS",
            options={"gtol": 2e-6, "maxiter": 300},
        )
        gradient = float(np.max(np.abs(fit.jac)))
        if not np.isfinite(fit.fun) or gradient > 2e-4:
            raise RuntimeError(
                f"Spline optimization failed: {fit.message}; gradient={gradient}"
            )
        fits.append(fit)
    best = min(fits, key=lambda fit: fit.fun)
    spread = float(np.ptp([fit.fun for fit in fits]))
    if spread > 1e-5:
        raise RuntimeError(f"Optimization restarts disagree: objective range {spread}")
    return best.x.reshape(2, b.shape[1]) @ np.asarray(basis).T, dict(
        method="MAP point optimization; no posterior uncertainty",
        objective=float(best.fun),
        restart_objectives=[float(f.fun) for f in fits],
        max_abs_gradients=[float(np.max(np.abs(f.jac))) for f in fits],
        iterations=[int(f.nit) for f in fits],
        restart_objective_range=spread,
        nominal_correlation_condition=condition,
        weights=best.x.reshape(2, b.shape[1]).tolist(),
    )


def independent_bin_bank(bank):
    """Keep each bin's real 6x6 XYZ block, discard inter-bin correlations."""
    d = bank.shape[-1]
    groups = np.tile(np.repeat(np.arange(d // 6), 3), 2)
    return bank * (groups[:, None] == groups[None, :])


def gaussian_score(bank, log_correction, observed, whitening):
    """Score independent data using the FULL covariance for every candidate."""
    transformed = np.einsum(
        "ia,ckab,jb->ckij", whitening, bank, whitening, optimize=True
    )
    covariance = np.einsum("ck,ckij->ij", np.exp(log_correction), transformed)
    y = observed @ whitening.T
    root = cho_factor(covariance, lower=True)
    q = np.einsum("ri,ir->r", y, cho_solve(root, y.T))
    logdet = 2 * np.log(np.diag(root[0])).sum()
    return dict(
        mean_q_per_dimension=float(q.mean() / len(covariance)),
        mean_nll_per_dimension=float((q.mean() + logdet) / (2 * len(covariance))),
        individual_q_per_dimension=(q / len(covariance)).tolist(),
    )


def recovery_errors(estimate, truth, frequency):
    diagonal = np.diagonal(estimate, axis1=-2, axis2=-1).real
    true_diag = np.diagonal(truth, axis1=-2, axis2=-1).real
    residual = np.log10(diagonal / true_diag)
    result = dict(
        auto_log10_rmse=float(np.sqrt(np.mean(residual**2))),
        auto_signed_log10_mean=float(residual.mean()),
        channel_log10_rmse=np.sqrt(np.mean(residual**2, axis=(0, 1))).tolist(),
    )
    scale = np.sqrt(true_diag[..., :, None] * true_diag[..., None, :])
    delta = (estimate - truth) / scale
    off = np.triu_indices(3, 1)
    result["complex_cross_normalized_rmse"] = float(
        np.sqrt(np.mean(np.abs(delta[..., off[0], off[1]]) ** 2))
    )
    coherence = np.abs(estimate) ** 2 / (
        diagonal[..., :, None] * diagonal[..., None, :]
    )
    true_coherence = np.abs(truth) ** 2 / scale**2
    result["coherence_rmse"] = float(
        np.sqrt(
            np.mean(
                (coherence[..., off[0], off[1]] - true_coherence[..., off[0], off[1]])
                ** 2
            )
        )
    )
    result["nulls"] = {}
    for null in (0.03, 0.06, 0.09):
        select = np.abs(frequency - null) < 0.0006
        local = residual[:, select]
        result["nulls"][str(int(null * 1000))] = dict(
            auto_log10_rmse=float(np.sqrt(np.mean(local**2))),
            signed_log10_mean=float(local.mean()),
            oscillation_rms=float(np.sqrt(np.mean(np.diff(local, axis=1) ** 2))),
            oscillation_definition="RMS adjacent-frequency first difference of signed log10 residual; native weekly grid",
        )
    return result


def main():
    import gaplike as gl

    cli = parser(__doc__, "outputs/03_seven_day_xyz")
    cli.add_argument("--orbit", choices=["keplerian", "esa"], default="keplerian")
    cli.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run only the exact compressed Gaussian control",
    )
    cli.add_argument(
        "--vi-steps",
        type=int,
        help="VI updates per Cholesky row and restart (default: 1500 quick / 6000 standard)",
    )
    cli.add_argument("--vi-restarts", type=int, default=3)
    args = cli.parse_args()
    if args.vi_restarts < 2 or (args.vi_steps is not None and args.vi_steps < 500):
        cli.error("use at least two VI restarts and at least 500 VI steps")
    create_output(args.output)
    started = timer.monotonic()
    duration, dt = 7 * DAY, 5.0
    n = int(duration / dt)
    n_bands = 16 if args.quick else 32
    centres, labels, spline = source_bands(n, dt, n_bands)
    epochs = (np.arange(8) + 0.5) * duration / 8
    delays, origin = orbit_light_times(epochs, model=args.orbit)
    sim = TVSimulator(n, dt, epochs, delays, labels)
    f = sim.frequency
    # Fixed design BEFORE data/truth: broad logarithmic support plus neighbouring
    # native bins at each response null to exercise frequency correlations.
    base = np.geomspace(0.4e-3, 0.099, 18 if args.quick else 36)
    bins = np.rint(base * duration).astype(int)
    null_bins = np.rint(np.array([0.03, 0.06, 0.09]) * duration).astype(int)
    bins = np.unique(
        np.r_[bins, (null_bins[:, None] + np.array([-2, -1, 0, 1])[None]).ravel()]
    )
    intervals = [(1.5 * DAY, 1.5 * DAY + 4 * 3600), (4.25 * DAY, 4.25 * DAY + 6 * 3600)]
    mask = gl.gaps.mask_from_intervals(n, dt, intervals)
    segment = gl.gaps.segment_window(n, 0.05)
    windows = dict(
        continuous=segment,
        gapped=gl.gaps.effective_window(
            gl.gaps.gate_from_mask(mask, dt, taper_s=1800), segment
        ),
    )
    print(
        f"Seven days, {n} samples, {len(bins)} Fourier bins / {6 * len(bins)} real features",
        flush=True,
    )
    print(
        "Building joint channel/frequency/quadrature covariance banks ...", flush=True
    )
    banks = covariance_bank(
        sim,
        windows,
        bins,
        n_bands,
        progress=lambda i, total: print(f"  source band {i}/{total}", flush=True),
    )
    # Smooth, deliberately non-parametric deviations; independent of basis knots.
    u = np.log(centres / 0.004)
    injected = np.stack(
        [
            0.35 + 0.45 * np.exp(-0.5 * ((u + 0.6) / 0.85) ** 2),
            -0.2 + 0.4 * np.exp(-0.5 * ((u - 2.0) / 0.7) ** 2),
        ]
    )
    series = sim.simulate(injected, np.random.default_rng(args.seed))
    observed = {
        name: feature_vector(series, w, bins, dt) for name, w in windows.items()
    }
    estimates, diagnostics = {}, {}
    for name in ("continuous", "gapped", "gapped_independent_bins"):
        print(f"Fitting {name}: 3 optimization restarts", flush=True)
        source = "gapped" if name.startswith("gapped") else "continuous"
        bank = (
            independent_bin_bank(banks[source])
            if name.endswith("independent_bins")
            else banks[source]
        )
        estimates[name], diagnostics[name] = fit_splines(
            bank,
            observed[source],
            np.asarray(spline.basis),
            np.asarray(spline.penalty),
            seed=args.seed,
        )
    # Separate simulation seeds for final predictive checks, never used in fits.
    validation = {name: [] for name in windows}
    for seed in range(args.seed + 1000, args.seed + 1004):
        validation_series = sim.simulate(injected, np.random.default_rng(seed))
        for name, w in windows.items():
            validation[name].append(feature_vector(validation_series, w, bins, dt))
    predictive = {}
    for name, correction in estimates.items():
        source = "gapped" if name.startswith("gapped") else "continuous"
        _, whitening, _ = whiten_bank(banks[source])
        predictive[name] = gaussian_score(
            banks[source], correction, np.asarray(validation[source]), whitening
        )
    oracle = {}
    for name, bank in banks.items():
        _, whitening, _ = whiten_bank(bank)
        oracle[name] = gaussian_score(
            bank, injected, np.asarray(validation[name]), whitening
        )
    # Fixed dense reconstruction grid, distinct from sparse fitting features.
    evaluate_bins = np.unique(
        np.r_[
            np.rint(np.geomspace(0.4e-3, 0.099, 480) * duration).astype(int),
            *[np.flatnonzero(np.abs(f - null) < 0.0006) for null in (0.03, 0.06, 0.09)],
        ]
    )
    fe = f[evaluate_bins]
    times = np.linspace(0, duration - dt, 17)
    truth = sim.spectrum(times, fe, injected, labels[evaluate_bins])
    spectra = {
        name: sim.spectrum(times, fe, value, labels[evaluate_bins])
        for name, value in estimates.items()
    }
    errors = {
        name: recovery_errors(value, truth, fe) for name, value in spectra.items()
    }
    report = dict(
        reconstruction=errors,
        independent_validation=predictive,
        injected_covariance_control=oracle,
        optimization=diagnostics,
        evaluation="Fixed 17 epochs, log-frequency grid plus native bins within 0.6 mHz of 30/60/90 mHz; truth used only after fitting. No model selection on these scores.",
    )
    (args.output / "errors.json").write_text(json.dumps(report, indent=2))
    table = [
        "| Fit | Auto log10 RMSE | Signed log10 residual | Complex cross RMSE* | Coherence RMSE | Validation q / dimension |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, e in errors.items():
        table.append(
            f"| {name.replace('_', ' ')} | {e['auto_log10_rmse']:.4f} | {e['auto_signed_log10_mean']:+.4f} | {e['complex_cross_normalized_rmse']:.4f} | {e['coherence_rmse']:.4f} | {predictive[name]['mean_q_per_dimension']:.3f} |"
        )
    table += [
        "",
        "*Complex cross errors normalized by the geometric mean of the corresponding true auto spectra. Four independent validation realizations; q/d should be near 1 in expectation, not forced by an in-sample amplitude fit.",
        "",
        "All fits use the same 7 cubic basis functions per component and the same proper Gaussian prior; no knot tuning or posterior uncertainty claim.",
    ]
    (args.output / "errors.md").write_text("\n".join(table) + "\n")
    np.savez_compressed(
        args.output / "recovery.npz",
        time_seconds=np.arange(n) * dt,
        xyz=series,
        window_continuous=windows["continuous"],
        window_gapped=windows["gapped"],
        mask=mask,
        selected_bins=bins,
        band_centres=centres,
        band_labels=labels,
        spline_basis=spline.basis,
        spline_penalty=spline.penalty,
        injected_corrections=injected,
        evaluation_time=times,
        evaluation_frequency=fe,
        truth=truth,
        **{f"spectrum_{name}": value for name, value in spectra.items()},
        **{f"correction_{name}": value for name, value in estimates.items()},
    )
    plot_comparison(
        args.output,
        sim,
        windows,
        centres,
        injected,
        estimates,
        times,
        fe,
        truth,
        spectra,
    )
    receipt(
        args.output,
        case="seven_day_time_varying_xyz_spline",
        seed=args.seed,
        quick=args.quick,
        duration_days=7,
        dt_seconds=dt,
        n_samples=n,
        epochs=8,
        orbit_model=args.orbit,
        orbit_origin_seconds=origin,
        directed_link_order=LINKS.tolist(),
        source_frequency_bands=n_bands,
        spline_functions_per_component=int(spline.basis.shape[1]),
        selected_bins=bins.tolist(),
        real_feature_dimension=6 * len(bins),
        gap_intervals_seconds=intervals,
        duty_cycle=float(mask.mean()),
        validation_seeds=list(range(args.seed + 1000, args.seed + 1004)),
        gaplike_commit="99c61931f3f47997d589f255191052ccdbb79f97",
        run_seconds=timer.monotonic() - started,
        likelihood="Exact joint real Gaussian for fixed compressed Fourier features of the declared discretized time-varying simulator; all XYZ/frequency/quadrature/epoch correlations included",
        prior="Two independent natural-log PSD correction vectors; precision 10 * library trace-normalized derivative penalty + I/0.8^2; fixed before seeing truth/data",
        limitations=[
            "Quasi-static interpolated TDI-2 frequency response, not noncommuting time-delay TDI or a lisainstrument simulation",
            "Stationary secondary noise with orbit-induced output nonstationarity; no time spline inferred",
            "Smooth corrections are piecewise constant in declared source-frequency bands; native-bin transfer resolution retained",
            "Likelihood exact only for retained features, not the full time series; no filtering/decimation/aliasing chain",
            "MAP point reconstruction, not VI, NUTS, posterior calibration or ensemble validation",
        ],
        source_paper="https://arxiv.org/html/2610.28726v1",
        reconstruction=errors,
        independent_validation=predictive,
        injected_covariance_control=oracle,
    )
    print("\n".join(table), flush=True)
    if not args.benchmark_only:
        from .multivar_case import run_comparison

        for filename in ("comparison.png", "errors.json", "errors.md", "run.json"):
            (args.output / filename).rename(args.output / f"benchmark_{filename}")
        run_comparison(
            args.output,
            sim,
            windows,
            series,
            injected,
            times,
            evaluate_bins,
            truth,
            spectra,
            errors,
            seed=args.seed,
            quick=args.quick,
            steps=args.vi_steps,
            restarts=args.vi_restarts,
        )


def plot_comparison(
    output, sim, windows, centres, injected, estimates, times, frequency, truth, spectra
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for i, link in enumerate(LINKS):
        axes[0, 0].plot(
            sim.epochs / DAY,
            1e3 * (sim.delays[:, i] - sim.delays[0, i]),
            label=str(link),
        )
    axes[0, 0].set(
        xlabel="Day",
        ylabel="Light-time change (ms)",
        title="Keplerian geometry evolves through the week",
    )
    axes[0, 0].legend(ncol=3, fontsize=8, frameon=False)
    for name, w in windows.items():
        axes[0, 1].plot(np.arange(sim.n)[::24] * sim.dt / DAY, w[::24], label=name)
    axes[0, 1].set(
        xlabel="Day",
        ylabel="Gate × taper",
        title="Paired continuous and gapped observations",
    )
    axes[0, 1].legend(frameon=False)
    colors = dict(
        continuous="#2676a6", gapped="#d07520", gapped_independent_bins="#888888"
    )
    for c, component in enumerate(("TM", "OMS")):
        axes[0, 2].semilogx(
            centres * 1000,
            np.exp(injected[c]),
            "k--",
            label=f"{component} truth" if c == 0 else None,
        )
        for name, correction in estimates.items():
            axes[0, 2].semilogx(
                centres * 1000,
                np.exp(correction[c]),
                color=colors[name],
                ls="-" if c == 0 else ":",
                label=name.replace("_", " ") if c == 0 else None,
            )
    axes[0, 2].set(
        xlabel="Frequency (mHz)",
        ylabel="Link PSD / nominal PSD",
        title="Cubic splines recover secondary spectra",
        xlim=(0.4, 99),
    )
    axes[0, 2].text(
        0.04,
        0.04,
        "TM: solid    OMS: dotted",
        transform=axes[0, 2].transAxes,
        fontsize=8,
    )
    axes[0, 2].legend(fontsize=8, frameon=False)
    for ch, color in enumerate(("#26547c", "#b35125", "#608248")):
        offset = 10.0**ch
        axes[1, 0].loglog(
            frequency * 1000,
            truth[8, :, ch, ch].real * offset,
            color=color,
            ls="--",
            lw=1,
            label=f"{'XYZ'[ch]} truth ×{offset:g}",
        )
        axes[1, 0].loglog(
            frequency * 1000,
            spectra["gapped"][8, :, ch, ch].real * offset,
            color=color,
            lw=0.9,
        )
    axes[1, 0].set(
        xlabel="Frequency (mHz)",
        ylabel="PSD (1/Hz), vertically offset",
        title="Midweek XYZ: truth dashed, gap-aware fit solid",
    )
    axes[1, 0].legend(fontsize=8, frameon=False)
    for name, spectrum in spectra.items():
        residual = np.log10(
            np.diagonal(spectrum, axis1=-2, axis2=-1).real
            / np.diagonal(truth, axis1=-2, axis2=-1).real
        )
        axes[1, 1].plot(
            frequency * 1000,
            residual.mean(axis=(0, 2)),
            color=colors[name],
            label=name.replace("_", " "),
        )
    axes[1, 1].axhline(0, color="k", lw=0.6)
    for null in (30, 60, 90):
        axes[1, 1].axvspan(null - 0.6, null + 0.6, color="0.8", alpha=0.5)
    axes[1, 1].set(
        xlabel="Frequency (mHz)",
        ylabel="Mean signed log10(estimate / truth)",
        title="Prespecified evaluation · null windows shaded",
    )
    axes[1, 1].legend(fontsize=8, frameon=False)
    close = np.abs(frequency - 0.03) < 0.00015
    for t, color in ((0, "#2676a6"), (-1, "#b35125")):
        axes[1, 2].semilogy(
            frequency[close] * 1000,
            truth[t, close, 0, 0].real,
            color=color,
            ls="--",
            label=f"Truth day {times[t] / DAY:.0f}",
        )
        axes[1, 2].semilogy(
            frequency[close] * 1000,
            spectra["gapped"][t, close, 0, 0].real,
            color=color,
            label=f"Gap-aware recovery day {times[t] / DAY:.0f}",
        )
    axes[1, 2].set(
        xlabel="Frequency (mHz)",
        ylabel="X PSD (1/Hz)",
        title="Moving response null · native seven-day resolution",
    )
    axes[1, 2].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Seven-day evolving XYZ · fixed orbit response + smooth log-P-splines · MAP points"
    )
    save_figure(fig, output / "comparison.png")


if __name__ == "__main__":
    main()
