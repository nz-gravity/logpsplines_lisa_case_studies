"""Run the full matrix reference correction alongside the exact gap benchmark."""

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from .common import receipt, save_figure
from .joint_gaussian import conditioned_reference
from .orbit_reference import DAY
from .reference_matrix import (
    adaptive_partition,
    coefficient_data,
    fit_reference_correction,
    marginal_predictive_score,
    matrix_mean,
    pooling_diagnostic,
)


def run_comparison(
    output,
    sim,
    windows,
    series,
    injected,
    evaluation_time,
    evaluation_bins,
    truth,
    benchmark_spectra,
    benchmark_errors,
    *,
    seed,
    quick,
    steps=None,
    restarts=3,
):
    from .spline_recovery import recovery_errors

    started = time.monotonic()
    f = sim.frequency
    bins = np.flatnonzero((f >= 0.4e-3) & (f <= 0.099))
    frequency = f[bins]
    print(
        "Building orbit + gap/window reference at every native Fourier bin ...",
        flush=True,
    )
    references = conditioned_reference(
        sim,
        windows,
        progress=lambda i, n: (
            print(f"  epoch cross terms {i}/{n}", flush=True) if i % 8 == 0 else None
        ),
    )
    reference_roots, observations, inputs = {}, {}, {}
    for name, window in windows.items():
        reference_roots[name] = np.linalg.cholesky(references[name][bins])
        observed = np.fft.rfft(series * window[:, None], axis=0)[bins] * np.sqrt(
            2 * sim.dt / sim.n
        )
        observations[name] = observed
        whitened = np.linalg.solve(reference_roots[name], observed[..., None])[..., 0]
        # ENBW is the library's approximate information correction; not an
        # exact treatment of window dependence or posterior calibration.
        enbw = sim.n * np.sum(window**2) / np.sum(window) ** 2
        inputs[name] = coefficient_data(whitened, frequency, enbw=enbw)
    print(
        "Selecting shared adaptive bins from gapped TRAINING data only ...", flush=True
    )
    partition = adaptive_partition(inputs["gapped"])
    print(
        f"{len(bins)} native bins → {partition.Nc} adaptive bins; native resolution near nulls",
        flush=True,
    )
    coordinate = np.log(frequency / frequency[0]) / np.log(frequency[-1] / frequency[0])
    fe = f[evaluation_bins]
    evaluation_coordinate = np.log(fe / frequency[0]) / np.log(
        frequency[-1] / frequency[0]
    )
    evaluation_coordinate = np.clip(evaluation_coordinate, 0, 1)
    nominal = sim.spectrum(
        evaluation_time, fe, np.zeros_like(injected), sim.band_labels[evaluation_bins]
    )
    physical_root = np.linalg.cholesky(nominal)
    estimates = dict(parametric_reference=nominal)
    reports, corrections, evaluation_corrections = {}, {}, {}
    pooling_checks = {}
    for name in windows:
        selected, components, diagnostic = fit_reference_correction(
            inputs[name],
            partition,
            output,
            name,
            seed=seed,
            quick=quick,
            steps=steps,
            restarts=restarts,
        )
        reports[name] = diagnostic
        corrections[name] = matrix_mean(selected.posterior, components, coordinate)
        pooling_checks[name] = pooling_diagnostic(
            inputs[name], partition, corrections[name]
        )
        evaluation_corrections[name] = matrix_mean(
            selected.posterior, components, evaluation_coordinate
        )
        estimates[name] = (
            physical_root
            @ evaluation_corrections[name][None]
            @ physical_root.conj().swapaxes(-1, -2)
        )
    if len({report["basis_penalty_sha256"] for report in reports.values()}) != 1:
        raise RuntimeError(
            "Continuous and gapped fits have different spline bases or penalties"
        )
    # Validation starts only after bins, priors, restarts and fits are frozen.
    validation = {name: [] for name in windows}
    for validation_seed in range(seed + 1000, seed + 1004):
        draw = sim.simulate(injected, np.random.default_rng(validation_seed))
        for name, w in windows.items():
            validation[name].append(
                np.fft.rfft(draw * w[:, None], axis=0)[bins]
                * np.sqrt(2 * sim.dt / sim.n)
            )
    predictive = {
        name: marginal_predictive_score(
            np.asarray(validation[name]), references[name][bins], corrections[name]
        )
        for name in windows
    }
    baseline_predictive = {
        name: marginal_predictive_score(
            np.asarray(validation[name]),
            references[name][bins],
            np.broadcast_to(np.eye(3), (len(bins), 3, 3)),
        )
        for name in windows
    }
    errors = {
        name: recovery_errors(value, truth, fe) for name, value in estimates.items()
    }
    all_errors = {
        **errors,
        "exact_continuous_benchmark": benchmark_errors["continuous"],
        "exact_gapped_benchmark": benchmark_errors["gapped"],
    }
    report = dict(
        reconstruction=all_errors,
        independent_whittle_validation=predictive,
        reference_only_whittle_validation=baseline_predictive,
        variational=reports,
        coarse_graining_checks=pooling_checks,
        adaptive=dict(
            native_frequencies=len(bins),
            coarse_bins=partition.Nc,
            min_width=int(partition.freq_bin_counts.min()),
            max_width=int(partition.freq_bin_counts.max()),
            total_counts=int(partition.freq_bin_counts.sum()),
            pilot="gapped training data only; no continuous gap samples, truth or validation",
            tolerance=0.35,
            pilot_window=301,
            max_bin_width=256,
            protected_null_half_width_hz=1e-4,
            frozen_before_inference=True,
            same_partition_for_both=True,
        ),
        estimator="VI mean of reconstructed positive-definite matrices, with exact conditional theta draws restored; exploratory point summary",
        benchmark_note="Exact joint Gaussian for 48 (quick:30) Fourier bins, two component corrections, MAP with its own priors. Different information budget and parameter family; no ranking by raw ELBO/likelihood or uncertainty.",
        model="S(t,f)=L0(t,f) R(f) L0(t,f)^H. Full 3x3 Cholesky spline R(f), frequency-only residual; known reference carries orbital time variation.",
        approximation="Observed coefficient covariance approximated by L0_window(f) R(f) L0_window(f)^H; conditioning and frequency-dependent residual corrections need not commute. Nominal window reference includes all epoch terms, but Whittle fit omits cross-frequency and pseudo-covariance.",
    )
    (output / "errors.json").write_text(json.dumps(report, indent=2))
    table = [
        "| Reconstruction | Fitting target | Auto log10 RMSE | Signed log10 residual | Complex cross RMSE* | Coherence RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, e in all_errors.items():
        target = (
            "reference only"
            if name == "parametric_reference"
            else (
                "exact compressed Gaussian MAP"
                if name.startswith("exact_")
                else "adaptive Whittle, collapsed VI"
            )
        )
        table.append(
            f"| {name.replace('_', ' ')} | {target} | {e['auto_log10_rmse']:.4f} | {e['auto_signed_log10_mean']:+.4f} | {e['complex_cross_normalized_rmse']:.4f} | {e['coherence_rmse']:.4f} |"
        )
    table += [
        "",
        "*Complex cross errors are normalized by the geometric mean of the true auto spectra. Fixed 17-epoch evaluation grid with native null neighbourhoods. The library fits use all retained native frequencies, pooled into a shared adaptive partition; the exact benchmark uses a much smaller feature set and a two-component model. These rows do not isolate the effect of the likelihood or establish posterior calibration.",
    ]
    (output / "errors.md").write_text("\n".join(table) + "\n")
    np.savez_compressed(
        output / "matrix_recovery.npz",
        frequency=frequency,
        native_bins=bins,
        bin_starts=partition.J_start,
        bin_stops=partition.J_stop,
        bin_counts=partition.freq_bin_counts,
        bin_centres=partition.f_coarse,
        evaluation_frequency=fe,
        evaluation_time=evaluation_time,
        truth=truth,
        parametric_reference=nominal,
        **{f"spectrum_{name}": value for name, value in estimates.items()},
        **{
            f"correction_{name}": value
            for name, value in evaluation_corrections.items()
        },
        **{f"window_reference_{name}": references[name][bins] for name in windows},
    )
    plot_comparison(
        output,
        evaluation_time,
        fe,
        truth,
        estimates,
        benchmark_spectra,
        partition,
        frequency,
    )
    plot_elbo(output, reports, len(bins))
    receipt(
        output,
        case="parametric_reference_multivariate_spline_with_gap_benchmark",
        seed=seed,
        quick=quick,
        duration_days=7,
        n_samples=sim.n,
        dt_seconds=sim.dt,
        orbit_epochs=len(sim.epochs),
        library_commit="132bc478aee7bb4de87259531e4c4a24b7f18d2f",
        benchmark_receipt="benchmark_run.json",
        adaptive=report["adaptive"],
        model=report["model"],
        approximation=report["approximation"],
        variational=reports,
        coarse_graining_checks=pooling_checks,
        reconstruction=all_errors,
        independent_whittle_validation=predictive,
        validation_seeds=list(range(seed + 1000, seed + 1004)),
        run_seconds=time.monotonic() - started,
        uncertainty="Exploratory VI points only; no NUTS-equivalent uncertainty or convergence claim",
    )
    print("\n".join(table), flush=True)


def plot_comparison(
    output, times, f, truth, estimates, benchmarks, partition, fit_frequency
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    colors = dict(parametric_reference="0.55", continuous="#2878a5", gapped="#d07822")
    for channel, color in enumerate(("#26547c", "#b35125", "#608248")):
        offset = 10.0**channel
        axes[0, 0].loglog(
            f * 1000,
            truth[8, :, channel, channel].real * offset,
            color=color,
            ls="--",
            label=f"{'XYZ'[channel]} truth ×{offset:g}",
        )
        axes[0, 0].loglog(
            f * 1000,
            estimates["gapped"][8, :, channel, channel].real * offset,
            color=color,
            lw=1,
        )
    axes[0, 0].set(
        xlabel="Frequency (mHz)",
        ylabel="PSD (1/Hz), vertically offset",
        title="Orbit reference + full XYZ spline correction",
    )
    axes[0, 0].legend(fontsize=8, frameon=False)
    for name, value in estimates.items():
        residual = np.log10(
            np.diagonal(value, axis1=-2, axis2=-1).real
            / np.diagonal(truth, axis1=-2, axis2=-1).real
        )
        axes[0, 1].plot(
            f * 1000,
            residual.mean(axis=(0, 2)),
            color=colors[name],
            label=name.replace("_", " "),
        )
    benchmark = np.log10(
        np.diagonal(benchmarks["gapped"], axis1=-2, axis2=-1).real
        / np.diagonal(truth, axis1=-2, axis2=-1).real
    )
    axes[0, 1].plot(
        f * 1000,
        benchmark.mean(axis=(0, 2)),
        "k:",
        lw=1,
        label="exact gapped benchmark (fewer data)",
    )
    axes[0, 1].axhline(0, color="k", lw=0.5)
    axes[0, 1].set(
        xlabel="Frequency (mHz)",
        ylabel="Mean signed log10 residual",
        title="Same final reconstruction support",
    )
    axes[0, 1].legend(fontsize=7, frameon=False)
    axes[0, 2].step(
        partition.f_coarse * 1000, partition.freq_bin_counts, where="mid", lw=0.8
    )
    axes[0, 2].set(
        xlabel="Frequency (mHz)",
        ylabel="Native frequencies per aggregate",
        title=f"Dynamic coarse-graining · {len(fit_frequency):,} → {partition.Nc:,}",
    )
    for column, null in enumerate((0.03, 0.06, 0.09)):
        select = np.abs(f - null) < 0.00015
        for t, linestyle in ((0, "--"), (-1, "-")):
            axes[1, column].semilogy(
                f[select] * 1000,
                truth[t, select, 0, 0].real,
                color="k",
                ls=linestyle,
                lw=1,
                label=f"Truth day {times[t] / DAY:.0f}",
            )
            for name in ("continuous", "gapped"):
                axes[1, column].semilogy(
                    f[select] * 1000,
                    estimates[name][t, select, 0, 0].real,
                    color=colors[name],
                    ls=linestyle,
                    lw=0.9,
                    label=f"{name}, day {times[t] / DAY:.0f}",
                )
        axes[1, column].set(
            xlabel="Frequency (mHz)",
            ylabel="X PSD (1/Hz)",
            title=f"{null * 1000:.0f} mHz null · geometry handles the dip",
        )
    axes[1, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Seven-day evolving XYZ · full matrix residual splines · marginalised θ · exploratory VI"
    )
    save_figure(fig, output / "comparison.png")


def plot_elbo(output, reports, native_bins):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for index, (name, report) in enumerate(reports.items()):
        traces = np.load(output / f"{name}_elbo.npz")
        for row in range(3):
            losses = [
                traces[f"restart_{restart}_row_{row}"]
                for restart in range(len(report["restarts"]))
            ]
            offset = min(
                float(loss[-max(2, len(loss) // 5) :].mean()) for loss in losses
            )
            for restart, loss in enumerate(losses):
                steps = np.linspace(100, report["steps_per_row"], len(loss))
                axes[index, row].plot(
                    steps,
                    (loss - offset) / native_bins,
                    label=f"restart {restart + 1}",
                    lw=0.9,
                )
            axes[index, row].set(
                xlabel="SVI updates",
                ylabel="Negative ELBO / native bin (offset)",
                title=f"{name} · Cholesky row {row + 1}",
            )
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "ELBO stability checks · conditional θ restored after fitting · no posterior-convergence claim"
    )
    save_figure(fig, output / "elbo.png")
