"""Infer acceleration/OMS PSD scales using full correlated XYZ statistics."""

import json

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from log_psplines import WishartGridData
from log_psplines.inference.nuts import run_nuts
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from numpyro import distributions as dist
from numpyro.diagnostics import summary

from .common import create_output, log_rmse, parser, receipt, save_figure
from .noise import coherence, sample_complex, xyz_components


def matrix_log_likelihood(spectrum, data):
    """Modified-Cholesky factors of the exact full matrix Whittle target."""
    lower = jnp.linalg.cholesky(spectrum)
    diagonal = jnp.diagonal(lower, axis1=-2, axis2=-1).real
    triangular = jnp.linalg.inv(lower / diagonal[..., None, :])
    theta = -triangular
    logs = 2 * jnp.log(diagonal)
    return sum(
        wishart_log_likelihood(
            logs[..., row],
            theta[..., row, :row].real,
            theta[..., row, :row].imag,
            data.u_re[..., row, :],
            data.u_im[..., row, :],
            data.u_re[..., :row, :],
            data.u_im[..., :row, :],
            counts=data.counts,
        )
        for row in range(3)
    )


def parametric_model(templates, data):
    """Two positive PSD scales; priors are fixed independently of injection."""
    acc, oms = map(jnp.asarray, templates)

    def model():
        log_acc = numpyro.sample("log_acc_psd_scale", dist.Normal(0.0, 0.7))
        log_oms = numpyro.sample("log_oms_psd_scale", dist.Normal(0.0, 0.7))
        spectrum = jnp.exp(log_acc) * acc + jnp.exp(log_oms) * oms
        value = matrix_log_likelihood(spectrum[None], data)
        numpyro.deterministic("log_likelihood", value)
        numpyro.factor("matrix_whittle", value)

    return model


def main():
    args = parser(__doc__, "outputs/02_xyz_parametric_noise").parse_args()
    create_output(args.output)
    rng = np.random.default_rng(args.seed)
    frequency = np.geomspace(0.4e-3, 25e-3, 64 if args.quick else 160)
    acc, oms = xyz_components(frequency)
    injected = np.array([1.6, 0.7])
    truth = injected[0] * acc + injected[1] * oms
    replicates = 12 if args.quick else 32
    coefficients = sample_complex(truth, replicates, rng)
    # Fixed baseline whitening avoids tiny physical PSDs and near-null T-mode
    # condition numbers. This invertible transform preserves all XYZ cross
    # information; its likelihood Jacobian is independent of fitted parameters.
    whitening = np.linalg.inv(np.linalg.cholesky(acc + oms))

    def adjoint(x):
        return np.swapaxes(x.conj(), -1, -2)

    templates = tuple(
        whitening @ component @ adjoint(whitening) for component in (acc, oms)
    )
    data = WishartGridData.from_coefficients(
        (whitening @ coefficients)[None],
        np.array([0.0]),
        frequency,
        units="fixed_baseline_whitened_XYZ",
    )
    model = parametric_model(templates, data)
    warmup, draws = (100, 100) if args.quick else (400, 600)
    print(
        f"Fitting full XYZ spectral matrix: 2 chains, {warmup} warmup + {draws} draws each",
        flush=True,
    )
    result = run_nuts(
        model,
        rng_key=jax.random.PRNGKey(args.seed + 1),
        init_values={"log_acc_psd_scale": 0.0, "log_oms_psd_scale": 0.0},
        n_warmup=warmup,
        n_samples=draws,
        num_chains=2,
        chain_method="sequential",
        dense_mass=True,
        target_accept_prob=0.9,
        max_tree_depth=8,
        extra_fields=("diverging", "accept_prob", "num_steps"),
        progress_bar=False,
    )
    names = ("log_acc_psd_scale", "log_oms_psd_scale")
    samples = {name: np.asarray(result.posterior[name]) for name in names}
    medians = np.array([np.exp(np.median(samples[name])) for name in names])
    estimate = medians[0] * acc + medians[1] * oms
    empirical = coefficients @ adjoint(coefficients) / replicates
    heldout = sample_complex(truth, replicates, rng)
    heldout_empirical = heldout @ adjoint(heldout) / replicates
    # An independent replicate draw diagnoses predictive recovery; it does not
    # select a model or tune the priors/grid/budget.
    diagnostic = summary(samples, group_by_chain=True)
    stats = {
        name: {key: float(value) for key, value in values.items()}
        for name, values in diagnostic.items()
    }
    parameters = {
        name.removeprefix("log_"): dict(
            injected=float(injected[i]),
            median=float(medians[i]),
            q05=float(np.percentile(np.exp(samples[name]), 5)),
            q95=float(np.percentile(np.exp(samples[name]), 95)),
        )
        for i, name in enumerate(names)
    }
    (args.output / "parameters.json").write_text(json.dumps(parameters, indent=2))
    result.posterior.to_netcdf(args.output / "posterior.nc")
    result.sample_stats.to_netcdf(args.output / "sample_stats.nc")
    np.savez_compressed(
        args.output / "data.npz",
        frequency=frequency,
        truth=truth,
        estimate=estimate,
        empirical=empirical,
        heldout_empirical=heldout_empirical,
        whitening=whitening,
        replicates=replicates,
    )
    f = frequency * 1000
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), layout="constrained")
    for i, color in enumerate(("#2878ad", "#c27600", "#824aa0")):
        axes[0, 0].loglog(f, empirical[:, i, i].real, color=color, alpha=0.25, lw=0.7)
        axes[0, 0].loglog(
            f, estimate[:, i, i].real, color=color, label=f"{'XYZ'[i]} fit"
        )
    axes[0, 0].loglog(
        f, truth[:, 0, 0].real, color="0.15", ls="--", label="Truth (equal XYZ)"
    )
    axes[0, 0].set(ylabel="PSD (1/Hz)", title="XYZ auto spectra")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].semilogx(
        f, empirical[:, 0, 1].real, color="0.6", lw=0.7, label="Observed Re XY"
    )
    axes[0, 1].semilogx(f, truth[:, 0, 1].real, color="0.15", ls="--", label="Truth")
    axes[0, 1].semilogx(
        f, estimate[:, 0, 1].real, color="#2878ad", label="Parametric fit"
    )
    axes[0, 1].set(
        ylabel="Cross PSD (1/Hz)", title="XY cross spectrum (can be negative)"
    )
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].semilogx(
        f, coherence(empirical)[:, 0, 1], color="0.6", lw=0.7, label="Observed"
    )
    axes[1, 0].semilogx(
        f, coherence(truth)[:, 0, 1], color="0.15", ls="--", label="Truth"
    )
    axes[1, 0].semilogx(f, coherence(estimate)[:, 0, 1], color="#2878ad", label="Fit")
    axes[1, 0].set(
        ylabel=r"$|S_{XY}|^2/(S_{XX}S_{YY})$",
        ylim=(0, 1),
        title="XY magnitude-squared coherence",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)
    for i, name in enumerate(names):
        axes[1, 1].hist(
            np.exp(samples[name]).ravel(),
            bins=25,
            density=True,
            histtype="step",
            label=name[4:],
            color=("#2878ad", "#c27600")[i],
        )
        axes[1, 1].axvline(injected[i], ls="--", color=("#2878ad", "#c27600")[i])
    axes[1, 1].set(
        xlabel="PSD multiplier (not ASD)",
        ylabel="Draw density",
        title="Noise scales · dashed = injected",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Frequency (mHz)")
    fig.suptitle(
        "Equal-arm LISA instrument noise · full XYZ likelihood"
        + (" · quick execution check" if args.quick else "")
    )
    save_figure(fig, args.output / "comparison.png")
    receipt(
        args.output,
        case="XYZ_parametric_instrument",
        seed=args.seed,
        quick=args.quick,
        simulation="Independent proper-complex local Fourier coefficients; equal-arm first-generation Michelson approximation",
        parameters=parameters,
        replicates=replicates,
        n_warmup=warmup,
        n_samples=draws,
        chains=2,
        diagnostics=stats,
        divergences=int(result.sample_stats["diverging"].sum()),
        max_tree_depth=8,
        tree_depth_limit_hits=int((result.sample_stats["n_steps"] >= 2**8 - 1).sum()),
        truth_auto_log10_rmse=log_rmse(estimate[:, 0, 0].real, truth[:, 0, 0].real),
        interpretation="Quick mode checks execution only; inspect R-hat, ESS, divergences and chain agreement before interpreting posterior uncertainty",
    )
    print(json.dumps(parameters, indent=2))
    print(f"Saved {args.output / 'comparison.png'}")


if __name__ == "__main__":
    main()
