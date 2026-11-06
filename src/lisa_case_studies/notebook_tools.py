"""Small presentation and configuration helpers; inference stays in the library."""

from contextlib import contextmanager
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from log_psplines import StationaryConfig
from log_psplines.inference import model as library_model
from log_psplines.inference.model import prepare_model

COLORS = {
    "X": "#2878a5",
    "Y": "#d07822",
    "Z": "#668448",
    "continuous": "#2878a5",
    "gapped": "#d07822",
}


def style():
    plt.rcParams.update(
        {
            "figure.figsize": (9, 3.5),
            "figure.dpi": 110,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def reference_config(data, *, seed=7, warmup=1000, draws=1000):
    """Collapsed NUTS with unchanged library likelihood, penalties and priors."""
    config = StationaryConfig(
        method="nuts",
        n_knots=10,
        degree=3,
        knot_kwargs={"method": "uniform"},
        roughness_scale=1.28,
        n_warmup=warmup,
        n_samples=draws,
        num_chains=2,
        chain_method="sequential",
        target_accept_prob=0.95,
        max_tree_depth=10,
        dense_mass=True,
        verbose=False,
        rng_key=seed,
    )
    _, components = prepare_model(data, config)
    return config, components


@contextmanager
def reference_start():
    """Initialize NUTS's log-diagonal correction at zero; theta stays integrated.

    Protected singleton Fourier bins have rank-one empirical matrices. Their
    zero conditional variances are unsuitable NUTS starts. This isolated,
    process-local compatibility shim changes only empirical_init_values;
    likelihoods, data, penalties, priors, coordinates and sampling are still
    provided unchanged by the pinned library. No installed files are edited.
    """

    def initial_values(kwargs, channel, coordinates):
        result = {
            f"sigma_delta_{channel}": 0.3,
            f"weights_delta_{channel}_raw": np.zeros(len(coordinates["eigenvalues"])),
        }
        for previous in range(channel):
            for part in ("re", "im"):
                result[f"sigma_theta_{part}_{channel}_{previous}"] = 0.3
        return result

    with patch.object(library_model, "empirical_init_values", initial_values):
        yield


def plot_nuts_traces(result, *, label=""):
    """Display one representative coefficient or scale for each NUTS factor.

    This is a reading aid; the diagnostic table checks all parameters,
    including restored conditional theta coefficients.
    """
    if result.metadata.get("data_type") == "multivariate":
        names = [f"weights_delta_{j}" for j in range(3)]
    else:
        names = ["sigma_time", "sigma_freq"]
    fig, axes = plt.subplots(
        len(names),
        1,
        figsize=(9, 1.7 * len(names)),
        layout="constrained",
        squeeze=False,
    )
    for ax, name in zip(axes[:, 0], names, strict=True):
        values = np.asarray(result.posterior[name])
        if values.ndim > 2:
            values = values[..., values.shape[-1] // 2]
        for chain, trace in enumerate(values):
            ax.plot(trace, lw=0.65, alpha=0.8, label=f"Chain {chain + 1}")
        ax.set(ylabel=name.replace("weights_", ""), xlabel="Post-warmup draw")
    axes[0, 0].set_title(f"{label} NUTS traces".strip())
    axes[0, 0].legend(ncol=2, frameon=False, fontsize=9)
    return fig


def matrix_errors(estimate, truth):
    """Compact point-reconstruction errors; never select a model on truth."""
    a = np.diagonal(estimate, axis1=-2, axis2=-1).real
    b = np.diagonal(truth, axis1=-2, axis2=-1).real
    scale = np.sqrt(b[..., :, None] * b[..., None, :])
    pairs = np.triu_indices(3, 1)
    return {
        "Auto log10 RMSE": float(np.sqrt(np.mean(np.log10(a / b) ** 2))),
        "Complex cross RMSE": float(
            np.sqrt(
                np.mean(
                    np.abs(((estimate - truth) / scale)[..., pairs[0], pairs[1]]) ** 2
                )
            )
        ),
    }
