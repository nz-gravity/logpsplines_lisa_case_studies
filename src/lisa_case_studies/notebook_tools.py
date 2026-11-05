"""Small presentation and initialization helpers; inference stays in the library."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from log_psplines import StationaryConfig
from log_psplines.inference.model import prepare_model
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value

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


def reference_config(data, *, steps=4000, seed=7, draws=96):
    """Library config + a reference-centred guide; no new likelihood or prior.

    Rank-one protected bins make empirical conditional variances unsuitable
    initial values. Start at R=I instead. All priors, marginalisation and
    inference are provided by the pinned library, as in reference_matrix.py.
    """
    config = StationaryConfig(
        method="vi",
        n_knots=10,
        degree=3,
        knot_kwargs={"method": "uniform"},
        roughness_scale=1.28,
        vi_steps=steps,
        vi_lr=0.01,
        vi_posterior_draws=draws,
        vi_early_stopping=False,
        vi_progress_bar=False,
        verbose=False,
        rng_key=seed,
    )
    _, components = prepare_model(data, config)
    initial = {}
    for key in components.expected_component_order:
        initial[f"sigma_{key.name}"] = 0.3
        if key.family == "delta":
            initial[f"weights_{key.name}_raw"] = np.zeros(
                components.component(key).n_basis
            )

    def guide(model):
        return AutoMultivariateNormal(model, init_loc_fn=init_to_value(values=initial))

    return replace(config, vi_guide=guide), components


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
