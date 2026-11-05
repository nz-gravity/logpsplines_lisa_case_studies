# LISA case studies for LogPSplinePSD

[**Read the case studies →**](https://nz-gravity.github.io/logpsplines_lisa_case_studies/)
 · [LogPSplinePSD docs](https://nz-gravity.github.io/LogPSplinePSD/intro.html)

Small, reproducible noise examples, from simple spectra to seven days of
evolving, correlated XYZ data with time-domain gaps. The website shares the
library's theme and branding, with a persistent return button.

| Case | Demonstration | Run |
|---|---|---|
| [Nonstationary A noise](docs/time_varying.md) | Tensor log-P-spline versus stationary spline on held-out rows | `uv run lisa-demo-time` |
| [Correlated XYZ noise](docs/xyz.md) | Full spectral matrix, TM/OMS scales, two-parameter NUTS | `uv run lisa-demo-xyz` |
| [Seven-day evolving XYZ](docs/gapped_xyz.md) | Parametric orbit response + full matrix spline correction, adaptive bins, marginalised theta; continuous/gapped comparisons | `uv run --extra gapped lisa-demo-gaps` |

## Run the seven-day case

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/nz-gravity/logpsplines_lisa_case_studies.git
cd logpsplines_lisa_case_studies
git checkout 6837a066c46afce78ff9cd58d372a1ae5ba0ac37
uv sync --locked --extra gapped
uv run --extra gapped lisa-demo-gaps --output outputs/seven_day
```

The checkout pins the scientific version used for the figure and works while
[PR #1](https://github.com/nz-gravity/logpsplines_lisa_case_studies/pull/1) is awaiting merge.

The model is `S(t,f) = L0(t,f) R(f) L0(t,f)†`: known Keplerian TDI-2 response
sets the dips, while a positive-definite multivariate Cholesky spline learns
the residual covariance. The short-segment residual is frequency-only;
the reference carries orbital time variation. The main fit uses 59,634
native frequencies, pooled into a shared training-selected adaptive partition.
It analytically marginalises six Gaussian theta fields and fits the remaining
parameters with three VI restarts. It restores conditional theta draws for
reconstruction. See [model and priors](docs/reference_matrix.md).

![Reference-corrected multivariate spline recovery](docs/assets/seven_day_xyz.png)

The main Whittle likelihood uses an orbit-and-window-aware reference but
omits inter-frequency and pseudo-covariance. An **exact joint Gaussian
benchmark** for a smaller retained feature set remains alongside it, fitting
two secondary-component corrections. Their data budgets and priors differ;
RMSE alone does not compare the likelihoods fairly. Both use the same real
seven-day simulation and GapLike masks. [Can GapLike fit XYZ?](docs/gaplike_xyz.md)

`--quick` keeps all seven days and all main-fit frequencies, with shorter VI
and a smaller benchmark. `--benchmark-only` runs just the exact control.
Existing output directories are never overwritten; use `--seed` and `--output`.
The numbered scripts in `scripts/` call the same entry points.

These are exploratory point reconstructions, without calibrated uncertainty
claims. The case adapts the short-segment response idea from
[Burke et al., Noisy Pipelines I](https://arxiv.org/html/2610.28726v1), not their
full instrument and conditioning pipeline. See [results and limits](docs/gapped_xyz.md).

## Reproducibility and development

The isolated environment pins LogPSplinePSD and GapLike to exact Git commits.
The newer library pin enables adaptive counts and marginalised theta here;
published paper environments are unchanged. Outputs retain actual adaptive
edges/counts, library bases/penalties, restored coefficients, ELBO traces,
independent validation scores and source-hash receipts. Large arrays stay in
ignored `outputs/`; curated previews and receipts are in `docs/assets/`.

```sh
uv run --extra gapped pytest -q
uv run ruff check .
uv run ruff format --check .
uv run --group docs jupyter-book build docs --warningiserror
```

Tests check independent dense gap covariance, reference-whitening likelihood
invariance, complex adaptive pooling/counts, collapsed model sites and native
reconstruction against the library. CI checks PRs and builds the website;
`main` publishes it to GitHub Pages.
