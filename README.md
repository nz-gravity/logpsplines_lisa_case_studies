# LISA case studies for LogPSplinePSD

[**Read the case studies →**](https://nz-gravity.github.io/logpsplines_lisa_case_studies/)
 · [LogPSplinePSD docs](https://nz-gravity.github.io/LogPSplinePSD/intro.html)

Small, reproducible noise examples, from simple spectra to seven days of
evolving, correlated XYZ data with time-domain gaps. The website uses the
library's Jupyter Book theme and branding, with a persistent return button.

| Case | Demonstration | Run |
|---|---|---|
| [Nonstationary A noise](docs/time_varying.md) | Tensor log-P-spline versus stationary spline on held-out rows | `uv run lisa-demo-time` |
| [Correlated XYZ noise](docs/xyz.md) | Full spectral matrix, TM/OMS scales, two-parameter NUTS | `uv run lisa-demo-xyz` |
| [Seven-day evolving XYZ](docs/gapped_xyz.md) | Keplerian TDI-2, spline secondary spectra, continuous/gapped observations, joint Gaussian MAP | `uv run --extra gapped lisa-demo-gaps` |

## Run the seven-day case

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/nz-gravity/logpsplines_lisa_case_studies.git
cd logpsplines_lisa_case_studies
uv sync --locked --extra gapped
uv run --extra gapped lisa-demo-gaps --output outputs/seven_day
```

Both standard and `--quick` modes simulate 120,960 samples at a five-second
cadence. Standard mode fits 48 Fourier bins / 288 real features with 32
source-frequency bands. All fits use the same seven cubic basis functions
per secondary component, fixed priors, evaluation support and three
optimization restarts. Four independent realizations check predictions.
Existing output directories are never overwritten; use `--seed` and `--output`.
The numbered scripts in `scripts/` call the same entry points.

![Seven-day spline recovery](docs/assets/seven_day_xyz.png)

The evolving orbit filters act on a shared link-noise realization. The
likelihood propagates the actual gaps/windows and includes XYZ,
frequency, quadrature and epoch correlations. It is exact **for the fixed
compressed features under the declared quasi-static simulator**, rather
than for the entire time series. The moving response determines dip
locations; splines learn smooth secondary spectra. See the
[method, results and limitations](docs/gapped_xyz.md) and
[likelihood derivation](docs/likelihood.md).

These are MAP point reconstructions, without posterior uncertainty claims.
This teaching simulation adapts the short-segment response idea from
[Burke et al., Noisy Pipelines I](https://arxiv.org/html/2610.28726v1);
it is not a reproduction of their instrument/conditioning pipeline.

## Reproducibility and development

LogPSplinePSD and GapLike are pinned to exact Git commits; `uv.lock` fixes a
separate demo environment. Published paper environments are not changed.
Outputs include the real data, masks, basis/penalty, fitted spectral matrices,
error table, optimization diagnostics and a version/source-hash receipt.
Large generated arrays stay in ignored `outputs/`; curated previews and
receipts are committed under `docs/assets/`.

```sh
uv run --extra gapped pytest -q
uv run ruff check .
uv run ruff format --check .
uv run --group docs jupyter-book build docs --warningiserror
```

Tests include an independent dense time-domain check of the nonstationary
gap covariance and a dense likelihood/prior/gradient check. CI tests pull
requests and builds the website; `main` publishes to GitHub Pages.
