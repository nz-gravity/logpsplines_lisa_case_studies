# LISA PSD case studies

[**Read the notebook walkthroughs →**](https://nz-gravity.github.io/logpsplines_lisa_case_studies/)
 · [LogPSplinePSD documentation](https://nz-gravity.github.io/LogPSplinePSD/intro.html)

Four executed Jupyter notebooks explain the data, call the library, and plot
what was recovered. The estimated quantities are channel PSDs and full
spectral matrices.

| Notebook | What you do |
|---|---|
| [1. Generate the data](docs/notebooks/01_generate_data.ipynb) | Inspect full XYZ spectra, generate real time series, add time variation and gaps |
| [2. Time-varying PSD](docs/notebooks/02_time_varying_psd.ipynb) | Prepare local channel-A powers and call `log_psplines.fit` |
| [3. Standard XYZ estimation](docs/notebooks/03_xyz_psd.ipynb) | Run marginalised NUTS for a multivariate correction around orbit-derived sine dips; inspect auto/cross spectra and coherence |
| [4. Short-duration XYZ with gaps](docs/notebooks/04_short_xyz_gaps.ipynb) | Use the approximate sine reference for continuous/gapped NUTS recovery, with an exact selected-feature covariance check |

Start with notebook 1. It saves the shared inputs used by the other three.
The saved notebook outputs are visible on GitHub and on the docs website.
The teaching sequence does not fit instrument-noise component levels.

## Run the notebooks

```sh
git clone https://github.com/nz-gravity/logpsplines_lisa_case_studies.git
cd logpsplines_lisa_case_studies
git checkout e8b35495a84554d2e03bd33dc73fd2a73b8eaeca
uv sync --locked --extra gapped --group notebooks --group docs
uv run --no-sync jupyter lab docs/notebooks
```

The checkout selects the illustrated notebook version while
[PR #1](https://github.com/nz-gravity/logpsplines_lisa_case_studies/pull/1) awaits merge.
Select the project environment as the kernel, then run notebook 1 followed
by the estimation lesson you need. To execute all four from the command line:

```sh
uv run --no-sync python scripts/execute_notebooks.py
```

Generated arrays live in ignored `outputs/notebooks/`. Notebook 1 can be rerun
to regenerate the seeded datasets. No archived data or ephemeris download is
needed; orbits are offline Keplerian approximations.

## Scope

The stationary lesson follows the multivariate spline construction in
[Vajpeyi et al.](https://arxiv.org/abs/2607.04833). The seven-day lesson takes
the short-segment idea from [Burke et al.](https://arxiv.org/html/2610.28726v1).
It is a controlled full-matrix Gaussian simulation, not a full instrument
pipeline. The reference carries orbital time variation; its inferred residual
is frequency-only. The fast gap treatment is a Whittle approximation; the
selected-feature covariance benchmark retains channel/frequency correlations.
The estimation notebooks run two NUTS chains. XYZ fits analytically
marginalise theta, then restore its joint conditional draws. They show
sampler diagnostics; posterior summaries remain conditional on approximate
Whittle observation models and fixed reference geometry.

The independent demo environment pins LogPSplinePSD and GapLike. Published
paper environments are unchanged. Historical scripts and notes remain under
source control for reproducibility; use the four notebooks for the tutorials.

## Development

```sh
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync jupyter-book build docs --warningiserror
uv run --no-sync python scripts/check_docs.py docs/_build/html
```

CI executes the four notebooks with a fresh kernel for each and builds the
docs from saved outputs. The website retains the library's theme and a
persistent **Back to LogPSplinePSD** button.
