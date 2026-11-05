# Install and run

The notebooks are already executed on this website. To change a setting or
rerun a lesson, use the download button at the top of its page or open the
notebooks in a repository checkout.

## One environment for all four notebooks

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/nz-gravity/logpsplines_lisa_case_studies.git
cd logpsplines_lisa_case_studies
git checkout f7fa624323db7dc2c1166d7f0185eaf8e02c1073
uv sync --locked --extra gapped --group notebooks --group docs
uv run --no-sync jupyter lab docs/notebooks
```

The checkout selects the illustrated notebook version, including while
[PR #1](https://github.com/nz-gravity/logpsplines_lisa_case_studies/pull/1)
awaits merge. Use the project `.venv` Python as the notebook kernel.

## Reading and execution order

1. **[Generate the data](notebooks/01_generate_data.ipynb)** first. It writes
   the shared seeded time series and observation windows.
2. Choose **[time-varying channel A](notebooks/02_time_varying_psd.ipynb)**
   or **[standard XYZ](notebooks/03_xyz_psd.ipynb)**.
3. Continue to **[short-duration XYZ with gaps](notebooks/04_short_xyz_gaps.ipynb)**
   for the orbit reference, adaptive bins and exact covariance check.

Each estimation notebook has its own imports and loads the saved input files.
It does not rely on variables left in another kernel. Use **Restart Kernel
and Run All Cells** to check a clean execution. If a dataset is missing, the
notebook tells you to run notebook 1.

All inputs and fitted arrays are written under `outputs/notebooks/`, which
is ignored by Git. Rerunning notebook 1 regenerates its seeded data. Downloading
one notebook alone is useful for reading; executing it also requires the
checkout's small simulation helpers and shared data.

## Execute everything without JupyterLab

```sh
uv run --no-sync python scripts/execute_notebooks.py
```

This executes the notebooks in order with a fresh project-environment kernel
for each, saving figures, tables and execution counts in the `.ipynb` files.
The execution time depends on your machine; the notebooks use VI to keep
the examples small. There is no hidden NUTS job or cluster submission.

## Reproducibility and limits

`pyproject.toml` pins LogPSplinePSD and GapLike to exact Git commits;
`uv.lock` fixes the notebook environment. The source generator hash and
package versions are recorded beside the generated data. No published paper
environment is changed. All examples estimate channel PSDs and spectral
matrices, rather than instrument-noise component levels.

Read the [simulation and likelihood notes](notebook_models.md) for the
observation approximations and the scope of the exact benchmark.
[GapLike and correlated XYZ](gaplike_xyz.md) explains why a scalar
independent-channel likelihood cannot simply be applied to XYZ.

## Build this website

```sh
uv run --no-sync jupyter-book build docs --warningiserror
uv run --no-sync python scripts/check_docs.py docs/_build/html
```

The book displays committed notebook outputs and does not run inference
while building. It shares the library's Jupyter Book theme, logo and return
navigation. Older case-study page URLs redirect to the corresponding notebooks.
