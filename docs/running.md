# Install and run

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).
Each example creates its own seeded simulation and refuses to overwrite an
existing output directory.

```sh
git clone https://github.com/nz-gravity/logpsplines_lisa_case_studies.git
cd logpsplines_lisa_case_studies
git checkout 6837a066c46afce78ff9cd58d372a1ae5ba0ac37
uv sync --locked --extra gapped
uv run --extra gapped lisa-demo-gaps --output outputs/seven_day
```

The checkout selects the illustrated scientific version, including while
[PR #1](https://github.com/nz-gravity/logpsplines_lisa_case_studies/pull/1) awaits merge.

The default seven-day case runs offline Keplerian orbits, a full matrix
reference correction, adaptive coarse-graining and marginalised-theta VI.
It also reruns the exact compressed Gaussian gap benchmark on the same record.
Three VI restarts use 6,000 updates per Cholesky row and 256 stored draws.

```sh
uv run --extra gapped lisa-demo-gaps --quick --output outputs/seven_day_quick
uv run --extra gapped lisa-demo-gaps --benchmark-only --output outputs/benchmark
```

Quick mode keeps the full seven days at five-second cadence and uses all
59,634 native main-fit frequencies. It shortens VI to 1,500 updates per row
and 64 stored draws, and uses fewer source bands and benchmark features.
It checks execution, not posterior accuracy.

`--vi-steps` changes updates per row and `--vi-restarts` changes the restart
count (at least two). All entry points accept `--seed` and `--output`; the
numbered scripts call the same implementations. The optional `--orbit esa`
path downloads lisaorbits' ESA trailing ephemerides; the published example
uses Keplerian orbits.

## Introductory cases

These need no optional orbit or gap dependencies:

```sh
uv run lisa-demo-time --quick --output outputs/time_quick
uv run lisa-demo-xyz --quick --output outputs/xyz_quick
```

Omit `--quick` for their illustrated settings. The scalar case is a
local-coefficient VI example; the basic XYZ case is a two-parameter NUTS fit.
Neither is the full seven-day matrix reconstruction.

## Reproducibility

`pyproject.toml` pins LogPSplinePSD and GapLike to exact Git commits and
`uv.lock` fixes this independent environment. The newer library pin is used
only here; no published paper reproduction environment is changed.
Each `run.json` records settings, source hashes and limitations. The exact
benchmark has a separate `benchmark_run.json`. Curated previews and receipts
are in `docs/assets/`; generated time series and fit arrays stay in `outputs/`.

`*_correction.nc` contains a library HDF5-backed `PSDResult`, not a generic
NetCDF-4 file. Read it with `PSDResult.from_netcdf`. Its spectra are in
reference coordinates and its frequency axis uses log-frequency spline
coordinates. Physical native-frequency reconstructions and the observation
references are saved separately in `matrix_recovery.npz`.

## Checks and documentation

```sh
uv run --extra gapped pytest -q
uv run ruff check .
uv run ruff format --check .
uv run --group docs jupyter-book build docs --warningiserror
```

The website uses the library's Jupyter Book / Sphinx Book Theme and logo.
It builds from committed results without running inference. Pull requests
build the site and run scientific tests plus the seven-day quick case;
`main` publishes to GitHub Pages.
