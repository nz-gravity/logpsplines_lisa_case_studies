# Install and run

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).
Each example creates its own seeded simulation and refuses to overwrite an
existing output directory.

```sh
git clone https://github.com/nz-gravity/logpsplines_lisa_case_studies.git
cd logpsplines_lisa_case_studies
uv sync --locked --extra gapped
uv run --extra gapped lisa-demo-gaps --output outputs/seven_day
```

The default seven-day case uses offline Keplerian orbits and runs on a laptop.
`--quick` uses fewer source bands and Fourier features, keeping all seven days
at the same five-second cadence. It is an execution check, not a resolution
change. Both modes fit three restarts for each of three observation models.

The introductory examples need no optional orbit or gap dependencies:

```sh
uv run lisa-demo-time --quick --output outputs/time_quick
uv run lisa-demo-xyz --quick --output outputs/xyz_quick
```

Omit `--quick` for the illustrated introductory settings. All entry points
accept `--seed` and `--output`; the numbered scripts in `scripts/` call the
same implementations. The optional `--orbit esa` path downloads lisaorbits'
ESA trailing ephemerides; the published example uses Keplerian orbits.

## Reproducibility

`pyproject.toml` pins LogPSplinePSD and GapLike to exact Git commits and
`uv.lock` fixes this independent environment. No paper reproduction
environment is modified. Each `run.json` records settings, versions, source
hashes, simulation and likelihood limitations. Curated previews and receipts
are in `docs/assets/`; generated time series and fit arrays stay in `outputs/`.

## Checks and documentation

```sh
uv run --extra gapped pytest -q
uv run ruff check .
uv run ruff format --check .
uv run --group docs jupyter-book build docs --warningiserror
```

The website uses the same Jupyter Book 1.x / Sphinx Book Theme and logo as
the library. Pages build from committed results and never run inference.
Pull requests build the site and test the scientific contracts; pushes to
`main` publish it to GitHub Pages.
