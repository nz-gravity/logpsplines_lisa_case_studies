# Nonstationary A noise

This introductory case changes the acceleration and optical-metrology-system
(OMS) PSD scales smoothly with time. It draws independent local real Gaussian
coefficients and sums their squared values. It does not simulate a time-domain
TDI record or apply a WDM transform.

```sh
uv run lisa-demo-time --output outputs/time_standard
```

The pinned library's `PowerData` and `PowerConfig(method="vi")` fit a
tensor log-P-spline and a stationary, constant-time-basis spline. Every fourth
time row is held out by a fixed rule. Scaling and knots use training data or
prespecified grids; evaluation truth never enters the fit.

![Nonstationary A recovery](assets/time_varying_noise.png)

VI summaries here are exploratory point reconstructions. They do not establish
NUTS-equivalent posterior uncertainty. The script saves ELBO losses for inspection.

Outputs include `comparison.png`, `data.npz`, `elbo.npz`, `errors.json`,
`run.json`, and two library `PSDResult` files. Read the HDF5-backed `.nc`
files with `PSDResult.from_netcdf`. Their spectra are numerically normalized:
multiply by `data.npz["scale"]` for physical PSD units. The saved comparison
arrays and figure already use physical units.
