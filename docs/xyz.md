# Correlated XYZ noise

The simplest XYZ case learns two positive PSD scales from equal-arm,
first-generation Michelson noise templates:

$$S_{XYZ}(f)=a_{TM}S_{TM}(f)+a_{OMS}S_{OMS}(f).$$

The scales multiply PSD, not amplitude spectral density. The simulation
draws proper complex XYZ vectors and fits the full Hermitian covariance,
including cross spectra, with two NumPyro NUTS chains.

```sh
uv run lisa-demo-xyz --output outputs/xyz_standard
```

![Parametric XYZ recovery](assets/xyz_parametric_noise.png)

The example uses the library's Wishart likelihood through a small full-matrix
adapter. Fixed nominal whitening preserves the likelihood up to a
parameter-independent constant. It does not treat XYZ as independent.
Equal arms make the injected cross spectra real; complex sample cross spectra
still fluctuate around them.

Outputs include `comparison.png`, `data.npz`, `posterior.nc`,
`sample_stats.nc`, `parameters.json`, and `run.json`. Inspect chain agreement,
R-hat, effective sample size and divergences before interpreting uncertainty.
Quick mode checks execution only.

The [seven-day case](gapped_xyz.md) replaces fixed templates with a physical
orbit response and uses splines to learn secondary-noise spectral shape.
