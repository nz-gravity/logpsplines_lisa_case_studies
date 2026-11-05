"""Self-contained teaching examples; not a paper reproduction pipeline."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
