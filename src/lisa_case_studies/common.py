"""Small CLI/output helpers shared by the two examples."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parser(description, default_output):
    result = argparse.ArgumentParser(description=description)
    result.add_argument(
        "--quick",
        action="store_true",
        help="small execution check, not posterior validation",
    )
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--output", type=Path, default=Path(default_output))
    return result


def create_output(path):
    # Refuse overwrites so a rerun cannot silently replace a prior example.
    path.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )


def receipt(path, **values):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        commit = None
    packages = {
        p: importlib.metadata.version(p)
        for p in ("LogPSplinePSD", "numpy", "jax", "numpyro")
    }
    root = Path(__file__).resolve().parents[2]
    source_hashes = {
        str(file.relative_to(root)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(root.glob("src/**/*.py"))
    }
    payload = dict(
        created_utc=datetime.now(UTC).isoformat(),
        source_commit=commit,
        source_sha256=source_hashes,
        dependencies=packages,
        **values,
    )
    (path / "run.json").write_text(json.dumps(payload, indent=2))


def log_rmse(estimate, truth):
    return float(np.sqrt(np.mean(np.log10(np.asarray(estimate) / truth) ** 2)))


def save_figure(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
