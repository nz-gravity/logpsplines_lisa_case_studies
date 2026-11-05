"""Execute the four tutorials in order using this environment's Python.

By default, save executed notebooks in docs/notebooks. --output-directory
keeps a separate CI execution copy while retaining the documented workdir.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "notebooks",
    nargs="*",
    help="Optional filenames; default is the four tutorials in order",
)
parser.add_argument("--output-directory", type=Path)
args = parser.parse_args()
paths = (
    [ROOT / "docs/notebooks" / name for name in args.notebooks]
    if args.notebooks
    else sorted((ROOT / "docs/notebooks").glob("0*.ipynb"))
)

with tempfile.TemporaryDirectory(prefix="lisa-notebook-kernel-") as directory:
    kernel_path = Path(directory) / "lisa-tutorial"
    kernel_path.mkdir()
    (kernel_path / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "LISA tutorial",
                "language": "python",
            }
        )
    )
    for path in paths:
        print(f"Executing {path.name} ...", flush=True)
        started = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        manager = AsyncKernelManager(
            kernel_name="lisa-tutorial",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]),
            transport="ipc",
            ip=str(Path(directory) / "channel"),
        )
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=1200,
            resources={"metadata": {"path": str(path.parent)}},
        )
        # The runner supplies the kernel spec; the client owns its lifecycle.
        client.owns_km = True
        client.execute()
        nbformat.validate(notebook)
        plots = sum(
            "image/png" in output.get("data", {})
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.outputs
        )
        if plots < 3:
            raise RuntimeError(f"Expected teaching plots in {path.name}, got {plots}")
        destination = (
            args.output_directory / path.name if args.output_directory else path
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination)
        print(
            f"Saved {destination.name}: {plots} figures, {time.monotonic() - started:.1f}s",
            flush=True,
        )
