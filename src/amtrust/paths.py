"""Location of the repository, data root, output root and results directory.

The data root holds the two Zenodo archives and everything derived from them.
It is resolved in this order: an explicit command-line value, the environment
variable ``AMTRUST_DATA_ROOT``, and otherwise ``<repository>/data``.  The output
root is resolved the same way with ``AMTRUST_OUTPUT_ROOT`` and
``<repository>/outputs``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "benchmark.json"
RESULTS_DIR = REPO_ROOT / "results"

DATA_ROOT_ENV = "AMTRUST_DATA_ROOT"
OUTPUT_ROOT_ENV = "AMTRUST_OUTPUT_ROOT"


def data_root(cli_value: str | os.PathLike | None = None) -> Path:
    """Return the data root directory."""
    if cli_value:
        return Path(cli_value).expanduser().absolute()
    env_value = os.environ.get(DATA_ROOT_ENV)
    if env_value:
        return Path(env_value).expanduser().absolute()
    return REPO_ROOT / "data"


def output_root(cli_value: str | os.PathLike | None = None) -> Path:
    """Return the output root directory used by the full pipeline."""
    if cli_value:
        return Path(cli_value).expanduser().absolute()
    env_value = os.environ.get(OUTPUT_ROOT_ENV)
    if env_value:
        return Path(env_value).expanduser().absolute()
    return REPO_ROOT / "outputs"


class Layout:
    """Standard file locations below a data root and an output root."""

    def __init__(self, data: Path, outputs: Path):
        self.data = Path(data)
        self.outputs = Path(outputs)

    # Data root -----------------------------------------------------------
    @property
    def extracted(self) -> Path:
        return self.data / "extracted"

    @property
    def pretrained(self) -> Path:
        return self.data / "pretrained"

    def archive(self, filename: str) -> Path:
        return self.data / filename

    # Output root ---------------------------------------------------------
    @property
    def manifest_dir(self) -> Path:
        return self.outputs / "manifest"

    @property
    def manifest_csv(self) -> Path:
        return self.manifest_dir / "benchmark_manifest.csv"

    @property
    def boxes_csv(self) -> Path:
        return self.manifest_dir / "reference_boxes.csv"

    @property
    def splits_dir(self) -> Path:
        return self.outputs / "splits"

    @property
    def master_split_csv(self) -> Path:
        return self.splits_dir / "all_split_assignments.csv"

    @property
    def runs_dir(self) -> Path:
        return self.outputs / "runs"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    @property
    def derived_dir(self) -> Path:
        return self.outputs / "derived"

    @property
    def results_dir(self) -> Path:
        return self.outputs / "results"

    @property
    def tables_dir(self) -> Path:
        return self.outputs / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.outputs / "figures"


def make_layout(data: str | None = None, outputs: str | None = None) -> Layout:
    """Build a :class:`Layout` from optional command-line values."""
    return Layout(data_root(data), output_root(outputs))
