"""Shared command-line setup for the scripts: import path, common arguments and layout."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from amtrust.config import load_config  # noqa: E402
from amtrust.paths import RESULTS_DIR, make_layout  # noqa: E402


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-root", default=None,
                   help="Folder with the Zenodo archives (default: AMTRUST_DATA_ROOT or <repo>/data).")
    p.add_argument("--output-root", default=None,
                   help="Folder for generated outputs (default: AMTRUST_OUTPUT_ROOT or <repo>/outputs).")
    p.add_argument("--config", default=None, help="Configuration file (default: configs/benchmark.json).")
    return p


def context(args):
    """Return (layout, config) for parsed arguments."""
    return make_layout(args.data_root, args.output_root), load_config(args.config)


__all__ = ["parser", "context", "RESULTS_DIR", "SRC"]
