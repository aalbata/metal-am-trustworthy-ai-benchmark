"""Loading of the benchmark configuration file."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .paths import CONFIG_PATH


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: str | Path | None = None) -> dict:
    """Return the parsed benchmark configuration."""
    return _load(str(Path(path) if path else CONFIG_PATH))


def build_tokens(config: dict | None = None) -> dict:
    """Return the mapping from build label (B1, B2, B3) to build token."""
    config = config or load_config()
    return dict(config["dataset"]["builds"])


def build_label(token: str, config: dict | None = None) -> str:
    """Return the build label (B1, B2 or B3) of a build token."""
    for label, value in build_tokens(config).items():
        if value == token:
            return label
    raise KeyError(f"Unknown build token: {token}")
