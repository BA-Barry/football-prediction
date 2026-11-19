"""Portable paths for the dated research scripts in this directory."""

from __future__ import annotations

import os
from pathlib import Path


ROOT_ENVIRONMENT_VARIABLE = "FOOTBALL_MODEL_ROOT"


def discover_project_root(explicit: str | Path | None = None) -> Path:
    """Use an explicit root, an environment override, or this checkout."""
    configured = explicit or os.environ.get(ROOT_ENVIRONMENT_VARIABLE)
    candidate = (Path(configured).expanduser() if configured
                 else Path(__file__).resolve().parents[1]).resolve()
    if not (candidate / "model_v2" / "pyproject.toml").is_file():
        raise RuntimeError(
            f"{candidate} is not a football-prediction repository root; "
            f"set {ROOT_ENVIRONMENT_VARIABLE} before running this script"
        )
    return candidate


PROJECT_ROOT = discover_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "work" / "cache"
