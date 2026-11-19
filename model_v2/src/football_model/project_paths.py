"""Portable project-directory discovery shared by package entry points."""

from __future__ import annotations

import os
from pathlib import Path


ROOT_ENVIRONMENT_VARIABLE = "FOOTBALL_MODEL_ROOT"


def project_root(explicit: str | Path | None = None) -> Path:
    """Return the repository root without assuming a drive letter or username.

    An explicit value has priority, followed by ``FOOTBALL_MODEL_ROOT``.  The
    source checkout layout is the final default.  This keeps installed commands
    configurable while making a cloned repository work in place.
    """
    configured = explicit or os.environ.get(ROOT_ENVIRONMENT_VARIABLE)
    candidate = (Path(configured).expanduser() if configured
                 else Path(__file__).resolve().parents[3]).resolve()
    if not (candidate / "model_v2" / "pyproject.toml").is_file():
        raise ValueError(
            f"{candidate} is not a football-prediction repository root; "
            f"set {ROOT_ENVIRONMENT_VARIABLE} to the cloned repository directory"
        )
    return candidate


def project_path(*parts: str, root: str | Path | None = None) -> Path:
    """Resolve a path below the configured project root."""
    return project_root(root).joinpath(*parts)
