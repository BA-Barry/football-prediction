from pathlib import Path

import pytest

from football_model.project_paths import (
    ROOT_ENVIRONMENT_VARIABLE,
    project_path,
    project_root,
)


def make_repository(tmp_path: Path) -> Path:
    root = tmp_path / "portable-checkout"
    model_directory = root / "model_v2"
    model_directory.mkdir(parents=True)
    (model_directory / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    return root


def test_source_checkout_is_discovered_without_drive_assumption():
    root = project_root()
    assert (root / "model_v2" / "pyproject.toml").is_file()
    assert project_path("data", root=root) == root / "data"


def test_explicit_root_has_priority(tmp_path, monkeypatch):
    explicit = make_repository(tmp_path)
    monkeypatch.setenv(ROOT_ENVIRONMENT_VARIABLE, str(tmp_path / "not-the-root"))
    assert project_root(explicit) == explicit.resolve()


def test_environment_root_supports_arbitrary_location(tmp_path, monkeypatch):
    configured = make_repository(tmp_path)
    monkeypatch.setenv(ROOT_ENVIRONMENT_VARIABLE, str(configured))
    assert project_root() == configured.resolve()


def test_invalid_root_fails_with_actionable_message(tmp_path):
    with pytest.raises(ValueError, match=ROOT_ENVIRONMENT_VARIABLE):
        project_root(tmp_path / "missing")
