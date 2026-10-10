"""Project-root discovery and the project layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from review_capacity.errors import ProjectRootNotFoundError, SpecificationError
from review_capacity.paths import ROOT_ENV, ProjectPaths


def _make_root(directory: Path) -> Path:
    (directory / "config").mkdir(parents=True)
    (directory / "pyproject.toml").write_text("", encoding="utf-8")
    (directory / "config" / "features.yaml").write_text("", encoding="utf-8")
    return directory.resolve()


def test_search_finds_root_from_a_subdirectory(tmp_path: Path) -> None:
    root = _make_root(tmp_path / "project")
    nested = root / "notebooks" / "deep"
    nested.mkdir(parents=True)
    assert ProjectPaths.discover(start=nested, environ={}).root == root


def test_explicit_root_wins_over_environment(tmp_path: Path) -> None:
    first = _make_root(tmp_path / "first")
    second = _make_root(tmp_path / "second")
    found = ProjectPaths.discover(first, start=tmp_path, environ={ROOT_ENV: str(second)})
    assert found.root == first


def test_environment_variable_used_before_search(tmp_path: Path) -> None:
    root = _make_root(tmp_path / "project")
    found = ProjectPaths.discover(start=tmp_path, environ={ROOT_ENV: str(root)})
    assert found.root == root


def test_no_root_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ProjectRootNotFoundError):
        ProjectPaths.discover(start=tmp_path, environ={})
    with pytest.raises(ProjectRootNotFoundError):
        ProjectPaths.discover(tmp_path, environ={})


def test_layout(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    assert paths.features_spec == tmp_path / "config" / "features.yaml"
    assert paths.dataset_spec == tmp_path / "config" / "dataset.yaml"
    assert paths.raw_dir == tmp_path / "data" / "raw"
    assert paths.manifest == tmp_path / "data" / "datapackage.json"
    assert paths.processed_file("train") == tmp_path / "data" / "processed" / "train.parquet"
    assert paths.provenance_file("train").name == "train.provenance.json"
    assert paths.resolve("data/raw/x.csv") == tmp_path / "data" / "raw" / "x.csv"
    with pytest.raises(SpecificationError):
        paths.resolve(tmp_path / "absolute.csv")
