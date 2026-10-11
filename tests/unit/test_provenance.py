"""Code provenance: the source fingerprint and graceful recording of unavailable facts."""

from __future__ import annotations

import logging
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from review_capacity.provenance import code_provenance, source_fingerprint


def _tree(root: Path) -> Path:
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("A = 1\n", encoding="utf-8")
    (root / "pkg" / "sub" / "b.py").write_text("B = 2\n", encoding="utf-8")
    return root / "pkg"


def test_fingerprint_is_stable_for_the_same_sources(tmp_path: Path) -> None:
    package = _tree(tmp_path)
    assert source_fingerprint(package) == source_fingerprint(package)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: (p / "a.py").write_text("A = 2\n", encoding="utf-8"),  # content
        lambda p: (p / "a.py").rename(p / "c.py"),  # name
        lambda p: (p / "sub" / "d.py").write_text("", encoding="utf-8"),  # another module
    ],
)
def test_fingerprint_changes_with_the_python_sources(tmp_path: Path, change: Any) -> None:
    package = _tree(tmp_path)
    before = source_fingerprint(package)
    change(package)
    assert source_fingerprint(package) != before


def test_fingerprint_ignores_files_that_are_not_python(tmp_path: Path) -> None:
    package = _tree(tmp_path)
    before = source_fingerprint(package)
    (package / "notes.txt").write_text("not code", encoding="utf-8")
    assert source_fingerprint(package) == before


def test_unavailable_git_and_package_are_null_with_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def no_git(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git")

    def version(name: str) -> str:
        if name == "pyarrow":
            raise metadata.PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr(subprocess, "run", no_git)
    monkeypatch.setattr(metadata, "version", version)
    with caplog.at_level(logging.WARNING, logger="review_capacity"):
        record = code_provenance(tmp_path)
    assert record["git_commit"] is None
    assert record["git_uncommitted_changes"] is None
    assert record["packages"]["pyarrow"] is None
    assert record["packages"]["pandas"] == "1.0"
    assert len(record["package"]["source_sha256"]) == 64  # kept without Git
    assert "unavailable" in caplog.text
    assert "pyarrow is not installed" in caplog.text
