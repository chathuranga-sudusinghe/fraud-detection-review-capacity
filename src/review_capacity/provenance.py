"""Provenance records for files the code writes (``*.provenance.json``).

A record states what a file is (name, SHA-256, rows, columns) and how it was made: the source
file hashes, the configuration hashes, the code and the environment. The code is identified by
the Git commit of the project root, whether its working tree had uncommitted changes, and a
SHA-256 fingerprint of the source files of the package that actually ran, so that two runs of
different uncommitted code can be told apart. Records for research evidence should come from a
clean, committed working tree.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import review_capacity
from review_capacity import __version__
from review_capacity.files import write_json

RECORDED_PACKAGES = ("numpy", "pandas", "pyarrow", "PyYAML")


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def package_directory() -> Path:
    """Directory of the ``review_capacity`` package that is running."""
    return Path(review_capacity.__file__).resolve().parent


def source_fingerprint(directory: Path) -> str:
    """SHA-256 over the relative path and bytes of every ``.py`` file under ``directory``,
    in sorted path order (tracked or not)."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        relative = path.relative_to(directory).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def code_provenance(root: Path) -> dict[str, Any]:
    """Code version (Git state of ``root`` and the running package's sources) and environment."""
    status = _git(root, "status", "--porcelain")
    packages: dict[str, str | None] = {}
    for name in RECORDED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    source_dir = package_directory()
    return {
        "package": {
            "name": "review_capacity",
            "version": __version__,
            "location": source_dir.as_posix(),
            "source_sha256": source_fingerprint(source_dir),
        },
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_uncommitted_changes": None if status is None else bool(status),
        "python": platform.python_version(),
        "packages": packages,
    }


def write_record(path: Path, record: dict[str, Any]) -> None:
    """Write a provenance record as indented UTF-8 JSON, atomically."""
    write_json(path, record)
