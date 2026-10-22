"""Reviewed promotion of a development run's evidence (Issue #120; Issue #96, decision 4).

Promotion copies a finished run's aggregate evidence from the Git-ignored ``outputs/`` into
``evidence/<run type>/<run id>/``:

* ``records/*.json``: the JSON records, each of which holds aggregate values and links to local
  files by path and SHA-256;
* ``summary.md`` and ``run.log``;
* ``manifest.json``: the run identity and the SHA-256 of every promoted file.

Nothing else is ever copied: the row-level score files, the fitted preprocessing artifacts and
the saved models stay local (the records only name them). Promotion is a separate, explicit step
(``review-capacity promote-evidence``) that is never called from a run, and it needs a closed,
completed run whose summary is exactly the one generated from its records.

The copy is made in a fresh, private staging directory and published only after it has been
checked as a whole: its file inventory is exactly the allowlist, its records satisfy their
contracts and the hashes of the run's closing record, and its summary is the one generated from
the validated run. A source that changes while it is being copied, or a stray file in the
staging area, is therefore rejected before anything reaches ``evidence/``. Promotion assumes
that no other process changes the project tree while it runs; links that exist when it starts
are refused.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from review_capacity.development.records import DevelopmentPartitions
from review_capacity.development.rendering import render_summary
from review_capacity.development.run_output import (
    END_KIND,
    LOG_NAME,
    RECORDS_DIR,
    SUMMARY_NAME,
    LoadedRun,
    load_run,
    timestamp,
)
from review_capacity.errors import FileAccessError, RecordContractError, SpecificationError
from review_capacity.files import write_json
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

EVIDENCE_ROOT = Path("evidence")
MANIFEST_NAME = "manifest.json"
_TOP_LEVEL = {RECORDS_DIR, SUMMARY_NAME, LOG_NAME}
_ALLOWED = re.compile(r"records/[A-Za-z0-9_.-]+\.json|summary\.md|run\.log")
_MANIFEST_KEYS = ("run_type", "run_id", "identity", "source", "promoted", "files")


def _inside(path: Path, root: Path, what: str) -> Path:
    resolved = path.resolve()
    if root not in resolved.parents:
        raise SpecificationError(f"{what} {resolved} must lie inside the project root")
    return resolved


def _files(directory: Path, run: LoadedRun) -> list[str]:
    """The promoted files of a run directory (relative POSIX paths): the allowlist."""
    names = [f"{RECORDS_DIR}/{name}" for name, _ in run.records]
    names.append(f"{RECORDS_DIR}/{END_KIND}.json")
    names += [SUMMARY_NAME, LOG_NAME]
    for name in names:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise SpecificationError(f"{path}: needed for promotion and must be a regular file")
    return sorted(names)


def _check_completed(directory: Path, run: LoadedRun) -> None:
    """The run must be closed and completed."""
    if run.end is None:
        raise SpecificationError(f"{directory}: the run never closed (no {END_KIND} record)")
    if run.end["outcome"] != "completed":
        raise SpecificationError(f"{directory}: the run {run.end['outcome']}; only completed runs")


def _check_summary(directory: Path, run: LoadedRun) -> bytes:
    """The summary file must be, byte for byte, the one generated from the records (read only
    after ``_files`` has shown it to be a regular file); returns those bytes."""
    summary = directory / SUMMARY_NAME
    generated = render_summary(run).encode("utf-8")
    try:
        current = summary.read_bytes() == generated
    except OSError as error:
        raise FileAccessError(f"{summary}: cannot be read ({error.strerror})") from error
    if not current:
        raise SpecificationError(f"{summary} differs from the records; regenerate it first")
    return generated


def _inventory(directory: Path) -> dict[str, str]:
    """Every file below ``directory`` (relative POSIX path -> SHA-256), the manifest excluded at
    the top level only. A link, or anything that is neither a file nor a directory, is an
    error."""
    found: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise RecordContractError(f"{directory}: {relative} is not a regular file")
        if path.is_file() and relative != MANIFEST_NAME:
            found[relative] = file_sha256(path)
    return found


def _check_staging(
    staging: Path,
    expected: list[str],
    run: LoadedRun,
    summary: bytes,
    partitions: DevelopmentPartitions,
) -> None:
    """The staged copy as a whole: exactly the allowlisted files, the records and closing
    record of the validated run, and the generated summary."""
    inventory = _inventory(staging)
    if sorted(inventory) != expected:
        extra = sorted(set(inventory) - set(expected))
        missing = sorted(set(expected) - set(inventory))
        raise SpecificationError(
            f"The staged copy differs from the allowlist: extra {extra}, missing {missing}"
        )
    staged = load_run(staging, partitions)
    if staged.end != run.end or staged.identity != run.identity:
        raise SpecificationError("The run changed while it was being copied; nothing is promoted")
    if (staging / SUMMARY_NAME).read_bytes() != summary:
        raise SpecificationError("The summary changed while it was being copied")


def _discard(path: Path) -> None:
    """Remove a staging directory; a failure is logged, never replacing the error in flight."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError as error:
        logger.warning("Staging directory %s could not be removed: %s", path, error)


def _discard_empty(path: Path) -> None:
    try:
        path.rmdir()
    except OSError as error:
        logger.warning("Empty directory %s could not be removed: %s", path, error)


def promote(
    directory: Path,
    root: Path,
    partitions: DevelopmentPartitions,
    *,
    evidence_root: Path | None = None,
) -> Path:
    """Copy the evidence of the completed run in ``directory`` to a new directory under
    ``evidence_root`` (default ``<root>/evidence``) and return it.

    Raises ``SpecificationError`` if the run is not closed and completed, its summary is not the
    generated one, a needed file is missing or a link, the evidence directory is a link, the run
    changes while it is copied, or the destination exists.
    """
    root = root.resolve()
    source = _inside(directory, root, "The run directory")
    base = _inside(evidence_root or root / EVIDENCE_ROOT, root, "The evidence directory")
    with log_step(logger, f"Promoting {source}"):
        run = load_run(source, partitions)
        _check_completed(source, run)
        names = _files(source, run)
        summary = _check_summary(source, run)
        type_directory = base / run.identity["run_type"]
        if type_directory.is_symlink():
            raise SpecificationError(f"{type_directory} must be a directory, not a link")
        destination: Path = type_directory / run.identity["run_id"]
        if destination.exists():
            raise SpecificationError(f"{destination} already exists; evidence is never replaced")
        _stage(source, destination, names, run, summary, root, partitions)
        excluded = sorted(p.name for p in source.iterdir() if p.name not in _TOP_LEVEL)
        logger.info("Promoted %d files to %s; left local: %s", len(names), destination, excluded)
        return destination


def _stage(
    source: Path,
    destination: Path,
    names: list[str],
    run: LoadedRun,
    summary: bytes,
    root: Path,
    partitions: DevelopmentPartitions,
) -> None:
    """Copy into a fresh staging directory next to ``destination``, check the copy as a whole,
    then publish it."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        )
    except OSError as error:
        raise FileAccessError(f"{destination.parent}: cannot be prepared ({error})") from error
    created = published = False
    try:
        for name in names:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        identity = run.identity
        manifest: dict[str, Any] = {
            "run_type": identity["run_type"],
            "run_id": identity["run_id"],
            "identity": identity,
            "source": source.relative_to(root).as_posix(),
            "promoted": timestamp(),
            "files": [{"path": n, "sha256": file_sha256(staging / n)} for n in names],
        }
        write_json(staging / MANIFEST_NAME, manifest)
        _check_staging(staging, names, run, summary, partitions)
        verify_promoted(staging)
        try:
            destination.mkdir()  # exclusive: fails if the destination appeared meanwhile
        except FileExistsError as error:
            raise SpecificationError(
                f"{destination} already exists; evidence is never replaced"
            ) from error
        created = True
        staging.replace(destination)
        published = True
    except FileAccessError:
        raise
    except OSError as error:
        raise FileAccessError(f"{destination}: cannot be written ({error.strerror})") from error
    finally:
        _discard(staging)
        if created and not published:
            _discard_empty(destination)


def _manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RecordContractError(f"{path}: the manifest must be a regular file")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RecordContractError(f"{path}: cannot be read as a manifest ({error})") from error
    if not isinstance(manifest, Mapping):
        raise RecordContractError(f"{path}: the manifest must be a JSON object")
    problems = [f"missing {key!r}" for key in _MANIFEST_KEYS if key not in manifest]
    files = manifest.get("files")
    if not isinstance(files, list) or not all(
        isinstance(entry, Mapping)
        and isinstance(entry.get("path"), str)
        and isinstance(entry.get("sha256"), str)
        for entry in files
    ):
        problems.append("'files' must be a list of {path, sha256}")
    if problems:
        raise RecordContractError(f"{path}: invalid manifest: {problems}")
    return dict(manifest)


def verify_promoted(directory: Path) -> dict[str, Any]:
    """Check a promoted directory against its manifest and the allowlist, and return the
    manifest: every listed path is an allowed one, listed once, present and unchanged; nothing
    else is present (only the root ``manifest.json`` is exempt); there are no links."""
    manifest = _manifest(directory / MANIFEST_NAME)
    paths = [entry["path"] for entry in manifest["files"]]
    outside = sorted(p for p in paths if not _ALLOWED.fullmatch(p))
    if outside or len(set(paths)) != len(paths):
        raise RecordContractError(
            f"{directory}: the manifest lists paths outside the allowlist, or a path twice: "
            f"{outside}"
        )
    listed = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    present = _inventory(directory)
    if listed != present:
        differing = sorted(
            name for name in listed.keys() | present.keys() if listed.get(name) != present.get(name)
        )
        raise RecordContractError(f"{directory}: differs from its manifest: {differing}")
    return manifest
