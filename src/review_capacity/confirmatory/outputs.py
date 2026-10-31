"""The summary and the figures of a closed confirmatory run, and their promotion (Issue #209).

``render_outputs`` returns the bytes of ``summary.md`` and of the figures, generated from the
records alone. ``write_outputs`` writes them into the run directory after the run is closed; they
are not listed in ``test_run_end`` (as for the development runs), and a failure to write them
never changes the closed run: ``summarise-run`` regenerates them.

``promote`` copies a closed, completed run's aggregate evidence from the Git-ignored
``outputs/confirmatory/`` into ``evidence/confirmatory/<run id>/``: ``records/*.json``,
``summary.md``, the figures, ``run.log`` and the resumption logs, and a ``manifest.json`` with the
SHA-256 of every file. The row-level score files, the attempt stamps and the bundles stay local.
The copy is staged privately and published only after it has been checked as a whole: its
inventory is exactly the allowlist, its records are those of the validated run, and its summary
and figures are exactly the ones generated from them.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from review_capacity.confirmatory.figures import render_figures
from review_capacity.confirmatory.records import ConfirmatoryRecords
from review_capacity.confirmatory.summary import render_summary
from review_capacity.development.promotion import (
    EVIDENCE_ROOT,
    MANIFEST_NAME,
    _discard,
    _discard_empty,
    _inside,
    _inventory,
    _manifest,
)
from review_capacity.development.run_output import (
    LOG_NAME,
    RECORDS_DIR,
    SUMMARY_NAME,
    LoadedRun,
    load_run,
    timestamp,
)
from review_capacity.errors import FileAccessError, RecordContractError, SpecificationError
from review_capacity.files import write_atomically, write_json
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

RUN_TYPE = "confirmatory"
RESUME_LOG = re.compile(r"resume-\d{4}-\d{2}-\d{2}T\d{6}Z\.log")
ALLOWED = re.compile(
    r"records/[A-Za-z0-9_.-]+\.json|summary\.md|run\.log|resume-\d{4}-\d{2}-\d{2}T\d{6}Z\.log"
    r"|figures/[A-Za-z0-9_.-]+\.svg"
)
MANIFEST_KEYS = ("run_type", "run_id", "identity", "source", "promoted", "files")


def render_outputs(run: LoadedRun) -> dict[str, bytes]:
    """The generated files of a run (relative path -> bytes): the summary and the figures."""
    records = [record for _, record in run.records]
    return {
        SUMMARY_NAME: render_summary(run).encode("utf-8"),
        **render_figures(records),
    }


def _write(path: Path, content: bytes) -> None:
    write_atomically(path, lambda target: target.write_bytes(content))


def write_outputs(directory: Path, records: ConfirmatoryRecords) -> list[Path]:
    """Regenerate the summary and the figures of a run directory from its records, atomically."""
    run = load_run(directory, records=records)
    written = []
    with log_step(logger, f"Writing the summary and figures of {directory.name}"):
        for name, content in render_outputs(run).items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(path, content)
            written.append(path)
    return written


def _check_completed(directory: Path, run: LoadedRun) -> None:
    if run.end is None:
        raise SpecificationError(f"{directory}: the run never closed (no closing record)")
    if run.end["outcome"] != "completed":
        raise SpecificationError(f"{directory}: the run {run.end['outcome']}; only completed runs")


def _files(
    directory: Path, run: LoadedRun, generated: dict[str, bytes], end_kind: str
) -> list[str]:
    """The promoted files (relative POSIX paths): the allowlist, each a regular file."""
    names = [f"{RECORDS_DIR}/{name}" for name, _ in run.records]
    names.append(f"{RECORDS_DIR}/{end_kind}.json")
    names += list(generated)
    names.append(LOG_NAME)
    names += sorted(p.name for p in directory.iterdir() if RESUME_LOG.fullmatch(p.name))
    for name in names:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise SpecificationError(f"{path}: needed for promotion and must be a regular file")
    return sorted(names)


def _check_generated(directory: Path, generated: dict[str, bytes]) -> None:
    """The summary and the figures on disk are, byte for byte, the ones generated from the
    records."""
    for name, content in generated.items():
        try:
            current = (directory / name).read_bytes() == content
        except OSError as error:
            raise FileAccessError(
                f"{directory / name}: cannot be read ({error.strerror})"
            ) from error
        if not current:
            raise SpecificationError(f"{directory / name} differs from the records; regenerate it")


def verify_promoted(directory: Path) -> dict[str, Any]:
    """Check a promoted directory against its manifest and the allowlist, and return the
    manifest: every listed path is allowed, listed once, present and unchanged; nothing else is
    present (only ``manifest.json`` is exempt); there are no links."""
    manifest = _manifest(directory / MANIFEST_NAME)
    paths = [entry["path"] for entry in manifest["files"]]
    outside = sorted(p for p in paths if not ALLOWED.fullmatch(p))
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


def promote(
    directory: Path,
    root: Path,
    records: ConfirmatoryRecords,
    *,
    evidence_root: Path | None = None,
) -> Path:
    """Copy the evidence of the completed confirmatory run in ``directory`` to a new directory
    under ``evidence_root`` (default ``<root>/evidence``) and return it."""
    root = root.resolve()
    source = _inside(directory, root, "The run directory")
    base = _inside(evidence_root or root / EVIDENCE_ROOT, root, "The evidence directory")
    with log_step(logger, f"Promoting {source}"):
        run = load_run(source, records=records)
        _check_completed(source, run)
        generated = render_outputs(run)
        names = _files(source, run, generated, records.end_kind)
        _check_generated(source, generated)
        type_directory = base / RUN_TYPE
        if type_directory.is_symlink():
            raise SpecificationError(f"{type_directory} must be a directory, not a link")
        destination = type_directory / run.identity["run_id"]
        if destination.exists():
            raise SpecificationError(f"{destination} already exists; evidence is never replaced")
        _stage(source, destination, names, run, generated, root, records)
        left = sorted(p.name for p in source.iterdir() if not ALLOWED.fullmatch(p.name))
        logger.info("Promoted %d files to %s; left local: %s", len(names), destination, left)
        return Path(destination)


def _stage(
    source: Path,
    destination: Path,
    names: list[str],
    run: LoadedRun,
    generated: dict[str, bytes],
    root: Path,
    records: ConfirmatoryRecords,
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
        before = {name: file_sha256(source / name) for name in names}
        for name in names:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        changed = sorted(n for n in names if file_sha256(staging / n) != before[n])
        if changed:
            raise SpecificationError(
                f"{changed} changed while the run was being copied; nothing is promoted"
            )
        write_json(
            staging / MANIFEST_NAME,
            {
                "run_type": run.identity["run_type"],
                "run_id": run.identity["run_id"],
                "identity": run.identity,
                "source": source.relative_to(root).as_posix(),
                "promoted": timestamp(),
                "files": [{"path": n, "sha256": file_sha256(staging / n)} for n in names],
            },
        )
        inventory = _inventory(staging)
        if sorted(inventory) != names:
            extra = sorted(set(inventory) - set(names))
            missing = sorted(set(names) - set(inventory))
            raise SpecificationError(
                f"The staged copy differs from the allowlist: extra {extra}, missing {missing}"
            )
        staged = load_run(staging, records=records)
        if staged.end != run.end or staged.identity != run.identity:
            raise SpecificationError(
                "The run changed while it was being copied; nothing is promoted"
            )
        _check_generated(staging, generated)
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
