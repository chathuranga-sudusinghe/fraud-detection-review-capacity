"""The one-pass guard of the Test evaluation (Issue #203; protocol, Authorised Test evaluation).

Three files in a fixed directory outside the dated run directories, named by the identity of the
frozen state (commit and hashes, not the tag name), so that a repeat in a new run directory still
finds them:

- the **lock** prevents concurrent runs. It is created exclusively and removed when the run ends,
  whatever the way. A lock left by a killed process is never removed by the code: the command
  stops and names the file, and the stale lock is a decision of the person who runs it.
- the **scoring marker** records that the scoring phase completed, with the run and the bundle
  hash of its scores. It is created atomically when the phase completes and never deleted; while
  it exists, no scoring phase may start for the same frozen state in any directory, because only
  a failed phase 1 may be repeated.
- the **consumption marker** records that the evaluation reached the Test labels. It is created
  atomically before the first label is read and is never deleted by the code. Once it exists, no
  new scoring phase may start for the same frozen state; a failed evaluation resumes from the
  saved, hashed scores.

Nothing here reads research data.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from review_capacity.errors import FileAccessError, SpecificationError
from review_capacity.files import write_new_atomically

logger = logging.getLogger(__name__)

STATE_DIRECTORY = Path("outputs") / "confirmatory" / "state"
LOCK_SUFFIX = ".lock"
MARKER_SUFFIX = ".consumed.json"
SCORED_SUFFIX = ".scored.json"
_IDENTITY = re.compile(r"[0-9a-f]{64}")


def state_directory(root: Path) -> Path:
    return root / STATE_DIRECTORY


def _checked(identity: str) -> str:
    if not _IDENTITY.fullmatch(identity):
        raise SpecificationError(f"The frozen-state identity {identity!r} is not a SHA-256 digest")
    return identity


def lock_path(root: Path, frozen_state_sha256: str) -> Path:
    return state_directory(root) / (_checked(frozen_state_sha256) + LOCK_SUFFIX)


def marker_path(root: Path, frozen_state_sha256: str) -> Path:
    return state_directory(root) / (_checked(frozen_state_sha256) + MARKER_SUFFIX)


def scored_path(root: Path, frozen_state_sha256: str) -> Path:
    return state_directory(root) / (_checked(frozen_state_sha256) + SCORED_SUFFIX)


def is_consumed(root: Path, frozen_state_sha256: str) -> bool:
    return marker_path(root, frozen_state_sha256).exists()


def is_consumed_by(root: Path, frozen_state_sha256: str, run_id: str) -> bool:
    """Whether the consumption marker exists and was created by the run ``run_id``."""
    path = marker_path(root, frozen_state_sha256)
    if not path.exists():
        return False
    found = json.loads(path.read_text(encoding="utf-8"))
    return bool(found.get("run_id") == run_id)


def refuse_if_consumed(root: Path, frozen_state_sha256: str) -> None:
    """Stop a new scoring phase once the labels have been read for this frozen state."""
    marker = marker_path(root, frozen_state_sha256)
    if marker.exists():
        raise SpecificationError(
            f"The evaluation of this frozen state is consumed ({marker.name}): the Test labels "
            "were read, so no new scoring phase may start; a failed evaluation resumes from the "
            "saved scores"
        )


def consume(root: Path, frozen_state_sha256: str, *, run_id: str, scores_sha256: str) -> Path:
    """Record that the labels are about to be read: created atomically, before the first label.
    A second call for the same frozen state is refused (the marker is never replaced)."""
    path = marker_path(root, frozen_state_sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        {
            "frozen_state_sha256": frozen_state_sha256,
            "run_id": run_id,
            "scores_sha256": scores_sha256,
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        },
        indent=2,
        sort_keys=True,
    )
    try:
        write_new_atomically(path, text + "\n")
    except FileAccessError as error:
        if not path.exists():
            raise
        raise SpecificationError(
            f"{path.name} already exists: the evaluation is consumed"
        ) from error
    logger.info("Consumption marker written: %s", path)
    return path


@contextmanager
def locked(root: Path, frozen_state_sha256: str, *, run_id: str) -> Iterator[Path]:
    """Hold the lock of the frozen state for the block. Refused while a lock exists."""
    path = lock_path(root, frozen_state_sha256)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise SpecificationError(
            f"{path} exists: another run holds the lock, or a killed run left it; the code never "
            "removes a lock"
        ) from error
    except OSError as error:
        raise FileAccessError(f"{path}: the lock cannot be created ({error.strerror})") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"{run_id} pid {os.getpid()}\n")
        logger.info("Lock acquired: %s", path.name)
        yield path
    finally:
        path.unlink(missing_ok=True)
        logger.info("Lock released: %s", path.name)


def refuse_if_scored(root: Path, frozen_state_sha256: str) -> None:
    """Stop a new scoring phase once a scoring phase completed for this frozen state."""
    marker = scored_path(root, frozen_state_sha256)
    if marker.exists():
        raise SpecificationError(
            f"The scoring phase of this frozen state completed ({marker.name}); only a failed "
            "scoring phase may be repeated, and the evaluation phase is next"
        )


def record_scoring(
    root: Path, frozen_state_sha256: str, *, run_id: str, scores_sha256: str
) -> Path:
    """Record, atomically and once, that the scoring phase completed. A second call is refused."""
    path = scored_path(root, frozen_state_sha256)
    text = json.dumps(
        {
            "frozen_state_sha256": frozen_state_sha256,
            "run_id": run_id,
            "scores_sha256": scores_sha256,
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        },
        indent=2,
        sort_keys=True,
    )
    try:
        write_new_atomically(path, text + "\n")
    except FileAccessError as error:
        if not path.exists():
            raise
        raise SpecificationError(f"{path.name} already exists: the scoring is recorded") from error
    logger.info("Scoring marker written: %s", path)
    return path


def begin_consumption(
    root: Path, frozen_state_sha256: str, *, run_id: str, scores_sha256: str
) -> Path:
    """Create the consumption marker before the first label is read, or accept the marker that
    this same run created for these same scores (a resumed evaluation). A marker of another run
    or of other scores is a consumed evaluation and is refused."""
    path = marker_path(root, frozen_state_sha256)
    if not path.exists():
        return consume(root, frozen_state_sha256, run_id=run_id, scores_sha256=scores_sha256)
    found = json.loads(path.read_text(encoding="utf-8"))
    if (found.get("run_id"), found.get("scores_sha256")) != (run_id, scores_sha256):
        raise SpecificationError(
            f"{path.name} belongs to another run or other scores: the evaluation of this frozen "
            "state is consumed"
        )
    logger.info("Resuming the evaluation consumed by this run: %s", path.name)
    return path
