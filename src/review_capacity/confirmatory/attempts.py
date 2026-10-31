"""The attempts of the evaluation phase (Issue #207; protocol, Authorised Test evaluation).

An evaluation attempt is allocated before the phase starts, by a stamped file that is created
exclusively, so that an attempt that was killed (and so wrote no ``test_phase`` record) is found
by the next invocation and recorded as interrupted. Every attempt, completed or not, ends with
exactly one ``test_phase`` record that names the scores bundle it used. ``labels_accessed`` is true
when **that attempt** reached the consumption marker, which is created before the first label is
read: an intent file of the attempt is written just before the marker, so that the answer survives
a kill and a later retry that fails before the marker is not reported as having reached it. The
marker itself is the run-level, irreversible guard.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from review_capacity.confirmatory.records import contract_versions
from review_capacity.development.run_output import RunDirectory, timestamp
from review_capacity.files import write_new_atomically

DIRECTORY = "evaluation"
PHASE = "evaluation"


def _directory(run: RunDirectory) -> Path:
    return run.path / DIRECTORY


def phase_records(run: RunDirectory, phase: str) -> list[dict[str, Any]]:
    """The ``test_phase`` records of ``phase`` already written in the run, in attempt order."""
    paths = sorted(run.records_dir.glob(f"test_phase-{phase}-*.json"))
    return [run.records.read(path, "test_phase") for path in paths]


def allocated(run: RunDirectory) -> list[int]:
    """The numbers of the evaluation attempts allocated in ``run``, ascending."""
    directory = _directory(run)
    if not directory.is_dir():
        return []
    return sorted(
        int(path.stem.removeprefix("attempt-")) for path in directory.glob("attempt-*.json")
    )


def next_attempt(run: RunDirectory) -> int:
    numbers = [r["attempt"] for r in phase_records(run, PHASE)] + allocated(run)
    return max(numbers, default=0) + 1


def allocate(run: RunDirectory, attempt: int, started: str) -> None:
    """Allocate ``attempt`` durably; an attempt number already allocated is refused."""
    write_new_atomically(
        _directory(run) / f"attempt-{attempt:02d}.json",
        json.dumps({"attempt": attempt, "started": started}) + "\n",
    )


def _intent(run: RunDirectory, attempt: int) -> Path:
    return _directory(run) / f"attempt-{attempt:02d}.consuming"


def mark_reaching(run: RunDirectory, attempt: int) -> None:
    """Record, durably and **before** the consumption marker is created, that ``attempt`` is about
    to reach it. A crash between the two leaves the attempt reported as having reached the
    marker, which is the conservative reading."""
    write_new_atomically(_intent(run, attempt), json.dumps({"attempt": attempt}) + "\n")


def reached(run: RunDirectory, attempt: int) -> bool:
    """Whether ``attempt`` reached the consumption marker (the meaning of ``labels_accessed``)."""
    return _intent(run, attempt).exists()


def started_of(run: RunDirectory, attempt: int) -> str:
    stamp = json.loads(
        (_directory(run) / f"attempt-{attempt:02d}.json").read_text(encoding="utf-8")
    )
    started: str = stamp["started"]
    return started


def phase_record(
    identity: Mapping[str, Any],
    attempt: int,
    started: str,
    outcome: str,
    error: str | None,
    *,
    scores_sha256: str,
    labels_accessed: bool,
) -> dict[str, Any]:
    """The ``test_phase`` record of one evaluation attempt."""
    return {
        "record": "test_phase",
        "schema_version": contract_versions()["test_phase"],
        "identity": dict(identity),
        "phase": PHASE,
        "attempt": attempt,
        "started": started,
        "finished": timestamp(),
        "outcome": outcome,
        "error": error,
        "labels_accessed": labels_accessed,
        "scores_sha256": scores_sha256,
    }


def close_orphans(
    run: RunDirectory,
    identity: Mapping[str, Any],
    *,
    scores_sha256: str,
) -> list[int]:
    """Record as interrupted every allocated attempt that left no terminal record (the process
    ended without one, for example after a kill); returns their numbers. Each record says
    whether that attempt had reached the consumption marker."""
    recorded = {r["attempt"] for r in phase_records(run, PHASE)}
    closed = []
    for attempt in allocated(run):
        if attempt in recorded:
            continue
        run.write_record(
            phase_record(
                identity,
                attempt,
                started_of(run, attempt),
                "interrupted",
                "the attempt left no terminal record: the process ended without one",
                scores_sha256=scores_sha256,
                labels_accessed=reached(run, attempt),
            ),
            PHASE,
            attempt,
        )
        closed.append(attempt)
    return closed
