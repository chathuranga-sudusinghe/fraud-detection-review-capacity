"""The check of the whole run and its closing record (Issue #207).

A run is complete when it holds the ``test_run`` record, a completed scoring phase, the nine
score records, a completed evaluation phase, the outcome records of every configuration, and the
inference record, and all of them agree (``records.run_problems`` with ``complete``). The check
runs **before** the completed evaluation attempt is recorded, with that record added in memory, so
that a run that does not pass is never recorded as completed. Only a run that passes is closed
with ``test_run_end``; the code never closes a run as failed on its own.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from review_capacity.confirmatory.records import run_problems
from review_capacity.development.run_output import RunDirectory
from review_capacity.errors import SpecificationError

logger = logging.getLogger(__name__)


def run_records(run: RunDirectory) -> list[dict[str, Any]]:
    """Every record of the run except the closing record, each validated as it is read."""
    return [
        run.records.read(path)
        for path in sorted(run.records_dir.glob("test_*.json"))
        if path.name != f"{run.records.end_kind}.json"
    ]


def whole_run_problems(run: RunDirectory, completed_phase: Mapping[str, Any] | None) -> list[str]:
    """The problems of the run as it would be with ``completed_phase`` (the record of the
    completed evaluation attempt, not yet written) added; empty when it can be completed."""
    records = run_records(run)
    if completed_phase is not None:
        records.append(dict(completed_phase))
    return run_problems(records, complete=True)


def close_run(run: RunDirectory) -> Path:
    """Close a run whose completed evaluation attempt is recorded: the check is made again on the
    records as written, and ``test_run_end`` lists every record by hash."""
    problems = whole_run_problems(run, None)
    if problems:
        raise SpecificationError("The run cannot be closed: " + "; ".join(problems))
    logger.info("The run is complete and consistent; closing it")
    return run.finish("completed")
