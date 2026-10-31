"""Run directories, record persistence and closing of a development run (Issue #120; Issue #96).

A run writes to a directory of its own, ``outputs/<run type>/<UTC start time>/`` (Git-ignored),
that did not exist before (or holds nothing but the run's log file):

* ``records/<kind>[-<part>...].json``: one JSON record per file, each validated against its
  contract and written atomically as soon as it is complete, so that an interruption leaves every
  completed record intact and never a partial one;
* ``run.log``: the run's log file;
* ``summary.md``: the Markdown generated from the records alone (``development.rendering``).

The ``run`` record is written first, the ``run_end`` record last: it states how the run ended and
the SHA-256 of every other record. A directory without ``run_end`` was interrupted before it could
close. There is no resume: a rerun uses a new directory, and a record file is never overwritten.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from review_capacity.development.records import (
    RECORD_CONTRACTS,
    DevelopmentPartitions,
    dump_record,
    read_record,
)
from review_capacity.development.schema import RecordContract
from review_capacity.errors import FileAccessError, RecordContractError, SpecificationError
from review_capacity.files import write_new_atomically
from review_capacity.hashing import file_sha256

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs")
RECORDS_DIR = "records"
LOG_NAME = "run.log"
SUMMARY_NAME = "summary.md"
RUN_KIND = "run"
SOURCE_LINK = "source"  # the link of an analysis run to the development run it reads
END_KIND = "run_end"

_PART = re.compile(r"[A-Za-z0-9_.-]+")


class RecordSet(Protocol):
    """The record contracts a run directory reads and writes: their kinds in production order,
    the kind that opens a run and the kind that closes it, and the validating writer and reader.
    The development pipeline uses ``DevelopmentRecords``; the confirmatory run has its own
    (``confirmatory.records``)."""

    @property
    def contracts(self) -> Mapping[str, RecordContract]: ...

    @property
    def run_kind(self) -> str: ...

    @property
    def end_kind(self) -> str: ...

    def dump(self, record: Mapping[str, Any]) -> str: ...

    def read(self, path: Path, kind: str | None = None) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DevelopmentRecords:
    """The record set of the development pipeline, validated against the development
    partitions."""

    partitions: DevelopmentPartitions

    @property
    def contracts(self) -> Mapping[str, RecordContract]:
        return RECORD_CONTRACTS

    @property
    def run_kind(self) -> str:
        return RUN_KIND

    @property
    def end_kind(self) -> str:
        return END_KIND

    def dump(self, record: Mapping[str, Any]) -> str:
        return dump_record(record, self.partitions)

    def read(self, path: Path, kind: str | None = None) -> dict[str, Any]:
        return read_record(path, self.partitions, kind)


def _record_set(partitions: DevelopmentPartitions | None, records: RecordSet | None) -> RecordSet:
    """``records`` if given, else the development record set of ``partitions``."""
    if records is not None:
        return records
    if partitions is None:
        raise SpecificationError("A run directory needs the development partitions or a record set")
    return DevelopmentRecords(partitions)


def source_file(reference: Mapping[str, str]) -> dict[str, str]:
    """A reference to a file of the source run, as written in an analysis run: the same path and
    SHA-256, with the path under the ``source/`` link."""
    return {"path": f"{SOURCE_LINK}/{reference['path']}", "sha256": reference["sha256"]}


def run_id(now: datetime | None = None) -> str:
    """The UTC start time as ``YYYY-MM-DDTHHMMSSZ``: the run's identifier and directory name."""
    return (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H%M%SZ")


def timestamp(now: datetime | None = None) -> str:
    """A UTC timestamp as the records write it (ISO 8601 with microseconds and ``Z``)."""
    return (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def default_directory(root: Path, run_type: str, started: str) -> Path:
    """``outputs/<run type>/<started>`` under the project root."""
    return root / OUTPUT_ROOT / run_type / started


def check_new_directory(output: Path, root: Path, *, log_opened: bool = False) -> Path:
    """The resolved ``output`` if a run may write there: inside the project root, and a directory
    that does not exist or holds nothing but the run's log file (``log_opened``: the command has
    already opened ``run.log`` there). A directory with anything else is never reused."""
    root = root.resolve()
    resolved = (output if output.is_absolute() else root / output).resolve()
    if resolved != root and root not in resolved.parents:
        raise SpecificationError(f"{resolved} is outside the project root; runs write inside it")
    allowed = {LOG_NAME} if log_opened else set()
    try:
        occupied = resolved.exists() and (
            not resolved.is_dir() or bool({p.name for p in resolved.iterdir()} - allowed)
        )
    except OSError as error:
        raise FileAccessError(f"{resolved}: cannot be inspected ({error.strerror})") from error
    if occupied:
        raise SpecificationError(f"{resolved} already exists; a run needs a new directory")
    return resolved


class RunDirectory:
    """One run's directory: writes validated records, closes the run and keeps its identity.

    Every record of a run carries the same identity, whose run identifier and run type are those
    of the directory; the ``run`` record comes first; nothing is written after ``run_end``.
    """

    def __init__(
        self,
        path: Path,
        partitions: DevelopmentPartitions | None,
        run_type: str,
        started: str,
        records: RecordSet | None = None,
    ) -> None:
        self.path = path
        self._partitions = partitions
        self.records = _record_set(partitions, records)
        self.run_type = run_type
        self.started = started
        self._identity: dict[str, Any] | None = None
        self._closed = False

    @classmethod
    def create(
        cls,
        root: Path,
        partitions: DevelopmentPartitions | None,
        run_type: str,
        started: str,
        output: Path | None = None,
        *,
        log_opened: bool = False,
        source: Path | None = None,
        records: RecordSet | None = None,
    ) -> RunDirectory:
        """Create the run's directory (``output``, or by default the dated one) and its
        ``records/`` subdirectory; refuse any directory that is already in use. An analysis run
        passes the ``source`` run directory it reads: ``source/`` in the new directory is a
        relative link to it, so that a reference ``source/<path>`` resolves inside the run."""
        root = root.resolve()
        target = output if output is not None else default_directory(root, run_type, started)
        path = check_new_directory(target, root, log_opened=log_opened)
        try:
            (path / RECORDS_DIR).mkdir(parents=True)
            if source is not None:
                (path / SOURCE_LINK).symlink_to(os.path.relpath(source.resolve(), path))
        except OSError as error:
            raise FileAccessError(f"{path}: cannot be created ({error.strerror})") from error
        logger.info("Run directory %s", path)
        return cls(path, partitions, run_type, started, records)

    @classmethod
    def reopen(cls, path: Path, run_type: str, records: RecordSet) -> RunDirectory:
        """Re-open the unfinished run in ``path`` to continue it: its first record fixes the
        identity again, and a run that was closed (it holds its closing record) is refused."""
        run_kind, end_kind = records.run_kind, records.end_kind
        if (path / RECORDS_DIR / f"{end_kind}.json").exists():
            raise SpecificationError(f"{path} is closed; a rerun uses a new directory")
        first = path / RECORDS_DIR / f"{run_kind}.json"
        if not first.is_file():
            raise SpecificationError(
                f"{path} holds no {run_kind} record; there is no run to resume"
            )
        identity = records.read(first, run_kind)["identity"]
        if identity["run_type"] != run_type:
            raise SpecificationError(
                f"{path} is a {identity['run_type']} run, not a {run_type} run"
            )
        run = cls(path, None, run_type, identity["run_id"], records)
        run._identity = dict(identity)
        logger.info("Run directory %s re-opened", path)
        return run

    @property
    def partitions(self) -> DevelopmentPartitions:
        """The development partitions of a development-pipeline run directory."""
        if self._partitions is None:
            raise SpecificationError("This run directory has no development partitions")
        return self._partitions

    @property
    def identity(self) -> dict[str, Any] | None:
        """The run's identity, fixed by its first record (``None`` before that)."""
        return self._identity

    @property
    def records_dir(self) -> Path:
        return self.path / RECORDS_DIR

    @property
    def log_file(self) -> Path:
        return self.path / LOG_NAME

    def record_path(self, kind: str, *parts: str | int) -> Path:
        """``records/<kind>[-<part>...].json``; a part is letters, digits, ``_``, ``.`` or ``-``."""
        names = [str(part) for part in parts]
        if kind not in self.records.contracts:
            raise SpecificationError(f"Unknown record kind {kind!r}")
        if not all(_PART.fullmatch(name) for name in names):
            raise SpecificationError(f"Record name parts {names} need letters, digits, _, . or -")
        return self.records_dir / ("-".join([kind, *names]) + ".json")

    def write_record(self, record: Mapping[str, Any], *parts: str | int) -> Path:
        """Validate ``record`` and write it atomically as a new file; return its path.

        Refused: a record that breaks its contract; an identity other than the run's; a first
        record that is not ``run``; the ``run`` record under any name but ``run.json``; a second
        record of the same name (also one that appears while writing: a record is published
        without replacing a file); any record once the run is closed (``run_end`` is written by
        ``finish`` only). The run's identity is fixed by the first ``run`` record that is
        actually written.
        """
        kind = record.get("record")
        run_kind, end_kind = self.records.run_kind, self.records.end_kind
        if self._closed:
            raise SpecificationError(f"{self.path} is closed; a rerun uses a new directory")
        if kind == end_kind:
            raise SpecificationError(f"{end_kind} is written by finish(), not as a record")
        if kind == run_kind and parts:
            raise SpecificationError(f"The {run_kind} record is {run_kind}.json, with no name part")
        text = self.records.dump(record)
        self._check_identity(record)
        path = self.record_path(str(kind), *parts)
        if path.exists():
            raise SpecificationError(f"{path} already exists; a record is never overwritten")
        write_new_atomically(path, text)
        if self._identity is None:
            self._identity = dict(record["identity"])
        logger.debug("Wrote %s", path)
        return path

    def _check_identity(self, record: Mapping[str, Any]) -> None:
        identity = record["identity"]
        if self._identity is None:
            if record["record"] != self.records.run_kind:
                raise SpecificationError(f"The first record of a run is {self.records.run_kind}")
            if identity["run_id"] != self.started or identity["run_type"] != self.run_type:
                raise SpecificationError(
                    f"The run identity {identity['run_type']}/{identity['run_id']} differs from "
                    f"the run {self.run_type}/{self.started}"
                )
        elif identity != self._identity:
            raise SpecificationError(f"{record['record']}: identity differs from the run's own")

    def finish(
        self, outcome: str, error: str | None = None, finished: datetime | None = None
    ) -> Path:
        """Write ``run_end`` with the SHA-256 of every record, and close the run (once)."""
        if self._closed:
            raise SpecificationError(f"{self.path} is already closed")
        end_kind = self.records.end_kind
        if self._identity is None:
            raise SpecificationError(
                f"No {self.records.run_kind} record was written; the run cannot close"
            )
        entries = [
            {"path": f"{RECORDS_DIR}/{path.name}", "sha256": file_sha256(path)}
            for path in _record_files(self.path, end_kind)
        ]
        end = {
            "record": end_kind,
            "schema_version": self.records.contracts[end_kind].version,
            "identity": self._identity,
            "finished": timestamp(finished),
            "outcome": outcome,
            "error": error,
            "records": entries,
        }
        text = self.records.dump(end)
        path = self.record_path(end_kind)
        write_new_atomically(path, text)
        self._closed = True
        logger.info("Run closed: %s (%d records)", outcome, len(entries))
        return path

    @contextmanager
    def closing(self, summarise: Callable[[Path], object] | None = None) -> Iterator[RunDirectory]:
        """Close the run when the block ends, whatever the way: completed, failed or interrupted
        (the exception is re-raised). ``summarise(path)`` then writes the Markdown summary from
        the records, also after a failure."""
        outcome, error = "completed", None
        try:
            yield self
        except KeyboardInterrupt:
            outcome, error = "interrupted", "KeyboardInterrupt"
            raise
        except BaseException as exc:
            outcome, error = "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._close(outcome, error, summarise)

    def _close(
        self, outcome: str, error: str | None, summarise: Callable[[Path], object] | None
    ) -> None:
        """Close the run and write its summary, each step attempted whatever the other does. A
        failure of a step is logged; it is raised only when the run itself had completed (a run
        that failed or was interrupted keeps raising its own, original exception)."""
        steps: list[tuple[str, Callable[[], object]]] = [
            ("closing the run", lambda: None if self._closed else self.finish(outcome, error))
        ]
        if summarise is not None:
            steps.append(("writing the summary", lambda: summarise(self.path)))
        first: BaseException | None = None
        for name, step in steps:
            try:
                step()
            except BaseException as failure:  # an interrupt too: the other step still runs
                logger.exception("%s failed (the run %s)", name, outcome)
                first = first or failure
        if first is not None and outcome == "completed":
            raise first


def _record_files(directory: Path, end_kind: str = END_KIND) -> list[Path]:
    """The record files of a run directory, other than the closing one, in increasing name order.
    Hidden files (the temporary files of an interrupted write) are not records."""
    records = directory / RECORDS_DIR
    if records.is_symlink():
        raise RecordContractError(f"{records}: must be a directory of the run, not a link")
    try:
        entries = sorted(p for p in records.iterdir() if not p.name.startswith("."))
    except OSError as error:
        raise FileAccessError(f"{records}: cannot be read ({error.strerror})") from error
    unexpected = [
        p.name for p in entries if p.suffix != ".json" or p.is_symlink() or not p.is_file()
    ]
    if unexpected:
        raise RecordContractError(f"{records}: unexpected entries {unexpected}")
    return [p for p in entries if p.name != f"{end_kind}.json"]


class LoadedRun:
    """The records of a run directory, validated: ``(file name, record)`` in production order
    (kind order, then name), and the ``run_end`` record if the run closed."""

    def __init__(
        self, path: Path, records: list[tuple[str, dict[str, Any]]], end: dict[str, Any] | None
    ) -> None:
        self.path = path
        self.records = records
        self.end = end

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [record for _, record in self.records if record["record"] == kind]

    @property
    def identity(self) -> dict[str, Any]:
        identity: dict[str, Any] = self.records[0][1]["identity"]
        return identity


def load_run(
    directory: Path,
    partitions: DevelopmentPartitions | None = None,
    records: RecordSet | None = None,
) -> LoadedRun:
    """Read and validate every record of ``directory`` (against the development contracts, or
    the contracts of ``records``).

    Raises ``RecordContractError`` for a record that breaks its contract, a file whose name does
    not match its kind, records of different runs, a directory without a ``run`` record, and, for
    a closed run, any record that is missing, extra or changed since ``run_end`` listed it.
    """
    record_set = _record_set(partitions, records)
    run_kind, end_kind = record_set.run_kind, record_set.end_kind
    kind_order = {kind: index for index, kind in enumerate(record_set.contracts)}
    files = _record_files(directory, end_kind)
    end_path = directory / RECORDS_DIR / f"{end_kind}.json"
    loaded: list[tuple[str, dict[str, Any]]] = []
    for path in files:
        record = record_set.read(path)
        kind = record["record"]
        if path.stem != kind and not path.stem.startswith(f"{kind}-"):
            raise RecordContractError(f"{path}: the name does not match its {kind} record")
        if kind in (run_kind, end_kind) and path.stem != kind:
            raise RecordContractError(f"{path}: a {kind} record is named {kind}.json")
        loaded.append((path.name, record))
    end = record_set.read(end_path, end_kind) if end_path.exists() else None
    if [record["record"] for _, record in loaded].count(run_kind) != 1:
        raise RecordContractError(f"{directory}: needs exactly one {run_kind} record")
    loaded.sort(key=lambda item: (kind_order[item[1]["record"]], item[0]))
    identity = loaded[0][1]["identity"]
    others = [name for name, record in loaded if record["identity"] != identity]
    if others or (end is not None and end["identity"] != identity):
        raise RecordContractError(f"{directory}: records of different runs: {others or end_kind}")
    if end is not None:
        _check_closed(directory, files, end, end_kind)
    return LoadedRun(directory, loaded, end)


def _check_closed(
    directory: Path, files: list[Path], end: Mapping[str, Any], end_kind: str = END_KIND
) -> None:
    listed = {entry["path"]: entry["sha256"] for entry in end["records"]}
    present = {f"{RECORDS_DIR}/{p.name}": file_sha256(p) for p in files}
    changes = sorted(
        [f"{name}: missing" for name in listed.keys() - present.keys()]
        + [f"{name}: not listed in {end_kind}" for name in present.keys() - listed.keys()]
        + [f"{n}: changed" for n in listed.keys() & present.keys() if listed[n] != present[n]]
    )
    if changes:
        raise RecordContractError(f"{directory}: records differ from {end_kind}: {changes}")
