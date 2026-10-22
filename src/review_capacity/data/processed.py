"""Reading the processed development files, verified against their provenance records.

After the Parquet build (``prepare-data``), development work reads the data only from
``data/processed/`` and only through this module; no raw file is opened. ``read_processed``
checks a file before returning any of its data:

* its provenance record is well formed and describes this file and its scope;
* the file's SHA-256, rows and columns equal the recorded values;
* the recorded raw-source hashes equal those declared in ``config/dataset.yaml``, so the file
  comes from the verified raw data;
* the recorded feature-declaration fingerprint equals that of the specification in use
  (``config.features_fingerprint``): the column roles and the missing label decide the stored
  values, so a file built under another declaration is stale and rejected, while a change to the
  library-handling blocks or to comments of ``config/features.yaml`` does not invalidate it; a
  record without a fingerprint (built before it existed) is rejected until ``migrate-provenance``
  has verified and re-recorded it;
* its columns are exactly the declared schema, with the declared physical types (integer, text
  or float64) in the Parquet schema; a partition file has no missing categorical label;
* its content reproduces the DR-001 anchors, from the identifier and time columns only and
  before any label is read: rows, identifier-hash prefix and relative days of every development
  partition it holds, and, in the joined file, the ``TransactionDT`` bounds and the partition of
  every row; identifiers are unique.

The generating commit is recorded, not required to equal the current one: a valid file stays
valid across code changes that do not alter what it contains (the integration tests rebuild the
partition files in memory from the joined file to show that). A changed ``config/dataset.yaml``
is reported, not rejected, because the partition content is checked against it directly.

Files and their scope:

* ``development_joined``: every joined raw column of the development rows, with
  ``relative_day``, ``partition`` and the target;
* one file per development partition: the identifier, ``relative_day``, the target and the
  canonical model inputs (``features.model_inputs``), in that order.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from review_capacity.config import DatasetSpec, FeatureSpec, features_fingerprint
from review_capacity.data.partitions import assign_partitions, relative_day
from review_capacity.errors import DataValidationError, FileAccessError
from review_capacity.hashing import file_sha256, id_set_sha256
from review_capacity.paths import ProjectPaths

JOINED = "development_joined"
DAY_COLUMN = "relative_day"
PARTITION_COLUMN = "partition"

logger = logging.getLogger(__name__)
_HEX_64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ProvenanceRecord:
    """The checked fields of a ``*.provenance.json`` record (``provenance.write_record``)."""

    file: str
    sha256: str
    rows: int
    columns: int
    scope: str
    transaction_id_sha256: str
    sources: dict[str, str]
    configuration: dict[str, str]
    code: dict[str, Any]


def stems(dataset: DatasetSpec) -> tuple[str, ...]:
    """Every processed file stem: the joined file, then one per development partition."""
    return (JOINED, *(p.name for p in dataset.development))


def _scope(stem: str, dataset: DatasetSpec) -> str:
    if stem == JOINED:
        return "development"
    if stem in {p.name for p in dataset.development}:
        return stem
    raise DataValidationError(f"{stem} is not a processed development file")


def schema(stem: str, features: FeatureSpec) -> tuple[str, ...]:
    """Declared columns of a processed file; ordered for partition files, in any order for the
    joined file (whose raw columns keep the file order of the raw data)."""
    if stem == JOINED:
        rest = (c for c in features.joined_columns if c not in (features.key, features.target))
        return (features.key, DAY_COLUMN, PARTITION_COLUMN, features.target, *rest)
    return (features.key, DAY_COLUMN, features.target, *features.model_inputs)


def _text(record: dict[str, Any], name: str, path: Path) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"{path.name}: {name} must be a non-empty string")
    return value


def _count(record: dict[str, Any], name: str, path: Path) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataValidationError(f"{path.name}: {name} must be a non-negative integer")
    return value


def _hashes(record: dict[str, Any], name: str, path: Path) -> dict[str, str]:
    value = record.get(name)
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and _HEX_64.fullmatch(v) for k, v in value.items()
    ):
        raise DataValidationError(f"{path.name}: {name} must map file names to SHA-256 values")
    return dict(value)


def read_record(path: Path) -> ProvenanceRecord:
    """Read and check the structure of a provenance record."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileAccessError(f"{path}: provenance record missing") from error
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error.strerror})") from error
    except json.JSONDecodeError as error:
        raise DataValidationError(f"{path.name}: not valid JSON") from error
    if not isinstance(record, dict):
        raise DataValidationError(f"{path.name}: a provenance record must be a JSON object")
    sha256 = _text(record, "sha256", path)
    id_hash = _text(record, "transaction_id_sha256", path)
    if not (_HEX_64.fullmatch(sha256) and _HEX_64.fullmatch(id_hash)):
        raise DataValidationError(f"{path.name}: sha256 values must be 64 hex characters")
    code = record.get("code")
    if not isinstance(code, dict):
        raise DataValidationError(f"{path.name}: code must be a JSON object")
    return ProvenanceRecord(
        file=_text(record, "file", path),
        sha256=sha256,
        rows=_count(record, "rows", path),
        columns=_count(record, "columns", path),
        scope=_text(record, "scope", path),
        transaction_id_sha256=id_hash,
        sources=_hashes(record, "sources", path),
        configuration=_hashes(record, "configuration", path),
        code=code,
    )


FINGERPRINT = "features_fingerprint"


def verify_processed(
    paths: ProjectPaths,
    stem: str,
    features: FeatureSpec,
    dataset: DatasetSpec,
    *,
    migrating: bool = False,
) -> ProvenanceRecord:
    """Check a processed file against its record, the declared sources, schema and anchors.

    ``migrating`` is for ``migrate-provenance`` only: the record predates the fingerprint, so its
    declaration is checked by the whole-file hash it recorded, which must still equal the current
    file's; everything else that this function checks is checked as usual."""
    scope = _scope(stem, dataset)
    path = paths.processed_file(stem)
    record = read_record(paths.provenance_file(stem))
    problems: list[str] = []
    if record.file != path.name:
        problems.append(f"record describes {record.file}")
    if record.scope != scope:
        problems.append(f"scope {record.scope}, expected {scope}")
    declared = {spec.path.name: spec.sha256 for spec in (dataset.transaction, dataset.identity)}
    if record.sources != declared:
        problems.append("recorded raw-source hashes differ from config/dataset.yaml")
    if stem != JOINED:
        prefix = dataset.partition(stem).transaction_id_sha256_prefix
        if not record.transaction_id_sha256.startswith(prefix):
            problems.append(f"identifier hash does not start with the DR-001 anchor {prefix}")
    declaration = paths.features_spec.name
    if migrating:
        if record.configuration.get(declaration) != file_sha256(paths.features_spec):
            problems.append(
                f"cannot migrate: the recorded {declaration} hash differs from the current file. "
                f"Migrate with the {declaration} the files were built with, before any status or "
                "comment edit; never rebuild from the raw data"
            )
    else:
        recorded = record.configuration.get(FINGERPRINT)
        if recorded is None:
            problems.append(
                f"has no {FINGERPRINT} (built before it existed); run migrate-provenance"
            )
        elif recorded != features_fingerprint(features):
            problems.append(f"built under another {declaration} declaration (stale)")
    if problems:
        raise DataValidationError(f"{path.name}: " + "; ".join(problems))
    digest = file_sha256(path)
    if digest != record.sha256:
        raise DataValidationError(f"{path.name}: SHA-256 {digest} differs from its record")
    try:
        parquet = pq.ParquetFile(path)
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error})") from error
    except ValueError as error:
        raise DataValidationError(f"{path.name}: not a readable Parquet file") from error
    arrow_schema = parquet.schema_arrow
    rows, names = int(parquet.metadata.num_rows), list(arrow_schema.names)
    if (rows, len(names)) != (record.rows, record.columns):
        raise DataValidationError(
            f"{path.name}: {rows} rows and {len(names)} columns, "
            f"record states {record.rows} and {record.columns}"
        )
    expected = schema(stem, features)
    # The joined file keeps the raw column order; a partition file has the declared order.
    same = sorted(names) == sorted(expected) if stem == JOINED else names == list(expected)
    if not same:
        raise DataValidationError(
            f"{path.name}: columns differ from the declared schema: "
            f"unexpected {sorted(set(names) - set(expected))}, "
            f"missing {sorted(set(expected) - set(names))}"
            + ("" if stem == JOINED else " (or their order differs)")
        )
    _check_types(arrow_schema, path, stem, features)
    _check_content(path, stem, features, dataset)
    _report_lineage(paths, path, record)
    return record


def _report_lineage(paths: ProjectPaths, path: Path, record: ProvenanceRecord) -> None:
    commit = record.code.get("git_commit")
    logger.info(
        "%s verified: SHA-256 matches its record; %d rows, %d columns; built by commit %s",
        path.name,
        record.rows,
        record.columns,
        commit[:12] if isinstance(commit, str) else "unknown",
    )
    if record.code.get("git_uncommitted_changes") is not False:
        logger.warning("%s was not built from a clean, committed working tree", path.name)
    split = paths.dataset_spec
    if split.exists() and record.configuration.get(split.name) != file_sha256(split):
        logger.info(
            "%s was built with another %s; its partition content is checked against the "
            "current one",
            path.name,
            split.name,
        )


def _check_types(arrow_schema: Any, path: Path, stem: str, features: FeatureSpec) -> None:
    """Physical Parquet types of every column, before any column is read."""
    integer = {features.key, features.time, DAY_COLUMN, features.target}
    text = (
        {PARTITION_COLUMN, *features.text_categorical}
        if stem == JOINED
        else set(features.categorical)
    )
    problems = []
    for field in arrow_schema:
        if field.name in integer:
            ok = pa.types.is_integer(field.type)
        elif field.name in text:
            ok = pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
        else:
            ok = pa.types.is_float64(field.type)
        if not ok:
            problems.append(f"{field.name} ({field.type})")
    if problems:
        raise DataValidationError(f"{path.name}: columns of an undeclared kind: {problems}")


def _check_content(path: Path, stem: str, features: FeatureSpec, dataset: DatasetSpec) -> None:
    """DR-001 anchors reproduced from the identifier and time columns (no label is read)."""
    columns = [features.key, DAY_COLUMN]
    if stem == JOINED:
        columns += [features.time, PARTITION_COLUMN]
    frame = pd.read_parquet(path, columns=columns)
    problems = []
    if frame[features.key].isna().any() or not frame[features.key].is_unique:
        problems.append("identifiers missing or repeated")
    if stem == JOINED:
        days = relative_day(frame[features.time])
        if not (frame[DAY_COLUMN] == days).all():
            problems.append("relative_day differs from TransactionDT // 86400")
        assigned = assign_partitions(days, dataset).astype(str)
        if not (frame[PARTITION_COLUMN] == assigned).all():
            problems.append("partition differs from the DR-001 boundaries")
        parts = list(dataset.development)
        if set(frame[PARTITION_COLUMN].unique()) - {p.name for p in parts}:
            problems.append("rows outside the development partitions")
    else:
        parts = [dataset.partition(stem)]
    for part in parts:
        rows = frame if stem != JOINED else frame[frame[PARTITION_COLUMN] == part.name]
        if len(rows) != part.rows:
            problems.append(f"{part.name}: {len(rows)} rows, DR-001 states {part.rows}")
        if not id_set_sha256(rows[features.key]).startswith(part.transaction_id_sha256_prefix):
            problems.append(f"{part.name}: identifier hash differs from the DR-001 anchor")
        if not rows[DAY_COLUMN].between(part.first_day, part.last_day).all():
            problems.append(f"{part.name}: relative days outside {part.first_day}-{part.last_day}")
        if stem == JOINED and len(rows):
            bounds = (int(rows[features.time].min()), int(rows[features.time].max()))
            if bounds != (part.transaction_dt_min, part.transaction_dt_max):
                problems.append(f"{part.name}: TransactionDT bounds {bounds} differ from DR-001")
    if problems:
        raise DataValidationError(f"{path.name}: content does not reproduce DR-001: {problems}")


def read_processed(
    paths: ProjectPaths,
    stem: str,
    features: FeatureSpec,
    dataset: DatasetSpec,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """A verified processed file; with ``columns``, only those columns are read from storage."""
    verify_processed(paths, stem, features, dataset)
    return _load(paths, stem, features, columns)


def _load(
    paths: ProjectPaths, stem: str, features: FeatureSpec, columns: Sequence[str] | None
) -> pd.DataFrame:
    path = paths.processed_file(stem)
    if columns is not None:
        unknown = sorted(set(columns) - set(schema(stem, features)))
        if unknown:
            raise DataValidationError(f"{path.name}: requested columns not in the file: {unknown}")
    try:
        frame = pd.read_parquet(path, columns=None if columns is None else list(columns))
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error})") from error
    if stem != JOINED:
        missing = [c for c in features.categorical if c in frame and frame[c].isna().any()]
        if missing:
            raise DataValidationError(f"{path.name}: missing categorical labels in {missing}")
    return frame


@dataclass(frozen=True)
class DevelopmentFeatures:
    """The joined raw columns of the development rows, without the target, each row's
    development partition, and the provenance record of the file they were read from."""

    joined: pd.DataFrame
    partition: pd.Series
    record: ProvenanceRecord


def read_development_features(
    paths: ProjectPaths, features: FeatureSpec, dataset: DatasetSpec
) -> DevelopmentFeatures:
    """The joined development rows without the target, which is never read from storage."""
    record = verify_processed(paths, JOINED, features, dataset)
    source = [c for c in features.joined_columns if c != features.target]
    frame = _load(paths, JOINED, features, [PARTITION_COLUMN, *source])
    partition = frame.pop(PARTITION_COLUMN)
    development = {p.name for p in dataset.development}
    outside = sorted(set(partition.unique()) - development)
    if outside:
        raise DataValidationError(f"{JOINED}: rows outside the development partitions: {outside}")
    logger.info("Development rows read (no target): %d rows, %d columns", *frame.shape)
    return DevelopmentFeatures(frame, partition, record)
