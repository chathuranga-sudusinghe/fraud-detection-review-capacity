"""Content-duplicate audit of the development partitions (Issue #86; descriptive only).

A *repeated input vector* is a set of two or more development rows, with different
``TransactionID`` values, whose values are identical on every compared column. Two definitions
are audited on the canonical model inputs of DR-008 (``features.model_inputs``):

* **primary:** all model inputs, the supplied categorical and numeric inputs and the derived
  time-of-day pair ``tod_sin`` and ``tod_cos``;
* **secondary:** the supplied inputs only, without the time-of-day pair, which also finds rows
  that repeat the same values at a different time of the relative day.

``TransactionID``, the raw ``TransactionDT``, the target, the relative day and the partition are
never compared. Values are compared exactly, in their canonical form: categorical inputs as
their text labels (missing as the declared missing label), numeric inputs as ``float64`` with no
rounding or tolerance; two missing numeric values are equal, and so are ``0.0`` and ``-0.0``.

For each definition the audit counts, per development partition and over all development rows,
the rows, the repeated-vector groups, the rows in such groups and the excess rows (rows beyond one
per group), and, across partitions, the vectors that occur in more than one development partition.

The rows come from the verified processed development file through ``read_development_features``,
which never reads the target; no raw file and no Test row is read. Nothing is removed, changed or
flagged: the result is a label-free set of counts. Why a vector repeats (an ingestion duplicate, a
retried transaction, card testing) cannot be established from the data and is not inferred.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from review_capacity.config import DatasetSpec, FeatureSpec
from review_capacity.data.processed import DevelopmentFeatures
from review_capacity.errors import DataValidationError
from review_capacity.features.model_inputs import build_canonical_frame
from review_capacity.logging_setup import ProgressLogger, log_step

logger = logging.getLogger(__name__)

PRIMARY = "primary"
SECONDARY = "secondary"
DEVELOPMENT = "development"

type VectorIds = npt.NDArray[np.int64]


@dataclass(frozen=True)
class RepeatCounts:
    """Repeated input vectors within one set of rows (a partition, or all development rows)."""

    rows: int
    groups: int  # vectors that occur on two or more rows
    rows_in_groups: int  # rows whose vector occurs on another row too
    excess_rows: int  # rows_in_groups - groups: rows beyond one per group


@dataclass(frozen=True)
class SharedVectors:
    """Vectors that occur in every partition of ``partitions``."""

    partitions: tuple[str, ...]
    vectors: int


@dataclass(frozen=True)
class DefinitionAudit:
    """The counts of one definition."""

    definition: str
    columns: tuple[str, ...]
    within: dict[str, RepeatCounts]  # per partition, then "development" for all rows
    vectors_in_several_partitions: int
    rows_with_such_vectors: int
    shared: tuple[SharedVectors, ...]  # per pair of partitions


@dataclass(frozen=True)
class DuplicateAudit:
    partitions: tuple[str, ...]
    definitions: tuple[DefinitionAudit, ...]


def vector_ids(frame: pd.DataFrame, task: str = "Compare columns") -> VectorIds:
    """An identifier per row, equal for two rows exactly when their values are equal in every
    column (missing equals missing; ``0.0`` equals ``-0.0``).

    Each column is factorised, and the running identifier is combined with the column's codes and
    factorised again, so that identifiers stay below the number of rows and never overflow.
    """
    rows = len(frame)
    if rows >= 2**31:
        raise DataValidationError(f"{rows} rows: too many to compare exactly")
    ids = np.zeros(rows, dtype=np.int64)
    progress = ProgressLogger(logger, task, len(frame.columns))
    for column in frame.columns:
        codes, uniques = pd.factorize(frame[column], use_na_sentinel=False)
        combined = ids * len(uniques) + codes.astype(np.int64)
        ids = pd.factorize(combined)[0].astype(np.int64)
        progress.advance()
    return ids


def repeat_counts(ids: VectorIds) -> RepeatCounts:
    """Repeated vectors among rows with identifiers ``ids``."""
    _, counts = np.unique(ids, return_counts=True)
    repeated = counts[counts >= 2]
    return RepeatCounts(
        rows=len(ids),
        groups=len(repeated),
        rows_in_groups=int(repeated.sum()),
        excess_rows=int(repeated.sum()) - len(repeated),
    )


def audit_definition(
    definition: str,
    frame: pd.DataFrame,
    partition: pd.Series,
    partitions: tuple[str, ...],
) -> DefinitionAudit:
    """The counts of one definition on the columns of ``frame``; ``partition`` names each row's
    development partition, in ``partitions`` order."""
    ids = vector_ids(frame, f"Compare columns ({definition})")
    names = partition.to_numpy()
    within = {name: repeat_counts(ids[names == name]) for name in partitions}
    within[DEVELOPMENT] = repeat_counts(ids)
    present = {name: np.unique(ids[names == name]) for name in partitions}
    occurrences = np.concatenate(list(present.values()))
    vectors, partition_counts = np.unique(occurrences, return_counts=True)
    several = vectors[partition_counts >= 2]
    shared = tuple(
        SharedVectors((a, b), len(np.intersect1d(present[a], present[b], assume_unique=True)))
        for a, b in itertools.combinations(partitions, 2)
    )
    return DefinitionAudit(
        definition=definition,
        columns=tuple(frame.columns),
        within=within,
        vectors_in_several_partitions=len(several),
        rows_with_such_vectors=int(np.isin(ids, several).sum()),
        shared=shared,
    )


def audit_duplicates(
    data: DevelopmentFeatures, features: FeatureSpec, dataset: DatasetSpec
) -> DuplicateAudit:
    """Both definitions on the development rows of ``data`` (read without the target)."""
    if features.target in data.joined.columns:
        raise DataValidationError("The duplicate audit must not receive the target column")
    aligned = data.partition.index.equals(data.joined.index)  # same rows, in the same order
    if len(data.partition) != len(data.joined) or not aligned:
        raise DataValidationError("Partition names do not match the development rows")
    partitions = tuple(p.name for p in dataset.development)
    unknown = sorted(set(data.partition.unique()) - set(partitions))
    if unknown:
        raise DataValidationError(f"Rows outside the development partitions: {unknown}")
    with log_step(logger, f"Duplicate audit of {len(data.joined)} development rows"):
        canonical = build_canonical_frame(data.joined, features)
        supplied = [c for c in features.model_inputs if c not in features.derived]
        definitions = (
            audit_definition(PRIMARY, canonical, data.partition, partitions),
            audit_definition(SECONDARY, canonical[supplied], data.partition, partitions),
        )
    return DuplicateAudit(partitions, definitions)


def audit_summary(audit: DuplicateAudit) -> dict[str, object]:
    """The audit as a JSON-ready mapping of counts (no row values, no labels)."""
    return {
        "partitions": list(audit.partitions),
        "definitions": [
            {
                "definition": d.definition,
                "columns": len(d.columns),
                "within": {
                    name: {
                        "rows": c.rows,
                        "groups": c.groups,
                        "rows_in_groups": c.rows_in_groups,
                        "excess_rows": c.excess_rows,
                    }
                    for name, c in d.within.items()
                },
                "vectors_in_several_partitions": d.vectors_in_several_partitions,
                "rows_with_such_vectors": d.rows_with_such_vectors,
                "shared": [
                    {"partitions": list(s.partitions), "vectors": s.vectors} for s in d.shared
                ],
            }
            for d in audit.definitions
        ],
    }
