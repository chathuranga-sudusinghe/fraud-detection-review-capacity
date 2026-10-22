"""The verified Train and V1 inputs of the development search, and the Train-only class weights
(DR-015 §1 and §3; DR-008 §5; Issue #133).

Both partitions are read from the processed files only, through ``data.processed`` (which checks
each file against its provenance record and the DR-001 anchors before returning any data):

* a partition file supplies the identifier, the relative day, the target and the canonical model
  inputs;
* the joined development file supplies ``TransactionDT``, which the partition files do not hold.
  ``TransactionDT`` is never a model input; it serves the V1 objective's daily grouping and the
  check of row order (CatBoost's ``has_time``).

The two sources must agree row for row: every partition identifier is in the joined file, and each
row's relative day equals its ``TransactionDT`` divided by 86,400 seconds, rounded down. The rows
must be in time order. The Test period is not a development partition and is never read.

Class weights come from Train labels only (``models.base.class_weights``). The V1 partition is
evaluated unweighted: its sample weights are all 1 and play no part in the objective.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from review_capacity.config import SECONDS_PER_DAY, DatasetSpec, FeatureSpec
from review_capacity.data.processed import (
    DAY_COLUMN,
    JOINED,
    ProvenanceRecord,
    read_processed,
    read_record,
)
from review_capacity.errors import DataValidationError
from review_capacity.logging_setup import log_step
from review_capacity.models.base import (
    ClassWeights,
    Keys,
    Labels,
    ModelData,
    class_weights,
    sample_weights,
)
from review_capacity.models.registry import Weighting
from review_capacity.paths import ProjectPaths
from review_capacity.training.timing import rows_identity

logger = logging.getLogger(__name__)

TRAIN = "train"
V1 = "v1"


@dataclass(frozen=True)
class PartitionRows:
    """One verified development partition: model inputs, labels, identifiers and times, in the
    file's time order, with the provenance record of the file they were read from."""

    name: str
    features: pd.DataFrame
    labels: Labels
    ids: Keys
    times: Keys
    provenance: ProvenanceRecord

    @property
    def rows(self) -> int:
        return len(self.labels)

    @property
    def frauds(self) -> int:
        return int(self.labels.sum())

    @property
    def ordered_ids_sha256(self) -> str:
        """SHA-256 of the identifiers in row order (the order the model sees)."""
        return rows_identity(self.ids)

    def model_data(self, weights: ClassWeights | None) -> ModelData:
        """The partition as ``ModelData``; ``weights=None`` gives every row weight 1."""
        row_weights = (
            np.ones(self.rows, dtype=np.float64)
            if weights is None
            else sample_weights(self.labels, weights)
        )
        return ModelData(self.features, self.labels, row_weights, self.times, self.ids)


@dataclass(frozen=True)
class SearchInputs:
    """The partitions the search reads: Train (fitting) and V1 (early stopping and selection)."""

    train: PartitionRows
    v1: PartitionRows

    def weights(self, weighting: Weighting) -> ClassWeights:
        """The class weights of ``weighting``, from the Train labels only."""
        return class_weights(self.train.labels, weighting)

    def train_data(self, weighting: Weighting) -> ModelData:
        """Train with the sample weights of ``weighting``."""
        return self.train.model_data(self.weights(weighting))

    def v1_data(self) -> ModelData:
        """V1, unweighted (the V1 objective ignores sample weights)."""
        return self.v1.model_data(None)


def _times_by_id(joined: pd.DataFrame, features: FeatureSpec) -> pd.Series:
    times = pd.Series(
        joined[features.time].to_numpy(dtype=np.int64),
        index=joined[features.key].to_numpy(dtype=np.int64),
    )
    if not times.index.is_unique:
        raise DataValidationError(f"{JOINED}: {features.key} is not unique")
    return times


def _partition(
    stem: str,
    frame: pd.DataFrame,
    times_of: pd.Series,
    features: FeatureSpec,
    provenance: ProvenanceRecord,
) -> PartitionRows:
    ids = frame[features.key].to_numpy(dtype=np.int64)
    missing = int((~pd.Index(ids).isin(times_of.index)).sum())
    if missing:
        raise DataValidationError(f"{stem}: {missing} identifiers are not in the joined file")
    times = times_of.loc[ids].to_numpy(dtype=np.int64)
    if not np.array_equal(frame[DAY_COLUMN].to_numpy(dtype=np.int64), times // SECONDS_PER_DAY):
        raise DataValidationError(
            f"{stem}: relative days differ from TransactionDT divided by {SECONDS_PER_DAY}"
        )
    labels = frame[features.target].to_numpy(dtype=np.int8)
    inputs = frame[list(features.model_inputs)].reset_index(drop=True)
    rows = PartitionRows(stem, inputs, labels, ids, times, provenance)
    if not rows.model_data(None).in_time_order():
        raise DataValidationError(
            f"{stem}: rows are not in time order (TransactionDT, then {features.key})"
        )
    return rows


def read_search_inputs(
    paths: ProjectPaths, features: FeatureSpec, dataset: DatasetSpec
) -> SearchInputs:
    """Train and V1, verified and checked against the joined file's ``TransactionDT``."""
    with log_step(logger, "Reading the verified Train and V1 inputs"):
        joined = read_processed(
            paths, JOINED, features, dataset, columns=[features.key, features.time]
        )
        times_of = _times_by_id(joined, features)
        parts: dict[str, PartitionRows] = {}
        for stem in (TRAIN, V1):
            frame = read_processed(paths, stem, features, dataset)
            provenance = read_record(paths.provenance_file(stem))
            parts[stem] = _partition(stem, frame, times_of, features, provenance)
            logger.info(
                "%s: %d rows, %d frauds, ordered identifiers %s",
                stem,
                parts[stem].rows,
                parts[stem].frauds,
                parts[stem].ordered_ids_sha256[:12],
            )
    return SearchInputs(parts[TRAIN], parts[V1])
