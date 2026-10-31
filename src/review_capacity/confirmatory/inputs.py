"""The Test features of the single evaluation, read without the labels (Issue #203).

The raw files are read once for the features of all rows (the target is never read with them),
checked against the recorded checksums and against the partition anchors of DR-001 (rows,
``TransactionDT`` bounds and identifier-hash prefixes of every partition) and, for the Test
partition, against the full ``transaction_id_sha256`` of the dataset manifest, which is the frozen
anchor. The Test rows are then turned into the canonical model inputs by the same function that
built the development partitions (``features.model_inputs.build_canonical_frame``).

Nothing here reads the target column: reading labels is the evaluation phase, through
``data.loading.load_labels`` only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from review_capacity.config import DatasetSpec, FeatureSpec
from review_capacity.data.loading import check_file_checksums, load_raw
from review_capacity.data.validation import check_partition_anchors, summarise_partitions
from review_capacity.errors import DataValidationError
from review_capacity.features.model_inputs import build_canonical_frame
from review_capacity.logging_setup import log_step
from review_capacity.models.base import Keys
from review_capacity.training.timing import rows_identity

logger = logging.getLogger(__name__)

TEST = "test"


@dataclass(frozen=True)
class TestInputs:
    """The Test partition's model inputs, identifiers and times, in the file's time order."""

    __test__ = False  # not a pytest class

    features: pd.DataFrame
    ids: Keys
    times: Keys
    days: Keys
    id_set_sha256: str

    @property
    def rows(self) -> int:
        return len(self.ids)

    @property
    def ordered_ids_sha256(self) -> str:
        """SHA-256 of the identifiers in row order (the order the model sees)."""
        return rows_identity(self.ids)


def read_test_inputs(features: FeatureSpec, dataset: DatasetSpec, anchor_sha256: str) -> TestInputs:
    """The Test partition's inputs, once the raw files and every partition anchor are verified and
    the Test identifier hash equals the manifest's ``anchor_sha256``."""
    with log_step(logger, "Reading the Test features (no target)"):
        check_file_checksums(dataset)
        raw = load_raw(features, dataset)
        summaries = summarise_partitions(raw, features, dataset)
        check_partition_anchors(summaries, dataset)
        test = next(s for s in summaries if s.name == TEST)
        if test.transaction_id_sha256 != anchor_sha256:
            raise DataValidationError(
                "The Test identifiers differ from the manifest's frozen anchor "
                f"({test.transaction_id_sha256} against {anchor_sha256})"
            )
        index = raw.joined.index[(raw.partition == TEST).to_numpy()]
        rows = raw.joined.loc[index]
        frame = build_canonical_frame(rows, features).reset_index(drop=True)
        ids = rows[features.key].to_numpy(dtype=np.int64)
        times = rows[features.time].to_numpy(dtype=np.int64)
        days = raw.relative_day.loc[index].to_numpy(dtype=np.int64)
    if not np.array_equal(np.lexsort((ids, times)), np.arange(len(ids))):
        raise DataValidationError(
            f"The Test rows are not in time order (TransactionDT, then {features.key})"
        )
    logger.info("Test inputs: %d rows, ordered identifiers %s", len(ids), rows_identity(ids)[:12])
    return TestInputs(frame, ids, times, days, test.transaction_id_sha256)
