"""The slice inputs of the Test false-negative analysis (Issue #205; protocol, False-Negative
Analysis).

Built for the Test frauds from sources that fix every boundary without a Test label: the V2 frauds
(the pooled levels), the Train amounts (the amount edges), the Train frauds (the recurring
pseudo-entities), the declared weeks, and the identity join of the raw files. The fine
pseudo-entity of each Test fraud is the key of ``evaluation.entities`` (relative day, ``D1``,
``card1``, ``addr1``, ``P_emaildomain``), also used by the entity-level bootstrap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from review_capacity.config import SECONDS_PER_DAY, FeatureSpec
from review_capacity.confirmatory.inputs import TestInputs
from review_capacity.development.inputs import PartitionRows
from review_capacity.errors import DataValidationError
from review_capacity.evaluation.declaration import EvaluationDeclaration
from review_capacity.evaluation.entities import pseudo_entity_keys
from review_capacity.evaluation.false_negatives import (
    POOLED_VARIABLES,
    Boundaries,
    SliceVariable,
    amount_codes,
    check_frozen,
    compute_boundaries,
    declared_levels,
    pooled_variable,
    week_codes,
)
from review_capacity.evaluation.resampling import Positions
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

AMOUNT = "TransactionAmt"


@dataclass(frozen=True)
class SliceInputs:
    """What the false-negative analysis and the entity-level bootstrap need for the Test frauds
    (in row order): the slice variables, their boundaries and the pseudo-entity index."""

    boundaries: Boundaries
    variables: list[SliceVariable]
    entity_index: Positions


def _text(frame: pd.DataFrame, name: str) -> npt.NDArray[np.str_]:
    return np.array(frame[name].to_numpy(dtype=object), dtype=np.str_)


def _fraud_keys(partition: PartitionRows, features: FeatureSpec) -> list[str]:
    """The fine pseudo-entity keys of the frauds of a development partition."""
    fraud = partition.labels == 1
    days = partition.times[fraud] // SECONDS_PER_DAY
    return pseudo_entity_keys(partition.features[fraud], days, features.missing_label)


def development_boundaries(
    declaration: EvaluationDeclaration, train: PartitionRows, v2: PartitionRows
) -> Boundaries:
    """The slice boundaries recomputed from the verified development partitions and checked
    against the frozen ones. It reads no Test row or label, so the evaluation phase calls it
    before the consumption marker (Issue #212)."""
    v2_frauds = v2.features[v2.labels == 1]
    boundaries = compute_boundaries(
        declaration,
        train.features[AMOUNT].to_numpy(dtype=np.float64),
        {name: _text(v2_frauds, name) for name in POOLED_VARIABLES},
    )
    check_frozen(boundaries, declaration)
    return boundaries


def build_slice_inputs(
    declaration: EvaluationDeclaration,
    features: FeatureSpec,
    test: TestInputs,
    labels: npt.NDArray[np.int8],
    train: PartitionRows,
    boundaries: Boundaries,
) -> SliceInputs:
    """The slice inputs of the Test frauds of ``test`` (``labels`` are its labels, read in the
    evaluation phase). ``train`` is the verified Train partition (the recurring entities) and
    ``boundaries`` the verified ones of ``development_boundaries``."""
    rule = declaration.false_negative
    if labels.shape != (test.rows,) or test.has_identity is None:
        raise DataValidationError("The slice inputs need one label and one join flag per Test row")
    fraud = labels == 1
    if not fraud.any():
        raise DataValidationError("The Test partition has no fraud: the slices are undefined")
    with log_step(logger, f"Slice inputs of {int(fraud.sum())} Test frauds"):
        edges = boundaries.amount_edges
        frame = test.features[fraud].reset_index(drop=True)
        days = test.days[fraud]
        levels = declared_levels(declaration)
        keys = pseudo_entity_keys(frame, days, features.missing_label)
        recurring = set(_fraud_keys(train, features))
        _, entity_index = np.unique(np.array(keys, dtype=np.str_), return_inverse=True)
        variables = [
            SliceVariable("identity", levels["identity"], test.has_identity[fraud].astype(np.intp)),
            *(pooled_variable(name, _text(frame, name), boundaries) for name in POOLED_VARIABLES),
            SliceVariable(
                "amount_band",
                levels["amount_band"],
                amount_codes(frame[AMOUNT].to_numpy(dtype=np.float64), edges),
            ),
            SliceVariable("week", levels["week"], week_codes(days, rule.test_weeks)),
            SliceVariable(
                "entity",
                levels["entity"],
                np.fromiter((key in recurring for key in keys), dtype=np.intp, count=len(keys)),
            ),
        ]
    return SliceInputs(boundaries, variables, entity_index.astype(np.intp))
