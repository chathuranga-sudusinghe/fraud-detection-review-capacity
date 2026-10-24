"""Ranking and capacity evaluation of the scored V1 and V2 partitions (DR-002; DR-015 §3; Issue
#134).

For a scored partition (``development.scoring``) this writes the ``evaluation`` records of the
selected model: R3 (daily top-K review) at each of the four capacities, with the fixed daily
budgets K_c of V2 (DR-002 §3), and the whole-ranking measures. The measures are computed by
``evaluation.ranking`` (Issue #122); this module only joins their inputs and records them.

The scores are read back from the local score file, and the run stops unless the ordered
``TransactionID`` sequence of the file is that of the partition's rows, whose labels and
transaction amounts are used: a score is never paired with another transaction's label or amount.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from review_capacity.config import SECONDS_PER_DAY
from review_capacity.development.inputs import PartitionRows
from review_capacity.development.scoring import (
    MODEL_ROLE,
    ScoredPartition,
    ScoringContext,
    read_scores,
)
from review_capacity.errors import DataValidationError
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import (
    Subject,
    Truth,
    capacity_budgets,
    r3_outcome,
    r3_record,
    ranking_outcome,
    ranking_record,
)
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

AMOUNT = "TransactionAmt"


def truth_of(rows: PartitionRows) -> Truth:
    """The labels, relative days and transaction amounts of a partition's rows."""
    if AMOUNT not in rows.features:
        raise DataValidationError(f"{AMOUNT} is not among the partition's model inputs")
    amounts = rows.features[AMOUNT].to_numpy(dtype=np.float64)
    return Truth.build(rows.times // SECONDS_PER_DAY, rows.labels, amounts)


def evaluate_partition(context: ScoringContext, scored: ScoredPartition) -> dict[str, Any]:
    """Write the R3 records at the four capacities and the ranking record of the selected model
    of ``scored``; a refit (``scored.role``) is evaluated under R3 only (the records' contract:
    refits are evaluated under R3 on V2). Returns the path of each record by name."""
    run, name, label = context.run, scored.configuration, scored.label
    selected = scored.role == MODEL_ROLE
    seed = context.seed if scored.seed is None else scored.seed
    with log_step(logger, f"Evaluating the {scored.tag} {name} on {label}"):
        scores = read_scores(run, scored.file, scored.rows)
        truth = truth_of(scored.rows)
        queue = DailyQueue.build(scored.rows.ids, truth.days)
        subject = Subject(name, scored.role, seed, scored.record, label)
        written: dict[str, Any] = {}
        for capacity in capacity_budgets(run.partitions.v2):
            outcome = r3_outcome(queue, truth, scores, capacity)
            record = r3_record(context.identity, subject, truth, capacity, outcome)
            part = f"R3-c{round(capacity.value * 100):02d}"
            written[part] = run.write_record(record, name, scored.tag, label, part)
        if selected:
            ranking = ranking_record(
                context.identity, subject, truth, ranking_outcome(truth, scores)
            )
            written["ranking"] = run.write_record(ranking, name, scored.tag, label, "ranking")
    return written
