"""R4 and the RQ3 cost analysis on the saved V2 scores (protocol Decision Policies and Cost
Model; DR-007; Issue #158).

For one configuration this reads the saved V2 scores through ``read_scores`` and, for each
calibrator of ``development.calibration`` (Platt and isotonic, primary and label-latency),

* builds R3's daily queues at the four capacities with the fixed budgets K_c of V2 (on all of V2
  for the primary calibrators, on the held-out days 148–154 for the label-latency ones);
* writes an ``evaluation`` record of R4 for every capacity and every ratio of the grid
  {2, 5, 10, 20, 50, 100, 200, 500, 1000}, reviewing the queue members whose calibrated
  probability is at least 1 / rho (``evaluation.cost``);
* writes one ``cost_analysis`` record per capacity: the saturation and break-even ratios, the
  grid table, the declined-set diagnostic, the amount proxies and the C_FP sensitivity.

The primary calibrators are fitted on V2 itself, so these results are in-sample pre-Test
expectations; the label-latency results are out of sample. R4 is not evaluated on V1 and the
confidence intervals of DR-007 §7 are part of the Test evaluation, not of this module.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from review_capacity.development.calibration import FittedCalibrator
from review_capacity.development.evaluation import truth_of
from review_capacity.development.records import DevelopmentPartitions
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.scoring import (
    MODEL_ROLE,
    HoldsRunFiles,
    ScoredPartition,
    check_selected,
    read_scores,
)
from review_capacity.errors import ModelError
from review_capacity.evaluation.cost import RATIO_GRID, analyse, r4_outcome, r4_record
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import Capacity, Subject, Truth, capacity_budgets
from review_capacity.logging_setup import ProgressLogger, log_step

logger = logging.getLogger(__name__)


def _share(capacity: Capacity) -> str:
    return f"c{round(capacity.value * 100):02d}"


def evaluate_r4(
    output: RunDirectory,
    source: HoldsRunFiles,
    partitions: DevelopmentPartitions,
    identity: Mapping[str, Any],
    seed: int,
    v2: ScoredPartition,
    calibrators: Sequence[FittedCalibrator],
) -> dict[str, Path]:
    """Write the R4 ``evaluation`` records and the ``cost_analysis`` records of ``v2``'s
    configuration for every calibrator in ``calibrators``; returns the path of each by name."""
    name = v2.configuration
    if v2.label != "V2":
        raise ModelError("R4 is evaluated on the V2 scores")
    check_selected(v2, seed)
    scores = read_scores(source, v2.file, v2.rows)
    full = truth_of(v2.rows)
    _, held_out = partitions.latency_ranges()
    written: dict[str, Path] = {}
    capacities = capacity_budgets(partitions.v2)
    total = len(calibrators) * len(capacities) * (len(RATIO_GRID) + 1)
    progress = ProgressLogger(logger, f"R4 evaluations of {name}", total)
    with log_step(logger, f"Evaluating R4 and the cost analysis of the selected {name}"):
        for fitted in calibrators:
            rows = np.ones(len(scores), dtype=bool)
            if fitted.period == "latency":
                rows = (full.days >= held_out[0]) & (full.days <= held_out[1])
            truth = (
                full
                if fitted.period == "primary"
                else Truth.build(full.days[rows], full.labels[rows], full.amounts[rows])
            )
            queue = DailyQueue.build(v2.rows.ids[rows], truth.days)
            raw = scores[rows]
            probabilities = fitted.calibrator.predict(raw)
            subject = Subject(name, MODEL_ROLE, seed, v2.record, "V2", fitted.period)
            tag = f"{fitted.method}-{fitted.period}"
            for capacity in capacities:
                selection = queue.select(raw, capacity.k)
                for ratio in RATIO_GRID:
                    outcome = r4_outcome(truth, selection, probabilities, capacity, float(ratio))
                    record = r4_record(
                        identity,
                        subject,
                        truth,
                        capacity,
                        float(ratio),
                        fitted.method,
                        fitted.record,
                        outcome,
                    )
                    part = f"R4-{tag}-{_share(capacity)}-rho{ratio}"
                    written[part] = output.write_record(record, name, MODEL_ROLE, "V2", part)
                    progress.advance()
                analysis = analyse(truth, selection, probabilities, capacity)
                record = {
                    "record": "cost_analysis",
                    "schema_version": 1,
                    "identity": dict(identity),
                    "configuration": name,
                    "calibrator": fitted.method,
                    "period": fitted.period,
                    "capacity": capacity.value,
                    "k": capacity.k,
                    "days": {"first": truth.first_day, "last": truth.last_day},
                    "score": dict(v2.record),
                    "calibration": dict(fitted.record),
                    **analysis,
                }
                part = f"{tag}-{_share(capacity)}"
                written[f"cost-{part}"] = output.write_record(record, name, part)
                progress.advance()
    return written
