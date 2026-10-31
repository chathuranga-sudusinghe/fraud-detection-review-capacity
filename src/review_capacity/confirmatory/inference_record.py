"""The ``test_inference`` record of the Test evaluation (Issue #207; protocol, Statistical Plan;
DR-003, DR-005, DR-006, DR-010).

``inference_inputs`` draws everything the confirmatory inference needs from the saved scores, the
Test labels and the frozen policies, for the eight configurations at the primary capacity:

- the capture matrix of R3 (one row per configuration, one column per Test fraud) and the observed
  Recall@2 %;
- the replicate sets of the declared streams, each drawn once and shared by all configurations so
  that the contrasts are paired: ``pigeonhole_h1_h2`` (H1, H2 and S1), ``naive_s1`` (S1),
  ``day_block_s2`` (S2) and ``day_block_h3`` (block lengths 3, 1 and 7);
- the pooled alert rate of R2 at its frozen primary-capacity threshold and its day-block
  replicates, for H3.

``build_inference_record`` runs H1, H2 and H3 (the functions of Issue #191), the Holm decision, S1,
S2, the SESOI views, the exact McNemar tests and the effect units, and assembles the record.
Nothing here reads a file or a label: the labels arrive as the ``Truth`` of the partition.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from review_capacity.confirmatory.frozen_policies import FrozenPolicies
from review_capacity.confirmatory.outcome_records import OutcomeSetup, primary_capacity
from review_capacity.confirmatory.records import to_json_value
from review_capacity.errors import DataValidationError
from review_capacity.evaluation.contrasts import estimands
from review_capacity.evaluation.declaration import EvaluationDeclaration
from review_capacity.evaluation.hypotheses import analyse_h1, analyse_h2, analyse_h3, decide
from review_capacity.evaluation.inference import effect_units
from review_capacity.evaluation.mcnemar import cell_mcnemar
from review_capacity.evaluation.outcomes import Partition
from review_capacity.evaluation.resampling import (
    Floats,
    Replicates,
    day_block_indices,
    day_ratios,
    h3_day_block_sets,
    naive_ratios,
    pigeonhole_ratios,
    stream_generator,
)
from review_capacity.evaluation.sensitivity import analyse_s1, analyse_s2, sesoi_views
from review_capacity.evaluation.thresholds import alert_mask
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

STREAMS = ("pigeonhole_h1_h2", "naive_s1", "day_block_s2", "day_block_h3")


@dataclass(frozen=True)
class InferenceInputs:
    """The observed values and replicate sets of the confirmatory inference."""

    captures: npt.NDArray[np.bool_]  # (configurations x Test frauds)
    recalls: Floats
    rates: Floats  # the pooled R2 alert rate at the primary capacity, per configuration
    pigeonhole: Replicates
    naive: Replicates
    day_block: Replicates
    h3_sets: Mapping[int, Replicates]
    frauds: int
    days: int
    k: int
    oracle_ceiling: float


def inference_inputs(
    declaration: EvaluationDeclaration,
    partition: Partition,
    scores: Mapping[str, npt.NDArray[Any]],
    policies: FrozenPolicies,
) -> InferenceInputs:
    """The inputs of H1, H2, H3, S1, S2 and the McNemar tests (see the module description)."""
    truth, resampling = partition.truth, declaration.resampling
    order = declaration.configurations.order
    primary = primary_capacity(declaration)
    fraud = truth.labels == 1
    days = len(truth.day_values)
    frauds = int(fraud.sum())
    if frauds < 1:
        raise DataValidationError("The Test partition has no fraud: the inference is undefined")
    with log_step(
        logger, f"Inference inputs: {len(order)} configurations, B = {resampling.replicates}"
    ):
        captures = np.vstack(
            [partition.queue.select(scores[name], primary.k).mask[fraud] for name in order]
        )
        numerators = captures.astype(np.float64)
        pigeonhole = pigeonhole_ratios(
            stream_generator(declaration, "pigeonhole_h1_h2"),
            numerators,
            partition.fraud_day_index,
            days,
            partition.entity_index,
            replicates=resampling.replicates,
        )
        naive = naive_ratios(
            stream_generator(declaration, "naive_s1"), numerators, replicates=resampling.replicates
        )
        fraud_days = partition.fraud_day_index
        captured_by_day = np.vstack(
            [np.bincount(fraud_days[row], minlength=days) for row in captures]
        ).astype(np.float64)
        day_block = day_ratios(
            captured_by_day,
            truth.frauds.astype(np.float64),
            day_block_indices(
                stream_generator(declaration, "day_block_s2"),
                days,
                resampling.block_length,
                replicates=resampling.replicates,
            ),
        )
        alerts = [
            alert_mask(scores[name], policies.thresholds[name].r2[primary.value]) for name in order
        ]
        alerts_by_day = np.vstack(
            [np.bincount(truth.day_index[mask], minlength=days) for mask in alerts]
        ).astype(np.float64)
        volumes = truth.transactions.astype(np.float64)
        h3_sets = {
            blocks.block_length: day_ratios(alerts_by_day, volumes, blocks.indices)
            for blocks in h3_day_block_sets(declaration, days)
        }
        reviewable = np.minimum(primary.k, truth.frauds)
    return InferenceInputs(
        captures=captures,
        recalls=captures.mean(axis=1),
        rates=alerts_by_day.sum(axis=1) / volumes.sum(),
        pigeonhole=pigeonhole,
        naive=naive,
        day_block=day_block,
        h3_sets=h3_sets,
        frauds=frauds,
        days=days,
        k=primary.k,
        oracle_ceiling=float(reviewable.sum() / frauds),
    )


def _largest_undefined(value: object) -> int:
    """The largest ``undefined`` count among the bootstrap intervals inside ``value``."""
    if isinstance(value, Mapping):
        here = (
            int(value["undefined"])
            if {"lower", "upper", "level", "undefined", "replicates"} <= value.keys()
            and value["replicates"] > 0
            else 0
        )
        return max([here, *(_largest_undefined(item) for item in value.values())])
    if isinstance(value, list):
        return max([0, *(_largest_undefined(item) for item in value)])
    return 0


def build_inference_record(
    setup: OutcomeSetup, inputs: InferenceInputs, h3_configuration: str
) -> dict[str, Any]:
    """The ``test_inference`` record of the confirmatory inference on ``inputs``."""
    declaration = setup.declaration
    table = estimands(declaration)
    h1 = analyse_h1(declaration, table, inputs.recalls, inputs.pigeonhole)
    h2 = analyse_h2(declaration, table, inputs.recalls, inputs.pigeonhole)
    h3 = analyse_h3(declaration, h3_configuration, inputs.rates, inputs.h3_sets)
    primary = decide(declaration, h1, h2, h3)
    s1 = analyse_s1(declaration, table, inputs.recalls, inputs.pigeonhole, inputs.naive, primary)
    s2 = analyse_s2(declaration, table, inputs.recalls, inputs.day_block, primary)
    h3_json = to_json_value(h3)
    resampling = declaration.resampling
    reasons = [
        f"{q.name}: the corrected variance is not positive, so the Wald interval is not estimable"
        for q in s1.quantities
        if q.se is None
    ]
    record: dict[str, Any] = {
        **setup.envelope("test_inference"),
        "resampling": {
            "replicates": resampling.replicates,
            "root_seed": resampling.root_seed,
            "block_length": resampling.block_length,
            "block_length_sensitivity": list(resampling.block_length_sensitivity),
        },
        "undefined_replicates": {
            "pigeonhole_h1_h2": inputs.pigeonhole.undefined_count,
            "naive_s1": inputs.naive.undefined_count,
            "day_block_s2": inputs.day_block.undefined_count,
            "day_block_h3": _largest_undefined(h3_json),
        },
        "h1": to_json_value(h1),
        "h2": to_json_value(h2),
        "h3": h3_json,
        "holm": to_json_value(primary.holm),
        "decisions": {
            "h1_rejected": primary.h1_rejected,
            "h2_equivalent": primary.h2_equivalent,
            "h3_preserved": primary.h3_preserved,
            "h1_category": primary.h1_category.value,
        },
        "s1": {
            "quantities": [{**to_json_value(q), "se": q.se} for q in s1.quantities],
            "p_h1": s1.p_h1,
            "interval_h1": to_json_value(s1.interval_h1),
            "position_h1": to_json_value(s1.position_h1),
            "p_h2": s1.p_h2,
            "intervals_h2": to_json_value(s1.intervals_h2),
            "comparison": to_json_value(s1.comparison),
            "reasons": reasons,
        },
        "s2": {
            "p_h1": to_json_value(s2.h1.p),
            "p_h2": to_json_value(s2.h2.p),
            "h1_interval": to_json_value(s2.h1.interval),
            "h1_position": to_json_value(s2.h1.position),
            "comparison": to_json_value(s2.comparison),
        },
        "sesoi_views": to_json_value(sesoi_views(declaration, primary)),
        "mcnemar": to_json_value(cell_mcnemar(declaration, inputs.captures)),
        "effect_basis": {
            "frauds": inputs.frauds,
            "days": inputs.days,
            "k": inputs.k,
            "oracle_ceiling": inputs.oracle_ceiling,
        },
    }
    record["effect_units"] = _effect_units(record)
    return to_json_value(record)  # type: ignore[no-any-return]


def _effect_units(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The units of every estimand of the record on its stated basis."""
    h1, h2, basis = record["h1"], record["h2"], record["effect_basis"]
    estimates: dict[str, float] = {
        "family": h1["estimate"],
        "interaction": h1["interaction"]["estimate"],
    }
    for field in ("by_weighting", "within_family", "cells"):
        estimates |= {item["name"]: item["estimate"] for item in h1[field]}
    estimates |= {f"weighting:{m['model']}": m["estimate"] for m in h2["models"]}
    return [
        {
            "estimand": name,
            **to_json_value(
                effect_units(
                    estimate,
                    frauds=basis["frauds"],
                    days=basis["days"],
                    k=basis["k"],
                    oracle_ceiling=basis["oracle_ceiling"],
                )
            ),
        }
        for name, estimate in estimates.items()
    ]
