"""The rolling-origin supplement for RQ2 (protocol, Experimental Split Structure; DR-001; DR-003;
DR-010; Issue #159).

An exploratory, development-only look at how variable the transfer of an R2 threshold is across
origins, for the primary configuration only (the H3 configuration's model and weighting). The
window design is ``config/dataset.yaml``'s ``rolling_origin``: at origin ``s`` the configuration
is refitted on relative days ``1`` to ``s - gap - 1``, scores the threshold window (days ``s`` to
``s + 27``) and the transfer window (the next 28 days), and the R2 threshold set on the threshold
window's scores (no labels) is applied to the transfer window against the origin's fixed budget
(``evaluation.rolling``).

The refit uses the selected trial's hyperparameters and its **fixed length** (the retained rounds
or epochs), the main seed, no search and no early stopping. Everything the fit learns from data
is fitted on the origin's training rows only: the class weights from their labels and, for a
neural model, the shared preprocessing (DR-012) refitted on them with the declared seed; the
fitted preprocessing of the main run is not reused. The batch plan of the selected trial is
reused: it fixes the micro-batches of the same effective batch.

The supplement is a retrospective sensitivity analysis conditional on the final development-
selected specification: the configuration, its weighting and its hyperparameters were chosen on
V1 and V2, which lie after the early origins. It is descriptive and never enters a selection.
Rows are read from the verified Train, V1 and V2 partitions; the Test period is never read.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from review_capacity.config import SECONDS_PER_DAY, FeatureSpec, RollingOriginSpec
from review_capacity.development.inputs import PartitionRows, SearchInputs
from review_capacity.development.neural_run import NeuralRun
from review_capacity.development.refits import RefitJob, RefitSpec, fit_refit
from review_capacity.development.scoring import SCORE_COLUMNS, ScoredModel, ScoringContext
from review_capacity.development.scoring import load_selected as load_model
from review_capacity.development.subjects import MODEL_FILES, NEURAL, kind_of
from review_capacity.development.trial import (
    MODELS_DIR,
    NEURAL_MODELS,
    SearchContext,
    TrialOutcome,
    run_file,
)
from review_capacity.errors import DataValidationError, SpecificationError
from review_capacity.evaluation.rolling import OriginOutcome, origin_outcome
from review_capacity.features.neural_preprocessing import fit_neural_preprocessing
from review_capacity.files import write_atomically
from review_capacity.logging_setup import log_step
from review_capacity.models.base import Scores
from review_capacity.models.neural_training import NeuralData, NeuralModel
from review_capacity.models.registry import Configuration, Subject

logger = logging.getLogger(__name__)

PRIMARY_SHARE = Fraction(2, 100)  # the primary capacity of DR-002 §1
KIND = "rolling_origin"
SCORES_DIR = "scores"
PREPROCESSING_DIR = "rolling-preprocessing"


def rows_between(
    parts: Sequence[PartitionRows], first: int, last: int, name: str = "rolling"
) -> PartitionRows:
    """The rows of ``parts`` (time-ordered partitions, one after the other) whose relative day
    lies in ``first..last``, in time order."""
    kept = []
    for part in parts:
        days = part.times // SECONDS_PER_DAY
        mask = (days >= first) & (days <= last)
        if mask.any():
            kept.append((part, mask))
    if not kept:
        raise DataValidationError(f"No rows lie in the relative days {first}..{last}")
    frame = pd.concat([part.features[mask] for part, mask in kept], ignore_index=True)
    return PartitionRows(
        name,
        frame,
        np.concatenate([part.labels[mask] for part, mask in kept]),
        np.concatenate([part.ids[mask] for part, mask in kept]),
        np.concatenate([part.times[mask] for part, mask in kept]),
        kept[0][0].provenance,
    )


def refitted_neural(
    base: NeuralRun,
    features: FeatureSpec,
    seed: int,
    training: SearchInputs,
    v2: PartitionRows,
    directory: Path,
) -> NeuralRun:
    """The neural inputs of one origin: the shared preprocessing (DR-012) fitted again on the
    origin's training rows with the declared seed, and the network inputs and class weights those
    rows give. The batch plans are those of the main run."""
    rows = training.train
    with log_step(logger, f"Neural preprocessing on {rows.rows} rows"):
        artifact = fit_neural_preprocessing(rows.features, rows.ids, rows.times, features, seed)
        sha256 = artifact.save(directory)
        inputs = artifact.to_neural_inputs(rows.features)
        v1_inputs = artifact.to_neural_inputs(training.v1.features)
        v2_inputs = artifact.to_neural_inputs(v2.features)
        data = training.v1_data()
    return replace(
        base,
        artifact=artifact,
        directory=directory,
        artifact_sha256=sha256,
        train={
            weighting: NeuralData(inputs, rows.labels, training.train_data(weighting).weights)
            for weighting in base.train
        },
        v1=NeuralData(v1_inputs, data.labels, data.weights, data.ids, data.times),
        v1_inputs=v1_inputs,
        v2_inputs=v2_inputs,
    )


def predict(model: ScoredModel, rows: PartitionRows, neural: NeuralRun | None) -> Scores:
    """The raw scores of ``rows``."""
    if isinstance(model, NeuralModel):
        if neural is None:
            raise SpecificationError("A neural model needs the neural inputs of its origin")
        return model.predict(neural.artifact.to_neural_inputs(rows.features))
    return model.predict(rows.features)


@dataclass(frozen=True)
class OriginGeometry:
    """The relative days of one origin."""

    origin: int
    training: tuple[int, int]
    window: tuple[int, int]
    transfer: tuple[int, int]

    @classmethod
    def of(cls, spec: RollingOriginSpec, origin: int, first_day: int) -> OriginGeometry:
        return cls(
            origin,
            (first_day, spec.training_last_day(origin)),
            (origin, spec.window_last_day(origin)),
            (spec.transfer_first_day(origin), spec.transfer_last_day(origin)),
        )


def origin_record(
    search: SearchContext,
    subject: Subject,
    selected: TrialOutcome,
    geometry: OriginGeometry,
    training: PartitionRows,
    neural: NeuralRun | None,
    outcome: OriginOutcome,
    fit: tuple[float, Mapping[str, str]],
    scores: Mapping[str, str],
    device: str,
) -> dict[str, object]:
    """The ``rolling_origin`` record of one origin."""
    seconds, model = fit
    window, transfer = geometry.window, geometry.transfer
    return {
        "record": KIND,
        "schema_version": 1,
        "identity": dict(search.identity),
        "configuration": subject.name,
        "seed": search.identity["seeds"]["main"],
        "origin": geometry.origin,
        "training": {
            "first_day": geometry.training[0],
            "last_day": geometry.training[1],
            "rows": training.rows,
            "frauds": training.frauds,
            "preprocessing_sha256": None if neural is None else neural.artifact_sha256,
        },
        "fit": {
            "selected_trial": selected.trial,
            "length": selected.fitted_iterations,
            "device": device,
            "seconds": seconds,
            "model": dict(model),
            "source_fit": dict(selected.fit),
        },
        "scores": dict(scores),
        "window": {
            "first_day": window[0],
            "last_day": window[1],
            "transactions": outcome.window_transactions,
            "alerts": outcome.window_alerts,
            "rate": outcome.window_rate,
        },
        "threshold": {
            "capacity": float(outcome.share),
            "rank": outcome.rank,
            "value": outcome.threshold,
            "k": outcome.k,
        },
        "transfer": {
            "first_day": transfer[0],
            "last_day": transfer[1],
            "transactions": outcome.transfer_transactions,
            "alerts": outcome.transfer_alerts,
            "rate": outcome.transfer_rate,
            "relative_deviation": outcome.relative_deviation,
            "total_overflow": outcome.overflow,
            "total_unused": outcome.unused,
            "days_above_budget": outcome.days_above,
            "days_below_budget": outcome.days_below,
            "zero_alert_days": outcome.zero_alert_days,
            "inside_band": outcome.inside_band,
            "rate_component": outcome.rate_component,
            "volume_component": outcome.volume_component,
            "total_component": outcome.total_component,
            "daily": outcome.daily,
        },
    }


def run_origin(
    search: SearchContext,
    scoring: ScoringContext,
    subject: Subject,
    selected: TrialOutcome,
    spec: RollingOriginSpec,
    origin: int,
    parts: Sequence[PartitionRows],
) -> Path:
    """Refit at one origin, score its windows, apply the R2 threshold and write the record;
    ``parts`` are the verified Train, V1 and V2 rows."""
    assert isinstance(subject, Configuration)
    run, name = search.run, subject.name
    seed: int = search.identity["seeds"]["main"]
    train = parts[0]
    geometry = OriginGeometry.of(spec, origin, int(train.times.min() // SECONDS_PER_DAY))
    with log_step(logger, f"Rolling origin {origin} of {name}"):
        training = rows_between([train], *geometry.training, name=f"train-s{origin}")
        inputs = SearchInputs(training, parts[1])
        neural: NeuralRun | None = None
        if kind_of(subject) == NEURAL:
            if search.neural is None:
                raise SpecificationError(f"{name}: the neural inputs of the run are missing")
            neural = refitted_neural(
                search.neural,
                search.features,
                search.declaration.preprocessing_seed,
                inputs,
                parts[2],
                run.path / PREPROCESSING_DIR / f"s{origin}",
            )
        context = replace(search, inputs=inputs, neural=neural)
        device = scoring.settings.training_device(subject.model.name)
        job = RefitJob(
            name,
            RefitSpec(KIND, seed, device),
            selected.trial,
            selected.hyperparameters,
            selected.fitted_iterations,
        )
        directory = run.path / MODELS_DIR / name / f"rolling-s{origin}"
        seconds = fit_refit(context, subject, job, directory)[0]  # the fitted model is released
        model = load_model(
            replace(scoring, neural=neural),
            subject,
            selected,
            directory=directory,
            seed=seed,
            device=device,
        )
        window = rows_between(parts, geometry.window[0], geometry.transfer[1], name=f"s{origin}")
        scores = predict(model, window, neural)
        frame = pd.DataFrame({SCORE_COLUMNS[0]: window.ids, SCORE_COLUMNS[1]: scores})
        path = run.path / SCORES_DIR / f"{name}-rolling-s{origin}.parquet"
        write_atomically(path, lambda target: frame.to_parquet(target, index=False))
        days = window.times // SECONDS_PER_DAY
        in_window = days <= geometry.window[1]
        outcome = origin_outcome(
            PRIMARY_SHARE,
            days[in_window],
            scores[in_window],
            geometry.window,
            days[~in_window],
            scores[~in_window],
            geometry.transfer,
        )
        files = run_file(run, directory / MODEL_FILES[subject.model.name])
        record = origin_record(
            search, subject, selected, geometry, training, neural, outcome,
            (seconds, files), run_file(run, path), device,
        )  # fmt: skip
        written = run.write_record(record, name, f"s{origin}")
    logger.info(
        "%s origin %d: threshold %.6g, K = %d, transfer rate %.4f",
        name,
        origin,
        outcome.threshold,
        outcome.k,
        outcome.transfer_rate,
    )
    return written


def check_batches(
    search: SearchContext,
    subject: Subject,
    selected: TrialOutcome,
    spec: RollingOriginSpec,
    train: PartitionRows,
) -> None:
    """Stop before any fit unless the selected trial's batch plan suits every origin's training
    rows: the last micro-batch of an effective batch must hold the rows the network needs. The
    plan was fixed for the whole of Train, whose row count differs from each origin's."""
    if kind_of(subject) != NEURAL or search.neural is None:
        return
    model = subject.model.name
    plan = search.neural.plans[model][selected.trial]
    minimum = NEURAL_MODELS[model].MIN_BATCH_ROWS
    days = train.times // SECONDS_PER_DAY
    for origin in spec.origins:
        rows = int((days <= spec.training_last_day(origin)).sum())
        final = rows % plan.effective or plan.effective
        last = min(plan.micro_sizes(size)[-1] for size in {plan.effective, final})
        if last < minimum:
            raise SpecificationError(
                f"{subject.name}: the {rows} training rows of origin {origin} in effective "
                f"batches of {plan.effective} and micro-batches of {plan.micro} leave a last "
                f"micro-batch of {last} rows; {model} needs at least {minimum}"
            )


def run_rolling(
    search: SearchContext,
    scoring: ScoringContext,
    subject: Subject,
    selected: TrialOutcome,
    spec: RollingOriginSpec,
    parts: Sequence[PartitionRows],
) -> list[Path]:
    """Every origin of the supplement for the primary configuration ``subject``."""
    check_batches(search, subject, selected, spec, parts[0])
    with log_step(logger, f"Rolling-origin supplement of {subject.name} ({len(spec.origins)})"):
        return [
            run_origin(search, scoring, subject, selected, spec, origin, parts)
            for origin in spec.origins
        ]
