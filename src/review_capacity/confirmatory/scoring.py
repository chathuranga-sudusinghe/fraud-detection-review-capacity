"""Phase 1 of the single Test evaluation: the one scoring pass (Issue #203; protocol, Authorised
Test evaluation).

Each of the eight configurations and the logistic-regression reference is reloaded from the file
the frozen inputs name, scored once in one call on the Test features (no labels exist here), and
its scores are written as a Parquet file of ``TransactionID`` and ``score``, hashed and described
by a ``test_scores`` record body. The scoring time of the single pass is recorded and the pass is
never repeated for timing. The recorded time is the **end-to-end** boundary of DR-015
§9 for the single pass: from the in-memory Test features (conversion, the fitted preprocessing for
the neural models, native containers, host-device transfers) to the scores on the host, with the
explicit CUDA synchronisation before the clock starts and after the work completes for the neural
models. The model-only boundary needs a second, prepared pass and is measured on the development
partitions only; no Test pass is repeated for timing. The file of a model is hashed again
immediately before it is loaded, so a file replaced after the binding is refused.

This module never touches the labels: it takes the scores of the model, nothing else.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from review_capacity.config import FeatureSpec, ModelSettings
from review_capacity.confirmatory.inputs import TestInputs
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.scoring import (
    SCORE_COLUMNS,
    SCORES_DIR,
    ScoredModel,
    check_sequence,
    load_saved_model,
)
from review_capacity.development.subjects import NEURAL, REFERENCE, kind_of, scoring_device
from review_capacity.development.trial import run_file
from review_capacity.errors import DataValidationError, SpecificationError
from review_capacity.features.neural_preprocessing import NeuralPreprocessing
from review_capacity.files import sync_chain, write_atomically, write_new_atomically
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models.base import Scores, checked_scores
from review_capacity.models.neural_training import NeuralModel
from review_capacity.models.registry import Subject
from review_capacity.training.timing import Clock, gpu_clock

logger = logging.getLogger(__name__)


def attempt_directory(run: RunDirectory, attempt: int) -> Path:
    """The directory of the score files of one scoring attempt."""
    return run.path / SCORES_DIR / f"attempt-{attempt:02d}"


def _predict(
    model: ScoredModel, inputs: TestInputs, artifact: NeuralPreprocessing | None
) -> Scores:
    if isinstance(model, NeuralModel):
        if artifact is None:
            raise SpecificationError("The neural models need the preprocessing artifact")
        return model.predict(artifact.to_neural_inputs(inputs.features))
    return model.predict(inputs.features)


def _timed_predict(
    model: ScoredModel, kind: str, inputs: TestInputs, artifact: NeuralPreprocessing | None
) -> tuple[Scores, float]:
    """The scores of the single pass and its end-to-end seconds (DR-015 §9): the neural models
    synchronise the explicit CUDA device before the clock starts and after the work completes."""
    clock = gpu_clock(Clock()) if kind == NEURAL else Clock()
    clock.synchronise()
    start = clock.now()
    scores = _predict(model, inputs, artifact)
    clock.synchronise()
    return scores, clock.now() - start


BUNDLE_FILE = "bundle.json"
ATTEMPT_FILE = "attempt.json"


def start_attempt(run: RunDirectory, attempt: int, started: str) -> Path:
    """Allocate the attempt durably, before any scoring: its directory is created (an existing
    one is refused, so no earlier attempt's files are ever overwritten) and stamped."""
    directory = attempt_directory(run, attempt)
    directory.mkdir(parents=True, exist_ok=False)
    sync_chain(directory, run.path)
    write_new_atomically(
        directory / ATTEMPT_FILE, json.dumps({"attempt": attempt, "started": started}) + "\n"
    )
    return directory


def attempt_numbers(run: RunDirectory) -> list[int]:
    """The numbers of the attempts allocated in ``run`` (stamped directories), ascending."""
    root = run.path / SCORES_DIR
    if not root.is_dir():
        return []
    return sorted(
        int(path.name.removeprefix("attempt-"))
        for path in root.glob("attempt-*")
        if (path / ATTEMPT_FILE).is_file()
    )


def attempt_started(run: RunDirectory, attempt: int) -> str:
    stamp = json.loads((attempt_directory(run, attempt) / ATTEMPT_FILE).read_text("utf-8"))
    started: str = stamp["started"]
    return started


def write_bundle(run: RunDirectory, attempt: int, bodies: Sequence[Mapping[str, Any]]) -> None:
    """Record that every model of the attempt was scored: the record bodies, listed once, before
    any score record is published, so that a failure while publishing can be recovered without
    scoring again."""
    write_new_atomically(
        attempt_directory(run, attempt) / BUNDLE_FILE, json.dumps(list(bodies), indent=2) + "\n"
    )


def read_bundle(
    run: RunDirectory, attempt: int, configurations: Sequence[str]
) -> list[dict[str, Any]] | None:
    """The bundle of an attempt if it lists exactly ``configurations`` and every score file is the
    one it names (by hash); ``None`` for a missing bundle and, with a logged warning that names the
    cause, for one that cannot be used (a new attempt then scores again: no score was published and
    no label was read)."""
    path = attempt_directory(run, attempt) / BUNDLE_FILE
    if not path.is_file():
        return None
    try:
        bodies: list[dict[str, Any]] = json.loads(path.read_text("utf-8"))
        names = [body["configuration"] for body in bodies]
        verified = all(
            file_sha256(run.path / body["file"]["path"]) == body["file"]["sha256"]
            for body in bodies
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.warning(
            "%s: the recovery bundle cannot be used (%s: %s)", path, type(error).__name__, error
        )
        return None
    if not verified or names != list(configurations):
        logger.warning(
            "%s: the recovery bundle does not match its score files or the nine configurations",
            path,
        )
        return None
    return bodies


def _write_scores(target: Path, frame: pd.DataFrame) -> None:
    write_atomically(target, lambda out: frame.to_parquet(out, index=False))


def score_test_models(
    *,
    run: RunDirectory,
    attempt: int,
    root: Path,
    models: Mapping[str, Mapping[str, str]],
    subjects: Sequence[Subject],
    settings: ModelSettings,
    features: FeatureSpec,
    seed: int,
    reference_fixed: Mapping[str, Any],
    artifact: NeuralPreprocessing | None,
    inputs: TestInputs,
    release: Callable[[], None] = lambda: None,
) -> list[dict[str, Any]]:
    """Score the Test rows once with every subject's frozen model. Returns the ``test_scores``
    record bodies (everything but the envelope), in the order of ``subjects``; no record is
    written here, so that a failed attempt leaves no score record behind."""
    bodies: list[dict[str, Any]] = []
    directory = attempt_directory(run, attempt)
    first, last = int(inputs.days.min()), int(inputs.days.max())
    with log_step(logger, f"Scoring the Test rows ({inputs.rows}) with {len(subjects)} models"):
        for subject in subjects:
            name = subject.name
            reference = models.get(name)
            if reference is None:
                raise SpecificationError(f"{name}: the frozen inputs name no model")
            path = root / reference["path"]
            if file_sha256(path) != reference["sha256"]:
                raise SpecificationError(f"{name}: {path.name} differs from the frozen model")
            kind = kind_of(subject)
            device = None if kind == REFERENCE else settings.training_device(subject.model.name)
            with log_step(logger, f"Scoring the Test rows with the frozen {name}"):
                model = load_saved_model(
                    subject,
                    path.parent,
                    seed=seed,
                    settings=settings,
                    features=features,
                    device=device,
                    artifact=artifact,
                    reference_fixed=reference_fixed if kind == REFERENCE else None,
                )
                raw_scores, seconds = _timed_predict(model, kind, inputs, artifact)
                scores = checked_scores(raw_scores, inputs.rows, name)
            del model
            if kind == NEURAL:
                release()
            target = directory / f"{name}.parquet"
            frame = pd.DataFrame({SCORE_COLUMNS[0]: inputs.ids, SCORE_COLUMNS[1]: scores})
            _write_scores(target, frame)
            written = pd.read_parquet(target)
            if tuple(written.columns) != SCORE_COLUMNS:
                raise DataValidationError(f"{target}: columns {tuple(written.columns)}")
            check_sequence(inputs.ids, written["TransactionID"].to_numpy(dtype=np.int64), name)
            bodies.append(
                {
                    "configuration": name,
                    "partition": "Test",
                    "days": {"first": first, "last": last},
                    "rows": inputs.rows,
                    "model": dict(reference),
                    "file": run_file(run, target),
                    "format": "parquet",
                    "min_score": float(scores.min()),
                    "max_score": float(scores.max()),
                    "distinct_scores": len(np.unique(scores)),
                    "device": scoring_device(settings, subject),
                    "timing": {
                        "passes": 1,
                        "seconds": seconds,
                        "rows_per_second": inputs.rows / seconds,
                    },
                }
            )
            logger.info("%s scored: %d distinct scores", name, bodies[-1]["distinct_scores"])
    return bodies
