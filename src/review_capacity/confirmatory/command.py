"""The command of the single Test evaluation: phase 1, the scoring phase (Issue #203).

``run_test_scoring`` follows the protocol's Authorised Test evaluation in this order, and every
step before the sixth stops the command without a Test row having been read:

1. the evaluating checkout is a clean tree at the commit of the annotated freeze tag;
2. every frozen input is verified by hash and recorded (``confirmatory.frozen.bind_frozen_inputs``);
3. the frozen state is not consumed (the labels were not read in an earlier run);
4. the lock of the frozen state is taken;
5. the run directory is created (or an unfinished one re-opened) and ``test_run`` is written;
6. the Test features are read without the target and the raw files and partition anchors verified;
7. each frozen model scores them once; the scores are hashed and described by ``test_scores``
   records, and the attempt by a ``test_phase`` record. A failed or interrupted attempt keeps its
   ``test_phase`` record and writes no score record; a repeat is a new attempt.

The run stays open: the evaluation phase (the Test labels, the outcomes, the closing record) is a
later step that resumes from the saved, hashed scores. Nothing here reads a label.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from review_capacity.config import load_dataset_spec, load_feature_spec
from review_capacity.confirmatory.frozen import (
    FrozenBinding,
    bind_frozen_inputs,
    evaluator_state,
)
from review_capacity.confirmatory.inputs import read_test_inputs
from review_capacity.confirmatory.records import (
    ConfirmatoryContext,
    ConfirmatoryRecords,
    contract_versions,
    scores_bundle_sha256,
)
from review_capacity.confirmatory.scoring import (
    attempt_numbers,
    attempt_started,
    read_bundle,
    score_test_models,
    start_attempt,
    write_bundle,
)
from review_capacity.confirmatory.state import (
    locked,
    record_scoring,
    refuse_if_consumed,
    refuse_if_scored,
    scored_path,
)
from review_capacity.development.inputs import TRAIN, read_partitions
from review_capacity.development.records import ALL_CONFIGURATIONS, DevelopmentPartitions
from review_capacity.development.run_output import RunDirectory, load_run, run_id, timestamp
from review_capacity.development.run_start import project_file
from review_capacity.development.runner import chosen_configurations
from review_capacity.errors import SpecificationError
from review_capacity.evaluation.declaration import load_declaration as load_evaluation_declaration
from review_capacity.features.neural_preprocessing import load_neural_preprocessing
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models.device import environment_record
from review_capacity.models.run_record import load_settings_with_snapshot
from review_capacity.paths import ProjectPaths
from review_capacity.provenance import code_provenance
from review_capacity.training.declaration import REFERENCE_MODEL
from review_capacity.training.declaration import load_declaration as load_training_declaration
from review_capacity.training.preflight import release_gpu_memory

logger = logging.getLogger(__name__)

RUN_TYPE = "confirmatory"
PHASES = ("scoring",)  # the evaluation phase is added with the labels


@dataclass(frozen=True)
class TestRunOptions:
    """The arguments of ``run-test-evaluation``."""

    __test__ = False  # not a pytest class

    tag: str
    development: Path
    analyses: tuple[Path, ...]
    phase: str = "scoring"
    output: Path | None = None
    resume: Path | None = None
    started: str | None = None


def run_log_name(started: str, resume: bool) -> str:
    """The log file of a run directory: ``run.log`` for a new run, one file for each resumption."""
    return f"resume-{started}.log" if resume else "run.log"


def _identity(
    started: str,
    tag: str,
    tag_commit: str,
    code: Mapping[str, Any],
    binding: FrozenBinding,
    hashes: Mapping[str, str],
    seeds: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": started,
        "run_type": RUN_TYPE,
        "evaluator": {
            "code_commit": code["git_commit"],
            "uncommitted_changes": code["git_uncommitted_changes"],
            "source_sha256": code["package"]["source_sha256"],
        },
        "freeze": {"tag": tag, "tag_commit": tag_commit},
        "training": dict(binding.training),
        **hashes,
        "seeds": dict(seeds),
        "frozen_state_sha256": binding.frozen_state_sha256,
    }


def _scoring_attempts(run: RunDirectory) -> list[dict[str, Any]]:
    """The ``test_phase`` records of the scoring phase already written in the run."""
    paths = sorted(run.records_dir.glob("test_phase-scoring-*.json"))
    return [run.records.read(path, "test_phase") for path in paths]


def _phase_record(
    identity: Mapping[str, Any],
    attempt: int,
    started: str,
    outcome: str,
    error: str | None,
    scores_sha256: str | None,
) -> dict[str, Any]:
    return {
        "record": "test_phase",
        "schema_version": contract_versions()["test_phase"],
        "identity": dict(identity),
        "phase": "scoring",
        "attempt": attempt,
        "started": started,
        "finished": timestamp(),
        "outcome": outcome,
        "error": error,
        "labels_accessed": False,
        "scores_sha256": scores_sha256,
    }


def _close_orphans(run: RunDirectory, identity: Mapping[str, Any]) -> None:
    """Give every allocated attempt that left no terminal record (the process ended without one,
    for example after a kill) an ``interrupted`` record, so that no attempt goes unreported."""
    recorded = {a["attempt"] for a in _scoring_attempts(run)}
    for attempt in attempt_numbers(run):
        if attempt in recorded:
            continue
        run.write_record(
            _phase_record(
                identity,
                attempt,
                attempt_started(run, attempt),
                "interrupted",
                "the attempt left no terminal record: the process ended without one",
                None,
            ),
            "scoring",
            attempt,
        )
        logger.warning("Scoring attempt %d left no record; recorded as interrupted", attempt)


def _next_attempt(run: RunDirectory) -> int:
    numbers = [a["attempt"] for a in _scoring_attempts(run)] + attempt_numbers(run)
    return max(numbers, default=0) + 1


def _recoverable(run: RunDirectory) -> list[dict[str, Any]] | None:
    """The bundle of the latest attempt whose nine models were all scored, if any: it is
    published without scoring again."""
    for attempt in reversed(attempt_numbers(run)):
        bodies = read_bundle(run, attempt, ALL_CONFIGURATIONS)
        if bodies is not None:
            return bodies
    return None


def _publish(
    run: RunDirectory, identity: Mapping[str, Any], bodies: Sequence[Mapping[str, Any]]
) -> None:
    """Write the ``test_scores`` record of every model. A record that exists already (an earlier
    publication that failed part-way) must be the one that would be written; it is kept."""
    for body in bodies:
        record = {
            "record": "test_scores",
            "schema_version": contract_versions()["test_scores"],
            "identity": dict(identity),
            **body,
        }
        path = run.record_path("test_scores", body["configuration"])
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(record)):
                raise SpecificationError(f"{path.name} exists and differs from the scored bundle")
            continue
        run.write_record(record, body["configuration"])


def run_test_scoring(
    paths: ProjectPaths,
    options: TestRunOptions,
    *,
    now: datetime | None = None,
    release: Callable[[], None] = release_gpu_memory,
) -> Path:
    """Phase 1 of the single Test evaluation; returns the run directory (left open)."""
    if options.phase not in PHASES:
        raise SpecificationError(f"The phase must be one of {list(PHASES)}, not {options.phase!r}")
    root = paths.root
    started = options.started or run_id(now)
    with log_step(logger, f"Test evaluation, scoring phase ({options.tag})"):
        settings, settings_file = load_settings_with_snapshot(paths)
        declaration, declaration_file = load_training_declaration(paths)
        evaluation, evaluation_file = load_evaluation_declaration(paths)
        dataset = load_dataset_spec(paths)
        partitions = DevelopmentPartitions.from_spec(dataset, paths)
        evaluator = evaluator_state(root, options.tag)
        code = code_provenance(root)
        development = load_run(options.development, partitions)
        analyses = [load_run(path, partitions) for path in options.analyses]
        manifest_sha256 = file_sha256(paths.manifest)
        binding = bind_frozen_inputs(
            root=root,
            evaluator=evaluator,
            development=development,
            analyses=analyses,
            manifest_path=paths.manifest,
            capacities=[float(share) for share in evaluation.capacity.shares],
            settings_sha256=settings_file.sha256,
            training_declaration_sha256=declaration_file.sha256,
            evaluation_declaration_sha256=evaluation_file.sha256,
            dataset_manifest_sha256=manifest_sha256,
        )
        refuse_if_consumed(root, binding.frozen_state_sha256)
        identity = _identity(
            started,
            options.tag,
            evaluator.tag_commit,
            code,
            binding,
            {
                "settings_sha256": settings_file.sha256,
                "training_declaration_sha256": declaration_file.sha256,
                "evaluation_declaration_sha256": evaluation_file.sha256,
                "dataset_manifest_sha256": manifest_sha256,
            },
            {
                "main": settings.main_seed,
                "additional": list(declaration.additional_seeds),
                "sampler": declaration.sampler_seed,
                "preprocessing": declaration.preprocessing_seed,
                "resampling": evaluation.resampling.root_seed,
            },
        )
        context = ConfirmatoryContext.from_specs(dataset, evaluation)
        record_set = ConfirmatoryRecords(context)
        state = binding.frozen_state_sha256
        if options.resume is None:
            refuse_if_scored(root, state)  # before any directory is created
        with locked(root, state, run_id=started):
            refuse_if_consumed(root, state)  # a run that ended meanwhile
            run = _open_run(root, options, started, identity, record_set)
            completed = [a for a in _scoring_attempts(run) if a["outcome"] == "completed"]
            if completed:
                if not scored_path(root, state).exists():  # a crash between record and marker
                    record_scoring(
                        root,
                        state,
                        run_id=run.path.name,
                        scores_sha256=str(completed[-1]["scores_sha256"]),
                    )
                raise SpecificationError(
                    "The scoring phase of this run is complete; the evaluation phase is next"
                )
            refuse_if_scored(root, state)  # a scoring that completed in another run
            if not (run.records_dir / "test_run.json").exists():
                test = dataset.partition("test")
                run.write_record(
                    {
                        "record": "test_run",
                        "schema_version": contract_versions()["test_run"],
                        "identity": identity,
                        "started": timestamp(now),
                        "environment": environment_record(),
                        "settings": project_file(paths, paths.models_spec, settings_file.sha256),
                        "training_declaration": project_file(
                            paths, paths.training_spec, declaration_file.sha256
                        ),
                        "evaluation_declaration": project_file(
                            paths, paths.evaluation_spec, evaluation_file.sha256
                        ),
                        "dataset_manifest": project_file(paths, paths.manifest, manifest_sha256),
                        "frozen_inputs": binding.frozen_inputs,
                        "test": {
                            "first_day": test.first_day,
                            "last_day": test.last_day,
                            "rows": test.rows,
                            "transaction_id_sha256": binding.test_anchor_sha256,
                        },
                        "contracts": contract_versions(),
                    }
                )
            _close_orphans(run, run.identity or identity)
            _score(paths, run, binding, declaration, settings, release)
    logger.info("Scoring phase written to %s (the run stays open)", run.path)
    return run.path


def _open_run(
    root: Path,
    options: TestRunOptions,
    started: str,
    identity: Mapping[str, Any],
    records: ConfirmatoryRecords,
) -> RunDirectory:
    """A new run directory, or the unfinished one to resume, which must be of this frozen state."""
    if options.resume is None:
        return RunDirectory.create(
            root, None, RUN_TYPE, started, options.output, log_opened=True, records=records
        )
    run = RunDirectory.reopen(options.resume, RUN_TYPE, records)
    kept = {k: v for k, v in (run.identity or {}).items() if k != "run_id"}
    now = {k: v for k, v in identity.items() if k != "run_id"}
    if kept != now:
        raise SpecificationError(
            f"{options.resume} was started under another identity; a resumed run keeps its own"
        )
    return run


def _check_frozen_files(root: Path, frozen: Mapping[str, Any], names: Sequence[str]) -> None:
    """The files read after the binding are the frozen ones: each is hashed again and compared
    with the hash the binding recorded for it."""
    for name in names:
        reference = frozen[name]
        if file_sha256(root / reference["path"]) != reference["sha256"]:
            raise SpecificationError(f"{reference['path']} differs from the frozen {name}")


def _score(
    paths: ProjectPaths,
    run: RunDirectory,
    binding: FrozenBinding,
    declaration: Any,
    settings: Any,
    release: Callable[[], None],
) -> None:
    """One attempt of the scoring phase: its ``test_phase`` record is written whatever happens,
    and the scoring marker when it completed. A bundle of nine verified score files left by an
    earlier attempt is published without scoring again."""
    attempt = _next_attempt(run)
    started = timestamp()
    identity = run.identity or {}
    outcome, error, bundle = "completed", None, None
    try:
        bodies = _recoverable(run)
        if bodies is None:
            features = load_feature_spec(paths.features_spec)
            dataset = load_dataset_spec(paths)
            frozen = binding.frozen_inputs
            _check_frozen_files(paths.root, frozen, ("preprocessing", "preprocessing_arrays"))
            train = read_partitions(paths, features, dataset, (TRAIN,))[TRAIN]
            manifest = paths.root / frozen["preprocessing"]["path"]
            artifact = load_neural_preprocessing(
                manifest.parent,
                features,
                declaration.preprocessing_seed,
                train_ids_sha256=train.ordered_ids_sha256,
            )
            inputs = read_test_inputs(features, dataset, binding.test_anchor_sha256)
            start_attempt(run, attempt, started)
            bodies = score_test_models(
                run=run,
                attempt=attempt,
                root=paths.root,
                models=frozen["models"],
                subjects=chosen_configurations(ALL_CONFIGURATIONS),
                settings=settings,
                features=features,
                seed=settings.main_seed,
                reference_fixed=dict(declaration.space(REFERENCE_MODEL).fixed),
                artifact=artifact,
                inputs=inputs,
                release=release,
            )
            write_bundle(run, attempt, bodies)
        else:
            logger.info("Publishing the verified bundle of an earlier attempt without scoring")
        _publish(run, identity, bodies)
        bundle = scores_bundle_sha256([body["file"]["sha256"] for body in bodies])
    except KeyboardInterrupt:
        outcome, error = "interrupted", "KeyboardInterrupt"
        raise
    except BaseException as failure:
        outcome, error = "failed", f"{type(failure).__name__}: {failure}"
        raise
    finally:
        run.write_record(
            _phase_record(identity, attempt, started, outcome, error, bundle), "scoring", attempt
        )
    record_scoring(
        paths.root,
        binding.frozen_state_sha256,
        run_id=run.path.name,
        scores_sha256=str(bundle),
    )
