"""The command of the single Test evaluation: phase 1 (scoring) and phase 2 (evaluation)
(Issues #203 and #207).

``run_test`` follows the protocol's Authorised Test evaluation in this order, and none of the
first six steps reads a Test row:

1. the evaluating checkout is a clean tree at the commit of the annotated freeze tag;
2. every frozen input is verified by hash and recorded (``confirmatory.frozen.bind_frozen_inputs``);
3. the frozen state is not consumed and, for a new scoring, not scored;
4. the lock of the frozen state is taken;
5. the run directory is created (or an unfinished one re-opened) and ``test_run`` is written;
6. a completed scoring phase has its global marker (a crash between the two is repaired);
7. **scoring** (``--phase scoring|all``): the Test features are read without the target and each
   frozen model scores them once; a failed or interrupted attempt keeps its ``test_phase`` record
   and writes no score record, a repeat is a new attempt;
8. **evaluation** (``--phase evaluation|all``): ``confirmatory.evaluation`` verifies the saved
   scores, creates the consumption marker and only then reads the labels, and writes the outcome
   records; the inference record follows; the whole run is checked as if the attempt were
   complete; the completed attempt is recorded and the run is closed with ``test_run_end``. A
   failed attempt keeps its record (``labels_accessed`` says whether *that attempt* reached the
   marker) and the evaluation resumes from the saved scores.

The labels are read only inside ``confirmatory.evaluation``; this module orchestrates.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from review_capacity.config import DatasetSpec, load_dataset_spec, load_feature_spec
from review_capacity.confirmatory.attempts import (
    allocate,
    close_orphans,
    mark_reaching,
    next_attempt,
    phase_record,
    phase_records,
    reached,
)
from review_capacity.confirmatory.closing import close_run, whole_run_problems
from review_capacity.confirmatory.evaluation import EvaluationSetup, evaluate_outcomes
from review_capacity.confirmatory.frozen import (
    FrozenBinding,
    bind_frozen_inputs,
    evaluator_state,
)
from review_capacity.confirmatory.inference_record import build_inference_record, inference_inputs
from review_capacity.confirmatory.inputs import read_test_inputs
from review_capacity.confirmatory.outcome_records import OutcomeSetup, publish_outcomes
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
from review_capacity.evaluation.declaration import EvaluationDeclaration
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
PHASES = ("scoring", "evaluation", "all")
EVALUATION = "evaluation"


@dataclass(frozen=True)
class TestRunOptions:
    """The arguments of ``run-test-evaluation``."""

    __test__ = False  # not a pytest class

    tag: str
    development: Path
    analyses: tuple[Path, ...]
    phase: str = "all"
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


@dataclass(frozen=True)
class _Prepared:
    """Everything an invocation loads and verifies before it touches a run directory."""

    paths: ProjectPaths
    options: TestRunOptions
    started: str
    settings: Any
    settings_file: Any
    declaration: Any
    declaration_file: Any
    evaluation: EvaluationDeclaration
    evaluation_file: Any
    dataset: DatasetSpec
    binding: FrozenBinding
    identity: dict[str, Any]
    records: ConfirmatoryRecords
    manifest_sha256: str


def _prepare(paths: ProjectPaths, options: TestRunOptions, started: str) -> _Prepared:
    """Load the declarations, check the evaluator and bind every frozen input by hash: nothing
    here reads a Test row."""
    root = paths.root
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
    return _Prepared(
        paths,
        options,
        started,
        settings,
        settings_file,
        declaration,
        declaration_file,
        evaluation,
        evaluation_file,
        dataset,
        binding,
        identity,
        ConfirmatoryRecords(ConfirmatoryContext.from_specs(dataset, evaluation)),
        manifest_sha256,
    )


def _write_test_run(prep: _Prepared, run: RunDirectory, now: datetime | None) -> None:
    """The ``test_run`` record, written once, first."""
    if (run.records_dir / "test_run.json").exists():
        return
    paths, test = prep.paths, prep.dataset.partition("test")
    run.write_record(
        {
            "record": "test_run",
            "schema_version": contract_versions()["test_run"],
            "identity": prep.identity,
            "started": timestamp(now),
            "environment": environment_record(),
            "settings": project_file(paths, paths.models_spec, prep.settings_file.sha256),
            "training_declaration": project_file(
                paths, paths.training_spec, prep.declaration_file.sha256
            ),
            "evaluation_declaration": project_file(
                paths, paths.evaluation_spec, prep.evaluation_file.sha256
            ),
            "dataset_manifest": project_file(paths, paths.manifest, prep.manifest_sha256),
            "frozen_inputs": prep.binding.frozen_inputs,
            "test": {
                "first_day": test.first_day,
                "last_day": test.last_day,
                "rows": test.rows,
                "transaction_id_sha256": prep.binding.test_anchor_sha256,
            },
            "contracts": contract_versions(),
        }
    )


def run_test(
    paths: ProjectPaths,
    options: TestRunOptions,
    *,
    now: datetime | None = None,
    release: Callable[[], None] = release_gpu_memory,
) -> Path:
    """The single Test evaluation: ``--phase scoring`` (phase 1, the run stays open),
    ``evaluation`` (phase 2 on the unfinished run ``--resume``) or ``all`` (phase 1 if it is not
    complete, then phase 2, under one lock). Returns the run directory."""
    if options.phase not in PHASES:
        raise SpecificationError(f"The phase must be one of {list(PHASES)}, not {options.phase!r}")
    if options.phase == "evaluation" and options.resume is None:
        raise SpecificationError(
            "The evaluation phase continues an existing run: give the unfinished run (--resume)"
        )
    root = paths.root
    started = options.started or run_id(now)
    with log_step(logger, f"Test evaluation, phase {options.phase} ({options.tag})"):
        prep = _prepare(paths, options, started)
        state = prep.binding.frozen_state_sha256
        scoring = options.phase in ("scoring", "all")
        if scoring and options.resume is None:
            refuse_if_consumed(root, state)
            refuse_if_scored(root, state)  # before any directory is created
        with locked(root, state, run_id=started):
            if options.phase == "scoring":
                refuse_if_consumed(root, state)  # a run that ended meanwhile
            run = _open_run(root, options, started, prep.identity, prep.records)
            scored = [a for a in _scoring_attempts(run) if a["outcome"] == "completed"]
            if scoring and not scored:
                refuse_if_consumed(root, state)
                refuse_if_scored(root, state)  # a scoring that completed in another run
                _write_test_run(prep, run, now)
                _close_orphans(run, run.identity or prep.identity)
                _score(paths, run, prep.binding, prep.declaration, prep.settings, release)
            if scored:
                _reconcile_scoring_marker(root, state, run, str(scored[-1]["scores_sha256"]))
            if options.phase == "scoring" and scored:
                raise SpecificationError(
                    "The scoring phase of this run is complete; the evaluation phase is next"
                )
            if options.phase in ("evaluation", "all"):
                _evaluation_phase(prep, run)
    logger.info("Test evaluation written to %s", run.path)
    return run.path


def _reconcile_scoring_marker(root: Path, state: str, run: RunDirectory, bundle: str) -> None:
    """A completed scoring phase has its global marker, whatever phase is run next: a crash
    between the completed record and the marker is repaired here, and an existing marker must be
    this run's, for these scores."""
    run_id_of = (run.identity or {})["run_id"]
    marker = scored_path(root, state)
    if not marker.exists():
        record_scoring(root, state, run_id=run_id_of, scores_sha256=bundle)
        return
    found = json.loads(marker.read_text(encoding="utf-8"))
    if (found.get("run_id"), found.get("scores_sha256")) != (run_id_of, bundle):
        raise SpecificationError(
            f"{marker.name} belongs to another run or other scores than the completed scoring "
            "of this run"
        )


def run_test_scoring(
    paths: ProjectPaths,
    options: TestRunOptions,
    *,
    now: datetime | None = None,
    release: Callable[[], None] = release_gpu_memory,
) -> Path:
    """Phase 1 alone (``--phase scoring``): the run is left open."""
    return run_test(paths, replace(options, phase="scoring"), now=now, release=release)


def _evaluation_phase(prep: _Prepared, run: RunDirectory) -> None:
    """One attempt of the evaluation phase on a run with a completed scoring phase."""
    paths = prep.paths
    scored = [a for a in _scoring_attempts(run) if a["outcome"] == "completed"]
    if not scored:
        raise SpecificationError("The evaluation phase follows a completed scoring phase")
    bundle = str(scored[-1]["scores_sha256"])
    identity = run.identity or prep.identity
    if any(a["outcome"] == "completed" for a in phase_records(run, EVALUATION)):
        # a crash between the completed attempt and the closing record
        close_run(run)
        return
    close_orphans(run, identity, scores_sha256=bundle)
    attempt, started = next_attempt(run), timestamp()
    allocate(run, attempt, started)
    outcome, error, completed_record = "completed", None, None
    try:
        features = load_feature_spec(paths.features_spec)
        setup = EvaluationSetup(
            paths,
            run,
            prep.binding,
            prep.evaluation,
            prep.dataset,
            features,
            before_consumption=lambda: mark_reaching(run, attempt),
        )
        result = evaluate_outcomes(setup)
        outcome_setup = OutcomeSetup(
            prep.evaluation,
            identity,
            int(result.partition.truth.day_values[0]),
            int(result.partition.truth.day_values[-1]),
            result.saved.files,
            result.partition.reference_volume,
        )
        inputs = inference_inputs(
            prep.evaluation, result.partition, result.saved.scores, result.policies
        )
        record = build_inference_record(
            outcome_setup, inputs, str(prep.binding.frozen_inputs["h3_configuration"])
        )
        publish_outcomes(run, [((), record)])
        completed_record = phase_record(
            identity, attempt, started, "completed", None,
            scores_sha256=bundle, labels_accessed=True,
        )  # fmt: skip
        problems = whole_run_problems(run, completed_record)
        if problems:
            raise SpecificationError("The run is not complete: " + "; ".join(problems))
    except KeyboardInterrupt:
        outcome, error = "interrupted", "KeyboardInterrupt"
        raise
    except BaseException as failure:
        outcome, error = "failed", f"{type(failure).__name__}: {failure}"
        raise
    finally:
        record_to_write = (
            completed_record
            if outcome == "completed" and completed_record is not None
            else phase_record(
                identity, attempt, started, outcome, error,
                scores_sha256=bundle,
                labels_accessed=reached(run, attempt),
            )
        )  # fmt: skip
        run.write_record(record_to_write, EVALUATION, attempt)
    close_run(run)


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
        run_id=(run.identity or {})["run_id"],
        scores_sha256=str(bundle),
    )
