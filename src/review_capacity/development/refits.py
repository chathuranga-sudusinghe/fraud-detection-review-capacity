"""The fixed-length refits of each configuration's selected setting, and the CatBoost CPU–GPU
comparison (DR-015 §4; DR-013 §6–§7; DR-006; Issue #156).

After a configuration's search selects a trial, its setting is refitted on Train at the selected
length (the retained rounds or epochs), with no search and no early stopping:

* **seed refits** with each additional seed (43 and 44);
* the **same-seed repeat**: the main seed again, in a fresh process (DR-013 §6, item 1);
* for CatBoost, the **CPU refit** at the GPU-selected settings and tree count with the main seed
  (DR-013 §7).

A refit is saved, reloaded and scored on V2 like the selected model (``development.scoring``) and
evaluated under R3 at the four capacities (``development.evaluation``); its ``fit`` and ``refit``
records link the fit, its V2 scores and the selected model's. A same-seed repeat and a CPU refit
carry the comparison of their V2 scores with the selected model's, exactly. The refits never
replace the selected model and never enter selection (descriptive only).

The CPU–GPU comparison of a CatBoost configuration is the ``device_comparison`` record: the R3
point differences (CPU − GPU) at each capacity and the paired two-way bootstrap interval of the
Recall@2% difference over the frauds of V2 (``evaluation.bootstrap``).

A fit that must run in a fresh process is delegated to ``fresh``, a callable that fits and saves
the model in the directory it is given and returns the seconds taken
(``development.refit_process`` runs it as a child process of the package).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from review_capacity.config import Device
from review_capacity.development.evaluation import evaluate_partition, truth_of
from review_capacity.development.scoring import (
    ScoredPartition,
    ScoringContext,
    load_selected,
    read_scores,
    score_model,
)
from review_capacity.development.subjects import MODEL_FILES, NEURAL, TREE, kind_of
from review_capacity.development.trial import (
    MODEL_RECORD_NAME,
    MODELS_DIR,
    NEURAL_MODELS,
    TREE_MODELS,
    Fitted,
    SearchContext,
    TrialOutcome,
    fit_record,
    hyperparameters,
    neural_of,
    run_file,
    save_fitted,
)
from review_capacity.errors import ModelError, ReviewCapacityError, SpecificationError
from review_capacity.evaluation.bootstrap import (
    LEVEL,
    REPLICATES,
    entity_indices,
    paired_difference,
)
from review_capacity.evaluation.bootstrap import SEED as BOOTSTRAP_SEED
from review_capacity.evaluation.entities import pseudo_entity_keys
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import capacity_budgets, r3_outcome
from review_capacity.logging_setup import log_step
from review_capacity.models.registry import Configuration, Subject
from review_capacity.training.preflight import fit_hyperparameters
from review_capacity.training.sampler import Candidate
from review_capacity.training.timing import Clock, gpu_clock, timed

logger = logging.getLogger(__name__)

SEED_REFIT = "seed_refit"
SAME_SEED_REPEAT = "same_seed_repeat"
CPU_REFIT = "cpu_refit"
PRIMARY_CAPACITY = 0.02  # the capacity of the Recall@K that the protocol's refits report


@dataclass(frozen=True)
class RefitSpec:
    """One refit: its kind, training seed, training device and whether it needs a fresh process."""

    kind: str
    seed: int
    device: Device
    fresh_process: bool = False

    @property
    def tag(self) -> str:
        """Names the refit's files and records (``seed_refit-s43``, ``same_seed_repeat``)."""
        return f"{self.kind}-s{self.seed}" if self.kind == SEED_REFIT else self.kind


def refit_specs(
    subject: Subject, device: Device, main_seed: int, additional: tuple[int, ...]
) -> tuple[RefitSpec, ...]:
    """The refits of a configuration whose training device is ``device``: one per additional
    seed, the same-seed repeat in a fresh process and, for CatBoost, the CPU refit."""
    if subject.family is None:
        raise SpecificationError(f"{subject.name}: the reference has no refits (DR-011 §5)")
    specs = [RefitSpec(SEED_REFIT, seed, device) for seed in additional]
    specs.append(RefitSpec(SAME_SEED_REPEAT, main_seed, device, fresh_process=True))
    if subject.model.name == "catboost":
        specs.append(RefitSpec(CPU_REFIT, main_seed, "cpu"))
    return tuple(specs)


def refit_directory(search_or_run: Path, configuration: str, spec: RefitSpec) -> Path:
    """Where a refit's model is saved: ``models/<configuration>/<tag>`` in the run directory."""
    return search_or_run / MODELS_DIR / configuration / spec.tag


@dataclass(frozen=True)
class RefitJob:
    """What a refit needs besides the shared contexts: the setting refitted and its length."""

    configuration: str
    spec: RefitSpec
    trial: int
    hyperparameters: Candidate
    length: int


def fit_refit(
    context: SearchContext, subject: Subject, job: RefitJob, directory: Path
) -> tuple[float, Fitted]:
    """Fit the selected setting for exactly ``job.length`` rounds or epochs on Train, with no
    validation and no early stopping, and save the model in ``directory``; returns the seconds
    of the fit (the persistence is outside it) and the fitted model."""
    assert isinstance(subject, Configuration)
    name, spec, model_name = subject.name, job.spec, subject.model.name
    with log_step(logger, f"{spec.tag} fit of {name} (seed {spec.seed}, {job.length} units)"):
        kind = kind_of(subject)
        model: Any
        if kind == TREE:
            model_class, _ = TREE_MODELS[model_name]
            tree = model_class(
                subject,
                spec.seed,
                settings=context.settings,
                features=context.features,
                device=spec.device,
            )
            settings = hyperparameters(context.declaration, subject, job.hyperparameters)
            train = context.inputs.train_data(subject.weighting)
            started = time.perf_counter()
            tree.fit(train, None, settings, iterations=job.length)
            seconds, model = time.perf_counter() - started, tree
        elif kind == NEURAL:
            neural = neural_of(context)
            network = NEURAL_MODELS[model_name](
                subject,
                spec.seed,
                settings=context.settings,
                layout=neural.artifact.layout,
                device=spec.device,
                artifact_sha256=neural.artifact_sha256,
            )
            settings = fit_hyperparameters(context.declaration, model_name, job.hyperparameters)
            plan = neural.plans[model_name][job.trial]
            data = neural.train[subject.weighting]
            clock = gpu_clock(Clock()) if spec.device == "gpu" else Clock()
            seconds = timed(
                lambda: network.fit(data, None, settings, iterations=job.length, plan=plan), clock
            )
            model = network
        else:
            raise SpecificationError(f"{name}: not a tree-based or neural configuration")
        fitted = Fitted(model, settings, seconds, None, None, 0.0, job.length, job.length, "fixed")
        save_fitted(context, subject, fitted, directory)
    return seconds, fitted


def compare_scores(reference: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """The exact comparison of two V2 score vectors (the ``refit`` record's ``comparison``)."""
    if reference.shape != scores.shape:
        raise ModelError("The scores to compare are for different numbers of rows")
    differing = reference != scores
    difference = np.abs(reference - scores)
    return {
        "rows": len(reference),
        "identical": bool(not differing.any()),
        "differing_rows": int(differing.sum()),
        "max_abs_difference": float(difference.max()),
    }


def _refit_record(
    context: ScoringContext,
    subject: Subject,
    selected: TrialOutcome,
    spec: RefitSpec,
    *,
    fit: Mapping[str, str],
    scored: ScoredPartition | None,
    reference: ScoredPartition,
    comparison: Mapping[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "record": "refit",
        "schema_version": 1,
        "identity": dict(context.identity),
        "configuration": subject.name,
        "kind": spec.kind,
        "seed": spec.seed,
        "device": spec.device,
        "selected_trial": selected.trial,
        "length": selected.fitted_iterations,
        "fit": dict(fit),
        "score": None if scored is None else dict(scored.record),
        "reference_score": dict(reference.record),
        "comparison": None if comparison is None else dict(comparison),
        "status": "completed" if error is None else "failed",
        "error": error,
    }


def run_refit(
    search: SearchContext,
    scoring: ScoringContext,
    subject: Subject,
    selected: TrialOutcome,
    spec: RefitSpec,
    reference: ScoredPartition,
    *,
    fresh: Callable[[RefitJob, Path], float] | None = None,
) -> ScoredPartition:
    """Fit, save, reload, score on V2 and evaluate one refit; write its ``fit``, ``score``,
    ``evaluation`` and ``refit`` records. ``reference`` is the selected model's scored V2.

    A fit in a fresh process is made by ``fresh(job, directory)``, which fits and saves the model
    there and returns the fit seconds. A failure writes the failed ``fit`` and ``refit`` records
    and is raised again."""
    run, name = search.run, subject.name
    job = RefitJob(name, spec, selected.trial, selected.hyperparameters, selected.fitted_iterations)
    directory = refit_directory(run.path, name, spec)
    fit_written: dict[str, str] | None = None
    with log_step(logger, f"Refit {spec.tag} of {name}"):
        try:
            if spec.fresh_process:
                if fresh is None:
                    raise SpecificationError(f"{name}: the {spec.kind} needs a fresh process")
                seconds = fresh(job, directory)
            else:
                seconds, _ = fit_refit(search, subject, job, directory)
            model = load_selected(
                scoring, subject, selected, directory=directory, seed=spec.seed, device=spec.device
            )
            files = (
                run_file(run, directory / MODEL_FILES[subject.model.name]),
                run_file(run, directory / MODEL_RECORD_NAME),
            )
            fit = run.write_record(
                fit_record(
                    search,
                    subject,
                    selected.trial,
                    model=model,
                    files=files,
                    seconds=seconds,
                    error=None,
                    purpose=spec.kind,
                    seed=spec.seed,
                    device=spec.device,
                    length=job.length,
                    fresh_process=spec.fresh_process,
                ),
                name,
                spec.tag,
            )
            fit_written = run_file(run, fit)
            scored = score_model(
                scoring,
                subject,
                run_file(run, fit),
                model,
                "V2",
                role=spec.kind,
                seed=spec.seed,
                tag=spec.tag,
            )
            evaluate_partition(scoring, scored)
            comparison = None
            if spec.kind != SEED_REFIT:
                comparison = compare_scores(
                    read_scores(run, reference.file, reference.rows),
                    read_scores(run, scored.file, scored.rows),
                )
            run.write_record(
                _refit_record(
                    scoring,
                    subject,
                    selected,
                    spec,
                    fit=run_file(run, fit),
                    scored=scored,
                    reference=reference,
                    comparison=comparison,
                    error=None,
                ),
                name,
                spec.tag,
            )
        except ReviewCapacityError as error:
            _record_failure(search, scoring, subject, selected, spec, reference, error, fit_written)
            raise
    return scored


def _record_failure(
    search: SearchContext,
    scoring: ScoringContext,
    subject: Subject,
    selected: TrialOutcome,
    spec: RefitSpec,
    reference: ScoredPartition,
    error: ReviewCapacityError,
    completed_fit: Mapping[str, str] | None,
) -> None:
    """Write the failed ``refit`` record of a refit that raised ``error``, and, when the fit
    itself did not complete (``completed_fit`` is None), its failed ``fit`` record. A fit that
    completed keeps its record: the failure is that of a later stage."""
    run, name = search.run, subject.name
    message = f"{type(error).__name__}: {error}"
    fit = completed_fit
    if fit is None:
        failed = fit_record(
            search,
            subject,
            selected.trial,
            model=None,
            files=None,
            seconds=None,
            error=message,
            purpose=spec.kind,
            seed=spec.seed,
            device=spec.device,
            length=selected.fitted_iterations,
            fresh_process=spec.fresh_process,
        )
        fit = run_file(run, run.write_record(failed, name, spec.tag))
    run.write_record(
        _refit_record(
            scoring,
            subject,
            selected,
            spec,
            fit=fit,
            scored=None,
            reference=reference,
            comparison=None,
            error=message,
        ),
        name,
        spec.tag,
    )


def device_comparison(
    scoring: ScoringContext,
    subject: Subject,
    gpu: ScoredPartition,
    cpu: ScoredPartition,
    refit: Mapping[str, str],
) -> Path:
    """Write the ``device_comparison`` record of a CatBoost configuration: the R3 point
    differences (CPU − GPU) at each capacity and the paired bootstrap interval of the
    Recall@2% difference over the frauds of V2 (DR-013 §7; DR-006)."""
    run, name = scoring.run, subject.name
    with log_step(logger, f"CPU–GPU comparison of {name}"):
        truth = truth_of(gpu.rows)
        queue = DailyQueue.build(gpu.rows.ids, truth.days)
        scores_gpu = read_scores(run, gpu.file, gpu.rows)
        scores_cpu = read_scores(run, cpu.file, cpu.rows)
        capacities = []
        primary = None
        for capacity in capacity_budgets(run.partitions.v2):
            on_gpu = r3_outcome(queue, truth, scores_gpu, capacity)
            on_cpu = r3_outcome(queue, truth, scores_cpu, capacity)
            differences = {}
            for metric, value in on_gpu.metrics.items():
                other = on_cpu.metrics[metric]
                if value is not None and other is not None:
                    differences[metric] = other - value
            capacities.append(
                {"capacity": capacity.value, "k": capacity.k, "differences": differences}
            )
            if capacity.value == PRIMARY_CAPACITY:
                primary = capacity
        if primary is None:
            raise SpecificationError(f"The capacities have no {PRIMARY_CAPACITY:.0%} scenario")
        fraud = truth.labels == 1
        captured_gpu = queue.select(scores_gpu, primary.k).mask[fraud]
        captured_cpu = queue.select(scores_cpu, primary.k).mask[fraud]
        keys = pseudo_entity_keys(gpu.rows.features, truth.days, scoring.features.missing_label)
        entities = entity_indices(
            [key for key, is_fraud in zip(keys, fraud, strict=True) if is_fraud]
        )
        interval = paired_difference(
            captured_cpu,
            captured_gpu,
            truth.day_index[fraud],
            len(truth.day_values),
            entities,
            replicates=REPLICATES,
            seed=BOOTSTRAP_SEED,
            level=LEVEL,
        )
        record = {
            "record": "device_comparison",
            "schema_version": 1,
            "identity": dict(scoring.identity),
            "configuration": name,
            "seed": scoring.seed,
            "gpu_score": dict(gpu.record),
            "cpu_score": dict(cpu.record),
            "refit": dict(refit),
            "capacities": capacities,
            "bootstrap": {
                "capacity": primary.value,
                "k": primary.k,
                "entity_key": "fine",
                "estimate": interval.estimate,
                "lower": interval.lower,
                "upper": interval.upper,
                "level": interval.level,
                "replicates": interval.replicates,
                "undefined_replicates": interval.undefined_replicates,
                "seed": interval.seed,
                "days": interval.days,
                "entities": interval.entities,
                "frauds": interval.frauds,
            },
        }
        return run.write_record(record, name)


def run_refits(
    search: SearchContext,
    scoring: ScoringContext,
    subject: Subject,
    selected: TrialOutcome,
    reference: ScoredPartition,
    *,
    fresh: Callable[[RefitJob, Path], float],
) -> dict[str, ScoredPartition]:
    """All refits of one configuration, then, for CatBoost, the CPU–GPU comparison; returns the
    scored V2 of each refit by tag. ``reference`` is the selected model's scored V2."""
    seeds = scoring.identity["seeds"]
    device: Device = scoring.settings.training_device(subject.model.name)
    specs = refit_specs(subject, device, seeds["main"], tuple(seeds["additional"]))
    scored: dict[str, ScoredPartition] = {}
    with log_step(logger, f"Refits of the selected {subject.name} ({len(specs)})"):
        for spec in specs:
            scored[spec.tag] = run_refit(
                search, scoring, subject, selected, spec, reference, fresh=fresh
            )
        if CPU_REFIT in scored:
            refit_path = search.run.record_path("refit", subject.name, CPU_REFIT)
            device_comparison(
                scoring, subject, reference, scored[CPU_REFIT], run_file(search.run, refit_path)
            )
    return scored
