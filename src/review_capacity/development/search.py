"""The sequential search of one configuration and its selection (DR-015 §1 and §5; Issue #133).

The trials of a configuration run one after another from the recorded candidate list of its model
(trial 1 is the reference configuration). The selected trial has the highest V1 objective; trials
whose objectives are exactly equal are tied, and the lowest trial number among them is selected.
Two objectives are equal exactly when their integer numerators are, since all share the V1 frauds
as denominator.

Besides the trial files (``development.trial``), the search writes:

* a ``timing`` record of the whole loop (``training.timing.search_record``): an interrupted loop
  is recorded as partial, with the trials it completed, and the interrupt is raised again;
* the ``configuration`` record: every trial and the selected trial;
* a ``timing`` record of the selected trial's own fit time.

The search time covers the loop, the selection and the ``configuration`` record (DR-015 §9).
A trial that fails ends the search with its error (``development.trial``): no ``configuration``
record is written for a search that did not complete.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from review_capacity.development.cost_timing import timed_on
from review_capacity.development.trial import (
    SearchContext,
    TrialOutcome,
    run_file,
    run_trial,
)
from review_capacity.errors import SpecificationError
from review_capacity.logging_setup import ProgressLogger, log_step
from review_capacity.models.registry import Configuration
from review_capacity.training.sampler import CandidateList
from review_capacity.training.timing import Timed, fit_record, search_record

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigurationResult:
    """The search result of one configuration."""

    configuration: str
    trials: tuple[TrialOutcome, ...]
    selected: TrialOutcome
    tied: tuple[int, ...]
    record: dict[str, str]
    search_seconds: float


def select(trials: Sequence[TrialOutcome]) -> tuple[TrialOutcome, tuple[int, ...]]:
    """The selected trial and the trials tied with it: the highest V1 objective, compared
    exactly, and the lowest trial number among the tied (``trials`` in trial order)."""
    if not trials:
        raise SpecificationError("A selection needs at least one completed trial")
    if [t.trial for t in trials] != list(range(1, len(trials) + 1)):
        raise SpecificationError("Trials are numbered 1, 2, ... in trial order")
    best = max(t.objective for t in trials)
    tied = tuple(t.trial for t in trials if t.objective == best)
    return trials[tied[0] - 1], tied


def _timed(
    context: SearchContext,
    configuration: Configuration,
    role: str | None,
    subject: Mapping[str, str] | None = None,
) -> Timed:
    """Who is timed: the configuration on its training device (DR-013 §1)."""
    device = context.settings.training_device(configuration.model.name)
    return timed_on(context.settings, context.session, configuration, role, device, subject)


def _write_search_timing(
    context: SearchContext,
    configuration: Configuration,
    *,
    seconds: float,
    completed: int,
    interrupted: bool,
) -> None:
    record = search_record(
        context.identity,
        _timed(context, configuration, None),
        context.session,
        seconds=seconds,
        trials_completed=completed,
        trials_declared=context.declaration.trials,
        interrupted=interrupted,
    )
    context.run.write_record(record, configuration.name, "search")


def _configuration_record(
    context: SearchContext,
    configuration: Configuration,
    candidate_file: Mapping[str, str],
    trials: Sequence[TrialOutcome],
    selected: TrialOutcome,
    tied: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "record": "configuration",
        "schema_version": 1,
        "identity": dict(context.identity),
        "configuration": configuration.name,
        "candidate_list": dict(candidate_file),
        "trials": [
            {
                "trial": t.trial,
                "objective": t.objective,
                "stopping_reason": t.stopping_reason,
                "record": t.record,
            }
            for t in trials
        ],
        "selected_trial": selected.trial,
        "selected_objective": selected.objective,
        "tied_trials": list(tied),
        "selected_hyperparameters": dict(selected.hyperparameters),
        "selected_length": selected.best_checkpoint,
        "selected_stopping_reason": selected.stopping_reason,
        "selected_fit": selected.fit,
        "selected_checkpoint": selected.checkpoint,
    }


def search_configuration(
    context: SearchContext,
    configuration: Configuration,
    listed: CandidateList,
    candidate_file: Mapping[str, str],
) -> ConfigurationResult:
    """Run the declared trials of ``configuration`` from ``listed`` one after another, select
    the best and record the search.

    ``candidate_file`` is the run-directory reference of the recorded candidate list of the
    model (``{path, sha256}``); the two configurations of a model share it.
    """
    name, declared = configuration.name, context.declaration.trials
    if listed.model != configuration.model.name:
        raise SpecificationError(f"{name}: the candidate list is that of {listed.model}")
    if len(listed.candidates) != declared:
        raise SpecificationError(
            f"{name}: the candidate list has {len(listed.candidates)} candidates, "
            f"{declared} are declared"
        )
    outcomes: list[TrialOutcome] = []
    with log_step(logger, f"Search {name} ({declared} trials)"):
        progress = ProgressLogger(logger, f"{name} trials", declared)
        started = time.perf_counter()
        try:
            for number, candidate in enumerate(listed.candidates, start=1):
                trial = run_trial(context, configuration, number, candidate, candidate_file)
                outcomes.append(trial)
                progress.advance()
        except KeyboardInterrupt:
            if len(outcomes) < declared:
                _write_search_timing(
                    context,
                    configuration,
                    seconds=time.perf_counter() - started,
                    completed=len(outcomes),
                    interrupted=True,
                )
            raise
        selected, tied = select(outcomes)
        path = context.run.write_record(
            _configuration_record(context, configuration, candidate_file, outcomes, selected, tied),
            name,
        )
        seconds = time.perf_counter() - started  # trial set-up to recording and selection (§9)
        _write_search_timing(
            context, configuration, seconds=seconds, completed=len(outcomes), interrupted=False
        )
        context.run.write_record(
            fit_record(
                context.identity,
                _timed(context, configuration, "selected", selected.fit),
                context.session,
                seconds=selected.fit_seconds,
                completed_length=selected.stopped_at,
                retained_length=selected.best_checkpoint,
                startup_seconds=None,
                preprocessing_seconds=None,
            ),
            name,
            "fit",
        )
    logger.info(
        "%s selected: trial %d, V1 objective %.6f (tied trials %s)",
        name,
        selected.trial,
        selected.objective,
        list(tied),
    )
    return ConfigurationResult(
        name, tuple(outcomes), selected, tied, run_file(context.run, path), seconds
    )
