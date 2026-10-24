"""A synthetic closed development run and the setup of an analysis run over it (no model is
trained and no project data is read)."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from review_capacity.config import load_dataset_spec
from review_capacity.development.analysis import AnalysisSetup, check_stages
from review_capacity.development.evaluation import truth_of
from review_capacity.development.inputs import PartitionRows, ScoringInputs
from review_capacity.development.records import CONFIGURATION_NAMES
from review_capacity.development.run_output import RunDirectory, load_run
from review_capacity.development.scoring import SCORE_COLUMNS, SCORES_DIR
from review_capacity.development.trial import SearchContext, run_file
from review_capacity.evaluation.queue import DailyQueue
from review_capacity.evaluation.ranking import (
    Subject,
    capacity_budgets,
    r3_outcome,
    r3_record,
    ranking_outcome,
    ranking_record,
)
from review_capacity.hashing import file_sha256
from review_capacity.models.run_record import load_settings_with_snapshot
from review_capacity.paths import ProjectPaths
from support.development import IDENTITY, example, file
from support.search import (
    V1_ROWS,
    V2_ROWS,
    partition,
    search_declaration,
    synthetic_partitions,
)
from support.synthetic import FEATURES, REPO_PATHS

PARTITIONS = synthetic_partitions()
SOURCE_ID = "2026-10-20T091530Z"
ANALYSIS_ID = "2026-10-21T091530Z"


def source_reference(source_path: Path) -> dict[str, object]:
    """The ``source`` field of an analysis ``run`` record: the closing record of the source."""
    end = source_path / "records" / "run_end.json"
    return {
        "run_id": SOURCE_ID,
        "run_end": {
            "path": f"outputs/development/{SOURCE_ID}/records/run_end.json",
            "sha256": file_sha256(end),
        },
    }


def inputs() -> ScoringInputs:
    return ScoringInputs(
        partition("v1", V1_ROWS, 2, 2_000_000), partition("v2", V2_ROWS, 3, 3_000_000)
    )


def scores_of(rows: PartitionRows, configuration: str, winner: str | None) -> np.ndarray:
    """Random scores that carry some of the labels' signal, the ``winner``'s more of it."""
    rng = np.random.default_rng(zlib.crc32(configuration.encode()))  # stable across processes
    signal = 0.6 if configuration == winner else 0.25
    return np.clip(0.5 * rng.random(rows.rows) + signal * rows.labels, 0.0, 1.0)


def build_source(
    root: Path, names: Sequence[str] = CONFIGURATION_NAMES, winner: str | None = "Cat-b"
) -> tuple[Path, ScoringInputs]:
    """A closed, completed development run with, for each of ``names``, the V1 and V2 scores
    and ``score`` records of its selected model and its V2 R3 and ranking evaluations."""
    run = RunDirectory.create(root, PARTITIONS, "development", SOURCE_ID)
    run.write_record(example("run") | {"configurations": list(names)})
    data = inputs()
    budgets = capacity_budgets(PARTITIONS.v2)
    for name in names:
        for label, rows in (("V1", data.v1), ("V2", data.v2)):
            scores = scores_of(rows, name, winner)
            path = run.path / SCORES_DIR / f"{name}-selected-{label}.parquet"
            path.parent.mkdir(exist_ok=True)
            frame = pd.DataFrame({SCORE_COLUMNS[0]: rows.ids, SCORE_COLUMNS[1]: scores})
            frame.to_parquet(path, index=False)
            days = rows.times // 86_400
            record = example("score") | {
                "configuration": name,
                "fit": file(f"fits/{name}/trial-01.json"),
                "partition": label,
                "first_day": int(days.min()),
                "last_day": int(days.max()),
                "rows": rows.rows,
                "file": run_file(run, path),
                "columns": list(SCORE_COLUMNS),
                "score_dtype": "float64",
                "device": "cpu",
                "batch_rows": None,
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
                "distinct_scores": len(np.unique(scores)),
            }
            reference = run_file(run, run.write_record(record, name, "selected", label))
            if label == "V2":
                truth = truth_of(rows)
                queue = DailyQueue.build(rows.ids, truth.days)
                subject = Subject(name, "selected", 42, reference, "V2")
                for capacity in budgets:
                    outcome = r3_outcome(queue, truth, scores, capacity)
                    part = f"R3-c{round(capacity.value * 100):02d}"
                    run.write_record(
                        r3_record(IDENTITY, subject, truth, capacity, outcome),
                        name,
                        "selected",
                        "V2",
                        part,
                    )
                ranking = ranking_record(IDENTITY, subject, truth, ranking_outcome(truth, scores))
                run.write_record(ranking, name, "selected", "V2", "ranking")
    run.finish("completed")
    return run.path, data


def build_setup(
    root: Path,
    stages: Sequence[str],
    names: Sequence[str] = CONFIGURATION_NAMES,
    winner: str | None = "Cat-b",
) -> AnalysisSetup:
    """The setup of an analysis run over a new synthetic source run under ``root``."""
    source_path, data = build_source(root, names, winner)
    identity = IDENTITY | {"run_id": ANALYSIS_ID, "run_type": "analysis"}
    run = RunDirectory.create(root, PARTITIONS, "analysis", ANALYSIS_ID, source=source_path)
    source = load_run(source_path, PARTITIONS)
    settings, _ = load_settings_with_snapshot(REPO_PATHS)
    reference = source_reference(source_path)
    record = example("run") | {
        "identity": identity,
        "configurations": list(names),
        "source": reference,
    }
    load_dataset_spec(REPO_PATHS)  # the real declaration must stay loadable beside the fixtures
    return AnalysisSetup(
        run=run,
        identity=identity,
        run_record=record,
        source=source,
        declaration=search_declaration(),
        settings=settings,
        features=FEATURES,
        scoring_inputs=data,
        configurations=tuple(names),
        stages=check_stages(stages),
        paths=ProjectPaths(root),
    )


def build_searched_source(root: Path) -> tuple[Path, ScoringInputs, SearchContext]:
    """A closed development run of ``LGB-u`` made by the real search and scoring code on the
    synthetic partitions (models saved, scores and R3 evaluated), for the refits."""
    from review_capacity.development.evaluation import evaluate_partition
    from review_capacity.development.scoring import load_selected, score_partition
    from review_capacity.development.search import search_configuration
    from review_capacity.models.registry import configuration
    from support.search import (
        candidate_file,
        candidate_list,
        scoring_context,
        search_context,
    )

    lgb = configuration("LGB-u")
    search = search_context(root)
    result = search_configuration(search, lgb, candidate_list(), candidate_file(search.run))
    scoring = scoring_context(search)
    model = load_selected(scoring, lgb, result.selected)
    for label in ("V1", "V2"):
        scored = score_partition(scoring, lgb, result.selected, model, label)
        evaluate_partition(scoring, scored)
    search.run.finish("completed")
    return search.run.path, scoring.inputs, search


def build_refit_setup(root: Path) -> AnalysisSetup:
    """The setup of an analysis run of the refits stage over a really searched source run."""
    from dataclasses import replace

    from review_capacity.development.analysis import refit_contexts
    from review_capacity.development.refits import RefitJob, fit_refit
    from review_capacity.models.registry import configuration

    source_path, data, search = build_searched_source(root)
    identity = IDENTITY | {"run_id": ANALYSIS_ID, "run_type": "analysis"}
    run = RunDirectory.create(root, PARTITIONS, "analysis", ANALYSIS_ID, source=source_path)
    settings, snapshot = load_settings_with_snapshot(REPO_PATHS)
    record = example("run") | {
        "identity": identity,
        "configurations": ["LGB-u"],
        "source": source_reference(source_path),
    }
    setup = AnalysisSetup(
        run=run,
        identity=identity,
        run_record=record,
        source=load_run(source_path, PARTITIONS),
        declaration=search.declaration,
        settings=settings,
        features=FEATURES,
        scoring_inputs=data,
        configurations=("LGB-u",),
        stages=check_stages(["refits"]),
        paths=ProjectPaths(root),
        settings_snapshot=snapshot,
        environment=search.environment,
        code=search.code,
        search_inputs=search.inputs,
        lightgbm_build=search.lightgbm_build,
        session=search.session,
    )

    def fresh(job: RefitJob, directory: Path) -> float:  # stands in for the child process
        context = refit_contexts(setup)[0]
        return fit_refit(context, configuration(job.configuration), job, directory)[0]

    return replace(setup, fresh=fresh)
