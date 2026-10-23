"""The neural inputs of a development run: the shared artifact, the network inputs, the batch plans
and the ``preprocessing`` and ``preflight`` records (Issue #151, Part 3), on synthetic data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from review_capacity.development.inputs import ScoringInputs
from review_capacity.development.neural_run import (
    NeuralRun,
    build_neural_run,
    neural_models_of,
    preflight_identity,
    preflight_records,
    write_neural_records,
)
from review_capacity.development.records import RECORD_CONTRACTS
from review_capacity.development.run_output import load_run
from review_capacity.errors import ModelError
from review_capacity.features.neural_preprocessing import fit_neural_preprocessing
from review_capacity.hashing import file_sha256
from review_capacity.models.registry import REFERENCE_SUBJECT, Weighting, configuration
from review_capacity.paths import ProjectPaths
from support.development import IDENTITY, file
from support.search import (
    TRAIN_ROWS,
    V1_ROWS,
    V2_ROWS,
    partition,
    search_context,
    search_declaration,
    search_inputs,
)
from support.synthetic import FEATURES

IDENTITY_ARGS: dict[str, Any] = {
    "declaration_sha256": "d" * 64,
    "settings_sha256": "e" * 64,
    "code": {"package": {"source_sha256": "s" * 64}},
    "environment": {"gpu": {"name": "Test GPU"}},
}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any, Any, Any]:
    """The artifact fitted on the synthetic Train, saved, with the search and scoring inputs."""
    search = search_inputs()
    scoring = ScoringInputs(
        partition("v1", V1_ROWS, 2, 2_000_000), partition("v2", V2_ROWS, 3, 3_000_000)
    )
    train = search.train
    artifact = fit_neural_preprocessing(train.features, train.ids, train.times, FEATURES, 0)
    directory = tmp_path_factory.mktemp("project") / "artifact"
    artifact.save(directory)
    return directory, artifact, search, scoring


def neural_run(prepared: tuple[Path, Any, Any, Any]) -> NeuralRun:
    directory, _, search, scoring = prepared
    return build_neural_run(
        FEATURES,
        search_declaration(),
        search,
        scoring,
        directory,
        preflight=None,
        candidates={},
        models=(),
        **IDENTITY_ARGS,
    )


def test_the_inputs_of_every_partition_are_built_once_from_the_artifact(
    prepared: tuple[Path, Any, Any, Any],
) -> None:
    run = neural_run(prepared)
    layout = run.artifact.layout
    assert run.train[Weighting.UNWEIGHTED].inputs is run.train[Weighting.BALANCED].inputs
    assert run.train[Weighting.UNWEIGHTED].inputs.rows == TRAIN_ROWS
    assert run.v1.inputs is run.v1_inputs
    assert run.v1_inputs.rows == V1_ROWS
    assert run.v2_inputs.rows == V2_ROWS
    assert run.scoring_inputs("V2") is run.v2_inputs
    for inputs in (run.v1_inputs, run.v2_inputs, run.train[Weighting.UNWEIGHTED].inputs):
        inputs.check(layout)
    assert run.plans == {}
    assert run.preflight_path is None


def test_the_weightings_differ_in_weights_only_and_v1_carries_its_metadata(
    prepared: tuple[Path, Any, Any, Any],
) -> None:
    run = neural_run(prepared)
    unweighted = run.train[Weighting.UNWEIGHTED]
    balanced = run.train[Weighting.BALANCED]
    assert (unweighted.weights == 1.0).all()
    assert balanced.weights.max() > 1.0
    assert np.array_equal(unweighted.labels, balanced.labels)
    assert run.v1.ids is not None
    assert run.v1.times is not None
    assert (run.v1.weights == 1.0).all()


def test_a_neural_model_needs_the_preflight(prepared: tuple[Path, Any, Any, Any]) -> None:
    directory, _, search, scoring = prepared
    with pytest.raises(ModelError, match="--preflight"):
        build_neural_run(
            FEATURES,
            search_declaration(),
            search,
            scoring,
            directory,
            preflight=None,
            candidates={},
            models=("resnet",),
            **IDENTITY_ARGS,
        )


def test_the_neural_models_of_a_run_are_listed_in_dr011_order() -> None:
    chosen = [configuration("FTT-b"), configuration("RES-u"), configuration("LGB-u")]
    assert neural_models_of(chosen) == ("resnet", "ft_transformer")
    assert neural_models_of([configuration("Cat-u")]) == ()
    assert neural_models_of([REFERENCE_SUBJECT]) == ()


def test_the_preflight_identity_names_the_artifact_the_declaration_the_source_and_the_gpu() -> None:
    code = {"package": {"source_sha256": "s" * 64}}
    environment = {"gpu": {"name": "Test GPU"}}
    assert preflight_identity("a" * 64, "d" * 64, "e" * 64, code, environment) == {
        "artifact_sha256": "a" * 64,
        "declaration_sha256": "d" * 64,
        "settings_sha256": "e" * 64,
        "source_sha256": "s" * 64,
        "gpu_name": "Test GPU",
    }
    assert preflight_identity("a", "d", "e", code, {})["gpu_name"] is None


# --- records -------------------------------------------------------------------------------------


def phase(
    *, error: str | None = None, free: int | None = 3000, passes: bool = True
) -> dict[str, Any]:
    return {
        "completed": error is None,
        "error": error,
        "baseline_used_mib": 1000,
        "sampled_max_used_mib": None if free is None else 8000 - free,
        "total_mib": 8000,
        "free_at_peak_mib": free,
        "samples": 10,
        "problems": [],
        "passes": passes,
    }


def preflight_json() -> dict[str, Any]:
    item = {
        "trial": 1,
        "candidate": {},
        "candidate_sha256": "c" * 64,
        "micro_batch": 512,
        "attempts": [
            {
                "micro_batch": 1024,
                "training": phase(error="out of GPU memory", free=None, passes=False),
            },
            {"micro_batch": 512, "training": phase()},
        ],
        "evaluation": phase(free=2700),
    }
    return {
        "margin_mib": 1024,
        "models": {
            "ft_transformer": {
                "effective_batch": 1024,
                "evaluation_batch": 16,
                "candidates": [item],
            }
        },
    }


def test_each_candidate_gets_a_preflight_record_that_satisfies_its_contract() -> None:
    records = preflight_records(
        preflight_json(), IDENTITY, {"ft_transformer": file("candidates/ft_transformer.json")},
        ("ft_transformer",),
    )  # fmt: skip
    record = records[("ft_transformer", 1)]
    RECORD_CONTRACTS["preflight"].validate(record)
    assert [(a["phase"], a["rows"], a["outcome"]) for a in record["attempts"]] == [
        ("training", 1024, "out_of_memory"),
        ("training", 512, "pass"),
        ("evaluation", 16, "pass"),
    ]
    assert record["micro_batch"] == 512


def test_an_attempt_that_failed_without_a_memory_shortfall_cannot_be_recorded() -> None:
    recorded = preflight_json()
    attempt = recorded["models"]["ft_transformer"]["candidates"][0]["attempts"][1]["training"]
    attempt.update(passes=False, problems=["2 memory samples; at least 3 are required"])
    with pytest.raises(ModelError, match="without a memory shortfall"):
        preflight_records(
            recorded, IDENTITY, {"ft_transformer": file("candidates/ft_transformer.json")},
            ("ft_transformer",),
        )  # fmt: skip


def test_the_preprocessing_record_is_written_and_read_back(
    tmp_path: Path, prepared: tuple[Path, Any, Any, Any]
) -> None:
    directory, _, search, _ = prepared
    run = neural_run(prepared)
    context = search_context(tmp_path)
    paths = ProjectPaths(directory.parent)
    written = write_neural_records(context.run, paths, IDENTITY, run, search, {}, ())
    assert written.preprocessing["path"] == "records/preprocessing.json"
    assert written.preflights == {}
    loaded = load_run(context.run.path, context.run.partitions)
    [record] = loaded.of_kind("preprocessing")
    assert record["artifact"]["path"] == "artifact/manifest.json"
    manifest = directory / "manifest.json"
    assert record["artifact"]["sha256"] == file_sha256(manifest)  # the file's bytes
    assert record["artifact_sha256"] == run.artifact_sha256  # the artifact's identity
    assert record["artifact"]["sha256"] != record["artifact_sha256"]
    assert record["inputs"]["total"] == len(run.artifact.layout.numeric) + len(
        run.artifact.layout.categorical
    )
    assert record["train"]["rows"] == TRAIN_ROWS
