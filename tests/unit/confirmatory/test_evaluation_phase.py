"""The evaluation phase up to the outcome records, with the heavy steps stubbed (Issue #205): the
verification of the saved scores, the order of the label-free work, the consumption marker and
the first label, and the refusals. No research data and no real model is read."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from review_capacity.config import FeatureSpec, load_dataset_spec
from review_capacity.confirmatory import evaluation as phase
from review_capacity.confirmatory.evaluation import (
    EvaluationSetup,
    evaluate_outcomes,
    verify_saved_scores,
)
from review_capacity.confirmatory.frozen import FrozenBinding
from review_capacity.confirmatory.frozen_policies import FrozenPolicies
from review_capacity.confirmatory.inputs import TestInputs
from review_capacity.confirmatory.outcome_records import publish_outcomes
from review_capacity.confirmatory.records import ConfirmatoryRecords, scores_bundle_sha256
from review_capacity.confirmatory.state import begin_consumption, consume, locked, marker_path
from review_capacity.development.records import ALL_CONFIGURATIONS
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.trial import run_file
from review_capacity.errors import DataValidationError, SpecificationError
from review_capacity.paths import ProjectPaths
from support import confirmatory as c
from support.synthetic import REPO_PATHS

CONTEXT = c.context()
ROWS = CONTEXT.test.rows
STATE = c.identity()["frozen_state_sha256"]


def inputs_for(ids: np.ndarray) -> TestInputs:
    sizes = np.full(28, ROWS // 28)
    sizes[: ROWS - sizes.sum()] += 1
    days = np.repeat(np.arange(155, 183), sizes).astype(np.int64)
    frame = pd.DataFrame({"TransactionAmt": np.linspace(1.0, 500.0, ROWS)})
    return TestInputs(frame, ids, days * 86_400, days, "e" * 64, np.zeros(ROWS, dtype=np.bool_))


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    paths = ProjectPaths(tmp_path)
    run = RunDirectory.create(
        tmp_path, None, "confirmatory", c.RUN_ID, records=ConfirmatoryRecords(CONTEXT)
    )
    run.write_record(c.test_run())
    ids = np.arange(5_000_000, 5_000_000 + ROWS, dtype=np.int64)
    rng = np.random.default_rng(1)
    hashes = []
    for name in ALL_CONFIGURATIONS:
        target = run.path / "scores" / "attempt-01" / f"{name}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"TransactionID": ids, "score": rng.random(ROWS)}).to_parquet(
            target, index=False
        )
        record = {**c.scores(name), "file": run_file(run, target)}
        run.write_record(record, name)
        hashes.append(record["file"]["sha256"])
    run.write_record(c.phase("scoring", 1, "completed", scores_bundle_sha256(hashes)), "scoring", 1)
    events: list[str] = []

    def read_raw(*_a: Any, **_k: Any) -> tuple[TestInputs, str]:
        events.append("features")
        return inputs_for(world_ids["ids"]), "raw"

    def load_labels(*_a: Any, **_k: Any) -> pd.Series:
        events.append("labels")
        return pd.Series((np.arange(ROWS) % 30 == 0).astype(np.int8))

    def consumption(*args: Any, **kwargs: Any) -> Path:
        events.append("consume")
        return begin_consumption(*args, **kwargs)

    world_ids = {"ids": ids}

    def tick(event: str, value: Any) -> Any:
        events.append(event)
        return value

    for name, value in {
        "read_test_raw": read_raw,
        "load_labels": load_labels,
        "begin_consumption": consumption,
        "load_policies": lambda *_a, **_k: tick("policies", FrozenPolicies({}, {}, {})),
        "read_partitions": lambda *_a, **_k: tick(
            "development", {"train": SimpleNamespace(), "v2": SimpleNamespace()}
        ),
        "build_slice_inputs": lambda *_a, **_k: tick("slices", c.outcome_slices()),
        "build_outcome_records": lambda *_a, **_k: tick("build", []),
        "publish_outcomes": lambda *_a, **_k: tick("publish", 0),
    }.items():
        monkeypatch.setattr(phase, name, value)
    binding = FrozenBinding(c.frozen_inputs(), c.identity()["training"], "a" * 64, STATE)
    setup = EvaluationSetup(
        paths,
        run,
        binding,
        c.declaration(),
        load_dataset_spec(REPO_PATHS),
        cast(FeatureSpec, SimpleNamespace(missing_label="__MISSING__")),
    )
    return {"setup": setup, "events": events, "ids": world_ids, "root": tmp_path, "run": run}


def test_the_saved_scores_are_verified_by_hash_and_bundle(world: dict[str, Any]) -> None:
    saved = verify_saved_scores(world["run"], world["setup"].binding)
    assert set(saved.scores) == set(ALL_CONFIGURATIONS)
    assert all(len(values) == ROWS for values in saved.scores.values())
    assert saved.bundle_sha256 == scores_bundle_sha256(list(saved.files.values()))
    assert all(np.array_equal(ids, world["ids"]["ids"]) for ids in saved.ids.values())


def test_a_changed_score_file_is_refused_before_anything_is_read(world: dict[str, Any]) -> None:
    path = world["run"].path / "scores" / "attempt-01" / "LGB-b.parquet"
    pd.DataFrame({"TransactionID": world["ids"]["ids"], "score": np.zeros(ROWS)}).to_parquet(
        path, index=False
    )
    with (
        locked(world["root"], STATE, run_id="r"),
        pytest.raises(SpecificationError, match="differs from its record"),
    ):
        evaluate_outcomes(world["setup"])
    assert world["events"] == []
    assert not marker_path(world["root"], STATE).exists()


def test_the_consumption_marker_comes_after_the_label_free_work_and_before_the_labels(
    world: dict[str, Any],
) -> None:
    with locked(world["root"], STATE, run_id="r"):
        evaluate_outcomes(world["setup"])
    assert world["events"] == [
        "features",
        "policies",
        "development",
        "consume",
        "labels",
        "slices",
        "build",
        "publish",
    ]
    marker = json.loads(marker_path(world["root"], STATE).read_text(encoding="utf-8"))
    assert marker["run_id"] == c.RUN_ID
    assert (
        marker["scores_sha256"]
        == verify_saved_scores(world["run"], world["setup"].binding).bundle_sha256
    )


def test_the_lock_must_be_held(world: dict[str, Any]) -> None:
    with pytest.raises(SpecificationError, match="lock of the frozen state"):
        evaluate_outcomes(world["setup"])
    assert world["events"] == []


def test_a_marker_of_another_run_refuses_the_evaluation_before_a_label_is_read(
    world: dict[str, Any],
) -> None:
    consume(world["root"], STATE, run_id="another-run", scores_sha256="c" * 64)
    with (
        locked(world["root"], STATE, run_id="r"),
        pytest.raises(SpecificationError, match="another run or other scores"),
    ):
        evaluate_outcomes(world["setup"])
    assert "labels" not in world["events"]


def test_a_resumed_evaluation_of_the_run_that_consumed_it_is_accepted(
    world: dict[str, Any],
) -> None:
    with locked(world["root"], STATE, run_id="r"):
        evaluate_outcomes(world["setup"])
    world["events"].clear()
    with locked(world["root"], STATE, run_id="r2"):
        evaluate_outcomes(world["setup"])  # the same run and the same scores: a resume
    assert world["events"].count("labels") == 1


def test_scores_for_other_rows_stop_the_phase_before_the_marker(world: dict[str, Any]) -> None:
    world["ids"]["ids"] = world["ids"]["ids"] + 1  # the Test rows now have other identifiers
    with (
        locked(world["root"], STATE, run_id="r"),
        pytest.raises(DataValidationError, match="identifier sequence differs"),
    ):
        evaluate_outcomes(world["setup"])
    assert "labels" not in world["events"]
    assert not marker_path(world["root"], STATE).exists()  # no label was reached: not consumed


def test_the_result_keeps_what_the_inference_needs(world: dict[str, Any]) -> None:
    with locked(world["root"], STATE, run_id="r"):
        result = evaluate_outcomes(world["setup"])
    assert result.written == 0
    assert result.partition.truth.rows == ROWS
    assert set(result.saved.scores) == set(ALL_CONFIGURATIONS)
    assert result.slices.boundaries == c.outcome_slices().boundaries  # the stubbed slice inputs


def test_an_evaluation_interrupted_while_publishing_resumes_through_the_entry_point(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outcome records without a completed evaluation do not stop the verification of the scoring
    records; the publication then completes the missing ones and keeps the written one."""
    named = [((n,), c.workload(n)) for n in ("LGB-b", "Cat-b", "RES-u")]
    monkeypatch.setattr(phase, "build_outcome_records", lambda *_a, **_k: named)
    real_publish = publish_outcomes

    def interrupted(run: RunDirectory, records: Any) -> int:
        written = real_publish(run, records[:1])  # one record written, then the process ends
        raise OSError("the disk is full") if written else RuntimeError("unreachable")

    monkeypatch.setattr(phase, "publish_outcomes", interrupted)
    with locked(world["root"], STATE, run_id="r"), pytest.raises(OSError, match="disk is full"):
        evaluate_outcomes(world["setup"])
    assert len(list(world["run"].records_dir.glob("test_workload-*.json"))) == 1
    monkeypatch.setattr(phase, "publish_outcomes", real_publish)
    with locked(world["root"], STATE, run_id="r2"):
        result = evaluate_outcomes(world["setup"])
    assert result.written == 2  # the missing two; the first was kept
    assert len(list(world["run"].records_dir.glob("test_workload-*.json"))) == 3


def test_a_record_of_another_identity_in_the_run_is_refused(world: dict[str, Any]) -> None:
    source = world["run"].records_dir / "test_scores-LR.json"
    other = json.loads(source.read_text(encoding="utf-8"))
    other["identity"]["run_id"] = "2026-12-01T000000Z"
    other["configuration"] = "LR"
    source.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(SpecificationError, match="another identity"):
        verify_saved_scores(world["run"], world["setup"].binding)
