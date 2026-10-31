"""The single scoring pass on stub models and synthetic Test inputs (Issue #203)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from review_capacity.confirmatory.inputs import TestInputs
from review_capacity.confirmatory.records import RECORD_CONTRACTS, ConfirmatoryRecords
from review_capacity.confirmatory.scoring import (
    attempt_directory,
    attempt_numbers,
    attempt_started,
    read_bundle,
    score_test_models,
    start_attempt,
    write_bundle,
)
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.runner import chosen_configurations
from review_capacity.errors import ModelError, SpecificationError
from review_capacity.hashing import file_sha256
from review_capacity.models.run_record import load_settings_with_snapshot
from review_capacity.training.timing import Clock
from support import confirmatory as c
from support.synthetic import FEATURES, REPO_PATHS

ROWS = 6
SUBJECTS = chosen_configurations(["LR", "LGB-u"])


class Stub:
    """A frozen model that scores by the row position, once per call."""

    calls = 0

    def __init__(self, count: int = ROWS) -> None:
        self.count = count

    def predict(self, _frame: pd.DataFrame) -> np.ndarray:
        Stub.calls += 1
        return np.linspace(0.01, 0.99, self.count)


def inputs() -> TestInputs:
    ids = np.arange(9_000, 9_000 + ROWS, dtype=np.int64)
    frame = pd.DataFrame({name: np.zeros(ROWS) for name in FEATURES.model_inputs})
    return TestInputs(
        frame, ids, ids * 10, np.array([155, 155, 156, 156, 157, 157], dtype=np.int64), "e" * 64
    )


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    Stub.calls = 0
    models = {}
    for subject in SUBJECTS:
        path = tmp_path / "models" / subject.name / "model.bin"
        path.parent.mkdir(parents=True)
        path.write_text(subject.name, encoding="utf-8")
        models[subject.name] = {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": file_sha256(path),
        }
    loaded: list[str] = []

    def load(subject: Any, _directory: Path, **_kwargs: Any) -> Stub:
        loaded.append(subject.name)
        return Stub()

    monkeypatch.setattr("review_capacity.confirmatory.scoring.load_saved_model", load)
    run = RunDirectory.create(
        tmp_path, None, "confirmatory", c.RUN_ID, records=ConfirmatoryRecords(c.context())
    )
    settings, _ = load_settings_with_snapshot(REPO_PATHS)
    return {"root": tmp_path, "models": models, "run": run, "settings": settings, "loaded": loaded}


def score(world: dict[str, Any], **changes: Any) -> list[dict[str, Any]]:
    options: dict[str, Any] = {
        "run": world["run"],
        "attempt": 1,
        "root": world["root"],
        "models": world["models"],
        "subjects": SUBJECTS,
        "settings": world["settings"],
        "features": FEATURES,
        "seed": 42,
        "reference_fixed": {},
        "artifact": None,
        "inputs": inputs(),
    }
    return score_test_models(**{**options, **changes})


def test_each_frozen_model_scores_the_test_rows_once(world: dict[str, Any]) -> None:
    bodies = score(world)
    assert [b["configuration"] for b in bodies] == ["LR", "LGB-u"]
    assert world["loaded"] == ["LR", "LGB-u"]
    assert Stub.calls == 2  # one call for each model, never repeated
    for body in bodies:
        assert body["rows"] == ROWS
        assert body["days"] == {"first": 155, "last": 157}
        assert body["timing"]["passes"] == 1
        assert body["model"] == world["models"][body["configuration"]]
        record = {**c.envelope("test_scores"), **body}
        assert RECORD_CONTRACTS["test_scores"].problems(record) == []
        path = world["run"].path / body["file"]["path"]
        assert path.parent == attempt_directory(world["run"], 1)
        assert file_sha256(path) == body["file"]["sha256"]
        frame = pd.read_parquet(path)
        assert list(frame.columns) == ["TransactionID", "score"]
        assert frame["TransactionID"].tolist() == inputs().ids.tolist()
    assert bodies[0]["distinct_scores"] == ROWS
    assert bodies[0]["device"] == "cpu"


def test_a_repeat_writes_its_own_attempt_directory(world: dict[str, Any]) -> None:
    first, second = score(world, attempt=1), score(world, attempt=2)
    assert first[0]["file"]["path"] != second[0]["file"]["path"]
    assert first[0]["file"]["sha256"] == second[0]["file"]["sha256"]  # the same scores


def test_a_model_file_replaced_after_the_binding_is_refused(world: dict[str, Any]) -> None:
    path = world["root"] / world["models"]["LR"]["path"]
    path.write_text("another model", encoding="utf-8")
    with pytest.raises(SpecificationError, match="differs from the frozen model"):
        score(world)
    assert world["loaded"] == []  # nothing was loaded


def test_a_model_the_frozen_inputs_do_not_name_is_refused(world: dict[str, Any]) -> None:
    models = {k: v for k, v in world["models"].items() if k != "LGB-u"}
    with pytest.raises(SpecificationError, match="name no model"):
        score(world, models=models)


def test_scores_for_other_rows_are_refused(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "review_capacity.confirmatory.scoring.load_saved_model",
        lambda *_a, **_k: Stub(ROWS - 1),
    )
    with pytest.raises(ModelError, match="shape"):
        score(world)


def test_the_recorded_time_is_the_end_to_end_pass_and_neural_models_synchronise(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """DR-015 §9: the clock starts after and stops before an explicit CUDA synchronisation for the
    neural models only; the pass is timed once and the scoring call is the timed work."""
    events: list[str] = []

    class Recording(Clock):
        def __init__(self, label: str) -> None:
            super().__init__(
                now=lambda: float(len(events)),
                synchronise=lambda: events.append(f"sync-{label}"),
            )

    monkeypatch.setattr(
        "review_capacity.confirmatory.scoring.gpu_clock", lambda _c: Recording("gpu")
    )
    monkeypatch.setattr("review_capacity.confirmatory.scoring.Clock", lambda: Recording("cpu"))
    subjects = chosen_configurations(["RES-u", "LGB-u"])
    models = {}
    for subject in subjects:
        path = world["root"] / "models" / subject.name / "model.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(subject.name, encoding="utf-8")
        models[subject.name] = {
            "path": path.relative_to(world["root"]).as_posix(),
            "sha256": file_sha256(path),
        }
    released: list[str] = []
    bodies = score(world, subjects=subjects, models=models, release=lambda: released.append("r"))
    assert events == ["sync-cpu", "sync-cpu", "sync-gpu", "sync-gpu"]  # LGB-u first (DR-011 order)
    assert released == ["r"]  # the GPU is released after the neural model only
    assert [b["timing"]["passes"] for b in bodies] == [1, 1]
    assert all(b["timing"]["seconds"] > 0 for b in bodies)


def test_an_attempt_is_allocated_durably_and_a_bundle_is_verified(world: dict[str, Any]) -> None:
    run = world["run"]
    assert attempt_numbers(run) == []
    start_attempt(run, 1, "2026-11-02T09:00:00.000000Z")
    with pytest.raises(FileExistsError):
        start_attempt(run, 1, "2026-11-02T09:00:00.000000Z")  # never reused or overwritten
    assert attempt_numbers(run) == [1]
    assert attempt_started(run, 1) == "2026-11-02T09:00:00.000000Z"
    bodies = score(world, attempt=2)
    names = [b["configuration"] for b in bodies]
    assert read_bundle(run, 2, names) is None  # no bundle yet
    write_bundle(run, 2, bodies)
    assert read_bundle(run, 2, names) == bodies
    assert read_bundle(run, 2, ["LR"]) is None  # another set of models
    (run.path / bodies[0]["file"]["path"]).write_text("changed", encoding="utf-8")
    assert read_bundle(run, 2, names) is None  # a score file that is not the one it names
