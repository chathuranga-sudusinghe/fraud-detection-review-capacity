"""Model-run records: the settings snapshot, JSON conversion and the record contents
(no data read; no model library involved)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from review_capacity.config import ModelSettings, load_model_settings
from review_capacity.models.base import ModelData
from review_capacity.models.device import LightGBMIdentity
from review_capacity.models.registry import configuration
from review_capacity.models.run_record import (
    RECORD_SCHEMA_VERSION,
    jsonable,
    load_settings_with_snapshot,
    model_run_record,
    write_model_run_record,
)
from review_capacity.paths import ProjectPaths
from support.models import MeanModel
from support.synthetic import FEATURES, REPO_PATHS

BUILD = LightGBMIdentity(version="4.7.0", library=Path("/lib/lib_lightgbm.so"), sha256="a" * 64)


@pytest.fixture
def project(tmp_path: Path) -> ProjectPaths:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO_PATHS.models_spec, tmp_path / "config" / "models.yaml")
    return ProjectPaths(tmp_path)


def test_snapshot_is_taken_from_the_loaded_bytes(project: ProjectPaths) -> None:
    settings, snapshot = load_settings_with_snapshot(project)
    content = project.models_spec.read_bytes()
    assert snapshot.path == "config/models.yaml"
    assert snapshot.sha256 == hashlib.sha256(content).hexdigest()
    assert snapshot.content == content.decode("utf-8")
    assert settings.main_seed == 42


def test_settings_are_parsed_from_the_captured_bytes(
    project: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edit of the file after it was read (the A -> B -> A race) cannot change what is used:
    the loader receives exactly the bytes that are hashed and recorded."""
    captured: list[bytes | None] = []

    def edit_then_load(paths: ProjectPaths, *, content: bytes | None = None) -> ModelSettings:
        paths.models_spec.write_text("main_seed: edited\n", encoding="utf-8")  # B on disk
        captured.append(content)
        return load_model_settings(paths, content=content)

    before = project.models_spec.read_bytes()
    monkeypatch.setattr("review_capacity.models.run_record.load_model_settings", edit_then_load)
    settings, snapshot = load_settings_with_snapshot(project)
    assert captured == [before]
    assert snapshot.sha256 == hashlib.sha256(before).hexdigest()
    assert settings.main_seed == 42  # parsed from A, not from the edited file


def test_jsonable_converts_library_values() -> None:
    value = {
        "a": np.int64(3),
        "b": np.float32(0.5),
        "c": (1, 2),
        "d": np.array([1, 2]),
        "e": np.bool_(True),
        "f": np.str_("x"),
        "g": [{"h": np.float64(0.25)}],
    }
    assert jsonable(value) == {
        "a": 3,
        "b": 0.5,
        "c": [1, 2],
        "d": [1, 2],
        "e": True,
        "f": "x",
        "g": [{"h": 0.25}],
    }
    assert jsonable(Path("lib") / "lib_lightgbm.so") == "lib/lib_lightgbm.so"


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (float("nan"), ValueError),
        (np.float64("inf"), ValueError),
        ({"nested": [1.0, float("nan")]}, ValueError),
        ({1: "x"}, TypeError),
        ({1, 2}, TypeError),
        (np.longdouble("nan"), ValueError),
        (np.longdouble("inf"), ValueError),
        ({"nested": np.array([np.longdouble("-inf")])}, ValueError),
        (np.longdouble(1.5), TypeError),
        (np.array([1.5], dtype=np.longdouble), TypeError),
        (np.datetime64("2026-10-13"), TypeError),
        (np.array([1 + 2j]), TypeError),
    ],
)
def test_jsonable_rejects_unrecordable_values(value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        jsonable(value)


def _fitted(name: str, settings: ModelSettings) -> MeanModel:
    labels = np.array([0, 1, 0, 0], dtype=np.int8)
    train = ModelData(pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}), labels, np.ones(4))
    model = MeanModel(configuration(name), 42, settings=settings, features=FEATURES, device="gpu")
    model.fit(train, train, {"depth": 6})
    return model


def _record(model: MeanModel, snapshot: Any, build: LightGBMIdentity | None) -> dict[str, Any]:
    return model_run_record(
        model=model,
        settings=snapshot,
        hyperparameters={"depth": np.int64(6)},
        lightgbm_build=build,
        environment={"python": "3.12.3"},
        code={"git_commit": None},
    )


def test_record_contents_and_round_trip(project: ProjectPaths, tmp_path: Path) -> None:
    settings, snapshot = load_settings_with_snapshot(project)
    record = _record(_fitted("Cat-u", settings), snapshot, BUILD)
    assert record["schema_version"] == RECORD_SCHEMA_VERSION == 1
    assert record["configuration"] == "Cat-u"
    assert (record["model"], record["family"], record["weighting"]) == ("catboost", "tree", "u")
    assert (record["device"], record["seed"], record["fitted_iterations"]) == ("gpu", 42, 1)
    assert record["hyperparameters"] == {"depth": 6}
    assert record["parameters"] == {
        "requested": {"seed": 42, "depth": 6},
        "resolved": {"rate": 0.25, "rows": 4},
    }
    assert record["settings"] == {
        "path": "config/models.yaml",
        "sha256": snapshot.sha256,
        "content": snapshot.content,
    }
    assert record["lightgbm_build"] == {
        "version": "4.7.0",
        "library": "/lib/lib_lightgbm.so",
        "sha256": "a" * 64,
    }
    path = tmp_path / "records" / "Cat-u.json"
    write_model_run_record(path, record)
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_lightgbm_record_requires_its_build_identity(project: ProjectPaths) -> None:
    settings, snapshot = load_settings_with_snapshot(project)
    with pytest.raises(ValueError, match="requires its verified build identity"):
        _record(_fitted("LGB-u", settings), snapshot, None)
    assert _record(_fitted("Cat-u", settings), snapshot, None)["lightgbm_build"] is None
