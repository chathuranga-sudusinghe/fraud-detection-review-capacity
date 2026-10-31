"""Stubs for the Test command tests (Issues #203 and #207): everything heavy or external is
replaced, so that the order of the steps, the records, the markers and the refusals are tested on
a temporary project. No research data and no model is read."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from review_capacity.config import load_dataset_spec
from review_capacity.confirmatory import command
from review_capacity.confirmatory.frozen import EvaluatorState, FrozenBinding, check_evaluator
from review_capacity.confirmatory.identity import artifact_hashes, frozen_state_id
from review_capacity.confirmatory.inputs import TestInputs
from review_capacity.confirmatory.scoring import attempt_directory
from review_capacity.data import loading
from review_capacity.development.records import ALL_CONFIGURATIONS
from review_capacity.development.trial import run_file
from review_capacity.paths import ProjectPaths
from support import confirmatory as c
from support.synthetic import REPO_PATHS

TAG = "protocol-freeze"
COMMIT = "a" * 40


class Counters:
    def __init__(self) -> None:
        self.read_test_rows = 0
        self.scoring_calls = 0
        self.fail_next = False


def install_stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A temporary project and the stubbed steps of the command."""
    shutil.copytree(REPO_PATHS.config_dir, tmp_path / "config")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "datapackage.json").write_text("{}", encoding="utf-8")
    paths = ProjectPaths(tmp_path)
    counters = Counters()
    state = {"evaluator": EvaluatorState(COMMIT, COMMIT, True, True)}
    anchor_prefix = load_dataset_spec(paths).partition("test").transaction_id_sha256_prefix

    def bind(**kwargs: Any) -> FrozenBinding:
        check_evaluator(kwargs["evaluator"])
        inputs = c.frozen_inputs()
        declarations = {
            "settings": kwargs["settings_sha256"],
            "training": kwargs["training_declaration_sha256"],
            "evaluation": kwargs["evaluation_declaration_sha256"],
            "dataset_manifest": kwargs["dataset_manifest_sha256"],
        }
        return FrozenBinding(
            frozen_inputs=inputs,
            training=c.identity()["training"],
            test_anchor_sha256=anchor_prefix + "0" * (64 - len(anchor_prefix)),
            frozen_state_sha256=frozen_state_id(
                commit=kwargs["evaluator"].tag_commit,
                declarations=declarations,
                artifacts=artifact_hashes(inputs),
            ),
        )

    def bodies(**kwargs: Any) -> list[dict[str, Any]]:
        counters.scoring_calls += 1
        if counters.fail_next:
            counters.fail_next = False
            raise RuntimeError("a model failed to load")
        directory = attempt_directory(kwargs["run"], kwargs["attempt"])
        result = []
        for name in ALL_CONFIGURATIONS:
            body = {
                k: v
                for k, v in c.scores(name).items()
                if k not in ("record", "schema_version", "identity")
            }
            target = directory / f"{name}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(kwargs["attempt"] * 100 + ALL_CONFIGURATIONS.index(name))
            rows = body["rows"]
            pd.DataFrame(
                {"TransactionID": np.arange(rows, dtype=np.int64), "score": rng.random(rows)}
            ).to_parquet(target, index=False)
            body["file"] = run_file(kwargs["run"], target)
            result.append(body)
        return result

    def read_inputs(*_a: Any, **_k: Any) -> TestInputs:
        counters.read_test_rows += 1
        ids = np.arange(3, dtype=np.int64)
        return TestInputs(pd.DataFrame({"x": [0.0, 0.0, 0.0]}), ids, ids, ids, "e" * 64)

    def refuse_labels(*_a: Any, **_k: Any) -> None:
        raise AssertionError("the Test labels were read in the scoring phase")

    for name, value in {
        "evaluator_state": lambda *_a, **_k: state["evaluator"],
        "code_provenance": lambda _root: {
            "git_commit": COMMIT,
            "git_uncommitted_changes": False,
            "package": {"source_sha256": "f" * 64},
        },
        "load_run": lambda *_a, **_k: object(),
        "bind_frozen_inputs": bind,
        "environment_record": lambda: {"python": "3.12"},
        "read_partitions": lambda *_a, **_k: {
            "train": SimpleNamespace(ordered_ids_sha256="1" * 64)
        },
        "load_neural_preprocessing": lambda *_a, **_k: object(),
        "read_test_inputs": read_inputs,
        "score_test_models": bodies,
        "_check_frozen_files": lambda *_a, **_k: None,
    }.items():
        monkeypatch.setattr(command, name, value)
    monkeypatch.setattr(loading, "load_labels", refuse_labels)
    return {"paths": paths, "counters": counters, "state": state}
