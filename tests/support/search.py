"""A small synthetic search context for the trial, search and selection tests (no project data).

Train and V1 are synthetic partitions of canonical model inputs in time order. The training
declaration is the project's, with a few trials, a small V1 budget and short early stopping so
that LightGBM fits in a moment on the CPU. The development partitions are those of the project's
``config/dataset.yaml`` with the Train row count set to the synthetic Train, because a ``fit``
record covers every Train row.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from review_capacity.data.processed import ProvenanceRecord
from review_capacity.development.inputs import PartitionRows, SearchInputs
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.trial import SearchContext, run_file
from review_capacity.models.device import LightGBMIdentity
from review_capacity.models.run_record import load_settings_with_snapshot
from review_capacity.models.v1_objective import V1Objective
from review_capacity.training.declaration import Stopping, TrainingDeclaration, load_declaration
from review_capacity.training.sampler import Candidate, CandidateList
from review_capacity.training.timing import session_conditions
from support.development import IDENTITY, PARTITIONS, example
from support.runs import RUN_ID
from support.synthetic import DAY, FEATURES, REPO_PATHS

TRAIN_ROWS, V1_ROWS = 600, 300
TRIALS, PATIENCE, MAXIMUM, DAILY_K = 3, 5, 40, 15

CANDIDATES: tuple[Candidate, ...] = (
    {
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_data_in_leaf": 10,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "lambda_l2": 0.0,
    },
    {
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "lambda_l2": 1.0,
    },
    {
        "learning_rate": 0.2,
        "num_leaves": 5,
        "min_data_in_leaf": 15,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "lambda_l2": 0.5,
    },
)


def candidate_list(candidates: tuple[Candidate, ...] = CANDIDATES) -> CandidateList:
    return CandidateList("lightgbm", 0, (0,), candidates)


def _frame(rows: int, seed: int) -> tuple[pd.DataFrame, npt.NDArray[np.int8]]:
    """Canonical model inputs and labels that depend on a categorical and a numeric input."""
    rng = np.random.default_rng(seed)
    product = rng.choice(["W", "C", "H", FEATURES.missing_label], rows)
    amount = rng.gamma(2.0, 50.0, rows)
    columns: dict[str, Any] = {
        "ProductCD": product,
        "card1": rng.choice(["1001", "1002", "1003", "1004"], rows),
        "DeviceType": rng.choice(["desktop", "mobile", FEATURES.missing_label], rows),
        "id_13": rng.choice(["49", "52", FEATURES.missing_label], rows),
        "TransactionAmt": amount,
        "V1": rng.normal(size=rows),
        "id_01": rng.normal(size=rows),
        "tod_sin": rng.uniform(-1, 1, rows),
        "tod_cos": rng.uniform(-1, 1, rows),
    }
    frame = pd.DataFrame(columns)
    for name in FEATURES.categorical:
        frame[name] = frame[name].astype("str")
    risk = (product == "C") * 1.5 + amount / 200.0 - 2.0
    labels = (rng.random(rows) < 1 / (1 + np.exp(-risk))).astype(np.int8)
    return frame[list(FEATURES.model_inputs)], labels


def partition(name: str, rows: int, seed: int, first_id: int) -> PartitionRows:
    """A synthetic partition over relative days 1 to 3, in time order (ids rise with time)."""
    features, labels = _frame(rows, seed)
    rng = np.random.default_rng(seed + 100)
    times = np.sort(rng.integers(1, 4, rows) * DAY + rng.integers(0, DAY, rows)).astype(np.int64)
    ids = np.arange(first_id, first_id + rows, dtype=np.int64)
    provenance = ProvenanceRecord(
        file=f"{name}.parquet",
        sha256="a" * 64,
        rows=rows,
        columns=len(features.columns) + 3,
        scope=name,
        transaction_id_sha256="b" * 64,
        sources={},
        configuration={},
        code={},
    )
    return PartitionRows(name, features, labels, ids, times, provenance)


def search_inputs() -> SearchInputs:
    return SearchInputs(
        partition("train", TRAIN_ROWS, 1, 1_000_000), partition("v1", V1_ROWS, 2, 2_000_000)
    )


def search_declaration() -> TrainingDeclaration:
    """The project's declaration with a few trials and short LightGBM early stopping."""
    declaration, _ = load_declaration(REPO_PATHS)
    space = declaration.space("lightgbm")
    fixed = {**space.fixed, "num_iterations": MAXIMUM, "early_stopping_round": PATIENCE}
    return replace(
        declaration,
        trials=TRIALS,
        v1_daily_k=DAILY_K,
        tree_stopping=Stopping(PATIENCE, MAXIMUM),
        models={**declaration.models, "lightgbm": replace(space, fixed=fixed)},
    )


def search_context(root: Path) -> SearchContext:
    """A new run directory under ``root`` holding its ``run`` record, and the shared search
    state on the synthetic partitions."""
    partitions = replace(PARTITIONS, train=replace(PARTITIONS.train, rows=TRAIN_ROWS))
    run = RunDirectory.create(root, partitions, "development", RUN_ID)
    run.write_record(example("run"))
    settings, snapshot = load_settings_with_snapshot(REPO_PATHS)
    declaration = search_declaration()
    inputs = search_inputs()
    return SearchContext(
        run=run,
        identity=dict(IDENTITY),
        declaration=declaration,
        settings=settings,
        settings_snapshot=snapshot,
        features=FEATURES,
        inputs=inputs,
        objective=V1Objective(inputs.v1_data(), declaration.v1_daily_k),
        environment={"python": "test"},
        code={"git_commit": None},
        lightgbm_build=LightGBMIdentity("4.7.0", Path("lib_lightgbm.so"), "c" * 64),
        session=session_conditions(True, telemetry=lambda: {"available": False}),
    )


def candidate_file(run: RunDirectory) -> dict[str, str]:
    """A stand-in for the model's recorded candidate list, inside the run directory."""
    path = run.path / "candidates" / "lightgbm.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return run_file(run, path)
