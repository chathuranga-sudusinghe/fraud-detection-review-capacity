"""Synthetic inputs of the GPU acceptance probe (``docs/gpu-acceptance-probe.md``, Section 1).

No research data are read. The inputs take only the shape of the study's inputs: the declared
model-input names and roles (``config/features.yaml``) and the documented number of Train levels
of each categorical input (``docs/feature-catalog.md``, read through ``data.catalog``). All values
come from one NumPy generator seeded with the probe's data seed (``data_generator``), passed
through both builders in turn, so a model refit never changes its inputs.

* **Tree inputs:** the canonical frame of DR-008 (categorical text labels with ``"__MISSING__"``,
  float64 numeric inputs with NaN, the time-of-day inputs), rows in time order with increasing
  identifiers, labels from a logistic model with a categorical interaction, and a score set
  holding categories unseen in Train.
* **Neural inputs:** the form of DR-012: standard-normal transformed numeric values, binary
  missingness indicators, and category indices with ``"__MISSING__"`` at index 0, to which
  unseen categories are mapped.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from review_capacity.config import SECONDS_PER_DAY, FeatureSpec
from review_capacity.data.catalog import Catalog
from review_capacity.errors import SpecificationError
from review_capacity.models.base import Labels, ModelData, class_weights, sample_weights
from review_capacity.models.neural_training import InputLayout, NeuralData, NeuralInputs
from review_capacity.models.registry import Weighting
from review_capacity.models.v1_objective import V1Objective

DATA_SEED = 77  # the probe's data seed (Issue #77), separate from the model seeds
MISSING_SHARE = 0.05  # share of missing categorical values in every input
UNSEEN_SHARE = 0.01  # share of score-set categorical values unseen in Train, per input
NUMERIC_MISSING_SHARE = 0.10  # share of NaN in half of the numeric inputs
FRAUD_RATE = 0.035  # approximate fraud rate of the synthetic labels
INDICATOR_COUNT = 324  # missingness indicators of DR-012
CATEGORY_DROPOUT = 0.01  # DR-012
V1_DAYS = 21  # relative days of the synthetic V1 rows (the study's V1 spans 21 days)
V1_DAILY_K = 20  # daily budget of the synthetic V1 objective (probe value, not a research value)


@dataclass(frozen=True)
class ProbeSizes:
    """Row counts of the synthetic partitions (specification, Section 1)."""

    train: int = 368_632
    valid: int = 100_000
    score: int = 100_000
    neural_train: int = 16_384
    neural_valid: int = 4_096
    neural_score: int = 4_096


@dataclass(frozen=True)
class TreeData:
    """Synthetic canonical inputs: Train and V1 for each weighting, and the score set."""

    train: dict[Weighting, ModelData]
    valid: dict[Weighting, ModelData]
    score: pd.DataFrame
    score_ids: npt.NDArray[np.int64]
    unseen: dict[str, int]
    diagnostics: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class NeuralSet:
    """Synthetic neural inputs: the layout, Train and V1 for each weighting, the score set."""

    layout: InputLayout
    train: dict[Weighting, NeuralData]
    valid: dict[Weighting, NeuralData]
    score: NeuralInputs
    score_ids: npt.NDArray[np.int64]
    objective: V1Objective
    diagnostics: dict[str, Any]
    sha256: str


def data_generator(seed: int = DATA_SEED) -> np.random.Generator:
    """The probe's one generator of synthetic data."""
    return np.random.default_rng(seed)


def category_levels(catalog: Catalog, features: FeatureSpec) -> dict[str, int]:
    """The documented Train levels of every categorical input, from the feature catalog."""
    levels: dict[str, int] = {}
    for name in features.categorical:
        entry = catalog.entries.get(name)
        if entry is None or entry.levels is None or entry.levels < 1:
            raise SpecificationError(f"The feature catalog gives no Train levels for {name}")
        levels[name] = entry.levels
    return levels


def _labels(score: npt.NDArray[np.float64], rng: np.random.Generator) -> Labels:
    """Labels from a logistic model of ``score``, with the intercept set for the fraud rate."""
    low, high = -20.0, 20.0
    for _ in range(60):  # bisection on the expected fraud rate
        middle = (low + high) / 2
        rate = float(np.mean(1.0 / (1.0 + np.exp(-(score + middle)))))
        low, high = (middle, high) if rate < FRAUD_RATE else (low, middle)
    probability = 1.0 / (1.0 + np.exp(-(score + low)))
    return (rng.random(len(score)) < probability).astype(np.int8)


def _codes(rng: np.random.Generator, rows: int, levels: int) -> npt.NDArray[np.int64]:
    """Category codes: 1 to ``levels`` for observed labels and 0 for the missing label."""
    codes = rng.integers(1, levels + 1, rows)
    codes[rng.random(rows) < MISSING_SHARE] = 0
    return codes.astype(np.int64)


def _interaction(
    codes_a: npt.NDArray[np.int64], codes_b: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """A categorical interaction: rows whose two category codes agree in parity."""
    agree = (codes_a > 0) & (codes_b > 0) & ((codes_a % 2) == (codes_b % 2))
    interaction: npt.NDArray[np.float64] = agree.astype(np.float64)
    return interaction


def _hash(*arrays: npt.NDArray[np.generic]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def tree_data(
    features: FeatureSpec,
    levels: dict[str, int],
    sizes: ProbeSizes,
    rng: np.random.Generator | None = None,
) -> TreeData:
    """The synthetic canonical inputs of the tree-based models (specification, Section 1), drawn
    from ``rng`` (a new data generator when none is given)."""
    rng = rng if rng is not None else data_generator()
    rows = sizes.train + sizes.valid + sizes.score
    starts = (0, sizes.train, sizes.train + sizes.valid, rows)
    columns: dict[str, Any] = {}
    codes_by_input: dict[str, npt.NDArray[np.int64]] = {}
    unseen: dict[str, int] = {}
    for name in features.categorical:
        codes = _codes(rng, rows, levels[name])
        score_part = codes[starts[2] :]
        unseen_rows = rng.random(len(score_part)) < UNSEEN_SHARE
        score_part[unseen_rows] = -1  # labels absent from Train
        unseen[name] = int(unseen_rows.sum())
        codes_by_input[name] = codes
        labels = np.empty(rows, dtype=object)
        labels[codes > 0] = codes[codes > 0].astype(str)
        labels[codes == 0] = features.missing_label
        labels[codes < 0] = np.char.add("unseen-", np.arange(int((codes < 0).sum())).astype(str))
        columns[name] = pd.Series(labels, dtype="str")
    numeric = np.empty((rows, len(features.numeric)), dtype=np.float64)
    for index in range(len(features.numeric)):
        kind = index % 3
        if kind == 0:
            values = rng.normal(size=rows)
        elif kind == 1:
            values = rng.lognormal(mean=0.0, sigma=1.0, size=rows)
        else:
            values = np.where(rng.random(rows) < 0.6, 0.0, rng.gamma(2.0, 50.0, rows))
        if index % 2 == 0:
            values[rng.random(rows) < NUMERIC_MISSING_SHARE] = np.nan
        numeric[:, index] = values
    for index, name in enumerate(features.numeric):
        columns[name] = numeric[:, index]
    seconds = rng.integers(0, SECONDS_PER_DAY, rows)
    theta = 2.0 * np.pi * seconds.astype(np.float64) / SECONDS_PER_DAY
    derived = {"tod_sin": np.sin(theta), "tod_cos": np.cos(theta)}
    for name in features.derived:
        if name not in derived:
            raise SpecificationError(f"No synthetic rule for derived input {name}")
        columns[name] = derived[name]
    frame = pd.DataFrame(columns)[list(features.model_inputs)]
    first, second = features.categorical[0], features.categorical[1]
    signal = (
        np.nan_to_num(numeric[:, 0])
        - 0.8 * np.nan_to_num(np.log1p(numeric[:, 1]))
        + 1.5 * _interaction(codes_by_input[first], codes_by_input[second])
    )
    labels_all = _labels(signal, rng)
    times = np.sort(rng.integers(SECONDS_PER_DAY, 106 * SECONDS_PER_DAY, rows)).astype(np.int64)
    ids = np.arange(1, rows + 1, dtype=np.int64)
    train_rows, valid_rows = slice(starts[0], starts[1]), slice(starts[1], starts[2])
    train_labels = labels_all[train_rows]
    train: dict[Weighting, ModelData] = {}
    valid: dict[Weighting, ModelData] = {}
    for weighting in (Weighting.UNWEIGHTED, Weighting.BALANCED):
        weights = class_weights(train_labels, weighting)
        for target, part in ((train, train_rows), (valid, valid_rows)):
            labels = labels_all[part]
            target[weighting] = ModelData(
                features=frame.iloc[part].reset_index(drop=True),
                labels=labels,
                weights=sample_weights(labels, weights),
                times=times[part],
                ids=ids[part],
            )
    score = frame.iloc[starts[2] :].reset_index(drop=True)
    digest = _hash(
        labels_all,
        numeric,
        *(codes_by_input[name] for name in features.categorical),
        seconds,
        times,
        *(data.weights for data in (*train.values(), *valid.values())),
    )
    train_codes = {name: codes[starts[0] : starts[1]] for name, codes in codes_by_input.items()}
    diagnostics = {
        "rows": {"train": sizes.train, "valid": sizes.valid, "score": sizes.score},
        "fraud_rows": {
            "train": int(train_labels.sum()),
            "valid": int(labels_all[valid_rows].sum()),
        },
        "fraud_row_weight": {w.value: class_weights(train_labels, w).positive for w in train},
        "train_missing": {name: int((c == 0).sum()) for name, c in train_codes.items()},
        "train_levels": {name: len(np.unique(c[c > 0])) for name, c in train_codes.items()},
        "score_unseen": unseen,
        "numeric_missing_train": int(np.isnan(numeric[starts[0] : starts[1]]).sum()),
    }
    return TreeData(train, valid, score, ids[starts[2] :], unseen, diagnostics, digest)


def neural_layout(features: FeatureSpec, levels: dict[str, int]) -> InputLayout:
    """The DR-012 layout: the 384 numeric inputs, 324 missingness indicators (named after the first
    324 numeric inputs), and the 49 categorical inputs with ``"__MISSING__"`` at index 0."""
    numeric = tuple(features.numeric_inputs)
    if len(numeric) < INDICATOR_COUNT:
        raise SpecificationError("Fewer numeric inputs than missingness indicators")
    indicators = tuple(f"{name}__missing" for name in numeric[:INDICATOR_COUNT])
    return InputLayout(
        numeric=numeric + indicators,
        categorical=tuple(features.categorical),
        cardinalities=tuple(levels[name] + 1 for name in features.categorical),
        missing_indices=(0,) * len(features.categorical),
        category_dropout=CATEGORY_DROPOUT,
    )


def _valid_rows(rows: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Identifiers and times of the synthetic V1 rows: increasing identifiers on ``V1_DAYS``
    consecutive relative days, the first being day 1 (metadata of the V1 objective, never
    inputs)."""
    ids = np.arange(1, rows + 1, dtype=np.int64)
    days = ids * V1_DAYS // (rows + 1) + 1
    return ids, (days * SECONDS_PER_DAY + 1).astype(np.int64)


def neural_data(
    layout: InputLayout, sizes: ProbeSizes, rng: np.random.Generator | None = None
) -> NeuralSet:
    """The synthetic neural inputs in the form of DR-012 (specification, Section 1), drawn from
    ``rng`` (a new data generator when none is given)."""
    rng = rng if rng is not None else data_generator()
    rows = sizes.neural_train + sizes.neural_valid + sizes.neural_score
    starts = (0, sizes.neural_train, sizes.neural_train + sizes.neural_valid, rows)
    transformed = len(layout.numeric) - INDICATOR_COUNT
    numeric = np.empty((rows, len(layout.numeric)), dtype=np.float32)
    numeric[:, :transformed] = rng.normal(size=(rows, transformed))
    numeric[:, transformed:] = rng.random((rows, INDICATOR_COUNT)) < NUMERIC_MISSING_SHARE
    categorical = np.empty((rows, len(layout.categorical)), dtype=np.int64)
    for column, size in enumerate(layout.cardinalities):
        codes = _codes(rng, rows, size - 1)
        score_part = codes[starts[2] :]
        score_part[rng.random(len(score_part)) < UNSEEN_SHARE] = 0  # unseen -> "__MISSING__"
        categorical[:, column] = codes
    signal = (
        numeric[:, 0].astype(np.float64)
        - 0.8 * numeric[:, 1]
        + 1.5 * _interaction(categorical[:, 0], categorical[:, 1])
    )
    labels_all = _labels(signal, rng)
    train_rows, valid_rows = slice(starts[0], starts[1]), slice(starts[1], starts[2])
    train_labels = labels_all[train_rows]
    train: dict[Weighting, NeuralData] = {}
    valid: dict[Weighting, NeuralData] = {}
    valid_ids, valid_times = _valid_rows(sizes.neural_valid)
    for weighting in (Weighting.UNWEIGHTED, Weighting.BALANCED):
        weights = class_weights(train_labels, weighting)
        for target, part in ((train, train_rows), (valid, valid_rows)):
            labels = labels_all[part]
            is_valid = target is valid
            target[weighting] = NeuralData(
                inputs=NeuralInputs(numeric[part].copy(), categorical[part].copy()),
                labels=labels,
                weights=sample_weights(labels, weights),
                ids=valid_ids if is_valid else None,
                times=valid_times if is_valid else None,
            )
    objective = V1Objective.from_rows(valid_ids, valid_times, labels_all[valid_rows], V1_DAILY_K)
    score = NeuralInputs(numeric[starts[2] :].copy(), categorical[starts[2] :].copy())
    ids = np.arange(1, sizes.neural_score + 1, dtype=np.int64)
    layout_bytes = np.frombuffer(
        json.dumps(layout.to_json(), sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    weight_arrays = [data.weights for data in (*train.values(), *valid.values())]
    digest = _hash(numeric, categorical, labels_all, layout_bytes, *weight_arrays)
    diagnostics = {
        "rows": {
            "train": sizes.neural_train,
            "valid": sizes.neural_valid,
            "score": sizes.neural_score,
        },
        "fraud_rows": {
            "train": int(train_labels.sum()),
            "valid": int(labels_all[valid_rows].sum()),
        },
        "fraud_row_weight": {w.value: class_weights(train_labels, w).positive for w in train},
        "missing_index_values": {
            "train": int((categorical[train_rows] == 0).sum()),
            "score": int((categorical[starts[2] :] == 0).sum()),
        },
        "layout": layout.to_json(),
    }
    return NeuralSet(layout, train, valid, score, ids, objective, diagnostics, digest)


def top_k(rows: int, share: float = 0.02) -> int:
    """K of checks C9 and C11: ``share`` of the scored rows, rounded up (specification, C9)."""
    return math.ceil(share * rows)
