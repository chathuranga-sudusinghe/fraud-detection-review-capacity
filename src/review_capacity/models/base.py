"""The common interface of the eight model configurations (DR-011 §3, common procedure).

* ``ModelData``: the inputs, binary labels and sample weights of one partition, checked once.
  Rows are matched by position: features, labels and weights come from the same ordered source;
* ``class_weights`` and ``sample_weights``: the protocol's explicit Train-only class weighting
  (each fraud row weighted n_neg / n_pos, each legitimate row 1), identical for all models,
  rather than any library's own "balanced" switch;
* ``merge_parameters``: the fixed DR-013 settings combined with a configuration's
  hyperparameters, after mapping each library's parameter aliases to canonical names; a
  hyperparameter can never override a fixed setting, under any name;
* ``ScoringModel``: what every model implements. A model is created with its configuration, seed,
  the loaded model settings, the feature declaration (DR-008: the categorical inputs are named
  from ``config/features.yaml``, never inferred) and its device, and fitted either with early
  stopping on V1 or at a fixed iteration count without validation (the CPU refit of DR-013 §7);
* ``checked_scores``: every score a model returns is a real number, finite and within [0, 1].

The model settings are loaded once per run and passed in explicitly; no model reads YAML.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from review_capacity.config import DEVICES, SEED_LIMIT, Device, FeatureSpec, ModelSettings
from review_capacity.errors import DataValidationError, ModelError, SpecificationError
from review_capacity.models.registry import Configuration, Weighting

type Labels = npt.NDArray[np.int8]
type Weights = npt.NDArray[np.float64]
type Scores = npt.NDArray[np.float64]


def _check_labels(labels: object) -> None:
    if (
        not isinstance(labels, np.ndarray)
        or labels.ndim != 1
        or labels.size == 0
        or labels.dtype != np.int8
        or not np.isin(labels, (0, 1)).all()
    ):
        raise DataValidationError("Labels must be a non-empty one-dimensional int8 array of 0 or 1")


@dataclass(frozen=True)
class ModelData:
    """Model inputs, labels (1 = fraud, 0 = legitimate) and sample weights of one partition.

    Rows are matched by position, not by index: build all three from the same ordered source.
    V1 evaluation weights are an explicit input set by the training rules; balanced weights are
    never recomputed from V1 labels.
    """

    features: pd.DataFrame
    labels: Labels
    weights: Weights

    def __post_init__(self) -> None:
        if not isinstance(self.features, pd.DataFrame):
            raise DataValidationError("Features must be a DataFrame")
        if self.features.empty:
            raise DataValidationError("A model partition must not be empty")
        if not self.features.columns.is_unique:
            raise DataValidationError("Model feature columns must be unique")
        _check_labels(self.labels)
        if not isinstance(self.weights, np.ndarray) or self.weights.dtype != np.float64:
            raise DataValidationError("Sample weights must be a float64 NumPy array")
        rows = len(self.features)
        if self.labels.shape != (rows,) or self.weights.shape != (rows,):
            raise DataValidationError(
                f"{rows} feature rows, but labels {self.labels.shape} and weights "
                f"{self.weights.shape}"
            )
        if not (np.isfinite(self.weights).all() and (self.weights > 0).all()):
            raise DataValidationError("Sample weights must be finite and positive")


@dataclass(frozen=True)
class ClassWeights:
    """Sample weight of each fraud (positive) row; each legitimate row weighs 1."""

    positive: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.positive) and self.positive > 0):
            raise DataValidationError(f"Fraud-row weight must be finite and positive: {self}")


def class_weights(train_labels: Labels, weighting: Weighting) -> ClassWeights:
    """The protocol's class weights, from Train labels only: with balanced weighting, each fraud
    row weighs n_neg / n_pos (legitimate over fraud Train rows) and each legitimate row 1;
    unweighted, every row weighs 1."""
    _check_labels(train_labels)
    if weighting is Weighting.UNWEIGHTED:
        return ClassWeights(positive=1.0)
    if weighting is not Weighting.BALANCED:
        raise ValueError(f"Unsupported weighting {weighting!r}")
    positives = int((train_labels == 1).sum())
    negatives = len(train_labels) - positives
    if positives == 0 or negatives == 0:
        raise DataValidationError(
            f"Balanced weights need both classes: {positives} fraud, {negatives} legitimate"
        )
    return ClassWeights(positive=negatives / positives)


def sample_weights(labels: Labels, weights: ClassWeights) -> Weights:
    """The sample weight of every row of a partition."""
    _check_labels(labels)
    return np.where(labels == 1, weights.positive, 1.0).astype(np.float64)


def merge_parameters(
    fixed: Mapping[str, Any], hyperparameters: Mapping[str, Any], *, aliases: Mapping[str, str]
) -> dict[str, Any]:
    """The fixed DR-013 settings and the configuration's hyperparameters, as one mapping.

    ``aliases`` maps each alternative parameter name of the library to its canonical name (the
    complete table of the pinned library, supplied by its adapter). Both mappings are first
    rewritten to canonical names; two names for one parameter, or a hyperparameter naming a
    fixed setting, stop the run: the device policy is never tuned or overridden (DR-013).
    """

    def canonical(values: Mapping[str, Any], what: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in values.items():
            key = aliases.get(name, name)
            if key in result:
                raise SpecificationError(f"{what}: more than one name for parameter {key!r}")
            result[key] = value
        return result

    fixed_values = canonical(fixed, "Fixed settings")
    requested = canonical(hyperparameters, "Hyperparameters")
    clash = sorted(fixed_values.keys() & requested.keys())
    if clash:
        raise SpecificationError(f"Hyperparameters cannot override fixed DR-013 settings: {clash}")
    return {**fixed_values, **requested}


def checked_scores(scores: npt.ArrayLike, rows: int, model: str) -> Scores:
    """``scores`` as float64, after checking that there is one real, finite score in [0, 1] per
    row. Scores of exactly 0 or 1 are valid; ties are handled by the decision policies."""
    if np.ma.isMaskedArray(scores):
        raise ModelError(f"{model} returned masked scores")
    try:
        raw = np.asarray(scores)
    except (TypeError, ValueError) as error:
        raise ModelError(f"{model} returned invalid scores") from error
    if raw.shape != (rows,):
        raise ModelError(f"{model} returned scores of shape {raw.shape} for {rows} rows")
    if raw.dtype.kind not in "fiu":
        raise ModelError(f"{model} returned non-real-numeric scores (dtype {raw.dtype})")
    values = raw.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ModelError(f"{model} returned non-finite scores")
    if (values < 0).any() or (values > 1).any():
        raise ModelError(f"{model} returned scores outside [0, 1]")
    return values


def check_fit_arguments(valid: ModelData | None, iterations: int | None) -> None:
    """Exactly one of V1 early stopping (``valid``) or a fixed positive iteration count without
    validation (``iterations``, the CPU refit of DR-013 §7)."""
    if (valid is None) == (iterations is None):
        raise ValueError("Fit with exactly one of valid (early stopping) or iterations")
    if iterations is not None and (
        isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0
    ):
        raise ValueError(f"iterations must be a positive integer, not {iterations!r}")


class ScoringModel(ABC):
    """What every configuration implements.

    A model is created with its configuration, seed, the loaded model settings (from which each
    adapter takes its own library or neural settings), the feature declaration and its device.
    Its inputs are the canonical model inputs of DR-008, in declared order. It is fitted once and
    then scores any partition. Scores are the model's raw fraud scores in [0, 1] (calibration,
    for R4 only, is separate). Tree models always score through their CPU prediction backend
    (DR-013 §5); neural models score in float32 on their device.
    """

    def __init__(
        self,
        configuration: Configuration,
        seed: int,
        *,
        settings: ModelSettings,
        features: FeatureSpec,
        device: Device,
    ) -> None:
        if device not in DEVICES:
            raise ValueError(f"Unsupported device {device!r}; expected one of {DEVICES}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"Seed must be an integer, not {seed!r}")
        if not 0 <= seed < SEED_LIMIT:
            raise ValueError(f"Seed {seed} outside 0 to {SEED_LIMIT - 1}")
        self.configuration = configuration
        self.seed = seed
        self.settings = settings
        self.features = features
        self.device = device

    def check_inputs(self, frame: pd.DataFrame) -> None:
        """Check that ``frame`` holds the canonical model inputs of DR-008, in declared order."""
        if tuple(frame.columns) != self.features.model_inputs:
            raise DataValidationError("Model inputs differ from the declared canonical inputs")

    @abstractmethod
    def fit(
        self,
        train: ModelData,
        valid: ModelData | None,
        hyperparameters: Mapping[str, Any],
        *,
        iterations: int | None = None,
    ) -> None:
        """Fit on ``train`` with early stopping on ``valid`` (V1), or, with ``valid=None``, at
        exactly ``iterations`` trees or epochs without validation. Implementations begin with
        ``check_fit_arguments(valid, iterations)``."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> Scores:
        """Fraud scores of ``features``, one per row, checked with ``checked_scores``."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Save the fitted model, with a manifest of its format version, configuration and seed."""

    @classmethod
    @abstractmethod
    def load(
        cls,
        directory: Path,
        configuration: Configuration,
        seed: int,
        *,
        settings: ModelSettings,
        features: FeatureSpec,
        device: Device,
    ) -> Self:
        """A fitted model saved by ``save``; its manifest must match ``configuration`` and
        ``seed``, and its format version must be supported."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """The requested parameters, the library's resolved parameters, and the seeds, thread
        settings and backends the fitted model actually used."""

    @property
    @abstractmethod
    def fitted_iterations(self) -> int:
        """The trees (or epochs) the fitted model keeps."""
