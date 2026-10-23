"""The logistic-regression reference of DR-011 §5 (fit and search: DR-015 §8; Issue #151).

One unweighted, L2-regularised logistic regression, reported as a descriptive reference. It is
outside the eight configurations, so it has no ``Configuration`` and is a class of its own beside
``models.base.ScoringModel``.

* **Inputs:** the float64 CSR matrix of ``features.logistic_inputs`` from the shared Train-fitted
  preprocessing artifact (DR-012). The artifact is given when the model is built; its SHA-256 and
  the matrix identity (artifact hash, block sizes and the hash of the ordered column names) are
  recorded with the fit and checked when a saved model is reloaded. Rows of the canonical frame are
  converted here, so the model scores the same canonical inputs as every other model.
* **Fitting:** scikit-learn ``LogisticRegression`` with the fixed settings of the training
  declaration (``config/training.yaml``: ``solver``, ``l1_ratio``, ``fit_intercept``,
  ``class_weight``, ``warm_start``, ``tol`` and ``max_iter``) and the candidate's ``C``, the only
  searched hyperparameter, on unweighted Train rows, to convergence and without V1 early stopping.
  Any other hyperparameter, or an override of a fixed setting, stops the run. Non-convergence (a
  ``ConvergenceWarning``), non-finite coefficients or intercept and invalid probabilities raise
  ``ModelError``: there is no solver substitution, candidate replacement or iteration extension.
  Other library warnings are logged and recorded.
* **Scores:** the positive-class probabilities as float64, checked with ``checked_scores``.
* **Persistence:** the float64 coefficient and intercept arrays (``arrays.npz``, no pickle) and a
  manifest with the format version, the fixed settings, ``C``, the iteration count, the warnings,
  the matrix identity, the package versions and the SHA-256 of the arrays and of the manifest.
  Reloading checks all of them against the request and the artifact before the estimator is
  rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from review_capacity.errors import (
    DataValidationError,
    EnvironmentMismatchError,
    FileAccessError,
    ModelError,
    SpecificationError,
)
from review_capacity.features.logistic_inputs import (
    build_logistic_matrix,
    logistic_columns,
    matrix_identity,
)
from review_capacity.features.neural_preprocessing import NeuralPreprocessing
from review_capacity.files import write_atomically, write_json
from review_capacity.logging_setup import log_step
from review_capacity.models.base import ModelData, Scores, checked_scores
from review_capacity.models.run_record import jsonable

logger = logging.getLogger(__name__)

KIND = "logistic_regression_reference"
FORMAT_VERSION = 1
ARRAYS_FILE = "arrays.npz"
MANIFEST_FILE = "manifest.json"
FIXED_NAMES = frozenset(
    {"solver", "l1_ratio", "fit_intercept", "class_weight", "warm_start", "tol", "max_iter"}
)
TUNED = "C"
PACKAGES = ("numpy", "scipy", "scikit-learn")
MANIFEST_KEYS = frozenset(
    {
        "kind",
        "format_version",
        "C",
        "fixed",
        "n_iter",
        "warnings",
        "matrix",
        "versions",
        "arrays_sha256",
        "state_sha256",
    }
)
CLASSES = np.array([0, 1], dtype=np.int64)

type Floats = npt.NDArray[np.float64]


def _versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError as error:
            raise EnvironmentMismatchError(f"Package {name} is not installed") from error
    return found


def _arrays_sha256(coefficients: Floats, intercept: Floats) -> str:
    digest = hashlib.sha256()
    for name, array in (("coefficients", coefficients), ("intercept", intercept)):
        data = np.ascontiguousarray(array, dtype="<f8")
        for part in (name.encode(), b"<f8", repr(data.shape).encode(), data.tobytes()):
            digest.update(part)
    return digest.hexdigest()


def _state_sha256(manifest: Mapping[str, Any]) -> str:
    body = {key: value for key, value in manifest.items() if key != "state_sha256"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_fixed(fixed: Mapping[str, Any]) -> dict[str, Any]:
    """The declared fixed settings: exactly the DR-015 §8 names, an unweighted L2 lbfgs fit."""
    if set(fixed) != FIXED_NAMES:
        raise SpecificationError(
            f"Fixed settings must be exactly {sorted(FIXED_NAMES)}, not {sorted(fixed)}"
        )
    if fixed["solver"] != "lbfgs" or fixed["class_weight"] is not None or fixed["warm_start"]:
        raise SpecificationError(
            "The reference is fitted with solver lbfgs, no class weighting and no warm start"
        )
    return dict(fixed)


def _check_hyperparameters(hyperparameters: Mapping[str, Any]) -> float:
    """``C``, the only searched hyperparameter, as a positive finite float."""
    clash = sorted(FIXED_NAMES & set(hyperparameters))
    if clash:
        raise SpecificationError(f"Hyperparameters cannot override fixed settings: {clash}")
    if set(hyperparameters) != {TUNED}:
        raise SpecificationError(
            f"The reference has one hyperparameter, {TUNED}; got {sorted(hyperparameters)}"
        )
    value = hyperparameters[TUNED]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpecificationError(f"{TUNED} must be a real number, not {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise SpecificationError(f"{TUNED} must be positive and finite, not {value!r}")
    return float(value)


def _estimator(c: float, fixed: Mapping[str, Any]) -> Any:
    return LogisticRegression(C=c, **fixed)


class LogisticRegressionReference:
    """The fitted or reloaded reference model for one preprocessing artifact."""

    def __init__(self, artifact: NeuralPreprocessing, fixed: Mapping[str, Any]) -> None:
        self.artifact = artifact
        self._fixed = _check_fixed(fixed)
        self._model: Any = None
        self._c: float | None = None
        self._n_iter = 0
        self._warnings: list[str] = []

    def _fitted(self) -> Any:
        if self._model is None:
            raise ModelError("The logistic-regression reference is not fitted")
        return self._model

    # --- fitting and scoring ----------------------------------------------------------------

    def fit(self, train: ModelData, hyperparameters: Mapping[str, Any]) -> None:
        """Fit on ``train`` (unweighted) to convergence with the candidate's ``C``."""
        c = _check_hyperparameters(hyperparameters)
        if not (train.weights == 1.0).all():
            raise SpecificationError("The reference is fitted unweighted (all weights 1)")
        matrix = build_logistic_matrix(self.artifact, train.features)
        model = _estimator(c, self._fixed)
        max_iter = self._fixed["max_iter"]
        with log_step(logger, f"Fitting the logistic-regression reference (C = {c:g})"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    model.fit(matrix, train.labels)
                except ValueError as error:
                    raise ModelError(f"The regression could not be fitted: {error}") from error
                finally:  # the solver's own messages are kept, whatever the outcome
                    for item in caught:
                        logger.warning(
                            "Logistic regression %s: %s", item.category.__name__, item.message
                        )
            notes = [f"{item.category.__name__}: {item.message}" for item in caught]
            unconverged = [
                str(item.message)
                for item in caught
                if issubclass(item.category, ConvergenceWarning)
            ]
            if unconverged:
                raise ModelError(
                    f"The logistic regression did not converge within max_iter = {max_iter}: "
                    + " ".join(unconverged)
                )
            coefficients = np.asarray(model.coef_, dtype=np.float64)
            intercept = np.asarray(model.intercept_, dtype=np.float64)
            width = len(logistic_columns(self.artifact))
            if coefficients.shape != (1, width) or intercept.shape != (1,):
                raise ModelError("The fitted coefficients have an unexpected shape")
            if not (np.isfinite(coefficients).all() and np.isfinite(intercept).all()):
                raise ModelError("The fitted coefficients or intercept are not finite")
            if not np.array_equal(model.classes_, CLASSES):
                raise ModelError("The fitted classes are not [0, 1]")
            n_iter = int(model.n_iter_[0])
            if not 0 <= n_iter <= max_iter:
                raise ModelError(f"The solver reports {n_iter} iterations (limit {max_iter})")
        self._model, self._c = model, c
        self._n_iter, self._warnings = n_iter, notes
        logger.info("Logistic regression fitted: C = %g, %d iterations", c, n_iter)

    def prepare(self, features: pd.DataFrame) -> Any:
        """The float64 CSR matrix of the canonical rows: the model's native input, built apart
        from prediction so that the two can be timed separately."""
        self._fitted()
        return build_logistic_matrix(self.artifact, features)

    def predict_prepared(self, matrix: Any) -> Scores:
        """The fraud probability of each row of a prepared matrix, float64 in [0, 1]."""
        model = self._fitted()
        probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ModelError("The logistic regression returned unexpected probabilities")
        return checked_scores(probabilities[:, 1], matrix.shape[0], "Logistic regression")

    def predict(self, features: pd.DataFrame) -> Scores:
        """The fraud probability of each row, float64 in [0, 1]."""
        return self.predict_prepared(self.prepare(features))

    @property
    def parameters(self) -> dict[str, Any]:
        """The requested hyperparameter, the fixed settings, the iterations used, the warnings and
        the matrix identity."""
        self._fitted()
        return {
            "requested": {TUNED: self._c},
            "fixed": jsonable(self._fixed),
            "n_iter": self._n_iter,
            "warnings": list(self._warnings),
            "matrix": matrix_identity(self.artifact),
        }

    @property
    def fitted_iterations(self) -> int:
        self._fitted()
        return self._n_iter

    # --- persistence --------------------------------------------------------------------------

    def _manifest(self, coefficients: Floats, intercept: Floats) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "kind": KIND,
            "format_version": FORMAT_VERSION,
            "C": self._c,
            "fixed": jsonable(self._fixed),
            "n_iter": self._n_iter,
            "warnings": list(self._warnings),
            "matrix": matrix_identity(self.artifact),
            "versions": _versions(),
            "arrays_sha256": _arrays_sha256(coefficients, intercept),
        }
        manifest["state_sha256"] = _state_sha256(manifest)
        return manifest

    def save(self, directory: Path) -> str:
        """Write ``arrays.npz`` and ``manifest.json`` to ``directory``, never replacing either;
        returns the SHA-256 of the saved state."""
        model = self._fitted()
        coefficients = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        intercept = np.asarray(model.intercept_, dtype=np.float64)
        arrays_path, manifest_path = directory / ARRAYS_FILE, directory / MANIFEST_FILE
        for path in (arrays_path, manifest_path):
            if path.exists():
                raise FileAccessError(f"{path}: already exists and is never replaced")

        def write_arrays(target: Path) -> None:
            with target.open("wb") as handle:
                np.savez(handle, coefficients=coefficients, intercept=intercept)

        manifest = self._manifest(coefficients, intercept)
        with log_step(logger, f"Saving the logistic-regression reference to {directory}"):
            write_atomically(arrays_path, write_arrays)
            write_json(manifest_path, manifest)
        digest: str = manifest["state_sha256"]
        return digest

    @classmethod
    def load(cls, directory: Path, artifact: NeuralPreprocessing, fixed: Mapping[str, Any]) -> Self:
        """A saved model for ``artifact`` and the declared fixed settings. Checks, in order, the
        files, kind and format, package versions, both digests, the fixed settings, the matrix
        identity, then the arrays, before the estimator is rebuilt."""
        model = cls(artifact, fixed)
        try:
            manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
            with np.load(directory / ARRAYS_FILE, allow_pickle=False) as archive:
                if set(archive.files) != {"coefficients", "intercept"}:
                    raise DataValidationError(f"{directory}: unexpected arrays {archive.files}")
                coefficients = np.asarray(archive["coefficients"])
                intercept = np.asarray(archive["intercept"])
        except OSError as error:
            raise FileAccessError(f"{directory}: saved model cannot be read") from error
        except ValueError as error:
            raise DataValidationError(f"{directory}: saved model is not valid ({error})") from error
        if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
            raise DataValidationError(f"{directory}: not a {KIND} manifest with the expected keys")
        if manifest["kind"] != KIND or manifest["format_version"] != FORMAT_VERSION:
            raise DataValidationError(f"{directory}: not a {KIND} model of this format")
        if manifest["versions"] != _versions():
            raise EnvironmentMismatchError(
                f"The model was fitted with {manifest['versions']}, installed: {_versions()}"
            )
        for name, array in (("coefficients", coefficients), ("intercept", intercept)):
            if array.dtype != np.float64 or not np.isfinite(array).all():
                raise DataValidationError(f"{name}: must be finite float64 values")
        if manifest["arrays_sha256"] != _arrays_sha256(coefficients, intercept):
            raise DataValidationError(f"{directory}: the arrays differ from their recorded SHA-256")
        if manifest["state_sha256"] != _state_sha256(manifest):
            raise DataValidationError(f"{directory}: the manifest does not match its SHA-256")
        if manifest["fixed"] != jsonable(model._fixed):
            raise SpecificationError("The saved model used different fixed settings")
        if manifest["matrix"] != matrix_identity(artifact):
            raise SpecificationError("The saved model was fitted on a different input matrix")
        c = manifest["C"]
        if isinstance(c, bool) or not isinstance(c, int | float) or not math.isfinite(c) or c <= 0:
            raise DataValidationError(f"{directory}: malformed C")
        n_iter, notes = manifest["n_iter"], manifest["warnings"]
        limit = model._fixed["max_iter"]
        valid = isinstance(n_iter, int) and not isinstance(n_iter, bool) and 0 <= n_iter <= limit
        if not valid or not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
            raise DataValidationError(f"{directory}: malformed iteration count or warnings")
        width = len(logistic_columns(artifact))
        if coefficients.shape != (width,) or intercept.shape != (1,):
            raise DataValidationError(f"{directory}: arrays do not match the matrix columns")
        estimator = _estimator(float(c), model._fixed)
        estimator.coef_ = coefficients.reshape(1, -1).copy()
        estimator.intercept_ = intercept.copy()
        estimator.classes_ = CLASSES.copy()
        estimator.n_iter_ = np.array([n_iter], dtype=np.int32)
        estimator.n_features_in_ = width
        model._model, model._c = estimator, float(c)
        model._n_iter, model._warnings = n_iter, list(notes)
        return model
