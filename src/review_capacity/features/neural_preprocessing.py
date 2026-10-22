"""The shared Train-only preprocessing of the two neural models (DR-012; Issue #149).

One artifact, fitted once on the canonical Train frame, is reused unchanged by every neural
configuration, candidate and refit, and applied unchanged to Validation 1 and Validation 2 (and,
once the protocol permits, Test). Fitting is the only step that draws random numbers; applying the
artifact never refits it and never draws noise.

* **Categorical inputs:** a vocabulary per input. ``"__MISSING__"`` has index 0 in every vocabulary
  (also where Train has no missing value), followed by the unique observed Train labels in sorted
  order. A missing or unseen value maps to index 0. The convention is recorded in DR-012
  (Issue #149) and persisted in the artifact.
* **Numeric inputs:** a quantile transformation to a normal output distribution, fitted per input
  on its non-missing Train values plus zero-mean Gaussian noise (scale ``1e-3 / max(s_i, 1e-3)``,
  ``s_i`` the population standard deviation of the non-missing values) and applied to the original
  noise-free values; missing values are 0 afterwards. One generator, ``default_rng(seed)``, is
  never reset: inputs in declared order, non-missing Train values in canonical row order, one
  ``standard_normal(size=m_i)`` vector per input (missing entries consume no draws, a constant
  input draws like any other). The requested quantile count is ``max(min(n_Train // 30, 1000), 10)``
  from the total Train rows; scikit-learn caps the count it uses at the rows it is given, so the
  count passed on is ``min(requested, n_Train)`` and both are recorded.
* **Indicators:** one binary indicator per numeric input with a missing Train value, in
  numeric-input order, named ``<input>__missing``.
* **Interfaces:** ``transform_numeric``, ``missingness_indicators`` and ``category_indices`` return
  float64 and int64 values (the representation a logistic-regression reference also reads);
  ``to_neural_inputs`` is the only conversion to float32.
* **Persistence:** ``manifest.json`` and ``arrays.npz`` (no pickle). The fitted transformer is
  rebuilt from the stored float64 quantiles and references and applied through scikit-learn's own
  ``transform``, so the artifact is coupled to the recorded NumPy, SciPy and scikit-learn versions
  and refuses to load under others. ``artifact_sha256`` covers the manifest and the array content
  (not the bytes of the archive, which carry file times).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from review_capacity.config import FeatureSpec, features_fingerprint
from review_capacity.errors import (
    DataValidationError,
    EnvironmentMismatchError,
    FileAccessError,
    SpecificationError,
)
from review_capacity.files import write_atomically, write_json
from review_capacity.logging_setup import ProgressLogger, log_step
from review_capacity.models.neural_training import InputLayout, NeuralInputs

logger = logging.getLogger(__name__)

KIND = "neural_preprocessing"
FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"
ARRAYS_FILE = "arrays.npz"
ARRAY_NAMES = ("quantiles", "references", "std", "noise_scale")  # all float64
PACKAGES = ("numpy", "scipy", "scikit-learn")

CATEGORY_DROPOUT = 0.01  # DR-012: fixed, not tuned
MISSING_INDEX = 0
INDICATOR_SUFFIX = "__missing"
NOISE_BASE = 1e-3
NOISE_FLOOR = 1e-3
QUANTILES_CAP = 1000
QUANTILES_FLOOR = 10
QUANTILES_ROWS_PER = 30

MANIFEST_KEYS = frozenset(
    {
        "kind",
        "format_version",
        "specification",
        "features_fingerprint",
        "seed",
        "missing_label",
        "train",
        "quantiles",
        "category_dropout",
        "numeric_inputs",
        "categorical_inputs",
        "vocabularies",
        "indicators",
        "versions",
        "arrays_sha256",
        "artifact_sha256",
    }
)

type Floats = npt.NDArray[np.float64]


def requested_quantiles(train_rows: int) -> int:
    """DR-012: ``max(min(n_Train // 30, 1000), 10)``, from the total Train rows."""
    return max(min(train_rows // QUANTILES_ROWS_PER, QUANTILES_CAP), QUANTILES_FLOOR)


def package_versions() -> dict[str, str]:
    """The installed versions of the packages the artifact's numbers depend on."""
    found: dict[str, str] = {}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError as error:
            raise EnvironmentMismatchError(f"Package {name} is not installed") from error
    return found


def ordered_ids_sha256(ids: npt.NDArray[np.int64]) -> str:
    """SHA-256 of the identifiers in row order, as little-endian int64 bytes."""
    return hashlib.sha256(np.ascontiguousarray(ids, dtype="<i8").tobytes()).hexdigest()


def arrays_sha256(arrays: Mapping[str, Floats]) -> str:
    """SHA-256 over the name, dtype, shape and values of each array, in ``ARRAY_NAMES`` order."""
    digest = hashlib.sha256()
    for name in ARRAY_NAMES:
        array = np.ascontiguousarray(arrays[name], dtype="<f8")
        for part in (name.encode(), b"<f8", repr(array.shape).encode(), array.tobytes()):
            digest.update(part)
    return digest.hexdigest()


def _digest_of(manifest: Mapping[str, Any]) -> str:
    """SHA-256 of the manifest without its own ``artifact_sha256``, as canonical JSON."""
    body = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def indicator_name(name: str) -> str:
    return f"{name}{INDICATOR_SUFFIX}"


def _check_canonical_order(
    ids: npt.NDArray[np.int64], times: npt.NDArray[np.int64], rows: int
) -> None:
    """Rows must be in ascending ``TransactionDT``, ties by ascending ``TransactionID`` (DR-008
    §5); neither key is a model input. One int64 identifier and time per fitted row."""
    for what, keys in (("identifiers", ids), ("times", times)):
        if not isinstance(keys, np.ndarray) or keys.dtype != np.int64 or keys.shape != (rows,):
            raise DataValidationError(f"Row {what} must be one int64 value for each of {rows} rows")
    step_time = np.diff(times)
    step_id = np.diff(ids)
    if not ((step_time > 0) | ((step_time == 0) & (step_id > 0))).all():
        raise DataValidationError(
            "Train rows are not in canonical order (TransactionDT, ties by TransactionID)"
        )


def _check_frame(frame: pd.DataFrame, columns: Sequence[str], *, exact: bool = True) -> None:
    """The frame is non-empty and has the declared columns (``exact``: in this order; otherwise
    these columns are present, in any order)."""
    names = tuple(frame.columns)
    if (names != tuple(columns)) if exact else (not set(columns) <= set(names)):
        raise DataValidationError("The frame's columns differ from the declared model inputs")
    if len(frame) == 0:
        raise DataValidationError("A partition must not be empty")


def _numeric_matrix(frame: pd.DataFrame, names: Sequence[str]) -> Floats:
    """The numeric inputs as one float64 matrix; infinite values are refused (missing is NaN)."""
    for name in names:
        if frame[name].dtype != np.float64:
            raise DataValidationError(f"{name}: numeric inputs must be float64")
    matrix: Floats = frame[list(names)].to_numpy(dtype="float64")
    if np.isinf(matrix).any():
        raise DataValidationError("Numeric inputs must not contain infinite values")
    return matrix


def _quantile_transformer(seed: int, actual: int, quantiles: Floats, references: Floats) -> Any:
    """A fitted scikit-learn transformer from stored state (float64 ``quantiles_`` and
    ``references_`` and the settings of the fit)."""
    transformer = QuantileTransformer(
        n_quantiles=actual,
        output_distribution="normal",
        subsample=None,
        random_state=seed,
        copy=True,
    )
    transformer.n_quantiles_ = actual
    transformer.quantiles_ = quantiles
    transformer.references_ = references
    transformer.n_features_in_ = quantiles.shape[1]
    return transformer


@dataclass(frozen=True, eq=False)
class NeuralPreprocessing:
    """A fitted, immutable preprocessing artifact. Building one validates its state in full, so a
    fitted artifact and a reloaded one are held to the same rules."""

    features_fingerprint: str
    seed: int
    missing_label: str
    train_rows: int
    train_ids_sha256: str
    requested_quantiles: int
    actual_quantiles: int
    numeric_inputs: tuple[str, ...]
    categorical_inputs: tuple[str, ...]
    vocabularies: tuple[tuple[str, ...], ...]
    indicators: tuple[str, ...]
    quantiles: Floats
    references: Floats
    std: Floats
    noise_scale: Floats
    versions: Mapping[str, str]
    transformer: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate()
        for name in ARRAY_NAMES:  # private, read-only copies: the artifact cannot be altered
            private = np.array(getattr(self, name), dtype=np.float64, copy=True)
            private.setflags(write=False)
            object.__setattr__(self, name, private)
        object.__setattr__(
            self,
            "transformer",
            _quantile_transformer(
                self.seed, self.actual_quantiles, self.quantiles, self.references
            ),
        )

    # --- validation ---------------------------------------------------------------------------

    def _validate(self) -> None:
        fail = DataValidationError
        if not (isinstance(self.train_rows, int) and self.train_rows > 0):
            raise fail("The Train row count must be a positive integer")
        if self.requested_quantiles != requested_quantiles(self.train_rows):
            raise fail("The requested quantile count disagrees with the Train rows (DR-012)")
        if self.actual_quantiles != min(self.requested_quantiles, self.train_rows):
            raise fail("The actual quantile count disagrees with the requested count")
        n = len(self.numeric_inputs)
        arrays = {name: getattr(self, name) for name in ARRAY_NAMES}
        shapes = {
            "quantiles": (self.actual_quantiles, n),
            "references": (self.actual_quantiles,),
            "std": (n,),
            "noise_scale": (n,),
        }
        for name, array in arrays.items():
            if not isinstance(array, np.ndarray) or array.dtype != np.float64:
                raise fail(f"{name}: must be a float64 array")
            if array.shape != shapes[name]:
                raise fail(f"{name}: shape {array.shape}, expected {shapes[name]}")
            if not np.isfinite(array).all():
                raise fail(f"{name}: values must be finite")
        if self.actual_quantiles > 1 and (np.diff(self.quantiles, axis=0) < 0).any():
            raise fail("quantiles: values must be nondecreasing within every input")
        if not np.array_equal(self.references, np.linspace(0, 1, self.actual_quantiles)):
            raise fail("references: not the grid of the recorded quantile count")
        if (self.std < 0).any() or (self.noise_scale <= 0).any():
            raise fail("std and noise_scale: negative or zero values")
        expected_scale = NOISE_BASE / np.maximum(self.std, NOISE_FLOOR)
        if not np.array_equal(self.noise_scale, expected_scale):
            raise fail("noise_scale: disagrees with std (DR-012)")
        names = (*self.numeric_inputs, *self.categorical_inputs)
        valid = all(isinstance(name, str) and name for name in names)
        if not valid or len(set(names)) != len(names):
            raise fail("Input names must be unique and non-empty")
        if len(self.vocabularies) != len(self.categorical_inputs):
            raise fail("One vocabulary per categorical input is required")
        for name, vocabulary in zip(self.categorical_inputs, self.vocabularies, strict=True):
            if not all(isinstance(label, str) for label in vocabulary):
                raise fail(f"{name}: vocabulary labels must be text")
            tail = vocabulary[1:]
            if not vocabulary or vocabulary[MISSING_INDEX] != self.missing_label:
                raise fail(f"{name}: index {MISSING_INDEX} must be {self.missing_label!r}")
            if self.missing_label in tail or list(tail) != sorted(set(tail)):
                raise fail(f"{name}: vocabulary must be the reserved label, then sorted labels")
        order = {name: i for i, name in enumerate(self.numeric_inputs)}
        positions = [order.get(name, -1) for name in self.indicators]
        if min(positions, default=0) < 0 or positions != sorted(set(positions)):
            raise fail("Indicators must be numeric inputs, unique and in numeric-input order")
        if not (isinstance(self.seed, int) and not isinstance(self.seed, bool) and self.seed >= 0):
            raise fail("The preprocessing seed must be a non-negative integer")
        _ = self.layout  # builds the layout, which checks the names once more

    # --- shape of the inputs --------------------------------------------------------------------

    @property
    def layout(self) -> InputLayout:
        """The neural input layout: numeric inputs, then the indicators; categorical inputs."""
        return InputLayout(
            numeric=(*self.numeric_inputs, *map(indicator_name, self.indicators)),
            categorical=self.categorical_inputs,
            cardinalities=tuple(len(vocabulary) for vocabulary in self.vocabularies),
            missing_indices=(MISSING_INDEX,) * len(self.categorical_inputs),
            category_dropout=CATEGORY_DROPOUT,
        )

    # --- applying the artifact --------------------------------------------------------------------

    def _check(self, frame: pd.DataFrame) -> None:
        _check_frame(frame, (*self.categorical_inputs, *self.numeric_inputs), exact=False)

    def transform_numeric(self, frame: pd.DataFrame) -> Floats:
        """The numeric inputs after the quantile transformation, float64, missing values 0."""
        self._check(frame)
        matrix = _numeric_matrix(frame, self.numeric_inputs)
        transformed: Floats = self.transformer.transform(matrix)
        transformed[np.isnan(matrix)] = 0.0
        if not np.isfinite(transformed).all():
            raise DataValidationError("The transformed numeric inputs are not all finite")
        return transformed

    def missingness_indicators(self, frame: pd.DataFrame) -> Floats:
        """One 0/1 float64 column per indicator, from the frame's own missing values."""
        self._check(frame)
        matrix = _numeric_matrix(frame, self.indicators)
        return np.isnan(matrix).astype(np.float64)

    def category_indices(self, frame: pd.DataFrame) -> npt.NDArray[np.int64]:
        """Vocabulary indices; a value unseen in Train maps to the reserved index 0."""
        self._check(frame)
        return self._indices(frame)[0]

    def _indices(self, frame: pd.DataFrame) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """The indices and, per input, the number of values unseen in Train."""
        columns = np.empty((len(frame), len(self.categorical_inputs)), dtype=np.int64)
        unseen = np.zeros(len(self.categorical_inputs), dtype=np.int64)
        for j, (name, vocabulary) in enumerate(
            zip(self.categorical_inputs, self.vocabularies, strict=True)
        ):
            series = frame[name]
            if series.isna().any():
                raise DataValidationError(
                    f"{name}: missing values must carry the label {self.missing_label!r}"
                )
            found = pd.Index(vocabulary).get_indexer(pd.Index(series.to_numpy(dtype=object)))
            unseen[j] = int((found < 0).sum())
            found[found < 0] = MISSING_INDEX
            columns[:, j] = found
        return columns, unseen

    def to_neural_inputs(self, frame: pd.DataFrame) -> NeuralInputs:
        """The inputs of the neural loop. The only float32 conversion: transformed numeric
        values followed by the indicators, and the category indices."""
        self._check(frame)
        numeric = np.concatenate(
            [self.transform_numeric(frame), self.missingness_indicators(frame)], axis=1
        ).astype(np.float32)
        inputs = NeuralInputs(numeric, self.category_indices(frame))
        inputs.check(self.layout)
        return inputs

    # --- diagnostics ------------------------------------------------------------------------------

    def diagnostics(self, name: str, frame: pd.DataFrame) -> dict[str, Any]:
        """DR-012 diagnostics of a partition: per numeric input the share of non-missing values
        outside the fitted range (the endpoints of the fitted transformation, strictly outside),
        the missing values of inputs without an indicator, and per categorical input the share of
        values unseen in Train. A share is ``None`` (undefined) where its denominator is 0."""
        self._check(frame)
        matrix = _numeric_matrix(frame, self.numeric_inputs)
        present = ~np.isnan(matrix)
        non_missing = present.sum(axis=0)
        below = (matrix < self.quantiles[0]).sum(axis=0)
        above = (matrix > self.quantiles[-1]).sum(axis=0)
        numeric = [
            {
                "name": column,
                "non_missing": int(non_missing[j]),
                "below": int(below[j]),
                "above": int(above[j]),
                "share_outside": _share(int(below[j] + above[j]), int(non_missing[j])),
            }
            for j, column in enumerate(self.numeric_inputs)
        ]
        indicated = set(self.indicators)
        no_indicator = [
            {"name": column, "missing": int(len(frame) - non_missing[j])}
            for j, column in enumerate(self.numeric_inputs)
            if column not in indicated
        ]
        _, unseen = self._indices(frame)
        categorical = []
        for j, column in enumerate(self.categorical_inputs):
            labelled = int((frame[column].to_numpy(dtype=object) != self.missing_label).sum())
            categorical.append(
                {
                    "name": column,
                    "non_missing": labelled,
                    "unseen": int(unseen[j]),
                    "share_unseen": _share(int(unseen[j]), labelled),
                }
            )
        return {
            "partition": name,
            "rows": len(frame),
            "artifact_sha256": self.artifact_sha256,
            "numeric": numeric,
            "missing_without_indicator": no_indicator,
            "missing_without_indicator_total": sum(entry["missing"] for entry in no_indicator),
            "categorical": categorical,
        }

    # --- persistence ---------------------------------------------------------------------

    @property
    def arrays(self) -> dict[str, Floats]:
        return {name: getattr(self, name) for name in ARRAY_NAMES}

    def manifest(self) -> dict[str, Any]:
        """The manifest, including both digests."""
        body: dict[str, Any] = {
            "kind": KIND,
            "format_version": FORMAT_VERSION,
            "specification": "DR-012",
            "features_fingerprint": self.features_fingerprint,
            "seed": self.seed,
            "missing_label": self.missing_label,
            "train": {"rows": self.train_rows, "ordered_ids_sha256": self.train_ids_sha256},
            "quantiles": {
                "requested": self.requested_quantiles,
                "actual": self.actual_quantiles,
            },
            "category_dropout": CATEGORY_DROPOUT,
            "numeric_inputs": list(self.numeric_inputs),
            "categorical_inputs": list(self.categorical_inputs),
            "vocabularies": {
                name: list(vocabulary)
                for name, vocabulary in zip(self.categorical_inputs, self.vocabularies, strict=True)
            },
            "indicators": list(self.indicators),
            "versions": dict(self.versions),
            "arrays_sha256": arrays_sha256(self.arrays),
        }
        body["artifact_sha256"] = _digest_of(body)
        return body

    @property
    def artifact_sha256(self) -> str:
        """The identity every neural trial and refit records."""
        digest: str = self.manifest()["artifact_sha256"]
        return digest

    def save(self, directory: Path) -> str:
        """Write ``manifest.json`` and ``arrays.npz`` to ``directory`` (created if missing),
        refusing to replace either; returns the artifact's SHA-256."""
        manifest_path, arrays_path = directory / MANIFEST_FILE, directory / ARRAYS_FILE
        for path in (manifest_path, arrays_path):
            if path.exists():
                raise FileAccessError(f"{path}: already exists and is never replaced")

        def write_arrays(target: Path) -> None:
            with target.open("wb") as handle:
                np.savez(
                    handle,
                    quantiles=self.quantiles,
                    references=self.references,
                    std=self.std,
                    noise_scale=self.noise_scale,
                )

        with log_step(logger, f"Writing the neural preprocessing artifact to {directory}"):
            write_atomically(arrays_path, write_arrays)
            write_json(manifest_path, self.manifest())
        return self.artifact_sha256


def _share(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def fit_neural_preprocessing(
    frame: pd.DataFrame,
    ids: npt.NDArray[np.int64],
    times: npt.NDArray[np.int64],
    features: FeatureSpec,
    seed: int,
) -> NeuralPreprocessing:
    """Fit the shared preprocessing on the canonical Train frame (DR-012).

    ``ids`` and ``times`` (``TransactionID`` and ``TransactionDT`` of each row) only verify and
    record the row order; they are never inputs. Raises ``DataValidationError`` for rows out of
    canonical order, for numeric inputs that are infinite or entirely missing in Train (checked
    before any draw), and for categorical missing values that are not labelled."""
    _check_frame(frame, features.model_inputs)
    _check_canonical_order(ids, times, len(frame))
    rows = len(frame)
    numeric_inputs = features.numeric_inputs
    matrix = _numeric_matrix(frame, numeric_inputs)
    observed = (~np.isnan(matrix)).sum(axis=0)
    empty = [name for name, count in zip(numeric_inputs, observed, strict=True) if not count]
    if empty:
        raise DataValidationError(f"Numeric inputs entirely missing in Train: {empty}")
    requested = requested_quantiles(rows)
    actual = min(requested, rows)
    with log_step(logger, f"Fitting the neural preprocessing on {rows} Train rows"):
        vocabularies = []
        for name in features.categorical:
            if frame[name].isna().any():
                raise DataValidationError(
                    f"{name}: missing values must carry the label {features.missing_label!r}"
                )
            labels = set(frame[name].unique().tolist()) - {features.missing_label}
            vocabularies.append((features.missing_label, *sorted(labels)))
        rng = np.random.default_rng(seed)
        quantiles = np.empty((actual, len(numeric_inputs)), dtype=np.float64)
        std = np.empty(len(numeric_inputs), dtype=np.float64)
        scale = np.empty(len(numeric_inputs), dtype=np.float64)
        progress = ProgressLogger(logger, "Fitting numeric inputs", len(numeric_inputs))
        for j in range(len(numeric_inputs)):
            column = matrix[:, j]
            present = ~np.isnan(column)
            values = column[present]
            std[j] = np.std(values)
            scale[j] = NOISE_BASE / max(float(std[j]), NOISE_FLOOR)
            noisy = column.copy()
            draws = rng.standard_normal(size=len(values), dtype=np.float64)
            noisy[present] = values + draws * scale[j]
            fitted = QuantileTransformer(
                n_quantiles=actual,
                output_distribution="normal",
                subsample=None,
                random_state=seed,
                copy=True,
            ).fit(noisy.reshape(-1, 1))
            quantiles[:, j] = fitted.quantiles_[:, 0]
            progress.advance()
        references = np.linspace(0, 1, actual)
        missing = np.isnan(matrix).sum(axis=0)
        indicators = tuple(
            name for name, count in zip(numeric_inputs, missing, strict=True) if count
        )
    artifact = NeuralPreprocessing(
        features_fingerprint=features_fingerprint(features),
        seed=seed,
        missing_label=features.missing_label,
        train_rows=rows,
        train_ids_sha256=ordered_ids_sha256(ids),
        requested_quantiles=requested,
        actual_quantiles=actual,
        numeric_inputs=tuple(numeric_inputs),
        categorical_inputs=tuple(features.categorical),
        vocabularies=tuple(vocabularies),
        indicators=indicators,
        quantiles=quantiles,
        references=references,
        std=std,
        noise_scale=scale,
        versions=package_versions(),
    )
    logger.info(
        "Neural preprocessing fitted: %d numeric, %d indicators, %d categorical; sha256 %s",
        len(numeric_inputs),
        len(indicators),
        len(features.categorical),
        artifact.artifact_sha256,
    )
    return artifact


def _read_arrays(path: Path) -> dict[str, Floats]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(ARRAY_NAMES):
                found = sorted(archive.files)
                raise DataValidationError(f"{path}: arrays {found}, expected {ARRAY_NAMES}")
            return {name: np.asarray(archive[name]) for name in ARRAY_NAMES}
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error})") from error
    except ValueError as error:
        raise DataValidationError(f"{path}: not a valid array archive ({error})") from error


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error.strerror})") from error
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as error:
        raise DataValidationError(f"{path}: not valid JSON ({error})") from error
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise DataValidationError(f"{path}: not a {KIND} manifest with the expected keys")
    return manifest


def _is_text_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_manifest_types(manifest: Mapping[str, Any]) -> None:
    """The manifest's nested structure and types, before any value is used."""
    train, quantiles = manifest["train"], manifest["quantiles"]
    vocabularies, versions = manifest["vocabularies"], manifest["versions"]
    checks = {
        "kind": isinstance(manifest["kind"], str),
        "format_version": _is_integer(manifest["format_version"]),
        "features_fingerprint": isinstance(manifest["features_fingerprint"], str),
        "seed": _is_integer(manifest["seed"]),
        "missing_label": isinstance(manifest["missing_label"], str),
        "train": isinstance(train, dict)
        and set(train) == {"rows", "ordered_ids_sha256"}
        and _is_integer(train["rows"])
        and isinstance(train["ordered_ids_sha256"], str),
        "quantiles": isinstance(quantiles, dict)
        and set(quantiles) == {"requested", "actual"}
        and _is_integer(quantiles["requested"])
        and _is_integer(quantiles["actual"]),
        "category_dropout": isinstance(manifest["category_dropout"], float),
        "numeric_inputs": _is_text_list(manifest["numeric_inputs"]),
        "categorical_inputs": _is_text_list(manifest["categorical_inputs"]),
        "indicators": _is_text_list(manifest["indicators"]),
        "vocabularies": isinstance(vocabularies, dict)
        and all(_is_text_list(labels) for labels in vocabularies.values())
        and _is_text_list(manifest["categorical_inputs"])
        and set(vocabularies) == set(manifest["categorical_inputs"]),
        "versions": isinstance(versions, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in versions.items()),
        "arrays_sha256": isinstance(manifest["arrays_sha256"], str),
        "artifact_sha256": isinstance(manifest["artifact_sha256"], str),
    }
    wrong = [name for name, valid in checks.items() if not valid]
    if wrong:
        raise DataValidationError(f"The manifest has malformed entries: {wrong}")


def _declared_inputs_agree(manifest: Mapping[str, Any], features: FeatureSpec) -> bool:
    return bool(
        manifest["numeric_inputs"] == list(features.numeric_inputs)
        and manifest["categorical_inputs"] == list(features.categorical)
        and manifest["missing_label"] == features.missing_label
        and manifest["category_dropout"] == CATEGORY_DROPOUT
    )


def load_neural_preprocessing(
    directory: Path,
    features: FeatureSpec,
    seed: int,
    *,
    train_ids_sha256: str | None = None,
) -> NeuralPreprocessing:
    """Reload an artifact for the given feature declaration and declared preprocessing seed
    (and, if given, the ordered Train identity it must have been fitted on).

    Checks, in this order: the manifest and array files and the manifest's structure; kind and
    format version; the recorded package versions against the installed ones; both digests; the
    declaration fingerprint, the declared inputs, the seed and the Train identity; then the full
    semantic validation of the state (shapes, ordering, vocabularies, counts), which runs before
    any scikit-learn object is rebuilt."""
    with log_step(logger, f"Loading the neural preprocessing artifact from {directory}"):
        manifest = _read_manifest(directory / MANIFEST_FILE)
        _check_manifest_types(manifest)
        arrays = _read_arrays(directory / ARRAYS_FILE)
        if manifest["kind"] != KIND or manifest["format_version"] != FORMAT_VERSION:
            raise DataValidationError(f"{directory}: not a {KIND} artifact of this format")
        installed = package_versions()
        if manifest["versions"] != installed:
            raise EnvironmentMismatchError(
                f"The artifact was fitted with {manifest['versions']}, installed: {installed}"
            )
        try:
            recomputed = arrays_sha256(arrays)
        except (ValueError, TypeError) as error:
            raise DataValidationError(f"{directory}: arrays are not float64 data") from error
        if manifest["arrays_sha256"] != recomputed:
            raise DataValidationError(f"{directory}: the arrays differ from their recorded SHA-256")
        if manifest["artifact_sha256"] != _digest_of(manifest):
            raise DataValidationError(f"{directory}: the manifest does not match its SHA-256")
        if manifest["features_fingerprint"] != features_fingerprint(features):
            raise SpecificationError("The artifact was fitted for a different feature declaration")
        if not _declared_inputs_agree(manifest, features):
            raise SpecificationError("The artifact's inputs differ from the feature declaration")
        recorded = manifest["train"]["ordered_ids_sha256"]
        if train_ids_sha256 is not None and recorded != train_ids_sha256:
            raise SpecificationError("The artifact was fitted on different Train rows")
        if manifest["seed"] != seed:
            raise SpecificationError(
                f"The artifact used preprocessing seed {manifest['seed']}, declared: {seed}"
            )
        artifact = NeuralPreprocessing(
            features_fingerprint=manifest["features_fingerprint"],
            seed=manifest["seed"],
            missing_label=manifest["missing_label"],
            train_rows=manifest["train"]["rows"],
            train_ids_sha256=manifest["train"]["ordered_ids_sha256"],
            requested_quantiles=manifest["quantiles"]["requested"],
            actual_quantiles=manifest["quantiles"]["actual"],
            numeric_inputs=tuple(manifest["numeric_inputs"]),
            categorical_inputs=tuple(manifest["categorical_inputs"]),
            vocabularies=tuple(
                tuple(manifest["vocabularies"][name]) for name in manifest["categorical_inputs"]
            ),
            indicators=tuple(manifest["indicators"]),
            quantiles=arrays["quantiles"],
            references=arrays["references"],
            std=arrays["std"],
            noise_scale=arrays["noise_scale"],
            versions=manifest["versions"],
        )
        if artifact.artifact_sha256 != manifest["artifact_sha256"]:
            raise DataValidationError(f"{directory}: the rebuilt state differs from its manifest")
    return artifact
