"""CatBoost model of DR-011 on the common interface (``models.base``).

* **Inputs (DR-008 §5):** the canonical model inputs; the 49 categorical inputs, named from the
  feature declaration, are passed as their canonical text labels (``to_catboost``). CatBoost's
  ordered target statistics follow the row order, so with ``has_time`` the Train rows must be in
  time order: ``TransactionDT`` non-decreasing, ties by increasing ``TransactionID``. The order is
  asserted from the rows' times and identifiers, never assumed and never repaired by sorting.
* **Settings (DR-013 §3):** the shared and device-specific settings of ``config/models.yaml`` and
  the run's seed. Hyperparameters are first mapped to canonical names with CatBoost's own
  synonym groups (``catboost.core._process_synonyms``, checked by a unit test). A hyperparameter
  naming a reserved setting stops the run. Reserved are the settings of both device sections,
  the seed, the declared inputs, CatBoost's own class weighting (the protocol's weights are
  explicit), the evaluation and early-stopping controls this module sets, the categorical
  statistics (CTRs) beyond the device lists, and the settings that write files, redirect logging
  or skip the final statistics. Every hyperparameter must be a CatBoost parameter and must not be
  ``None`` (which would leave CatBoost's default).
* **Explicit settings (DR-013 §3):** the settings that DR-013 fixes in the training rules,
  explicitly and identically on both devices, must be given: ``learning_rate`` (never the
  automatic, device-dependent value), ``model_size_reg``, ``fold_permutation_block``,
  ``permutation_count``, ``random_strength``, ``l2_leaf_reg``, ``leaf_estimation_iterations``,
  ``bagging_temperature``, ``eval_metric`` and ``metric_period``, and ``iterations``.
* **V1 objective (DR-015 §3, §6):** the research configuration sets ``eval_metric`` to
  ``"v1_objective"`` and ``boost_from_average`` to ``False``, explicitly. With early stopping, the
  shared objective (``models.v1_objective``) is passed as ``objective`` and becomes CatBoost's
  user-defined evaluation metric, which runs on the GPU learner as on the CPU; the recorded
  parameters (also stored inside the model) carry the name and the objective's identity instead
  of the metric object, so they stay JSON. In a fixed-length refit the name is recorded and not
  passed to CatBoost, which has no validation set to evaluate. After fitting, the resolved
  ``boost_from_average`` must be ``False``, and CatBoost's best iteration must equal the earliest
  best of the objective's own history. Without the research configuration, a built-in metric name
  is required (the GPU acceptance probe's path).
* **Fitting:** with V1 early stopping, ``od_wait`` must also be given; the overfitting detector is
  ``Iter`` and the model is cut back to its best iteration. With a fixed iteration count (the CPU
  refit of DR-013 §7), exactly that many trees are grown without validation, or the run stops.
  After fitting, the resolved parameters must confirm every fixed setting, including the loss,
  time order, Train-only counter statistics, CTR types, seed and device.
* **Prediction (DR-013 §5):** always ``predict_proba(..., task_type="CPU")`` with the kept tree
  count and the CPU thread count of ``config/models.yaml``.
* **Logging:** CatBoost's output goes to the run log; its error stream as warnings. User
  callbacks are not supported on GPU, so progress is CatBoost's own line every
  ``metric_period`` iterations.
* **Persistence:** the binary model (``cbm``), which carries the complete requested-parameter
  record in its own metadata, and a manifest with the model's SHA-256. Reloading checks every
  manifest field against the request, the requested parameters against the record inside the
  model, and the loaded model's inputs, tree count, classes and resolved settings.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

from review_capacity.config import Device, FeatureSpec, ModelSettings
from review_capacity.errors import (
    DataValidationError,
    FileAccessError,
    ModelError,
    SpecificationError,
)
from review_capacity.features.model_inputs import to_catboost
from review_capacity.files import write_atomically, write_json
from review_capacity.logging_setup import log_step
from review_capacity.models.base import (
    ModelData,
    Scores,
    ScoringModel,
    check_fit_arguments,
    checked_scores,
    merge_parameters,
)
from review_capacity.models.registry import Configuration
from review_capacity.models.run_record import jsonable
from review_capacity.models.v1_objective import (
    OBJECTIVE_NAME,
    CatBoostObjective,
    TrainingHistory,
    V1Objective,
    check_identity,
)

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2
MODEL_FILE = "model.cbm"
MANIFEST_FILE = "manifest.json"
# CatBoost's synonym groups (catboost.core._process_synonyms, CatBoost 1.2.10): alternative
# name -> the name CatBoost keeps.
SYNONYMS = {
    "objective": "loss_function",
    "eta": "learning_rate",
    "max_bin": "border_count",
    "max_depth": "depth",
    "colsample_bylevel": "rsm",
    "random_state": "random_seed",
    "reg_lambda": "l2_leaf_reg",
    "n_estimators": "iterations",
    "num_boost_round": "iterations",
    "num_trees": "iterations",
    "early_stopping_rounds": "od_wait",
    "custom_loss": "custom_metric",
    "num_leaves": "max_leaves",
    "min_child_samples": "min_data_in_leaf",
}
# Parameters of the native options that the Python constructor does not list (DR-013 §3).
NATIVE_ONLY = frozenset({"permutation_count"})
# Settings DR-013 §3 requires explicitly in the training rules, and the tree count.
EXPLICIT = (
    "iterations",
    "learning_rate",
    "model_size_reg",
    "fold_permutation_block",
    "permutation_count",
    "random_strength",
    "l2_leaf_reg",
    "leaf_estimation_iterations",
    "bagging_temperature",
    "eval_metric",
    "metric_period",
)
POSITIVE_INTEGERS = (
    "iterations",
    "fold_permutation_block",
    "permutation_count",
    "leaf_estimation_iterations",
    "metric_period",
)
# Reserved besides the device settings: the seed; the inputs (from the feature declaration);
# CatBoost's own class weighting; the evaluation and early-stopping controls set here; and the
# settings that write files, redirect logging or change the declared quantisation or bundling.
RESERVED = frozenset(
    {
        "random_seed",
        "cat_features",
        "text_features",
        "embedding_features",
        "ignored_features",
        "tokenizers",
        "dictionaries",
        "feature_calcers",
        "text_processing",
        "class_weights",
        "auto_class_weights",
        "scale_pos_weight",
        "class_names",
        "classes_count",
        "target_border",
        "eval_fraction",
        "use_best_model",
        "best_model_min_trees",
        "od_type",
        "od_pval",
        "train_dir",
        "allow_writing_files",
        "save_snapshot",
        "snapshot_file",
        "snapshot_interval",
        "name",
        "verbose",
        "silent",
        "logging_level",
        "callback",
        "input_borders",
        "output_borders",
        "per_float_feature_quantization",
        "sparse_features_conflict_fraction",
        "dev_efb_max_buckets",
        "per_feature_ctr",
        "ctr_description",
        "final_ctr_computation_mode",
        "metadata",
    }
)
# Controls this module sets on every fit; with EXPLICIT, they must be in every saved record.
CONTROLS = ("use_best_model", "allow_writing_files", "logging_level")
# The model-metadata key that holds the requested-parameter record inside the model file.
RECORD_KEY = "review_capacity_requested"
# Fixed settings that CatBoost's resolved parameters do not report: get_all_params() omits
# thread_count (DR-013 §3) and, in CatBoost 1.2.10, sampling_unit; devices is GPU-only. The
# system settings (thread_count, devices) are checked against the model's own system options
# (metadata ``params``); sampling_unit, which CatBoost does not serialise, through the saved
# record (manifest against the fixed settings, and the manifest against the record in the model).
NOT_RESOLVED = frozenset({"thread_count", "devices", "sampling_unit"})
SYSTEM_SETTINGS = ("thread_count", "devices")
CTR_LISTS = ("simple_ctr", "combinations_ctr")
TASK_TYPES = {"cpu": "CPU", "gpu": "GPU"}


@cache
def catboost_parameters() -> frozenset[str]:
    """Every canonical parameter name of the installed CatBoost: the parameters of its classifier
    constructor without the synonyms, and the native options it does not list."""
    from catboost import CatBoostClassifier

    names = set(inspect.signature(CatBoostClassifier.__init__).parameters) - {"self"}
    return frozenset((names - set(SYNONYMS)) | NATIVE_ONLY)


class _LogStream:
    """A stream for CatBoost's output: complete lines go to this module's logger."""

    def __init__(self, level: int) -> None:
        self.level = level
        self.pending = ""

    def write(self, message: str) -> None:
        self.pending += message
        *lines, self.pending = self.pending.split("\n")
        for line in lines:
            if line.strip():
                logger.log(self.level, "%s", line.rstrip())

    def flush(self) -> None:
        if self.pending.strip():
            logger.log(self.level, "%s", self.pending.rstrip())
        self.pending = ""


def _check_values(
    params: Mapping[str, Any], early_stopping: bool, objective: V1Objective | None
) -> None:
    required = [*EXPLICIT, "od_wait"] if early_stopping else list(EXPLICIT)
    missing = [name for name in required if name not in params]
    if missing:
        raise SpecificationError(f"These CatBoost settings must be set explicitly: {missing}")
    for name in POSITIVE_INTEGERS + (("od_wait",) if early_stopping else ()):
        value = params[name]
        if type(value) is not int or value <= 0:
            raise SpecificationError(f"{name} must be a positive integer, not {value!r}")
    rate = params["learning_rate"]
    if (
        isinstance(rate, bool)
        or not isinstance(rate, int | float)
        or not math.isfinite(rate)
        or rate <= 0
    ):
        raise SpecificationError(f"learning_rate must be a positive number, not {rate!r}")
    metric = params["eval_metric"]
    if not isinstance(metric, str) or not metric.strip():
        raise SpecificationError(f"eval_metric must name an evaluation metric, not {metric!r}")
    if objective is not None and metric != OBJECTIVE_NAME:
        raise SpecificationError(
            f"The shared V1 objective is the evaluation: eval_metric must be "
            f"{OBJECTIVE_NAME!r}, not {metric!r}"
        )
    if metric == OBJECTIVE_NAME:
        if early_stopping and objective is None:
            raise SpecificationError(f"eval_metric {OBJECTIVE_NAME!r} needs the objective argument")
        if params.get("boost_from_average") is not False:
            raise SpecificationError(
                f"boost_from_average must be set explicitly to False (DR-015 §6), "
                f"not {params.get('boost_from_average')!r}"
            )


def _binary_classes(classes: Any) -> bool:
    """Whether the model's class labels are exactly 0 and 1, in that order, stored as numbers or
    as text; no conversion that could turn another label into 0 or 1."""
    labels = list(classes)
    if [str(label) for label in labels] == ["0", "1"]:
        return all(
            isinstance(label, str | int | np.integer) and not isinstance(label, bool)
            for label in labels
        )
    return False


def _check_resolved(resolved: Mapping[str, Any], fixed: Mapping[str, Any]) -> None:
    """The resolved parameters must show every fixed setting (asserted, not assumed: DR-013 §3).

    Scalars are compared exactly; each CTR list by the types of its resolved descriptors, which
    CatBoost expands from the declared type names (``Borders`` -> ``Borders:CtrBorderCount=...``);
    the settings CatBoost does not report are checked through the saved record.
    """
    wrong: dict[str, Any] = {}
    for name, value in fixed.items():
        if name in NOT_RESOLVED:
            continue
        observed = resolved.get(name)
        if name in CTR_LISTS:
            descriptors = observed if isinstance(observed, list) else []
            if (
                not descriptors
                or not all(isinstance(d, str) for d in descriptors)
                or {d.split(":", 1)[0] for d in descriptors} != set(value)
            ):
                wrong[name] = observed
        elif type(observed) is not type(value) or observed != value:
            wrong[name] = observed
    if wrong:
        raise ModelError(f"Resolved CatBoost parameters differ from the fixed settings: {wrong}")


def _check_boost_from_average(resolved: Mapping[str, Any]) -> None:
    """The resolved parameters must show ``boost_from_average`` as ``False`` (DR-015 §6): a
    missing entry or any other value stops the run."""
    observed = resolved.get("boost_from_average")
    if observed is not False:
        raise ModelError(f"Resolved boost_from_average must be False, not {observed!r}")


def _canonical(value: object) -> str:
    """JSON text that keeps value types apart (``false``, ``0`` and ``0.0`` differ), independent of
    key order."""
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _check_system(metadata: Mapping[str, Any], fixed: Mapping[str, Any]) -> None:
    """The model's own system options (metadata ``params``) must show the fixed thread count and
    devices, by value and type."""
    try:
        system = json.loads(metadata["params"])["system_options"]
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError("The model's metadata has no readable system options") from error
    if not isinstance(system, dict):
        raise ModelError("The model's system options are not a JSON object")
    wrong = {
        name: system.get(name)
        for name in SYSTEM_SETTINGS
        if name in fixed
        and (type(system.get(name)) is not type(fixed[name]) or system.get(name) != fixed[name])
    }
    if wrong:
        raise ModelError(f"The model's system options differ from the fixed settings: {wrong}")


def _check_record(requested: Mapping[str, Any], embedded: object) -> None:
    """The manifest's requested parameters must equal the record saved inside the model, value
    and JSON type alike, and the record must hold every explicit setting and control."""
    if not isinstance(embedded, dict):
        raise ModelError("Saved requested parameters do not match the model's own record")
    try:
        same = _canonical(embedded) == _canonical(requested)
    except (TypeError, ValueError) as error:
        raise ModelError("Saved requested parameters are not valid JSON values") from error
    if not same:
        raise ModelError("Saved requested parameters do not match the model's own record")
    missing = sorted({*EXPLICIT, *CONTROLS} - embedded.keys())
    if missing:
        raise ModelError(f"Saved requested parameters lack required settings: {missing}")


def _manifest_fields(
    manifest: object,
    *,
    configuration: Configuration,
    seed: int,
    features: FeatureSpec,
    device: Device,
    fixed: Mapping[str, Any],
    library_version: str,
) -> tuple[int, dict[str, Any], str]:
    """The tree count, requested parameters and model SHA-256 of a manifest, after checking
    every field against the request."""
    if not isinstance(manifest, dict):
        raise ModelError("Saved model manifest must be a JSON object")
    try:
        version = manifest["format_version"]
        identity = (
            manifest["library"],
            manifest["library_version"],
            manifest["configuration"],
            manifest["seed"],
            manifest["device"],
            manifest["inputs"],
            manifest["categorical"],
        )
        iterations = manifest["iterations"]
        requested = manifest["requested"]
        model_sha256 = manifest["model_sha256"]
    except KeyError as error:
        raise ModelError(f"Saved model manifest is incomplete: missing {error}") from error
    if version != FORMAT_VERSION or type(version) is not int:
        raise ModelError(f"Unsupported model format version {version!r}")
    if type(manifest["seed"]) is not int:
        raise ModelError(f"Saved model seed must be an integer, not {manifest['seed']!r}")
    expected = (
        "catboost",
        library_version,
        configuration.name,
        seed,
        device,
        list(features.model_inputs),
        list(features.categorical),
    )
    if identity != expected:
        raise ModelError(
            "Saved model does not match the requested library version, configuration, seed, "
            "training device or inputs"
        )
    if type(iterations) is not int or iterations <= 0:
        raise ModelError(f"Saved tree count must be a positive integer, not {iterations!r}")
    if not isinstance(requested, dict):
        raise ModelError("Saved requested parameters must be a JSON object")
    for name, value in fixed.items():
        saved = requested.get(name)
        if type(saved) is not type(value) or saved != value:  # 0 is not False, 1.0 is not 1
            raise ModelError(f"Saved requested parameter {name!r} does not match the fixed setting")
    if (
        requested.get("eval_metric") == OBJECTIVE_NAME
        and requested.get("boost_from_average") is not False
    ):
        raise ModelError("Saved requested boost_from_average must be False for the V1 objective")
    if OBJECTIVE_NAME in requested:
        check_identity(requested[OBJECTIVE_NAME])
        if requested.get("eval_metric") != OBJECTIVE_NAME:
            raise ModelError("A model fitted on the V1 objective must record its eval_metric")
    if not isinstance(model_sha256, str) or len(model_sha256) != 64:
        raise ModelError("Saved model SHA-256 is missing or malformed")
    return iterations, requested, model_sha256


class CatBoostModel(ScoringModel):
    """CatBoost on the common interface: GPU or CPU training, CPU prediction."""

    def __init__(
        self,
        configuration: Configuration,
        seed: int,
        *,
        settings: ModelSettings,
        features: FeatureSpec,
        device: Device,
    ) -> None:
        super().__init__(configuration, seed, settings=settings, features=features, device=device)
        if configuration.model.name != "catboost":
            raise ValueError(f"{configuration.name} is not a CatBoost configuration")
        self._model: Any = None
        self._requested: dict[str, Any] = {}
        self._iterations = 0
        self._history: TrainingHistory | None = None

    def _fixed_parameters(self) -> dict[str, Any]:
        return {**self.settings.catboost.params(self.device), "random_seed": self.seed}

    def _reserved(self) -> set[str]:
        catboost = self.settings.catboost
        return set(catboost.params("cpu")) | set(catboost.params("gpu")) | RESERVED

    def _prediction_threads(self) -> int:
        return int(self.settings.catboost.params("cpu")["thread_count"])

    def _fitted(self) -> Any:
        if self._model is None:
            raise ModelError(f"{self.configuration.name} is not fitted")
        return self._model

    def _pool(self, features: pd.DataFrame, data: ModelData | None = None) -> Any:
        from catboost import Pool

        self.check_inputs(features)
        return Pool(
            to_catboost(features, self.features),
            label=None if data is None else data.labels,
            weight=None if data is None else data.weights,
            cat_features=list(self.features.categorical),
        )

    def _parameters_for_fit(
        self,
        hyperparameters: Mapping[str, Any],
        iterations: int | None,
        objective: V1Objective | None = None,
    ) -> dict[str, Any]:
        requested = merge_parameters({}, hyperparameters, aliases=SYNONYMS)
        clash = sorted(self._reserved() & requested.keys())
        if clash:
            raise SpecificationError(
                f"Hyperparameters cannot override fixed research settings: {clash}"
            )
        known = catboost_parameters()
        for name, value in requested.items():
            if name not in known:
                raise SpecificationError(f"Unknown CatBoost parameter {name!r}")
            if value is None:
                raise SpecificationError(f"{name}: None would leave CatBoost's default")
        early_stopping = iterations is None
        if not early_stopping:
            requested.pop("od_wait", None)
            requested["iterations"] = iterations
        _check_values(requested, early_stopping, objective)
        params = {
            **self._fixed_parameters(),
            **requested,
            "use_best_model": early_stopping,
            "allow_writing_files": False,
            "logging_level": "Verbose",  # one line every metric_period iterations
        }
        if early_stopping:
            params["od_type"] = "Iter"
        if objective is not None:
            params[OBJECTIVE_NAME] = objective.identity
        return params

    def fit(
        self,
        train: ModelData,
        valid: ModelData | None,
        hyperparameters: Mapping[str, Any],
        *,
        iterations: int | None = None,
        objective: V1Objective | None = None,
    ) -> None:
        """Fit as ``ScoringModel.fit``; with ``objective`` (the shared V1 objective, built from
        ``valid``), CatBoost evaluates and early-stops on it."""
        from catboost import CatBoostClassifier, CatBoostError

        check_fit_arguments(valid, iterations)
        if objective is not None:
            if valid is None:
                raise SpecificationError("The V1 objective needs early stopping on V1")
            objective.check_partition(valid)
        params = self._parameters_for_fit(hyperparameters, iterations, objective)
        self.check_inputs(train.features)
        if not train.in_time_order():
            raise DataValidationError(
                "CatBoost Train rows must be in time order (TransactionDT, then TransactionID)"
            )
        name = self.configuration.name
        rows = f"{len(train.labels)} Train rows"
        if valid is not None:
            rows += f", early stopping on {len(valid.labels)} V1 rows"
        out, err = _LogStream(logging.INFO), _LogStream(logging.WARNING)
        try:
            with log_step(logger, f"Fit {name} on {self.device} ({rows})"):
                model = CatBoostClassifier()
                record = json.dumps(jsonable(params), sort_keys=True)
                native = {k: v for k, v in params.items() if k != OBJECTIVE_NAME}
                recorder = None if objective is None else objective.recorder()
                if recorder is not None:
                    native["eval_metric"] = CatBoostObjective(recorder)
                elif native["eval_metric"] == OBJECTIVE_NAME:
                    del native["eval_metric"]  # no validation set to evaluate (fixed-length refit)
                model.set_params(**native, metadata={RECORD_KEY: record})
                model.fit(
                    self._pool(train.features, train),
                    eval_set=None if valid is None else self._pool(valid.features, valid),
                    log_cout=out,
                    log_cerr=err,
                )
                kept = int(model.tree_count_)
                if valid is not None and kept != model.get_best_iteration() + 1:
                    raise ModelError(f"{name}: the model was not cut back to its best iteration")
                if iterations is not None and kept != iterations:
                    raise ModelError(
                        f"{name}: {iterations} refit trees requested, CatBoost grew {kept}"
                    )
                if not _binary_classes(model.classes_):
                    raise ModelError(f"{name}: class labels are not 0 and 1")
                resolved = model.get_all_params()
                _check_resolved(resolved, self._fixed_parameters())
                _check_system(dict(model.get_metadata().items()), self._fixed_parameters())
                if params["eval_metric"] == OBJECTIVE_NAME:
                    _check_boost_from_average(resolved)
                history = None
                if recorder is not None:
                    history = recorder.history(
                        patience=params["od_wait"], maximum=params["iterations"]
                    )
                    evaluated = model.get_evals_result()["validation"]
                    trained = {len(values) for values in evaluated.values()}
                    if trained != {history.stopped_at}:
                        raise ModelError(
                            f"{name}: CatBoost's evaluation history has {sorted(trained)} rounds, "
                            f"{history.stopped_at} V1 evaluations were recorded"
                        )
                    if kept != history.best_checkpoint:
                        raise ModelError(
                            f"{name}: CatBoost kept {kept} trees, the earliest best V1 "
                            f"checkpoint is {history.best_checkpoint}"
                        )
                    if recorder.other_evaluations:
                        logger.info(
                            "%s: %d evaluations of other sets were not scored",
                            name,
                            recorder.other_evaluations,
                        )
        except CatBoostError as error:
            raise ModelError(f"{name}: CatBoost training failed: {error}") from error
        finally:
            out.flush()
            err.flush()
        self._model = model
        self._requested = params
        self._iterations = kept
        self._history = history
        logger.info("%s fitted: %d trees kept", name, kept)

    def predict(self, features: pd.DataFrame) -> Scores:
        from catboost import CatBoostError

        model = self._fitted()
        try:
            probabilities = model.predict_proba(
                self._pool(features),
                ntree_end=self._iterations,
                thread_count=self._prediction_threads(),
                task_type="CPU",
            )
        except CatBoostError as error:
            raise ModelError(f"{self.configuration.name}: prediction failed: {error}") from error
        values = np.asarray(probabilities)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ModelError(f"{self.configuration.name}: unexpected probability shape")
        return checked_scores(values[:, 1], len(features), self.configuration.name)

    def save(self, directory: Path) -> None:
        import catboost
        from catboost import CatBoostError

        model = self._fitted()
        path = directory / MODEL_FILE
        try:
            write_atomically(path, lambda target: model.save_model(str(target), format="cbm"))
        except CatBoostError as error:
            name = self.configuration.name
            raise ModelError(f"{name}: model cannot be saved: {error}") from error
        try:
            content = path.read_bytes()
        except OSError as error:
            raise FileAccessError(f"{path}: saved model cannot be read back") from error
        write_json(  # written last, with the model's SHA-256
            directory / MANIFEST_FILE,
            {
                "format_version": FORMAT_VERSION,
                "library": "catboost",
                "library_version": catboost.__version__,
                "configuration": self.configuration.name,
                "seed": self.seed,
                "device": self.device,
                "iterations": self._iterations,
                "inputs": list(self.features.model_inputs),
                "categorical": list(self.features.categorical),
                "requested": jsonable(self._requested),
                "model_sha256": hashlib.sha256(content).hexdigest(),
            },
        )

    @classmethod
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
        """A saved model; ``device`` is the device it was trained on (prediction is on CPU)."""
        import catboost
        from catboost import CatBoostClassifier, CatBoostError

        try:
            manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
            content = (directory / MODEL_FILE).read_bytes()
        except OSError as error:
            raise FileAccessError(f"{directory}: saved model cannot be read") from error
        except ValueError as error:
            raise ModelError(f"{directory}: saved model manifest is not valid JSON") from error
        model = cls(configuration, seed, settings=settings, features=features, device=device)
        iterations, requested, model_sha256 = _manifest_fields(
            manifest,
            configuration=configuration,
            seed=seed,
            features=features,
            device=device,
            fixed=model._fixed_parameters(),
            library_version=catboost.__version__,
        )
        if hashlib.sha256(content).hexdigest() != model_sha256:
            raise ModelError(f"{directory}: saved model and manifest do not match")
        try:
            loaded = CatBoostClassifier().load_model(blob=content)
            resolved = loaded.get_all_params()
            metadata = dict(loaded.get_metadata().items())
            names = list(loaded.feature_names_)
            cat_indices = list(loaded.get_cat_feature_indices())
            trees = int(loaded.tree_count_)
            classes = loaded.classes_
        except CatBoostError as error:
            raise ModelError(f"{directory}: CatBoost model cannot be loaded: {error}") from error
        if names != list(features.model_inputs):
            raise ModelError("Saved model's feature names do not match the inputs")
        declared = [features.model_inputs.index(name) for name in features.categorical]
        if sorted(cat_indices) != sorted(declared):
            raise ModelError("Saved model's categorical inputs do not match the declaration")
        if trees != iterations:
            raise ModelError("Saved tree count does not match the model")
        if not _binary_classes(classes):
            raise ModelError("Saved model's class labels are not 0 and 1")
        _check_resolved(resolved, model._fixed_parameters())
        _check_system(metadata, model._fixed_parameters())
        if requested.get("eval_metric") == OBJECTIVE_NAME:
            _check_boost_from_average(resolved)
        try:
            embedded = json.loads(metadata[RECORD_KEY])
        except (KeyError, ValueError) as error:
            raise ModelError(f"{directory}: the model holds no valid parameter record") from error
        _check_record(requested, embedded)
        model._model = loaded
        model._requested = requested
        model._iterations = iterations
        return model

    @property
    def parameters(self) -> dict[str, Any]:
        model = self._fitted()
        return {
            "requested": self._requested,
            "resolved": model.get_all_params(),
            "metadata": dict(model.get_metadata().items()),
            "prediction": {
                "backend": "cpu",
                "task_type": "CPU",
                "thread_count": self._prediction_threads(),
            },
        }

    @property
    def training_history(self) -> TrainingHistory | None:
        """The V1 checkpoint history of the fit that early-stopped on the shared objective
        (``None`` for any other fit, and for a reloaded model: the history belongs to the trial's
        record)."""
        self._fitted()
        return self._history

    @property
    def fitted_iterations(self) -> int:
        self._fitted()
        return self._iterations
