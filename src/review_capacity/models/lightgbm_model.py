"""LightGBM model of DR-011 on the common interface (``models.base``).

* **Inputs (DR-008 §5):** the canonical model inputs; the 49 categorical inputs, named from the
  feature declaration, become pandas categoricals with category lists fixed from Train rows only,
  so a category unseen in Train becomes missing, as in LightGBM's own prediction behaviour.
* **Device (DR-013 §1–§2):** training on the CPU only, with deterministic training; LightGBM's
  CUDA learner fails on categorical features with more than 256 categories, so the GPU is refused.
* **Settings (DR-013 §2):** the shared and CPU settings of ``config/models.yaml`` and the run's
  seed. Hyperparameters are first mapped to canonical names with LightGBM's own alias table; a
  hyperparameter naming a reserved setting stops the run. Reserved are the CPU settings, LightGBM's
  device parameters, the seed and every seed LightGBM derives from it, the categorical
  declaration, LightGBM's own class weighting (``is_unbalance``, ``scale_pos_weight``), which
  would add to the protocol's explicit sample weights, and class-dependent bagging
  (``pos_bagging_fraction``, ``neg_bagging_fraction``), which is under-sampling. LightGBM passes
  parameters to its library as one string of ``name=value`` pairs and ignores unknown names (with
  a warning), so every hyperparameter must be a LightGBM parameter, must not be ``None`` (which
  LightGBM drops, leaving its default) and must encode to exactly one ``name=value`` pair of
  printable ASCII characters that the library reads unchanged.
* **Fitting:** with V1 early stopping, the hyperparameters (training rules) must set a positive
  ``num_iterations`` and ``early_stopping_round`` and an explicit ``metric``, so that no library
  default decides them; with a fixed iteration count, exactly that many trees are grown without
  validation, or the run stops.
* **Prediction (DR-013 §5):** always through LightGBM's host (CPU) predictor, with the tree count
  the fitted model keeps and the CPU thread count of ``config/models.yaml``.
* **Logging:** LightGBM's own messages, including its warnings, go to the run log; the fit is a
  logged step with bounded progress.
* **Persistence:** the model text and a manifest with the model's SHA-256; reloading checks every
  manifest field against the request and against the loaded model itself.

The run that uses this model verifies the recorded LightGBM build at start-up
(``models.device.verify_lightgbm_build``); this module does not.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from review_capacity.config import Device, FeatureSpec, ModelSettings
from review_capacity.errors import FileAccessError, ModelError, SpecificationError
from review_capacity.features.model_inputs import fit_category_lists, to_lightgbm
from review_capacity.files import write_atomically, write_json
from review_capacity.logging_setup import ProgressLogger, log_step
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

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
MODEL_FILE = "model.txt"
MANIFEST_FILE = "manifest.json"
# Parameters that the training rules must set explicitly for V1 early stopping.
EARLY_STOPPING_PARAMETERS = ("num_iterations", "early_stopping_round", "metric")
# Seeds LightGBM derives from ``seed`` (DR-013 §2): never set separately.
DERIVED_SEEDS = frozenset(
    {
        "data_random_seed",
        "bagging_seed",
        "feature_fraction_seed",
        "extra_seed",
        "drop_seed",
        "objective_seed",
    }
)
# LightGBM's device parameters: the device is fixed by DR-013 §2, never by a hyperparameter.
DEVICE_PARAMETERS = frozenset(
    {
        "device_type",
        "gpu_platform_id",
        "gpu_device_id",
        "gpu_device_id_list",
        "num_gpu",
        "gpu_use_dp",
    }
)
# Reserved besides the device settings: the seeds, the categorical declaration (from the feature
# declaration), LightGBM's own class weighting and class-dependent bagging (under-sampling): the
# protocol's class weights are explicit, and resampling is excluded.
RESERVED = (
    DERIVED_SEEDS
    | DEVICE_PARAMETERS
    | {
        "seed",
        "categorical_feature",
        "is_unbalance",
        "scale_pos_weight",
        "pos_bagging_fraction",
        "neg_bagging_fraction",
    }
)
DISABLED_METRICS = frozenset({"none", "null", "na", "custom"})
TRAINING_DEVICE_TYPES = {"cpu": "cpu"}


@cache
def lightgbm_aliases() -> dict[str, str]:
    """Every parameter alias of the installed LightGBM mapped to its canonical name.

    The table is LightGBM's own (``lightgbm.basic._ConfigAliases``, read from the library), so it
    is complete for the pinned version; the private attribute is guarded by a unit test.
    """
    from lightgbm.basic import _ConfigAliases

    table = _ConfigAliases._get_all_param_aliases()
    return {alias: name for name, names in table.items() for alias in names if alias != name}


@cache
def lightgbm_parameters() -> frozenset[str]:
    """The canonical name of every parameter of the installed LightGBM: the keys of the same native
    table, which lists every parameter, not only those with aliases."""
    from lightgbm.basic import _ConfigAliases

    return frozenset(_ConfigAliases._get_all_param_aliases())


class _LibraryLogger:
    """LightGBM's logger interface, forwarding to this module's logger. LightGBM sends every
    native message, warnings included, through ``info``, so warnings are recognised by their
    prefix."""

    @staticmethod
    def info(message: str) -> None:
        text = message.rstrip()
        if text.startswith("[LightGBM] [Warning]"):
            logger.warning("%s", text)
        elif text:
            logger.info("%s", text)

    @staticmethod
    def warning(message: str) -> None:
        logger.warning("%s", message.rstrip())


@cache
def _route_library_messages() -> None:
    """Send LightGBM's messages to the project's logging (once per process)."""
    import lightgbm

    lightgbm.register_logger(_LibraryLogger())


def resolved_parameters(model_text: str) -> dict[str, str]:
    """The ``[name: value]`` entries of the parameter block LightGBM writes into a model."""
    _, start, remainder = model_text.partition("\nparameters:\n")
    block, end, _ = remainder.partition("\nend of parameters")
    if not start or not end:
        raise ModelError("LightGBM model text has no complete parameter block")
    resolved: dict[str, str] = {}
    for line in block.splitlines():
        if not line:
            continue
        if not (line.startswith("[") and line.endswith("]") and ": " in line):
            raise ModelError(f"Malformed LightGBM parameter entry {line!r}")
        name, value = line[1:-1].split(": ", 1)
        if not name or name in resolved:
            raise ModelError(f"Empty or repeated LightGBM parameter name {name!r}")
        resolved[name] = value
    if not resolved:
        raise ModelError("LightGBM parameter block is empty")
    return resolved


def _check_parameter_encoding(requested: Mapping[str, Any]) -> None:
    """Check that every canonical hyperparameter reaches LightGBM's library as written.

    Each value is encoded with LightGBM's own serializer (``lightgbm.basic._param_dict_to_str``),
    and the result must be one ``name=value`` pair of printable ASCII characters only: no control
    character (a NUL ends the C string the library receives), no whitespace (which separates
    pairs), no second ``=`` (which separates name and value) and no surrounding quotes (which the
    library removes).
    """
    from lightgbm.basic import _param_dict_to_str

    known = lightgbm_parameters()
    for name, value in requested.items():
        if name not in known:
            raise SpecificationError(f"Unknown LightGBM parameter {name!r}")
        if value is None:
            raise SpecificationError(f"{name}: None would leave the value to LightGBM's default")
        try:
            encoded = _param_dict_to_str({name: value})
        except TypeError as error:
            raise SpecificationError(f"{name}: unsupported value {value!r}") from error
        key, _, text = encoded.partition("=")
        if not text:
            raise SpecificationError(f"{name}: an empty value would leave LightGBM's default")
        if (
            key != name
            or "=" in text
            or not (encoded.isascii() and encoded.isprintable())
            or any(character.isspace() for character in encoded)
            or text[0] in "'\""
            or text[-1] in "'\""
        ):
            raise SpecificationError(
                f"{name}: value {value!r} cannot be passed to LightGBM unchanged"
            )


def _check_early_stopping(params: dict[str, Any]) -> None:
    missing = [name for name in EARLY_STOPPING_PARAMETERS if name not in params]
    if missing:
        raise SpecificationError(
            f"V1 early stopping needs these parameters set explicitly: {missing}"
        )
    for name in ("num_iterations", "early_stopping_round"):
        value = params[name]
        if type(value) is not int or value <= 0:
            raise SpecificationError(f"{name} must be a positive integer, not {value!r}")
    metric = params["metric"]
    if isinstance(metric, str):
        supplied = [metric]
    elif isinstance(metric, list | tuple):
        supplied = list(metric)
    else:
        raise SpecificationError(f"metric must name an explicit evaluation metric: {metric!r}")
    if not supplied or not all(isinstance(value, str) for value in supplied):
        raise SpecificationError(f"metric must enable an explicit evaluation metric: {metric!r}")
    metrics = [part for value in supplied for part in value.split(",")]
    if any(not value or value.lower() in DISABLED_METRICS for value in metrics):
        raise SpecificationError(f"metric must enable an explicit evaluation metric: {metric!r}")
    params["metric"] = metrics  # one name per entry, as LightGBM resolves it
    boosting = params.get("boosting", "gbdt")
    if isinstance(boosting, str) and boosting.lower() == "dart":
        raise SpecificationError("DART boosting does not support the required V1 early stopping")


def _manifest_fields(
    manifest: object,
    *,
    configuration: Configuration,
    seed: int,
    features: FeatureSpec,
    device: Device,
    fixed: Mapping[str, Any],
    library_version: str,
) -> tuple[int, dict[str, list[str]], dict[str, Any], str]:
    """The tree count, category lists, requested parameters and model SHA-256 of a manifest,
    after checking every field against the request."""
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
        categories = manifest["categories"]
        requested = manifest["requested"]
        model_sha256 = manifest["model_sha256"]
    except KeyError as error:
        raise ModelError(f"Saved model manifest is incomplete: missing {error}") from error
    if version != FORMAT_VERSION or type(version) is not int:
        raise ModelError(f"Unsupported model format version {version!r}")
    if type(manifest["seed"]) is not int:
        raise ModelError(f"Saved model seed must be an integer, not {manifest['seed']!r}")
    expected = (
        "lightgbm",
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
    if not isinstance(categories, dict) or set(categories) != set(features.categorical):
        raise ModelError("Saved category lists do not match the categorical inputs")
    for name, values in categories.items():
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) for value in values)
            or len(values) != len(set(values))
        ):
            raise ModelError(f"Invalid saved category list for {name}")
    if not isinstance(requested, dict):
        raise ModelError("Saved requested parameters must be a JSON object")
    for name, value in fixed.items():
        saved = requested.get(name)
        # Same value and same JSON type (0 is not False, 1.0 is not 1); histogram_pool_size is
        # any number in the settings schema.
        numeric = name == "histogram_pool_size" and type(saved) in (int, float)
        if not (numeric or type(saved) is type(value)) or saved != value:
            raise ModelError(f"Saved requested parameter {name!r} does not match the fixed setting")
    if not isinstance(model_sha256, str) or len(model_sha256) != 64:
        raise ModelError("Saved model SHA-256 is missing or malformed")
    return iterations, categories, requested, model_sha256


class LightGBMModel(ScoringModel):
    """LightGBM on the common interface: CPU training and CPU prediction (DR-013 §2)."""

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
        if configuration.model.name != "lightgbm":
            raise ValueError(f"{configuration.name} is not a LightGBM configuration")
        if device not in TRAINING_DEVICE_TYPES:
            raise SpecificationError(
                "LightGBM trains on the CPU only (DR-013 §2: its CUDA learner fails on "
                f"categorical features above 256 categories), not on {device!r}"
            )
        self._booster: Any = None
        self._categories: dict[str, list[str]] = {}
        self._requested: dict[str, Any] = {}
        self._iterations = 0

    def _fixed_parameters(self) -> dict[str, Any]:
        return {**self.settings.lightgbm.params(self.device), "seed": self.seed}

    def _reserved(self) -> set[str]:
        return set(self.settings.lightgbm.params("cpu")) | RESERVED

    def _prediction_threads(self) -> int:
        return int(self.settings.lightgbm.params("cpu")["num_threads"])

    def _fitted(self) -> Any:
        if self._booster is None:
            raise ModelError(f"{self.configuration.name} is not fitted")
        return self._booster

    def _dataset(self, data: ModelData, reference: Any = None) -> Any:
        import lightgbm

        self.check_inputs(data.features)
        return lightgbm.Dataset(
            to_lightgbm(data.features, self.features, self._categories),
            label=data.labels,
            weight=data.weights,
            categorical_feature=list(self.features.categorical),
            reference=reference,
            free_raw_data=False,
        )

    def _parameters_for_fit(
        self, hyperparameters: Mapping[str, Any], iterations: int | None
    ) -> dict[str, Any]:
        requested = merge_parameters({}, hyperparameters, aliases=lightgbm_aliases())
        clash = sorted(self._reserved() & requested.keys())
        if clash:
            raise SpecificationError(
                f"Hyperparameters cannot override fixed research settings: {clash}"
            )
        _check_parameter_encoding(requested)
        params = {**self._fixed_parameters(), **requested}
        verbosity = params.setdefault("verbosity", 1)
        if type(verbosity) is not int or verbosity < 0:
            raise SpecificationError("LightGBM verbosity must be a non-negative integer")
        if iterations is None:
            _check_early_stopping(params)
        else:
            params.pop("early_stopping_round", None)
            params["num_iterations"] = iterations
        return params

    def fit(
        self,
        train: ModelData,
        valid: ModelData | None,
        hyperparameters: Mapping[str, Any],
        *,
        iterations: int | None = None,
    ) -> None:
        import lightgbm
        from lightgbm.basic import LightGBMError

        check_fit_arguments(valid, iterations)
        _route_library_messages()
        params = self._parameters_for_fit(hyperparameters, iterations)
        self.check_inputs(train.features)
        self._categories = fit_category_lists(train.features, self.features)
        train_set = self._dataset(train)
        valid_sets = [] if valid is None else [self._dataset(valid, reference=train_set)]
        name = self.configuration.name
        progress = ProgressLogger(logger, f"{name} boosting", params["num_iterations"])

        def advance(_environment: Any) -> None:
            progress.advance()

        rows = f"{len(train.labels)} Train rows"
        if valid is not None:
            rows += f", early stopping on {len(valid.labels)} V1 rows"
        step = f"Fit {name} on {self.device} ({rows})"
        try:
            with log_step(logger, step):  # a fit that keeps the wrong tree count is a failed step
                booster = lightgbm.train(
                    params,
                    train_set,
                    valid_sets=valid_sets,
                    valid_names=["v1"][: len(valid_sets)],
                    callbacks=[advance],
                )
                kept = booster.best_iteration if valid is not None else booster.current_iteration()
                if kept <= 0:
                    raise ModelError(f"{name} kept no trees")
                if iterations is not None and kept != iterations:
                    raise ModelError(
                        f"{name}: {iterations} refit trees requested, LightGBM grew {kept}"
                    )
        except LightGBMError as error:
            raise ModelError(f"{name}: LightGBM training failed: {error}") from error
        self._booster = booster
        self._requested = params
        self._iterations = kept
        logger.info("%s fitted: %d trees kept", name, kept)

    def predict(self, features: pd.DataFrame) -> Scores:
        from lightgbm.basic import LightGBMError

        booster = self._fitted()
        self.check_inputs(features)
        try:
            scores = booster.predict(
                to_lightgbm(features, self.features, self._categories),
                num_iteration=self._iterations,
                num_threads=self._prediction_threads(),
            )
        except LightGBMError as error:
            raise ModelError(f"{self.configuration.name}: prediction failed: {error}") from error
        return checked_scores(
            cast(npt.NDArray[np.float64], scores), len(features), self.configuration.name
        )

    def _model_text(self, iterations: int | None = None) -> str:
        from lightgbm.basic import LightGBMError

        try:
            text: str = self._fitted().model_to_string(num_iteration=iterations)
        except LightGBMError as error:
            name = self.configuration.name
            raise ModelError(f"{name}: model text unavailable: {error}") from error
        return text

    def save(self, directory: Path) -> None:
        import lightgbm

        content = self._model_text(self._iterations).encode("utf-8")
        write_atomically(directory / MODEL_FILE, lambda target: target.write_bytes(content))
        write_json(  # written last, with the model's SHA-256
            directory / MANIFEST_FILE,
            {
                "format_version": FORMAT_VERSION,
                "library": "lightgbm",
                "library_version": lightgbm.__version__,
                "configuration": self.configuration.name,
                "seed": self.seed,
                "device": self.device,
                "iterations": self._iterations,
                "inputs": list(self.features.model_inputs),
                "categorical": list(self.features.categorical),
                "categories": self._categories,
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
        import lightgbm
        from lightgbm.basic import LightGBMError

        try:
            manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
            content = (directory / MODEL_FILE).read_bytes()
        except OSError as error:
            raise FileAccessError(f"{directory}: saved model cannot be read") from error
        except ValueError as error:
            raise ModelError(f"{directory}: saved model manifest is not valid JSON") from error
        model = cls(configuration, seed, settings=settings, features=features, device=device)
        iterations, categories, requested, model_sha256 = _manifest_fields(
            manifest,
            configuration=configuration,
            seed=seed,
            features=features,
            device=device,
            fixed=model._fixed_parameters(),
            library_version=lightgbm.__version__,
        )
        if hashlib.sha256(content).hexdigest() != model_sha256:
            raise ModelError(f"{directory}: saved model and manifest do not match")
        try:
            text = content.decode("utf-8")
            booster = lightgbm.Booster(model_str=text)
        except (UnicodeDecodeError, LightGBMError) as error:
            raise ModelError(f"{directory}: LightGBM model cannot be loaded: {error}") from error
        resolved = resolved_parameters(text)
        if booster.feature_name() != list(features.model_inputs):
            raise ModelError("Saved model's feature names do not match the inputs")
        if booster.current_iteration() != iterations:
            raise ModelError("Saved tree count does not match the model")
        if booster.pandas_categorical != [categories[name] for name in features.categorical]:
            raise ModelError("Saved category lists do not match the model")
        trained = (resolved.get("seed"), resolved.get("device_type"))
        if trained != (str(seed), TRAINING_DEVICE_TYPES[device]):
            raise ModelError("Saved model's seed or training device does not match")
        model._booster = booster
        model._categories = categories
        model._requested = requested
        model._iterations = iterations
        return model

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "requested": self._requested,
            "resolved": resolved_parameters(self._model_text()),
            "prediction": {"backend": "cpu", "num_threads": self._prediction_threads()},
        }

    @property
    def fitted_iterations(self) -> int:
        self._fitted()
        return self._iterations
