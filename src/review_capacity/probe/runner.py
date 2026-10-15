"""The GPU acceptance probe: checks C1–C13 and tolerances T1–T4 of
``docs/gpu-acceptance-probe.md`` (approved before implementation, Issue #77).

``run_probe`` builds the synthetic inputs, fits all eight configurations on their training
devices of DR-013 §1 with V1 early stopping (LightGBM on the CPU, the others on the GPU), refits
CatBoost on the CPU, runs every check, and writes the records before and after every fit and on
failure (``probe.json``, the generated ``probe.md`` and the comparison files). A first run ends
as ``pending second run``: only a second run, in a fresh process and a separate output directory,
given the first run's ``probe.json``, completes C11.

A GPU run is the acceptance run. A CPU run exists for the unit tests of this module only: it runs
the same code paths on small inputs with every model on the CPU, so that every check that needs the
GPU is evaluated against the CPU instead; a CPU run is marked as not being acceptance evidence.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from review_capacity.config import Device, FeatureSpec, ModelSettings, load_feature_spec
from review_capacity.data.catalog import read_catalog
from review_capacity.errors import FileAccessError, ModelError, SpecificationError
from review_capacity.features.model_inputs import fit_category_lists, to_lightgbm
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models.base import ScoringModel, class_weights, sample_weights
from review_capacity.models.catboost_model import (
    CTR_LISTS,
    SYSTEM_SETTINGS,
    CatBoostModel,
)
from review_capacity.models.device import environment_record, verify_lightgbm_build
from review_capacity.models.ft_transformer_model import FTTransformer, FTTransformerModel
from review_capacity.models.lightgbm_model import (
    TRAINING_DEVICE_TYPES,
    LightGBMModel,
    _route_library_messages,
    resolved_parameters,
)
from review_capacity.models.neural_training import (
    InputLayout,
    NeuralData,
    NeuralInputs,
    NeuralModel,
    derived_seeds,
    drop_categories,
    epoch_batches,
    float32_default,
    generator,
    predict_logits,
    probabilities,
    torch_device,
)
from review_capacity.models.registry import Weighting, configuration
from review_capacity.models.resnet_model import ResNetModel
from review_capacity.models.run_record import jsonable, load_settings_with_snapshot
from review_capacity.paths import ProjectPaths
from review_capacity.probe import diagnostics
from review_capacity.probe.records import (
    SCHEMA_VERSION,
    CheckBook,
    check_output,
    write_records,
)
from review_capacity.probe.synthetic import (
    DATA_SEED,
    NeuralSet,
    ProbeSizes,
    TreeData,
    category_levels,
    data_generator,
    neural_data,
    neural_layout,
    top_k,
    tree_data,
)
from review_capacity.provenance import code_provenance

logger = logging.getLogger(__name__)

SPECIFICATION = "docs/gpu-acceptance-probe.md"
MODEL_SEED = 42  # the main seed (DR-013 §6)
T1_TOLERANCE = 1e-4
CPU_TIMING_ROWS = 512  # neural scoring timed on the CPU, scores discarded [I]
DEVICE_PASS_ROWS = 64  # rows of the C1 forward and backward pass
PREDICTION_THREADS = 24  # the CPU thread count of config/models.yaml
PROCESS_TOKEN = uuid.uuid4().hex  # identifies this process, so that a run never compares itself
LIGHTGBM_LOGGER = "review_capacity.models.lightgbm_model"
ACCEPTED_WARNING = "No further splits with positive gain"
LIGHTGBM_WARNING_PREFIX = "[LightGBM] [Warning] "
# LightGBM's notice that a categorical feature has more bins than max_bin, which does not cap
# categorical features (DR-013 §2; [L] REF-117: dataset_loader.cpp). Accepted only when the probe's
# own Dataset has such a feature (C4).
CATEGORICAL_BIN_WARNINGS = (
    "Categorical features with more bins than the configured maximum bin number found.",
    "For categorical features, max_bin and max_bin_by_feature may be ignored with a large number "
    "of categories.",
)
FAILING_WARNING = re.compile(r"ignor|unsupported|not supported|fall ?back", re.IGNORECASE)
SAMPLING_UNIT_NOTE = (
    "CatBoost 1.2.10 does not serialise sampling_unit (bootstrap_options.h), so its effective "
    "value cannot be read back; the requested value Object is CatBoost's documented default"
)

# Probe values of the specification, Section 2 [I]: not research values.
TREE_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "lightgbm": {
        "num_iterations": 300,
        "early_stopping_round": 20,
        "metric": "binary_logloss",
        "learning_rate": 0.1,
        "num_leaves": 63,
    },
    "catboost": {
        "iterations": 300,
        "od_wait": 20,
        "eval_metric": "Logloss",
        "metric_period": 1,  # early stopping evaluates every iteration (Issue #77)
        "learning_rate": 0.1,
        "depth": 6,
        "model_size_reg": 0.5,
        "fold_permutation_block": 1,
        "permutation_count": 4,
        "random_strength": 1.0,
        "l2_leaf_reg": 3.0,
        "leaf_estimation_iterations": 10,
        "bagging_temperature": 1.0,
    },
}
NEURAL_TRAINING: dict[str, Any] = {
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "max_epochs": 3,
    "patience": 1,
    "metric": "binary_logloss",
}
NEURAL_SHARED: dict[str, dict[str, Any]] = {
    "resnet": {
        "d_embedding": 64,
        "hidden_dropout": 0.1,
        "residual_dropout": 0.1,
        "batch_size": 1024,
        "eval_batch_size": 4096,
    },
    "ft_transformer": {
        "d_ffn_factor": 4 / 3,
        "attention_dropout": 0.1,
        "ffn_dropout": 0.1,
        "residual_dropout": 0.1,
        "eval_batch_size": 16,
    },
}
NEURAL_VARIANTS: dict[str, dict[str, dict[str, Any]]] = {
    "resnet": {
        "small": {"n_layers": 2, "d": 256, "d_hidden_factor": 2.0},
        "largest": {"n_layers": 16, "d": 1024, "d_hidden_factor": 4.0},
    },
    "ft_transformer": {
        "default": {"n_layers": 3, "d_token": 192, "batch_size": 256},
        "largest": {"n_layers": 6, "d_token": 512, "batch_size": 64},
        "padded": {"n_layers": 2, "d_token": 200, "batch_size": 128},
    },
}
LABELS = {"lightgbm": "LGB", "catboost": "Cat", "resnet": "RES", "ft_transformer": "FTT"}

type Scores = npt.NDArray[np.float64]
type Ids = npt.NDArray[np.int64]


def score_key(name: str, variant: str, device: str, repetition: str, path: str) -> str:
    """The key of a set of scores: configuration, variant, device, repetition and scoring path
    (Section 5), for example ``RES-b.largest.gpu.fit1.score``."""
    return f"{name}.{variant}.{device}.{repetition}.{path}"


class WarningCapture(logging.Handler):
    """Keeps every warning of the package and of Python's warnings with its phase (C2)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.phase = "start-up"
        self.records: list[dict[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage()
        self.records.append({"logger": record.name, "phase": self.phase, "text": text})


@contextmanager
def captured_warnings() -> Iterator[WarningCapture]:
    capture = WarningCapture()
    loggers = [logging.getLogger(name) for name in ("review_capacity", "py.warnings")]
    for target in loggers:
        target.addHandler(capture)
    try:
        yield capture
    finally:
        for target in loggers:
            target.removeHandler(capture)


def _no_checkpoint(_stage: str) -> None:
    """The checkpoint of a probe outside ``run_probe``: nothing is written."""


@dataclass
class Probe:
    """The state of one probe run."""

    paths: ProjectPaths
    output: Path
    device: Device
    sizes: ProbeSizes
    settings: ModelSettings
    features: FeatureSpec
    capture: WarningCapture
    book: CheckBook = field(default_factory=CheckBook)
    fits: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    arrays: dict[str, tuple[Scores, Ids]] = field(default_factory=dict)
    checkpoint: Callable[[str], object] = _no_checkpoint
    # Largest LightGBM bin count of a categorical feature in the probe's Dataset (C4), or None
    categorical_bins: int | None = None
    # SHA-256 of each LightGBM fit's saved model text, by configuration and repetition (C11)
    model_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def gpu(self) -> bool:
        return self.device == "gpu"

    def training_device(self, model: str) -> Device:
        """The device ``model`` trains on: its training device of DR-013 §1 on a GPU run, the CPU
        on a CPU run (unit tests)."""
        return self.settings.training_device(model) if self.gpu else "cpu"

    def synchronise(self) -> None:
        if self.gpu:
            torch.cuda.synchronize()

    def keep_scores(self, key: str, scores: Scores, ids: Ids) -> None:
        """Write a comparison file (Section 5) and keep the scores for the run comparison."""
        self.scores[key] = diagnostics.save_scores(self.output / "scores", key, scores, ids)
        self.arrays[key] = (scores, ids)

    def reset_peak(self) -> None:
        if self.gpu:
            torch.cuda.reset_peak_memory_stats()

    def peak_mib(self) -> dict[str, float | None]:
        if not self.gpu:
            return {"peak_allocated_mib": None, "peak_reserved_mib": None}
        return {
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }


def weights_item(
    probe: Probe,
    subject: str,
    train: Any,
    valid: Any,
    weighting: Weighting,
    kept: int,
    received: npt.NDArray[Any] | None = None,
) -> None:
    """C6: at least one tree or epoch kept; Train and V1 weights equal the protocol's class-weight
    rule fitted on Train labels (unit weights unweighted, two distinct weights balanced); and, when
    given, the weights the library received equal the Train weights in its float32 storage."""
    rule = class_weights(train.labels, weighting)
    train_ok = diagnostics.bitwise_equal(train.weights, sample_weights(train.labels, rule))
    valid_ok = diagnostics.bitwise_equal(valid.weights, sample_weights(valid.labels, rule))
    distinct = sorted(set(train.weights.tolist()))
    shape_ok = distinct == [1.0] if weighting is Weighting.UNWEIGHTED else len(distinct) == 2
    received_ok = received is None or diagnostics.bitwise_equal(
        np.asarray(received, dtype=np.float32), train.weights.astype(np.float32)
    )
    probe.book.add(
        "C6",
        subject,
        kept >= 1 and train_ok and valid_ok and shape_ok and received_ok,
        kept=kept,
        distinct_train_weights=distinct,
        train_weights_follow_rule=train_ok,
        valid_weights_follow_rule=valid_ok,
        received_weights_checked=received is not None,
        received_weights_match=received_ok,
    )


def _fit_record(model: Any, subject: str, device: Device, seconds: float) -> dict[str, Any]:
    return {
        "subject": subject,
        "device": device,
        "kept": model.fitted_iterations,
        "seconds": seconds,
        "parameters": jsonable(model.parameters),
    }


# --- Tree-based models ----------------------------------------------------------------------------


def _canonical(value: object) -> str:
    """LightGBM's text form of a fixed setting, as written into the model's parameter block."""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _lightgbm_checks(probe: Probe, model: LightGBMModel, device: Device, subject: str) -> None:
    """C1 (resolved training device) and C3 (every fixed setting in the resolved parameters)."""
    resolved = resolved_parameters(model._model_text())
    probe.book.add(
        "C1",
        subject,
        resolved.get("device_type") == TRAINING_DEVICE_TYPES[device],
        resolved_device_type=resolved.get("device_type"),
    )
    fixed = probe.settings.lightgbm.params(device)
    wrong = {
        name: {"fixed": value, "resolved": resolved.get(name)}
        for name, value in fixed.items()
        if resolved.get(name) != _canonical(value)
    }
    probe.book.add("C3", subject, not wrong, compared=len(fixed), differences=wrong)


def _catboost_checks(probe: Probe, model: CatBoostModel, device: Device, subject: str) -> None:
    """C1 (resolved task type) and C3: every fixed setting against ``get_all_params()``, the CTR
    lists by descriptor type, thread count and devices against the model's system options;
    ``sampling_unit``, which CatBoost does not serialise, is an item for review."""
    parameters = model.parameters
    resolved = parameters["resolved"]
    task_type = resolved.get("task_type")
    expected = "GPU" if device == "gpu" else "CPU"
    probe.book.add("C1", subject, task_type == expected, task_type=task_type)
    system = json.loads(parameters["metadata"]["params"])["system_options"]
    fixed = probe.settings.catboost.params(device)
    wrong: dict[str, Any] = {}
    for name, value in fixed.items():
        if name == "sampling_unit":
            continue
        observed = system.get(name) if name in SYSTEM_SETTINGS else resolved.get(name)
        if name in CTR_LISTS:
            descriptors = observed if isinstance(observed, list) else []
            ok = bool(descriptors) and {str(d).split(":", 1)[0] for d in descriptors} == set(value)
        else:
            ok = type(observed) is type(value) and observed == value
        if not ok:
            wrong[name] = {"fixed": value, "resolved": observed}
    probe.book.add("C3", subject, not wrong, compared=len(fixed) - 1, differences=wrong)
    if "sampling_unit" in fixed:
        probe.book.add(
            "C3",
            f"{subject} sampling_unit",
            "review",
            requested=fixed["sampling_unit"],
            note=SAMPLING_UNIT_NOTE,
        )


def catboost_combinations(model_json: Mapping[str, Any]) -> int:
    """Categorical statistics over two or more categorical inputs in a CatBoost JSON model."""
    count = 0
    for ctr in model_json.get("features_info", {}).get("ctrs", []):
        elements = ctr.get("elements", [])
        values = [e for e in elements if e.get("combination_element") == "cat_feature_value"]
        count += len(values) >= 2
    return count


def _catboost_json(model: CatBoostModel) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        model._fitted().save_model(str(path), format="json")
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _received_weights(model: LightGBMModel | CatBoostModel, train: Any) -> npt.NDArray[Any]:
    """The Train weights as the library stores them, through the adapter's own construction:
    LightGBM's native weight field, CatBoost's Pool weights. LightGBM 4.7.0 stores no weight field
    when every weight is 1 (``Dataset.set_weight`` replaces them with None, as unit weights equal
    no weights), so an absent field stands for unit weights; with any other weights it still
    differs from the Train weights and fails C6."""
    if isinstance(model, LightGBMModel):
        dataset = model._dataset(train).construct()
        stored = dataset.get_field("weight")
        if stored is None:
            return np.ones(len(train.labels), dtype=np.float32)
        return np.asarray(stored)
    return np.asarray(model._pool(train.features, train).get_weight())


def _native_prediction(
    model: LightGBMModel | CatBoostModel, frame: pd.DataFrame
) -> Callable[[], object]:
    """Model-only scoring: the library's CPU predictor on inputs already in its form (C13)."""
    if isinstance(model, LightGBMModel):
        converted = to_lightgbm(frame, model.features, model._categories)
        booster = model._fitted()

        def lightgbm_predict() -> object:
            return booster.predict(
                converted, num_iteration=model.fitted_iterations, num_threads=PREDICTION_THREADS
            )

        return lightgbm_predict
    pool = model._pool(frame)
    fitted = model._fitted()

    def catboost_predict() -> object:
        return fitted.predict_proba(
            pool,
            ntree_end=model.fitted_iterations,
            thread_count=PREDICTION_THREADS,
            task_type="CPU",
        )

    return catboost_predict


def _tree_timing(
    probe: Probe, model: LightGBMModel | CatBoostModel, frame: pd.DataFrame
) -> dict[str, Any]:
    latency = []
    for row in range(min(diagnostics.LATENCY_ROWS, len(frame))):
        single = frame.iloc[row : row + 1]
        if row == 0:
            for _ in range(diagnostics.WARM_UP_CALLS):
                model.predict(single)
        start = time.perf_counter()
        model.predict(single)
        latency.append(time.perf_counter() - start)
    return {
        "end_to_end": diagnostics.timed_calls(lambda: model.predict(frame), probe.synchronise),
        "model_only": diagnostics.timed_calls(_native_prediction(model, frame), probe.synchronise),
        "single_row_latency": diagnostics.latency_summary(latency),
        "rows": len(frame),
        "backend": "cpu prediction (DR-013 §5)",
    }


def _tree_scores(
    probe: Probe, model: ScoringModel, tree: TreeData, subject: str, key: Callable[[str], str]
) -> Scores:
    """C9, C10 and C12 of a fitted tree model, with every scored array kept; its scores."""
    scores = model.predict(tree.score)
    probe.keep_scores(key("score"), scores, tree.score_ids)
    probe.book.add("C9", subject, bool(np.isfinite(scores).all()))
    summary = diagnostics.top_k_summary(scores, tree.score_ids, top_k(len(scores)))
    probe.book.add("C9", subject, None, **summary)
    repeated = model.predict(tree.score)
    probe.keep_scores(key("repeat"), repeated, tree.score_ids)
    probe.book.add("C10", subject, diagnostics.bitwise_equal(scores, repeated))
    with tempfile.TemporaryDirectory() as directory:
        model.save(Path(directory))
        loaded = type(model).load(
            Path(directory),
            model.configuration,
            model.seed,
            settings=probe.settings,
            features=probe.features,
            device=model.device,
        )
        reloaded = loaded.predict(tree.score)
    probe.keep_scores(key("reload"), reloaded, tree.score_ids)
    probe.book.add("C12", subject, diagnostics.bitwise_equal(scores, reloaded))
    return scores


def _fit_tree(
    probe: Probe,
    library: str,
    name: str,
    device: Device,
    data: tuple[Any, Any],
    iterations: int | None,
) -> LightGBMModel | CatBoostModel:
    cls = LightGBMModel if library == "lightgbm" else CatBoostModel
    model = cls(
        configuration(name),
        MODEL_SEED,
        settings=probe.settings,
        features=probe.features,
        device=device,
    )
    subject = f"{name} {device}" + (" refit" if iterations is not None else "")
    probe.capture.phase = subject
    probe.checkpoint(f"fitting {subject}")  # a native abort leaves the running fit in the record
    start = time.perf_counter()
    model.fit(data[0], data[1], TREE_HYPERPARAMETERS[library], iterations=iterations)
    probe.synchronise()
    probe.fits.append(_fit_record(model, subject, device, time.perf_counter() - start))
    return model


def _tree_checks(
    probe: Probe, model: LightGBMModel | CatBoostModel, device: Device, subject: str
) -> None:
    if isinstance(model, LightGBMModel):
        _lightgbm_checks(probe, model, device, subject)
    else:
        _catboost_checks(probe, model, device, subject)


def _unseen_item(probe: Probe, model: LightGBMModel, tree: TreeData, subject: str) -> None:
    """C5: LightGBM's conversion turns every score-set label absent from the Train categories,
    including every designated unseen label, into a missing value, and nothing else."""
    converted = to_lightgbm(tree.score, probe.features, model._categories)
    exact = True
    absent_total = 0
    for name in probe.features.categorical:
        labels = tree.score[name]
        absent = ~labels.isin(model._categories[name])
        missing = converted[name].isna()
        designated = labels.str.startswith("unseen-")
        exact &= bool((missing == absent).all()) and bool(missing[designated].all())
        absent_total += int(absent.sum())
    designated_total = sum(tree.unseen.values())
    probe.book.add(
        "C5",
        f"{subject} unseen to missing",
        exact and designated_total > 0,
        absent_from_train=absent_total,
        designated_unseen=designated_total,
    )


def _model_text(model: LightGBMModel | CatBoostModel) -> str | None:
    """LightGBM's saved model text (trees and parameters); None for CatBoost."""
    return model._model_text() if isinstance(model, LightGBMModel) else None


def _keep_model_hash(probe: Probe, key: str, text: str | None) -> None:
    """C11: the SHA-256 of a LightGBM fit's complete saved model text, for the run comparison."""
    if text is not None:
        probe.model_hashes[key] = hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tree_fit_checks(
    probe: Probe, model: LightGBMModel | CatBoostModel, tree: TreeData, weighting: Weighting
) -> str:
    """C1, C3, C5 and C6 of a fit on the training device; its subject."""
    train, valid = tree.train[weighting], tree.valid[weighting]
    subject = f"{model.configuration.name} {model.device}"
    received = _received_weights(model, train)
    weights_item(probe, subject, train, valid, weighting, model.fitted_iterations, received)
    _tree_checks(probe, model, model.device, subject)
    if isinstance(model, LightGBMModel):
        _unseen_item(probe, model, tree, subject)
    else:
        combinations = catboost_combinations(_catboost_json(model))
        probe.book.add("C5", f"{subject} combinations", combinations > 0, count=combinations)
    return subject


def run_trees(probe: Probe, tree: TreeData) -> None:
    """All four tree configurations on their training devices with V1 early stopping (DR-013
    §1). LightGBM, trained deterministically on the CPU, is fitted twice with the same seed (C11);
    CatBoost is refitted on the CPU at the kept tree count (DR-013 §7). A checkpoint before and
    after every fit."""
    for library in ("lightgbm", "catboost"):
        device = probe.training_device(library)
        for weighting in (Weighting.UNWEIGHTED, Weighting.BALANCED):
            name = f"{LABELS[library]}-{weighting.value}"
            train, valid = tree.train[weighting], tree.valid[weighting]

            def key(
                path: str, repetition: str = "fit1", name: str = name, device: Device = device
            ) -> str:
                return score_key(name, "default", device, repetition, path)

            model = _fit_tree(probe, library, name, device, (train, valid), None)
            subject = _tree_fit_checks(probe, model, tree, weighting)
            scores = _tree_scores(probe, model, tree, subject, key)
            timing = _tree_timing(probe, model, tree.score)
            probe.book.add("C13", f"{subject} scoring", None, **timing)
            kept = model.fitted_iterations
            text = _model_text(model)
            _keep_model_hash(probe, score_key(name, "default", device, "fit1", "model"), text)
            del model
            probe.checkpoint(f"{subject} fitted")
            if library == "lightgbm":
                again = _fit_tree(probe, library, name, device, (train, valid), None)
                repeated = again.predict(tree.score)
                probe.keep_scores(key("score", "fit2"), repeated, tree.score_ids)
                same_scores = diagnostics.bitwise_equal(scores, repeated)
                again_text = _model_text(again)
                _keep_model_hash(
                    probe, score_key(name, "default", device, "fit2", "model"), again_text
                )
                same_text = again_text == text
                probe.book.add(
                    "C11",
                    f"{subject} same seed, same process",
                    same_scores and same_text,
                    identical_scores=same_scores,
                    identical_model_text=same_text,
                )
                del again
                gc.collect()
                probe.checkpoint(f"{subject} fit 2 checked")
                continue
            refit = _fit_tree(probe, library, name, "cpu", (train, None), kept)
            refit_subject = f"{name} cpu refit"
            _tree_checks(probe, refit, "cpu", refit_subject)

            def refit_key(path: str, name: str = name) -> str:
                return score_key(name, "default", "cpu", "refit", path)

            refit_scores = _tree_scores(probe, refit, tree, refit_subject, refit_key)
            comparison = diagnostics.compare_scores(
                scores, refit_scores, tree.score_ids, top_k(len(scores))
            )
            probe.book.add("C9", f"{subject} against cpu refit", None, **comparison)
            del refit
            gc.collect()
            probe.checkpoint(f"{refit_subject} fitted")


def dataset_reproducibility(probe: Probe, train: Any) -> None:
    """C4 and T4: two LightGBM Datasets built independently from the same rows, weights, category
    mappings, seed and CPU construction settings, and saved with ``save_binary`` to fresh files,
    must be byte-identical. Records the largest bin count of a categorical feature (C2).
    LightGBM's messages go to the run log, warnings included."""
    import lightgbm
    from lightgbm.basic import LightGBMError

    _route_library_messages()
    categories = fit_category_lists(train.features, probe.features)
    data = to_lightgbm(train.features, probe.features, categories)
    params = {**probe.settings.lightgbm.params("cpu"), "seed": MODEL_SEED, "verbosity": 1}
    digests: dict[str, str] = {}
    paths: list[Path] = []
    bins: list[int] = []
    with tempfile.TemporaryDirectory() as directory:
        for index in range(2):
            path = Path(directory) / f"{index}-cpu.bin"  # a fresh file per Dataset
            try:
                dataset = lightgbm.Dataset(
                    data,
                    label=train.labels,
                    weight=train.weights,
                    categorical_feature=list(probe.features.categorical),
                    params=params,
                    free_raw_data=False,
                )
                dataset.construct()
                dataset.save_binary(str(path))
                bins.append(max(dataset.feature_num_bin(n) for n in probe.features.categorical))
            except LightGBMError as error:
                raise ModelError(f"LightGBM Dataset construction failed: {error}") from error
            digests[f"cpu construction {index + 1}"] = file_sha256(path)
            paths.append(path)
            del dataset
        identical = diagnostics.files_identical(paths[0], paths[1])
    probe.categorical_bins = max(bins)
    probe.book.add(
        "C4",
        "LightGBM Datasets, two cpu constructions",
        identical,
        sha256=digests,
        largest_categorical_bins=probe.categorical_bins,
        max_bin=params["max_bin"],
    )


# --- Neural models --------------------------------------------------------------------------------


def dropout_replacements(
    train: NeuralData, layout: InputLayout, batch_size: int, seed: int, epochs: int
) -> int:
    """C7: the category-dropout replacements of a fit, by replaying its recorded seeds through
    the same batching and dropout functions as the training loop."""
    seeds = derived_seeds(seed)
    order, dropout = generator(seeds["batch_order"]), generator(seeds["category_dropout"])
    missing = torch.tensor(layout.missing_indices, dtype=torch.int64)
    count = 0
    for _ in range(epochs):
        for index in epoch_batches(train.inputs.rows, batch_size, order):
            original = torch.from_numpy(train.inputs.categorical[index])
            dropped = drop_categories(original, missing, layout.category_dropout, dropout)
            count += int((dropped != original).sum())
    return count


def reference_logits(
    probe: Probe, model: NeuralModel, inputs: NeuralInputs, batch_size: int
) -> npt.NDArray[np.float32]:
    """C8: the float32 reference. ResNet: the same network on the CPU. FT-Transformer: a second
    network built with the math attention backend and loaded with the same state, on the training
    device (an outer backend context cannot replace the network's own)."""
    from torch.nn.attention import SDPBackend

    network = model._network_or_raise()
    if isinstance(model, FTTransformerModel):
        architecture = model.parameters["resolved"]["architecture"]
        with float32_default():
            reference = FTTransformer(
                n_numeric=len(model.layout.numeric),
                cardinalities=model.layout.cardinalities,
                d_token=architecture["d_token"],
                n_layers=architecture["n_layers"],
                d_ffn_factor=architecture["d_ffn_factor"],
                attention_dropout=architecture["attention_dropout"],
                ffn_dropout=architecture["ffn_dropout"],
                residual_dropout=architecture["residual_dropout"],
                gpu_backend=SDPBackend.MATH,
            )
        reference.load_state_dict(network.state_dict(), strict=True)
        device = torch_device(probe.device)
        logits = predict_logits(reference.to(device), inputs, device=device, batch_size=batch_size)
        del reference
        return logits
    cpu_network = copy.deepcopy(network).to("cpu")
    return predict_logits(cpu_network, inputs, device=torch.device("cpu"), batch_size=batch_size)


def _fit_neural(
    probe: Probe,
    model_name: str,
    name: str,
    data: NeuralSet,
    weighting: Weighting,
    hyperparameters: Mapping[str, Any],
    subject: str,
) -> NeuralModel:
    cls: type[NeuralModel] = ResNetModel if model_name == "resnet" else FTTransformerModel
    model = cls(
        configuration(name),
        MODEL_SEED,
        settings=probe.settings,
        layout=data.layout,
        device=probe.device,
    )
    probe.capture.phase = subject
    probe.checkpoint(f"fitting {subject}")
    probe.reset_peak()
    start = time.perf_counter()
    model.fit(data.train[weighting], data.valid[weighting], hyperparameters)
    probe.synchronise()
    record = _fit_record(model, subject, probe.device, time.perf_counter() - start)
    probe.fits.append({**record, **probe.peak_mib()})
    return model


def _device_item(probe: Probe, model: NeuralModel, train: NeuralData, subject: str) -> None:
    """C1: parameters on the training device, and one forward and backward pass of a copy of the
    network (the fitted network is not changed) with inputs, output and gradients on the device,
    through the configured attention backend for the FT-Transformer."""
    expected = "cuda" if probe.gpu else "cpu"
    device = torch_device(probe.device)
    parameters = {p.device.type for p in model._network_or_raise().parameters()}
    network = copy.deepcopy(model._network_or_raise())
    rows = min(DEVICE_PASS_ROWS, train.inputs.rows)
    numeric = torch.from_numpy(train.inputs.numeric[:rows].copy()).to(device)
    categorical = torch.from_numpy(train.inputs.categorical[:rows].copy()).to(device)
    network.train()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits: torch.Tensor = network(numeric, categorical)
    logits.float().sum().backward()  # type: ignore[no-untyped-call]
    gradients = [p.grad for p in network.parameters()]
    devices = {numeric.device.type, categorical.device.type, logits.device.type}
    devices |= {g.device.type for g in gradients if g is not None}
    complete = all(g is not None for g in gradients)
    del network, logits
    probe.book.add(
        "C1",
        subject,
        parameters == {expected} and devices == {expected} and complete,
        parameter_devices=sorted(parameters),
        pass_devices=sorted(devices),
        every_parameter_has_a_gradient=complete,
    )


def _neural_settings_item(probe: Probe, model: NeuralModel, subject: str) -> None:
    """C3: every fixed neural setting of ``config/models.yaml`` in the fit's precision record and
    in PyTorch's live state."""
    neural = probe.settings.neural
    expected: dict[str, Any] = {
        "matmul_fp32_precision": neural.matmul_fp32_precision,
        "allow_bf16_reduced_precision_reduction": neural.allow_bf16_reduced_precision_reduction,
        "deterministic_algorithms": neural.deterministic_algorithms,
        "deterministic_warn_only": False,
        "cudnn_benchmark": neural.cudnn_benchmark,
        "training_dtype": neural.training_dtype,
        "prediction_dtype": neural.prediction_dtype,
    }
    if isinstance(model, FTTransformerModel):
        expected["gpu_attention_backend"] = neural.gpu_attention_backend
    recorded = model.parameters["resolved"]["precision"]
    live = {
        "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    wrong = {
        name: {"fixed": value, "recorded": recorded.get(name), "live": live.get(name, value)}
        for name, value in expected.items()
        if recorded.get(name) != value or live.get(name, value) != value
    }
    probe.book.add("C3", subject, not wrong, compared=len(expected), differences=wrong)


def _computation_item(
    probe: Probe,
    model: NeuralModel,
    data: NeuralSet,
    weighting: Weighting,
    hyperparameters: Mapping[str, Any],
    subject: str,
) -> None:
    """C7: strict determinism, observed dtypes, category dropout, finite gradients and
    parameters."""
    resolved = model.parameters["resolved"]
    network = model._network_or_raise()
    observed = resolved["observed_dtypes"]
    state = {dtype for dtypes in observed["optimizer_state"].values() for dtype in dtypes}
    replacements = dropout_replacements(
        data.train[weighting],
        model.layout,
        hyperparameters["batch_size"],
        MODEL_SEED,
        resolved["schedule"]["epochs_run"],
    )
    parameters_finite = all(bool(torch.isfinite(p).all()) for p in network.parameters())
    gradients_finite = all(
        bool(torch.isfinite(p.grad).all()) for p in network.parameters() if p.grad is not None
    )
    passed = (
        resolved["precision"]["deterministic_algorithms"] is True
        and observed["training_logits"] == ["torch.bfloat16"]
        and state == {"torch.float32"}
        and replacements > 0
        and parameters_finite
        and gradients_finite
    )
    probe.book.add(
        "C7",
        subject,
        passed,
        training_logits=observed["training_logits"],
        optimizer_state=observed["optimizer_state"],
        category_dropout_replacements=replacements,
        parameters_finite=parameters_finite,
        gradients_finite=gradients_finite,
    )


def _float32_item(
    probe: Probe, model: NeuralModel, data: NeuralSet, batch_size: int, subject: str
) -> npt.NDArray[np.float32]:
    """C8 and T1: the float32 path against its float32 reference; the reference logits."""
    device = torch_device(probe.device)
    network = model._network_or_raise()
    probe.reset_peak()
    required = predict_logits(network, data.score, device=device, batch_size=batch_size)
    required_peak = probe.peak_mib()["peak_allocated_mib"]
    probe.reset_peak()
    reference = reference_logits(probe, model, data.score, batch_size)
    reference_peak = probe.peak_mib()["peak_allocated_mib"]
    comparison = diagnostics.logit_comparison(required, reference, T1_TOLERANCE)
    backend = "not applicable (no attention)"
    if isinstance(model, FTTransformerModel):
        backend = "efficient attention (sdpa_kernel)" if probe.gpu else "PyTorch CPU selection"
    probe.book.add(
        "C8",
        subject,
        comparison["passed"],
        **comparison,
        required_backend=backend,
        required_peak_allocated_mib=required_peak,
        reference_peak_allocated_mib=reference_peak,
    )
    return reference


def _neural_timing(
    probe: Probe, model: NeuralModel, data: NeuralSet, resolved: Mapping[str, Any]
) -> dict[str, Any]:
    """C13: end-to-end and model-only scoring with their memory peaks, single-row latency, CPU
    scoring (scores discarded) and the full Train epoch extrapolated from the training-only time
    of the probe epochs."""
    network = model._network_or_raise()
    device = torch_device(probe.device)
    batch_size = model._hyperparameters["eval_batch_size"]
    batches = [
        (
            torch.from_numpy(data.score.numeric[start : start + batch_size].copy()).to(device),
            torch.from_numpy(data.score.categorical[start : start + batch_size].copy()).to(device),
        )
        for start in range(0, data.score.rows, batch_size)
    ]

    def model_only() -> None:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            for numeric, categorical in batches:
                network(numeric, categorical)

    latency = []
    for row in range(min(diagnostics.LATENCY_ROWS, data.score.rows)):
        single = NeuralInputs(
            data.score.numeric[row : row + 1].copy(), data.score.categorical[row : row + 1].copy()
        )
        if row == 0:
            for _ in range(diagnostics.WARM_UP_CALLS):
                model.predict(single)
        probe.synchronise()
        start = time.perf_counter()
        model.predict(single)
        probe.synchronise()
        latency.append(time.perf_counter() - start)
    rows = min(CPU_TIMING_ROWS, data.score.rows)
    cpu_inputs = NeuralInputs(
        data.score.numeric[:rows].copy(), data.score.categorical[:rows].copy()
    )
    cpu_network = copy.deepcopy(network).to("cpu")
    cpu_device = torch.device("cpu")

    def cpu_scoring() -> None:
        predict_logits(cpu_network, cpu_inputs, device=cpu_device, batch_size=batch_size)

    cpu = diagnostics.timed_calls(cpu_scoring, lambda: None)
    probe.reset_peak()
    end_to_end = diagnostics.timed_calls(lambda: model.predict(data.score), probe.synchronise)
    end_to_end_peak = probe.peak_mib()
    probe.reset_peak()
    only = diagnostics.timed_calls(model_only, probe.synchronise)
    model_only_peak = probe.peak_mib()
    history = resolved["history"]
    train_seconds = float(np.mean([epoch["train_seconds"] for epoch in history]))
    step_seconds = train_seconds / resolved["batches_per_epoch"]
    full_steps = -(-probe.sizes.train // model._hyperparameters["batch_size"])
    return {
        "end_to_end": {**end_to_end, **end_to_end_peak},
        "model_only": {**only, **model_only_peak},
        "single_row_latency": diagnostics.latency_summary(latency),
        "cpu_scoring": {**cpu, "rows": rows, "scores": "discarded"},
        "rows": data.score.rows,
        "train_seconds_per_epoch": train_seconds,
        "full_train_epoch_minutes_extrapolated": step_seconds * full_steps / 60,
        "extrapolation": "mean training-only seconds per probe epoch, divided by its batches, "
        "times the batches of one full Train epoch; excludes V1 scoring and other unmeasured work",
    }


def _neural_checks(
    probe: Probe,
    model: NeuralModel,
    data: NeuralSet,
    weighting: Weighting,
    hyperparameters: Mapping[str, Any],
    subject: str,
    key: Callable[[str], str],
) -> Scores:
    """C1, C3, C6–C10, C12 and C13 of the first fit, with every scored array kept; its scores."""
    resolved = model.parameters["resolved"]
    train, valid = data.train[weighting], data.valid[weighting]
    weights_item(probe, subject, train, valid, weighting, model.fitted_iterations)
    _device_item(probe, model, train, subject)
    _neural_settings_item(probe, model, subject)
    _computation_item(probe, model, data, weighting, hyperparameters, f"{subject} fit 1")
    reference = _float32_item(probe, model, data, hyperparameters["eval_batch_size"], subject)
    reference_scores = probabilities(reference)
    probe.keep_scores(key("reference"), reference_scores, data.score_ids)
    scores = model.predict(data.score)
    probe.keep_scores(key("score"), scores, data.score_ids)
    k = top_k(len(scores))
    probe.book.add("C9", subject, bool(np.isfinite(scores).all()))
    probe.book.add("C9", subject, None, **diagnostics.top_k_summary(scores, data.score_ids, k))
    against = diagnostics.compare_scores(scores, reference_scores, data.score_ids, k)
    probe.book.add("C9", f"{subject} float32 path against reference", None, **against)
    repeated = model.predict(data.score)
    probe.keep_scores(key("repeat"), repeated, data.score_ids)
    probe.book.add("C10", subject, diagnostics.bitwise_equal(scores, repeated))
    with tempfile.TemporaryDirectory() as directory:
        model.save(Path(directory))
        loaded = type(model).load(
            Path(directory),
            model.configuration,
            model.seed,
            settings=probe.settings,
            layout=model.layout,
            device=model.device,
        )
        reloaded = loaded.predict(data.score)
        del loaded
    probe.keep_scores(key("reload"), reloaded, data.score_ids)
    probe.book.add("C12", subject, diagnostics.bitwise_equal(scores, reloaded))
    timing = _neural_timing(probe, model, data, resolved)
    probe.book.add("C13", f"{subject} scoring", None, **timing)
    return scores


def run_neural(probe: Probe, data: NeuralSet) -> None:
    """All neural configurations and variants, each fitted twice with the same seed (C11), with a
    checkpoint after every fit."""
    for model_name in ("resnet", "ft_transformer"):
        for variant, values in NEURAL_VARIANTS[model_name].items():
            hyperparameters = {**NEURAL_TRAINING, **NEURAL_SHARED[model_name], **values}
            for weighting in (Weighting.UNWEIGHTED, Weighting.BALANCED):
                name = f"{LABELS[model_name]}-{weighting.value}"
                subject = f"{name} {variant}"

                def key(
                    path: str, repetition: str = "fit1", name: str = name, variant: str = variant
                ) -> str:
                    return score_key(name, variant, probe.device, repetition, path)

                with log_step(logger, f"Probe {subject}"):
                    fit = f"{subject} fit 1"
                    first = _fit_neural(
                        probe, model_name, name, data, weighting, hyperparameters, fit
                    )
                    scores = _neural_checks(
                        probe, first, data, weighting, hyperparameters, subject, key
                    )
                    del first
                    gc.collect()
                    probe.checkpoint(f"{fit} checked")
                    fit = f"{subject} fit 2"
                    second = _fit_neural(
                        probe, model_name, name, data, weighting, hyperparameters, fit
                    )
                    _computation_item(probe, second, data, weighting, hyperparameters, fit)
                    again = second.predict(data.score)
                    probe.keep_scores(key("score", "fit2"), again, data.score_ids)
                    probe.book.add(
                        "C11",
                        f"{subject} same seed, same process",
                        diagnostics.bitwise_equal(scores, again),
                    )
                    del second
                    gc.collect()
                    if probe.gpu:
                        torch.cuda.empty_cache()
                    probe.checkpoint(f"{fit} checked")


# --- Run comparison (C11), warnings (C2) and the run ---------------------------------------------


IDENTITY: tuple[tuple[str, ...], ...] = (
    ("schema_version",),
    ("code", "git_commit"),
    ("code", "package", "source_sha256"),
    ("specification", "sha256"),
    ("settings", "sha256"),
    ("run", "data_seed"),
    ("run", "model_seed"),
    ("run", "device"),
    ("run", "sizes"),
    ("synthetic", "tree_sha256"),
    ("synthetic", "neural_sha256"),
)


def _value(record: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def environment_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """The execution conditions two runs must share to be compared."""
    environment = record.get("environment", {})
    gpu = environment.get("gpu", {})
    cpu = environment.get("cpu", {})
    return {
        "python": environment.get("python"),
        "platform": environment.get("platform"),
        "packages": environment.get("packages"),
        "logical_processors": cpu.get("logical_processors"),
        "affinity": cpu.get("affinity"),
        "torch_threads": environment.get("torch_threads"),
        "environment_variables": environment.get("environment_variables"),
        "gpu": {
            key: gpu.get(key)
            for key in (
                "name",
                "driver",
                "torch_cuda",
                "cudnn",
                "compute_capability",
                "multiprocessors",
                "memory_bytes",
            )
        },
    }


def _read_previous(path: Path) -> dict[str, Any]:
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileAccessError(f"{path}: first-run record cannot be read") from error
    except ValueError as error:
        raise SpecificationError(f"{path}: first-run record is not valid JSON") from error
    if not isinstance(previous, dict) or not isinstance(previous.get("scores"), dict):
        raise SpecificationError(f"{path}: not a probe record")
    return previous


def compare_runs(probe: Probe, current: Mapping[str, Any], previous_path: Path) -> dict[str, Any]:
    """C11 across runs: the first run's scores against this run's, once the two runs are shown to
    be comparable: a finished first run of the same schema, in another process, with the same code
    and source, specification, settings, seeds, sizes, synthetic inputs (with layouts and weights),
    execution environment and score keys; for acceptance evidence, both from a clean committed
    tree."""
    previous = _read_previous(previous_path)
    differences = [
        ".".join(path) for path in IDENTITY if _value(previous, path) != _value(current, path)
    ]
    if environment_identity(previous) != environment_identity(current):
        differences.append("environment")
    if _value(previous, ("run", "stage")) != "finished":
        differences.append("first run not finished")
    token = ("run", "process", "token")
    if _value(previous, token) == _value(current, token):
        differences.append("same process")
    if set(previous["scores"]) != set(probe.arrays):
        differences.append("score keys")
    recorded = previous.get("model_hashes")
    previous_hashes: dict[str, Any] = recorded if isinstance(recorded, dict) else {}
    if not isinstance(recorded, dict) or set(previous_hashes) != set(probe.model_hashes):
        differences.append("model hash keys")
    if current["run"]["acceptance_evidence"] and (
        _value(previous, ("code", "git_uncommitted_changes")) is not False
        or _value(current, ("code", "git_uncommitted_changes")) is not False
    ):
        differences.append("uncommitted changes")
    if differences:
        probe.book.add(
            "C11", "first run against this run", False, comparable=False, differences=differences
        )
        return {"comparable": False, "differences": differences}
    results: dict[str, Any] = {}
    for key, (scores, ids) in probe.arrays.items():
        try:
            first, first_ids = diagnostics.load_scores(
                previous_path.parent / "scores", previous["scores"][key]
            )
        except (OSError, ValueError, KeyError) as error:
            raise FileAccessError(f"First-run comparison file for {key} is unreadable") from error
        if not diagnostics.bitwise_equal(first_ids, ids):
            probe.book.add("C11", f"{key} across runs", False, row_order_differs=True)
            continue
        comparison = diagnostics.compare_scores(first, scores, ids, top_k(len(scores)))
        results[key] = comparison
        # LightGBM (deterministic CPU training) and the neural models must repeat bit for bit;
        # CatBoost's GPU training is non-deterministic, so its differences are descriptive (T3)
        deterministic = key.startswith(("LGB", "RES", "FTT"))
        outcome = comparison["bitwise_identical"] if deterministic else None
        probe.book.add("C11", f"{key} across runs", outcome, **comparison)
    # LightGBM's deterministic training must also repeat its complete saved model (DR-013 §8)
    if not probe.model_hashes:
        probe.book.add("C11", "LightGBM model text across runs", False, model_hashes=0)
    for key, digest in probe.model_hashes.items():
        probe.book.add(
            "C11",
            f"{key} across runs",
            previous_hashes[key] == digest,
            first_run_sha256=previous_hashes[key],
            this_run_sha256=digest,
        )
    return {"comparable": True, "first_run": previous_path.as_posix(), "scores": results}


def classify_warnings(probe: Probe) -> None:
    """C2. From LightGBM's own logger, either categorical ``max_bin`` notice, exactly, is accepted
    when the probe's Dataset has a categorical feature with more bins than ``max_bin`` (C4). Any
    other warning of an ignored setting, an unsupported capability or a fallback fails. Then, from
    LightGBM's own logger, "no further splits" is accepted (C6 is checked separately). Any other
    warning needs a recorded review."""
    max_bin = probe.settings.lightgbm.params("cpu")["max_bin"]
    above = probe.categorical_bins is not None and probe.categorical_bins > max_bin
    for item in probe.capture.records:
        text = item["text"]
        lightgbm_message = item["logger"] == LIGHTGBM_LOGGER
        message = text.removeprefix(LIGHTGBM_WARNING_PREFIX).strip()
        outcome: bool | str
        if lightgbm_message and above and message in CATEGORICAL_BIN_WARNINGS:
            outcome = True
        elif FAILING_WARNING.search(text):
            outcome = False
        elif lightgbm_message and ACCEPTED_WARNING in text:
            outcome = True
        else:
            outcome = "review"
        probe.book.add("C2", item["phase"], outcome, logger=item["logger"], text=text)
    if not probe.capture.records:
        probe.book.add("C2", "whole run", True, warnings=0)


def run_probe(
    paths: ProjectPaths,
    output: Path,
    *,
    previous: Path | None = None,
    device: Device = "gpu",
    sizes: ProbeSizes | None = None,
    log_opened: bool = False,
) -> dict[str, Any]:
    """Run the probe, write its records to ``output`` at every stage, after every fit and on any
    failure (start-up and finalisation included), and return the final record. ``log_opened``:
    the caller has itself just created this run's ``probe.log`` in ``output``."""
    sizes = sizes or ProbeSizes()
    output = output.resolve()
    check_output(output, previous, log_opened=log_opened)
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FileAccessError(f"{output}: cannot be created ({error.strerror})") from error
    specification = paths.root / SPECIFICATION
    started = time.perf_counter()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "specification": {"path": SPECIFICATION, "sha256": None},
        "settings": None,
        "run": {
            "started": datetime.now(UTC).isoformat(timespec="seconds"),
            "finished": None,
            "stage": "start",
            "error": None,
            "device": device,
            "acceptance_evidence": device == "gpu",
            "process": {"pid": os.getpid(), "token": PROCESS_TOKEN},
            "data_seed": DATA_SEED,
            "model_seed": MODEL_SEED,
            "sizes": asdict(sizes),
            "phases": {},
        },
        "code": None,
        "environment": None,
        "lightgbm_build": None,
        "synthetic": {},
        "gpu_memory": None,
        "host_peak_bytes": None,
        "fits": [],
        "scores": {},
        "model_hashes": {},
        "comparison": None,
        "checks": CheckBook().to_json(),
        "overall": "in progress",
    }
    probe: Probe | None = None
    sampler: diagnostics.GpuMemorySampler | None = None
    classified = False

    def write(stage: str) -> dict[str, Any]:
        record["run"]["stage"] = stage
        if probe is not None:
            record["fits"], record["scores"] = probe.fits, probe.scores
            record["model_hashes"] = probe.model_hashes
            record["checks"] = probe.book.to_json()
        record["gpu_memory"] = None if sampler is None else sampler.snapshot()
        record["host_peak_bytes"] = diagnostics.host_peak_bytes()
        written: dict[str, Any] = jsonable(record)  # type: ignore[assignment]
        write_records(output, written, specification)
        return written

    with captured_warnings() as capture:
        try:
            record["specification"]["sha256"] = file_sha256(specification)
            settings, snapshot = load_settings_with_snapshot(paths)
            record["settings"] = {"path": snapshot.path, "sha256": snapshot.sha256}
            record["code"] = code_provenance(paths.root)
            record["environment"] = environment_record()
            build = verify_lightgbm_build(settings.lightgbm_build) if device == "gpu" else None
            record["lightgbm_build"] = None if build is None else jsonable(asdict(build))
            torch_device(device)  # a GPU run stops here if CUDA is unavailable: no CPU fallback
            features = load_feature_spec(paths.features_spec)
            levels = category_levels(read_catalog(paths.feature_catalog), features)
            probe = Probe(paths, output, device, sizes, settings, features, capture)
            probe.checkpoint = write
            phases = record["run"]["phases"]
            with diagnostics.sampled_gpu_memory(device == "gpu") as running:
                sampler = running
                phases["start_up_seconds"] = time.perf_counter() - started
                mark = time.perf_counter()
                with log_step(logger, "Probe synthetic inputs"):
                    rng = data_generator()
                    tree = tree_data(features, levels, sizes, rng)
                    layout = neural_layout(features, levels)
                    neural = neural_data(layout, sizes, rng)
                phases["input_preparation_seconds"] = time.perf_counter() - mark
                record["synthetic"] = {
                    "tree_sha256": tree.sha256,
                    "neural_sha256": neural.sha256,
                    "tree": tree.diagnostics,
                    "neural": neural.diagnostics,
                }
                unweighted = tree.train[Weighting.UNWEIGHTED].features
                probe.book.add(
                    "C5",
                    "synthetic categorical inputs",
                    None,
                    documented_levels_total=sum(levels.values()),
                    neural_embedding_rows=sum(layout.cardinalities),
                    realised_train_levels=int(
                        sum(unweighted[c].nunique() for c in features.categorical)
                    ),
                    unseen_score_values=sum(tree.unseen.values()),
                )
                del unweighted
                write("synthetic inputs")
                mark = time.perf_counter()
                with log_step(logger, "Probe LightGBM Dataset reproducibility (C4)"):
                    capture.phase = "dataset reproducibility"
                    dataset_reproducibility(probe, tree.train[Weighting.BALANCED])
                write("dataset reproducibility")
                with log_step(logger, "Probe tree-based models"):
                    run_trees(probe, tree)
                phases["tree_seconds"] = time.perf_counter() - mark
                del tree
                gc.collect()
                mark = time.perf_counter()
                with log_step(logger, "Probe neural models"):
                    run_neural(probe, neural)
                phases["neural_seconds"] = time.perf_counter() - mark
            _memory_item(probe, sampler)
            if previous is not None:
                capture.phase = "run comparison"
                with log_step(logger, "Probe comparison with the first run (C11)"):
                    record["comparison"] = compare_runs(probe, record, previous)
            probe.book.add("C13", "run phases", None, **phases)
            classify_warnings(probe)
            classified = True
            record["run"]["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
            record["overall"] = probe.book.overall(compared=previous is not None)
            final = write("finished")
        except BaseException as error:
            if probe is not None and not classified:
                classify_warnings(probe)
            interrupted = isinstance(error, KeyboardInterrupt)
            record["run"]["error"] = f"{type(error).__name__}: {error}"
            record["run"]["finished"] = None
            record["overall"] = "interrupted" if interrupted else "failed"
            try:
                write("interrupted" if interrupted else "failed")
            except Exception:  # the original error is the one that is raised
                logger.exception("The probe record could not be written after the failure")
            raise
    logger.info("GPU acceptance probe: %s (records in %s)", final["overall"], output)
    return final


def _memory_item(probe: Probe, sampler: diagnostics.GpuMemorySampler | None) -> None:
    """C13: the whole-device memory monitoring must hold valid samples over the whole run on the
    GPU; on the CPU there is nothing to monitor."""
    if sampler is None:
        probe.book.add("C13", "GPU memory monitoring", None, monitored=False, device="cpu")
        return
    snapshot = sampler.snapshot()
    valid = (
        not snapshot["problems"]
        and snapshot["samples"] > 1
        and snapshot["total_mib"] is not None
        and snapshot["baseline_used_mib"] is not None
    )
    probe.book.add("C13", "GPU memory monitoring", valid, **snapshot)
