"""The training declaration (``config/training.yaml``): the machine-readable form of the common
training rules, DR-015 §1–§4 and §6–§8 (Issue #118).

The file is read once; the same bytes are parsed, hashed and recorded (``load_declaration``), so a
run's records always show the declaration in use. The code fixes which keys exist and their types:
a missing, repeated or mistyped entry raises ``SpecificationError``, and no value falls back to a
library default. Device and numerical settings and the main training seed belong to
``config/models.yaml`` (DR-013), not here.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from review_capacity.config import SEED_LIMIT, _read_yaml
from review_capacity.errors import FileAccessError, SpecificationError
from review_capacity.models.registry import MODELS
from review_capacity.models.run_record import SettingsSnapshot
from review_capacity.paths import ProjectPaths

REFERENCE_MODEL = "logistic_regression"  # DR-011 §5; outside the eight configurations
DISTRIBUTIONS = (
    "uniform",
    "log_uniform",
    "uniform_int",
    "log_uniform_int",
    "scaled_uniform_int",
    "mixture_uniform",
    "mixture_log_uniform",
)
INTEGER_DISTRIBUTIONS = ("uniform_int", "log_uniform_int", "scaled_uniform_int")
LOG_DISTRIBUTIONS = ("log_uniform", "log_uniform_int", "mixture_log_uniform")
MIXTURES = ("mixture_uniform", "mixture_log_uniform")
MODEL_NAMES: tuple[str, ...] = (*(m.name for m in MODELS), REFERENCE_MODEL)

type Value = bool | int | float | str | list[Any] | None

# The fixed settings each model must declare, with the kind of each value; the values belong to
# config/training.yaml (DR-015). The code fixes which keys exist and their types, so that a missing
# or mistyped setting can never fall back silently to a library default.
FIXED_SCHEMAS: dict[str, dict[str, str]] = {
    "lightgbm": {
        "boosting": "str",
        "data_sample_strategy": "str",
        "bagging_freq": "int",
        "max_depth": "int",
        "min_sum_hessian_in_leaf": "number",
        "lambda_l1": "number",
        "min_gain_to_split": "number",
        "max_delta_step": "number",
        "path_smooth": "number",
        "extra_trees": "bool",
        "feature_fraction_bynode": "number",
        "boost_from_average": "bool",
        "max_cat_to_onehot": "int",
        "max_cat_threshold": "int",
        "cat_l2": "number",
        "cat_smooth": "number",
        "min_data_per_group": "int",
        "num_iterations": "int",
        "early_stopping_round": "int",
        "early_stopping_min_delta": "number",
        "metric": "str",
    },
    "catboost": {
        "random_strength": "number",
        "model_size_reg": "number",
        "permutation_count": "int",
        "fold_permutation_block": "int",
        "boost_from_average": "bool",
        "eval_metric": "str",
        "metric_period": "int",
        "iterations": "int",
        "od_wait": "int",
    },
    "resnet": {"weight_decay_scope": "str"},
    "ft_transformer": {
        "residual_dropout": "number",
        "d_ffn_factor": "number",
        "n_heads": "int",
        "activation": "str",
        "normalization": "str",
        "initialization": "str",
        "no_weight_decay": "strings",
    },
    "logistic_regression": {
        "solver": "str",
        "l1_ratio": "number",
        "fit_intercept": "bool",
        "class_weight": "null",
        "warm_start": "bool",
        "tol": "number",
        "max_iter": "int",
    },
}
OPTIMISER_SCHEMA: dict[str, str] = {
    "name": "str",
    "betas": "pair",
    "eps": "number",
    "amsgrad": "bool",
    "foreach": "bool",
    "fused": "bool",
    "learning_rate_schedule": "str",
}


@dataclass(frozen=True)
class Parameter:
    """One searched parameter: its distribution, bounds and the value of the reference
    configuration (trial 1)."""

    name: str
    distribution: str
    low: int | float
    high: int | float
    reference: int | float
    factor: int | None = None  # scaled_uniform_int only


@dataclass(frozen=True)
class ModelSpace:
    """A model's search space (parameters in draw order) and its fixed settings."""

    name: str
    parameters: tuple[Parameter, ...]
    fixed: Mapping[str, Value]


@dataclass(frozen=True)
class Stopping:
    """Early stopping in the family's training unit (rounds or epochs)."""

    patience: int
    maximum: int


@dataclass(frozen=True)
class NeuralTraining:
    """Training values common to both neural models (DR-015 §7)."""

    effective_batch: int
    micro_batch_candidates: tuple[int, ...]
    preflight_free_margin_mib: int
    evaluation_batch: Mapping[str, int]
    optimiser: Mapping[str, Value]


PERCENTILE_METHODS = ("linear",)


@dataclass(frozen=True)
class TimingDeclaration:
    """The computational-cost procedure's counts and intervals (DR-015 §9)."""

    scoring_passes: int
    warmup_tree: int
    warmup_neural: int
    latency_rows: int
    latency_warmup: int
    latency_rounds: int
    percentile_method: str
    settling_seconds: int
    host_sample_interval_ms: int


@dataclass(frozen=True)
class TrainingDeclaration:
    """Everything DR-015 fixes, as data."""

    trials: int
    sampler_seed: int
    streams: tuple[str, ...]
    additional_seeds: tuple[int, ...]
    preprocessing_seed: int
    v1_daily_k: int
    tree_stopping: Stopping
    neural_stopping: Stopping
    neural: NeuralTraining
    timing: TimingDeclaration
    models: Mapping[str, ModelSpace]

    def space(self, model: str) -> ModelSpace:
        return self.models[model]

    def stream_index(self, model: str) -> int:
        return self.streams.index(model)


def _mapping(raw: object, what: str, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SpecificationError(f"{what} must be a mapping, not {raw!r}")
    missing = [key for key in keys if key not in raw]
    unknown = [key for key in raw if key not in keys]
    if missing or unknown:
        raise SpecificationError(f"{what}: missing {missing}, unknown {unknown}")
    return raw


def _integer(value: object, what: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SpecificationError(f"{what} must be an integer of at least {minimum}, not {value!r}")
    return value


def _number(value: object, what: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise SpecificationError(f"{what} must be a finite number, not {value!r}")
    return value


def _seed(value: object, what: str) -> int:
    seed = _integer(value, what)
    if seed >= SEED_LIMIT:
        raise SpecificationError(f"{what} must be below {SEED_LIMIT}, not {seed}")
    return seed


def _integers(value: object, what: str, *, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise SpecificationError(f"{what} must be a non-empty list, not {value!r}")
    items = tuple(_integer(item, f"{what} entry", minimum=minimum) for item in value)
    if len(set(items)) != len(items):
        raise SpecificationError(f"{what} must not repeat a value: {list(items)}")
    return items


def _stopping(raw: object, what: str) -> Stopping:
    entry = _mapping(raw, what, ("patience", "maximum"))
    stopping = Stopping(
        _integer(entry["patience"], f"{what}.patience", minimum=1),
        _integer(entry["maximum"], f"{what}.maximum", minimum=1),
    )
    if stopping.patience >= stopping.maximum:
        raise SpecificationError(f"{what}: patience must be below the maximum")
    return stopping


def _parameter(raw: object, model: str) -> Parameter:
    keys = ("name", "distribution", "low", "high", "reference")
    entry = raw if isinstance(raw, dict) else {}
    scaled = entry.get("distribution") == "scaled_uniform_int"
    entry = _mapping(raw, f"{model} parameter", (*keys, "factor") if scaled else keys)
    name = entry["name"]
    what = f"{model}.{name}"
    distribution = entry["distribution"]
    if distribution not in DISTRIBUTIONS:
        raise SpecificationError(f"{what}: unknown distribution {distribution!r}")
    low, high = _number(entry["low"], f"{what}.low"), _number(entry["high"], f"{what}.high")
    reference = _number(entry["reference"], f"{what}.reference")
    if not low < high:
        raise SpecificationError(f"{what}: low must be below high")
    if distribution in LOG_DISTRIBUTIONS and low <= 0:
        raise SpecificationError(f"{what}: a log distribution needs a positive low bound")
    if distribution in INTEGER_DISTRIBUTIONS:
        _integer(low, f"{what}.low", minimum=1 if distribution == "log_uniform_int" else 0)
        _integer(high, f"{what}.high")
    factor = _integer(entry["factor"], f"{what}.factor", minimum=1) if scaled else None
    shown = reference / factor if factor is not None else reference
    if distribution in INTEGER_DISTRIBUTIONS and (
        isinstance(reference, float) or (factor is not None and reference % factor)
    ):
        raise SpecificationError(f"{what}: the reference must be an integer value of the space")
    if not (low <= shown <= high or (distribution in MIXTURES and reference == 0)):
        raise SpecificationError(f"{what}: the reference lies outside the declared bounds")
    as_float = distribution not in INTEGER_DISTRIBUTIONS
    return Parameter(
        _name(name, model),
        distribution,
        low,
        high,
        float(reference) if as_float else int(reference),
        factor,
    )


def _name(value: object, model: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecificationError(f"{model}: a parameter name must be a non-empty string")
    return value


def _is_number(value: object) -> bool:
    """A finite integer or float (never a Boolean)."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


_KINDS: dict[str, Callable[[object], bool]] = {
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": _is_number,
    "bool": lambda v: isinstance(v, bool),
    "str": lambda v: isinstance(v, str) and bool(v),
    "null": lambda v: v is None,
    "strings": lambda v: isinstance(v, list) and bool(v) and all(isinstance(i, str) for i in v),
    "pair": lambda v: isinstance(v, list) and len(v) == 2 and all(_is_number(i) for i in v),
}


def _typed(value: object, kind: str, what: str) -> Value:
    """``value`` if it is of ``kind`` (``int``, ``number``, ``bool``, ``str``, ``null``,
    ``strings`` or ``pair``), else ``SpecificationError``."""
    if not _KINDS[kind](value):
        raise SpecificationError(f"{what} must be of kind {kind}, not {value!r}")
    return cast(Value, value)


def _fixed(raw: object, what: str, schema: Mapping[str, str]) -> dict[str, Value]:
    """The fixed settings in ``raw``: exactly the keys of ``schema``, each of its kind."""
    entry = _mapping(raw, what, tuple(schema))
    return {key: _typed(entry[key], kind, f"{what}.{key}") for key, kind in schema.items()}


def _model_space(raw: object, model: str) -> ModelSpace:
    entry = _mapping(raw, f"models.{model}", ("parameters", "fixed"))
    listed = entry["parameters"]
    if not isinstance(listed, list) or not listed:
        raise SpecificationError(f"models.{model}.parameters must be a non-empty list")
    parameters = tuple(_parameter(item, model) for item in listed)
    names = [p.name for p in parameters]
    if len(set(names)) != len(names):
        raise SpecificationError(f"models.{model}: repeated parameter names {names}")
    fixed = _fixed(entry["fixed"], f"models.{model}.fixed", FIXED_SCHEMAS[model])
    overlap = sorted(set(names) & set(fixed))
    if overlap:
        raise SpecificationError(f"models.{model}: searched and fixed settings overlap: {overlap}")
    return ModelSpace(model, parameters, fixed)


def _agrees(space: ModelSpace, stopping: Stopping, maximum: str, patience: str) -> None:
    """The fixed stopping settings of a tree library are the early-stopping rule of DR-015 §2."""
    expected = {maximum: stopping.maximum, patience: stopping.patience}
    for key, value in expected.items():
        if space.fixed.get(key) != value:
            raise SpecificationError(
                f"models.{space.name}.fixed.{key} must equal early_stopping.tree ({value})"
            )


def _neural(raw: object) -> NeuralTraining:
    entry = _mapping(
        raw,
        "neural_training",
        (
            "effective_batch",
            "micro_batch_candidates",
            "preflight_free_margin_mib",
            "evaluation_batch",
            "optimiser",
        ),
    )
    batch = _integer(entry["effective_batch"], "neural_training.effective_batch", minimum=1)
    micro = _integers(entry["micro_batch_candidates"], "micro_batch_candidates", minimum=1)
    if list(micro) != sorted(micro, reverse=True) or micro[0] != batch:
        raise SpecificationError("micro_batch_candidates: largest first, starting at the batch")
    neural_models = ("resnet", "ft_transformer")
    evaluation = _mapping(entry["evaluation_batch"], "evaluation_batch", neural_models)
    return NeuralTraining(
        batch,
        micro,
        _integer(entry["preflight_free_margin_mib"], "preflight_free_margin_mib", minimum=1),
        {
            model: _integer(evaluation[model], f"evaluation_batch.{model}", minimum=1)
            for model in neural_models
        },
        _fixed(entry["optimiser"], "neural_training.optimiser", OPTIMISER_SCHEMA),
    )


def _timing(raw: object) -> TimingDeclaration:
    entry = _mapping(raw, "timing", ("scoring", "latency", "session", "memory"))
    scoring = _mapping(
        entry["scoring"], "timing.scoring", ("passes", "warmup_tree", "warmup_neural")
    )
    latency = _mapping(
        entry["latency"],
        "timing.latency",
        ("sample_rows", "warmup_calls", "rounds", "percentile_method"),
    )
    session = _mapping(entry["session"], "timing.session", ("settling_seconds",))
    memory = _mapping(entry["memory"], "timing.memory", ("host_sample_interval_ms",))
    method = latency["percentile_method"]
    if method not in PERCENTILE_METHODS:
        raise SpecificationError(
            f"timing.latency.percentile_method must be one of {list(PERCENTILE_METHODS)}"
        )
    return TimingDeclaration(
        scoring_passes=_integer(scoring["passes"], "timing.scoring.passes", minimum=1),
        warmup_tree=_integer(scoring["warmup_tree"], "timing.scoring.warmup_tree", minimum=1),
        warmup_neural=_integer(scoring["warmup_neural"], "timing.scoring.warmup_neural", minimum=1),
        latency_rows=_integer(latency["sample_rows"], "timing.latency.sample_rows", minimum=1),
        latency_warmup=_integer(latency["warmup_calls"], "timing.latency.warmup_calls", minimum=1),
        latency_rounds=_integer(latency["rounds"], "timing.latency.rounds", minimum=1),
        percentile_method=method,
        settling_seconds=_integer(session["settling_seconds"], "timing.session.settling_seconds"),
        host_sample_interval_ms=_integer(
            memory["host_sample_interval_ms"], "timing.memory.host_sample_interval_ms", minimum=1
        ),
    )


def parse_declaration(raw: dict[str, Any]) -> TrainingDeclaration:
    """The declaration in a parsed ``config/training.yaml``; ``SpecificationError`` if it is
    malformed or inconsistent."""
    top = _mapping(
        raw,
        "training.yaml",
        (
            "search",
            "seeds",
            "v1_objective",
            "early_stopping",
            "timing",
            "neural_training",
            "models",
        ),
    )
    search = _mapping(top["search"], "search", ("trials", "sampler_seed", "streams"))
    streams = search["streams"]
    if streams != list(MODEL_NAMES):
        raise SpecificationError(
            f"search.streams must be {list(MODEL_NAMES)} (DR-011 order, then DR-015 §8)"
        )
    seeds = _mapping(top["seeds"], "seeds", ("additional", "preprocessing"))
    additional = _integers(seeds["additional"], "seeds.additional")
    for seed in additional:
        _seed(seed, "seeds.additional entry")
    stopping = _mapping(top["early_stopping"], "early_stopping", ("tree", "neural"))
    tree = _stopping(stopping["tree"], "early_stopping.tree")
    models = _mapping(top["models"], "models", MODEL_NAMES)
    spaces = {name: _model_space(models[name], name) for name in MODEL_NAMES}
    _agrees(spaces["lightgbm"], tree, "num_iterations", "early_stopping_round")
    _agrees(spaces["catboost"], tree, "iterations", "od_wait")
    return TrainingDeclaration(
        trials=_integer(search["trials"], "search.trials", minimum=2),
        sampler_seed=_seed(search["sampler_seed"], "search.sampler_seed"),
        streams=tuple(MODEL_NAMES),
        additional_seeds=additional,
        preprocessing_seed=_seed(seeds["preprocessing"], "seeds.preprocessing"),
        v1_daily_k=_integer(
            _mapping(top["v1_objective"], "v1_objective", ("daily_k",))["daily_k"],
            "v1_objective.daily_k",
            minimum=1,
        ),
        tree_stopping=tree,
        neural_stopping=_stopping(stopping["neural"], "early_stopping.neural"),
        neural=_neural(top["neural_training"]),
        timing=_timing(top["timing"]),
        models=spaces,
    )


def load_declaration(paths: ProjectPaths) -> tuple[TrainingDeclaration, SettingsSnapshot]:
    """Load ``config/training.yaml`` once, with a snapshot of the bytes it was parsed from: the
    same bytes are parsed, hashed and recorded."""
    path = paths.training_spec
    try:
        content = path.read_bytes()
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error.strerror})") from error
    raw = _read_yaml(path, content=content)
    try:
        declaration = parse_declaration(raw)
    except SpecificationError as error:
        raise SpecificationError(f"{path}: {error}") from error
    snapshot = SettingsSnapshot(
        path=path.relative_to(paths.root).as_posix(),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content.decode("utf-8"),
    )
    return declaration, snapshot
