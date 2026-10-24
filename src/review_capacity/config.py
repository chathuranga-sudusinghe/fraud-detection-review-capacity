"""Typed access to the machine-readable declarations in ``config/``.

* ``config/features.yaml``: column roles and derived inputs (DR-008).
* ``config/dataset.yaml``: raw files, temporal partitions and integrity anchors (DR-001).
* ``config/models.yaml``: device and numerical settings of the models (DR-013).

Each loader reads values with their exact types (no silent coercion), rejects a key repeated in
one YAML mapping and a scalar where a list is required, checks the file's internal consistency and
raises ``SpecificationError`` for any malformed or inconsistent declaration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from review_capacity.errors import FileAccessError, SpecificationError
from review_capacity.paths import ProjectPaths

SECONDS_PER_DAY = 86_400  # DR-001: relative day = TransactionDT // 86400; not configurable
_HEX_16 = re.compile(r"[0-9a-f]{16}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects a key repeated in one mapping (no last-key-wins)."""

    def construct_mapping(
        self, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"repeated key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _read_yaml(path: Path, *, content: bytes | None = None) -> dict[str, Any]:
    """The mapping in ``path``, or in ``content`` (bytes already read from ``path``)."""
    try:
        if content is None:
            content = path.read_bytes()
        raw = yaml.load(content.decode("utf-8"), Loader=_UniqueKeyLoader)
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error.strerror})") from error
    except (yaml.YAMLError, KeyError, IndexError, TypeError, ValueError) as error:
        # PyYAML raises TypeError for an unhashable mapping key (a sequence used as a key) and
        # ValueError for a malformed explicitly tagged scalar (e.g. ``!!int invalid``); invalid
        # UTF-8 is a ValueError too.
        raise SpecificationError(f"{path}: cannot be read as YAML") from error
    if not isinstance(raw, dict):
        raise SpecificationError(f"{path} does not contain a mapping")
    return raw


def _integer(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecificationError(f"{what} must be an integer, not {value!r}")
    return value


def _boolean(value: object, what: str) -> bool:
    if not isinstance(value, bool):
        raise SpecificationError(f"{what} must be true or false, not {value!r}")
    return value


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecificationError(f"{what} must be a non-empty string, not {value!r}")
    return value


def _parse[T](path: Path, build: Callable[[], T]) -> T:
    """Run ``build``; report a missing or mistyped entry as ``SpecificationError``."""
    try:
        return build()
    except SpecificationError:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise SpecificationError(f"{path}: malformed declaration ({error!r})") from error


def _require_unique(names: Sequence[str], what: str) -> None:
    repeated = sorted(name for name, n in Counter(names).items() if n > 1)
    if repeated:
        raise SpecificationError(f"Repeated {what}: {repeated}")


@dataclass(frozen=True)
class FeatureSpec:
    """Column roles of the joined data and the derived model inputs (DR-008)."""

    key: str
    time: str
    target: str
    excluded: tuple[str, ...]
    categorical: tuple[str, ...]
    numeric_coded: frozenset[str]
    numeric: tuple[str, ...]
    derived: tuple[str, ...]
    missing_label: str = "__MISSING__"

    @property
    def joined_columns(self) -> tuple[str, ...]:
        """Every column of the joined data: target, excluded, categorical and numeric."""
        return (self.target, *self.excluded, *self.categorical, *self.numeric)

    @property
    def model_inputs(self) -> tuple[str, ...]:
        """The model inputs in canonical order: categorical, numeric, derived."""
        return (*self.categorical, *self.numeric, *self.derived)

    @property
    def numeric_inputs(self) -> tuple[str, ...]:
        """Numeric model inputs: the numeric joined columns and the derived inputs."""
        return (*self.numeric, *self.derived)

    @property
    def text_categorical(self) -> tuple[str, ...]:
        """Categorical columns stored as text in the raw files."""
        return tuple(c for c in self.categorical if c not in self.numeric_coded)

    def validate(self, *, joined_count: int | None = None, input_count: int | None = None) -> None:
        """Check internal consistency; optionally against the declared counts."""
        _require_unique(self.joined_columns, "joined columns")
        _require_unique(self.model_inputs, "model inputs")
        if not self.numeric_coded <= set(self.categorical):
            extra = sorted(self.numeric_coded - set(self.categorical))
            raise SpecificationError(f"numeric_coded columns that are not categorical: {extra}")
        for name in (self.key, self.time):
            if name not in self.excluded:
                raise SpecificationError(f"{name} must be an excluded column")
        if set(self.derived) & set(self.joined_columns):
            raise SpecificationError("Derived inputs must not reuse joined column names")
        if joined_count is not None and len(self.joined_columns) != joined_count:
            raise SpecificationError(
                f"{len(self.joined_columns)} joined columns declared, {joined_count} expected"
            )
        if input_count is not None and len(self.model_inputs) != input_count:
            raise SpecificationError(
                f"{len(self.model_inputs)} model inputs declared, {input_count} expected"
            )


def load_feature_spec(
    path: Path, *, key: str = "TransactionID", time: str = "TransactionDT"
) -> FeatureSpec:
    """Read ``config/features.yaml`` and check it against its own declared counts."""
    raw = _read_yaml(path)
    return _parse(path, lambda: _feature_spec(raw, key, time))


def features_fingerprint(spec: FeatureSpec) -> str:
    """SHA-256 over the column roles the processed files are built from: the key, time and target
    names, the excluded, categorical, numeric-coded, numeric and derived inputs, and the missing
    label. The fingerprint is taken from the loaded specification, not from the file, so it
    describes exactly what the build used, and a change to a comment, a library-handling block or
    a prose reason of ``config/features.yaml`` cannot change it: such a change cannot alter a
    processed file, so a file built before it stays valid. The rules the code applies to these
    roles (the canonical form, the float64 type, the time-of-day inputs) are fixed in the code,
    which a record identifies by its commit and source fingerprint."""
    decided = {
        "key": spec.key,
        "time": spec.time,
        "target": spec.target,
        "excluded": list(spec.excluded),
        "categorical": list(spec.categorical),
        "numeric_coded": sorted(spec.numeric_coded),
        "numeric": list(spec.numeric),
        "derived": list(spec.derived),
        "missing_label": spec.missing_label,
    }
    text = json.dumps(decided, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _names(values: object, what: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise SpecificationError(f"{what} entries must be a list, not {values!r}")
    return tuple(_text(value, what) for value in values)


def _feature_spec(raw: Mapping[str, Any], key: str, time: str) -> FeatureSpec:
    categorical = raw["categorical"]
    spec = FeatureSpec(
        key=key,
        time=time,
        target=_text(raw["target"], "target"),
        excluded=tuple(_text(item["name"], "excluded column") for item in raw["excluded"]),
        categorical=(
            *_names(categorical["transaction"], "categorical column"),
            *_names(categorical["identity"], "categorical column"),
        ),
        numeric_coded=frozenset(_names(categorical["numeric_coded"], "numeric_coded column")),
        numeric=_names(raw["numeric"]["columns"], "numeric column"),
        derived=tuple(_text(item["name"], "derived input") for item in raw["derived"]["numeric"]),
        missing_label=_text(
            raw["encoding"]["categorical_canonical_form"]["missing_token"], "missing_token"
        ),
    )
    if raw["derived"]["source_column"] != time:
        raise SpecificationError(f"Derived inputs must come from {time}")
    source = raw["source"]
    spec.validate(
        joined_count=_integer(source["joined_column_count"], "joined_column_count"),
        input_count=_integer(source["model_input_count"], "model_input_count"),
    )
    declared_columns = _integer(source["model_input_column_count"], "model_input_column_count")
    if len(spec.categorical) + len(spec.numeric) != declared_columns:
        raise SpecificationError("Declared input columns do not match model_input_column_count")
    if len(spec.derived) != _integer(source["derived_input_count"], "derived_input_count"):
        raise SpecificationError("Derived inputs do not match derived_input_count")
    return spec


@dataclass(frozen=True)
class PartitionSpec:
    """One chronological partition of DR-001 with its integrity anchors."""

    name: str
    label: str
    first_day: int
    last_day: int
    rows: int
    transaction_dt_min: int
    transaction_dt_max: int
    transaction_id_sha256_prefix: str
    sealed: bool

    def contains(self, day: int) -> bool:
        return self.first_day <= day <= self.last_day


@dataclass(frozen=True)
class RawFileSpec:
    path: Path
    sha256: str
    columns: int


@dataclass(frozen=True)
class RollingOriginSpec:
    """The window design of the rolling-origin supplement (protocol, Experimental Split
    Structure; Issue #159).

    At origin ``s`` the primary configuration is refitted on relative days ``1`` to
    ``s - gap_days - 1``; the R2 threshold is set on the scores of the window days ``s`` to
    ``s + window_days - 1`` and applied to the next ``transfer_days`` days.
    """

    origins: tuple[int, ...]
    gap_days: int
    window_days: int
    transfer_days: int
    minimum_training_days: int

    def training_last_day(self, origin: int) -> int:
        return origin - self.gap_days - 1

    def window_last_day(self, origin: int) -> int:
        return origin + self.window_days - 1

    def transfer_first_day(self, origin: int) -> int:
        return origin + self.window_days

    def transfer_last_day(self, origin: int) -> int:
        return origin + self.window_days + self.transfer_days - 1


@dataclass(frozen=True)
class DatasetSpec:
    """Raw files and temporal partitions (DR-001)."""

    transaction: RawFileSpec
    identity: RawFileSpec
    key: str
    time: str
    seconds_per_day: int
    total_rows: int
    partitions: tuple[PartitionSpec, ...]
    identity_columns: tuple[str, ...]  # identity-file columns besides the key
    # Last V2 day of the label-latency fitting range (protocol, Experimental Split Structure:
    # V2 days 127-147 fit, days 148-154 are held out). None only for a synthetic declaration
    # without a latency range.
    label_latency_fitting_last_day: int | None = None
    # The window design of the rolling-origin supplement; None only for a synthetic declaration
    # without a supplement.
    rolling_origin: RollingOriginSpec | None = None

    def source_table(self, column: str) -> str:
        """Raw file a joined column comes from: ``identity`` if declared there, else
        ``transaction`` (the key is in both)."""
        return "identity" if column in self.identity_columns else "transaction"

    @property
    def first_day(self) -> int:
        return self.partitions[0].first_day

    @property
    def last_day(self) -> int:
        return self.partitions[-1].last_day

    @property
    def development(self) -> tuple[PartitionSpec, ...]:
        """Partitions whose labels may be read before the protocol freeze."""
        return tuple(p for p in self.partitions if not p.sealed)

    @property
    def last_development_day(self) -> int:
        return max(p.last_day for p in self.development)

    def partition(self, name: str) -> PartitionSpec:
        for part in self.partitions:
            if part.name == name:
                return part
        raise KeyError(name)

    def validate(self) -> None:
        """Check the DR-001 rules: a fixed day length; well-formed anchors; contiguous, ordered
        partitions summing to the total; and the last partition sealed (Test), after every
        development partition."""
        _require_unique([p.name for p in self.partitions], "partition names")
        if not self.partitions:
            raise SpecificationError("No partition declared")
        if self.seconds_per_day != SECONDS_PER_DAY:
            raise SpecificationError(
                f"seconds_per_day must be {SECONDS_PER_DAY} (DR-001), not {self.seconds_per_day}"
            )
        for raw_file in (self.transaction, self.identity):
            if not _HEX_64.fullmatch(raw_file.sha256):
                raise SpecificationError(f"{raw_file.path.name}: sha256 must be 64 hex characters")
            if raw_file.columns <= 0:
                raise SpecificationError(f"{raw_file.path.name}: columns must be positive")
        _require_unique(self.identity_columns, "identity columns")
        if self.key in self.identity_columns:
            raise SpecificationError(f"identity_columns must not list the key {self.key}")
        if len(self.identity_columns) != self.identity.columns - 1:
            raise SpecificationError(
                f"{len(self.identity_columns)} identity columns declared besides the key, "
                f"{self.identity.columns - 1} expected"
            )
        if not self.partitions[-1].sealed:
            raise SpecificationError("The last partition (Test) must be sealed before the freeze")
        if not self.development:
            raise SpecificationError("No development (unsealed) partition declared")
        for previous, current in zip(self.partitions, self.partitions[1:], strict=False):
            if current.first_day != previous.last_day + 1:
                raise SpecificationError(
                    f"Partitions {previous.name} and {current.name} not contiguous"
                )
            if current.transaction_dt_min <= previous.transaction_dt_max:
                raise SpecificationError(f"TransactionDT bounds of {current.name} overlap")
        for part in self.partitions:
            if part.first_day > part.last_day or part.rows <= 0:
                raise SpecificationError(f"Invalid partition {part.name}")
            if part.transaction_dt_min > part.transaction_dt_max:
                raise SpecificationError(f"TransactionDT bounds of {part.name} are reversed")
            if not _HEX_16.fullmatch(part.transaction_id_sha256_prefix):
                raise SpecificationError(
                    f"{part.name}: transaction_id_sha256_prefix must be 16 hex characters"
                )
            days = (
                part.transaction_dt_min // self.seconds_per_day,
                part.transaction_dt_max // self.seconds_per_day,
            )
            if not (part.contains(days[0]) and part.contains(days[1])):
                raise SpecificationError(f"TransactionDT bounds of {part.name} outside its days")
        latency = self.label_latency_fitting_last_day
        if latency is not None:
            v2 = self.partition("v2")
            if not v2.first_day <= latency < v2.last_day:
                raise SpecificationError(
                    f"label_latency.fitting_last_day {latency} must lie in "
                    f"{v2.first_day}..{v2.last_day - 1} (a held-out part must remain)"
                )
        rolling = self.rolling_origin
        if rolling is not None:
            train, v2 = self.partition("train"), self.partition("v2")
            if not rolling.origins or list(rolling.origins) != sorted(set(rolling.origins)):
                raise SpecificationError("rolling_origin.origins must be increasing and distinct")
            if min(rolling.gap_days, rolling.window_days, rolling.transfer_days) < 1:
                raise SpecificationError("rolling_origin: the gap and the windows are positive")
            for origin in rolling.origins:
                last = rolling.training_last_day(origin)
                if last < train.first_day - 1 + rolling.minimum_training_days:
                    raise SpecificationError(
                        f"rolling_origin: origin {origin} trains on fewer than "
                        f"{rolling.minimum_training_days} days"
                    )
                if last > train.last_day:
                    raise SpecificationError(
                        f"rolling_origin: origin {origin} trains beyond {train.name}"
                    )
                if rolling.transfer_last_day(origin) > v2.last_day:
                    raise SpecificationError(
                        f"rolling_origin: the transfer window of origin {origin} ends after "
                        f"day {v2.last_day}"
                    )
        if sum(p.rows for p in self.partitions) != self.total_rows:
            raise SpecificationError("Partition rows do not sum to total_rows")
        sealed = [p for p in self.partitions if p.sealed]
        if sealed and any(p.last_day > sealed[0].first_day for p in self.development):
            raise SpecificationError("A development partition lies after the sealed period")


def _rolling_origin(raw: Mapping[str, Any]) -> RollingOriginSpec:
    return RollingOriginSpec(
        origins=tuple(_integer(day, "rolling_origin origin") for day in raw["origins"]),
        gap_days=_integer(raw["gap_days"], "rolling_origin gap_days"),
        window_days=_integer(raw["window_days"], "rolling_origin window_days"),
        transfer_days=_integer(raw["transfer_days"], "rolling_origin transfer_days"),
        minimum_training_days=_integer(
            raw["minimum_training_days"], "rolling_origin minimum_training_days"
        ),
    )


def _raw_file(entry: Mapping[str, Any], paths: ProjectPaths, name: str) -> RawFileSpec:
    return RawFileSpec(
        path=paths.resolve(_text(entry["path"], f"{name} path")),
        sha256=_text(entry["sha256"], f"{name} sha256"),
        columns=_integer(entry["columns"], f"{name} columns"),
    )


def _partition(p: Mapping[str, Any]) -> PartitionSpec:
    name = _text(p["name"], "partition name")
    first_day, last_day = p["days"]
    dt_min, dt_max = p["transaction_dt"]
    return PartitionSpec(
        name=name,
        label=_text(p["label"], f"{name} label"),
        first_day=_integer(first_day, f"{name} first day"),
        last_day=_integer(last_day, f"{name} last day"),
        rows=_integer(p["rows"], f"{name} rows"),
        transaction_dt_min=_integer(dt_min, f"{name} TransactionDT minimum"),
        transaction_dt_max=_integer(dt_max, f"{name} TransactionDT maximum"),
        transaction_id_sha256_prefix=_text(
            p["transaction_id_sha256_prefix"], f"{name} transaction_id_sha256_prefix"
        ),
        sealed=_boolean(p["sealed"], f"{name} sealed"),
    )


def load_dataset_spec(paths: ProjectPaths) -> DatasetSpec:
    """Read ``config/dataset.yaml``; its raw-file paths are relative to the project root."""
    path = paths.dataset_spec
    raw = _read_yaml(path)

    def build() -> DatasetSpec:
        spec = DatasetSpec(
            transaction=_raw_file(raw["raw_files"]["transaction"], paths, "transaction"),
            identity=_raw_file(raw["raw_files"]["identity"], paths, "identity"),
            key=_text(raw["key"], "key"),
            time=_text(raw["time"], "time"),
            seconds_per_day=_integer(raw["seconds_per_day"], "seconds_per_day"),
            total_rows=_integer(raw["total_rows"], "total_rows"),
            partitions=tuple(_partition(p) for p in raw["partitions"]),
            identity_columns=_names(
                raw["raw_files"]["identity"]["columns_besides_key"], "identity column"
            ),
            label_latency_fitting_last_day=_integer(
                raw["label_latency"]["fitting_last_day"], "label_latency fitting_last_day"
            ),
            rolling_origin=_rolling_origin(raw["rolling_origin"]),
        )
        spec.validate()
        return spec

    return _parse(path, build)


type Device = Literal["gpu", "cpu"]
type ParameterValue = bool | int | float | str | tuple[str, ...]
type ParameterParser = Callable[[object, str], ParameterValue]

DEVICES: tuple[Device, ...] = ("gpu", "cpu")
SEED_LIMIT = 2**32  # NumPy's legacy global generator accepts seeds 0 to 2**32 - 1
FP32_PRECISIONS = ("ieee", "tf32")  # PyTorch's float32 matrix-multiplication precisions
TRAINING_DTYPES = ("bfloat16",)  # the values the neural-model code supports (DR-013 §4)
PREDICTION_DTYPES = ("float32",)
GPU_ATTENTION_BACKENDS = ("efficient",)

# The parameter that selects each library's device, and its value for each device the library is
# declared for (DR-013): LightGBM trains on the CPU only (§2).
_DEVICE_KEYS: dict[str, tuple[str, dict[Device, str]]] = {
    "lightgbm": ("device_type", {"cpu": "cpu"}),
    "catboost": ("task_type", {"gpu": "GPU", "cpu": "CPU"}),
}
# The models of DR-011, by package name (``models.registry``), each with a training device.
TRAINED_MODELS = ("lightgbm", "catboost", "resnet", "ft_transformer")


def _possibly_empty_text(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise SpecificationError(f"{what} must be a string, not {value!r}")
    return value


def _positive(value: object, what: str) -> int:
    number = _integer(value, what)
    if number <= 0:
        raise SpecificationError(f"{what} must be a positive integer, not {number}")
    return number


def _max_bin(value: object, what: str) -> int:
    number = _integer(value, what)
    if number <= 1:  # LightGBM 4.7.0 requires max_bin > 1 (config_auto.cpp)
        raise SpecificationError(f"{what} must be greater than 1, not {number}")
    return number


def _non_negative(value: object, what: str) -> int:
    number = _integer(value, what)
    if number < 0:
        raise SpecificationError(f"{what} must be a non-negative integer, not {number}")
    return number


def _number(value: object, what: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpecificationError(f"{what} must be numeric, not {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise SpecificationError(f"{what} must be finite, not {value!r}")
    return value


def _string_list(value: object, what: str) -> tuple[str, ...]:
    values = _names(value, what)
    if not values:
        raise SpecificationError(f"{what} must be a non-empty list")
    _require_unique(values, what)
    return values


def _choice(value: object, what: str, choices: tuple[str, ...]) -> str:
    text = _text(value, what)
    if text not in choices:
        raise SpecificationError(f"{what} must be one of {choices}, not {text!r}")
    return text


def _seed(value: object, what: str) -> int:
    seed = _integer(value, what)
    if not 0 <= seed < SEED_LIMIT:
        raise SpecificationError(f"{what} must be between 0 and {SEED_LIMIT - 1}, not {seed}")
    return seed


# The parameters each section must declare, with the parser of each value. The values belong to
# config/models.yaml (DR-013); the code fixes which keys exist and their types, so that a missing
# key can never fall back silently to a library default.
_SCHEMAS: dict[str, dict[str, dict[str, ParameterParser]]] = {
    "lightgbm": {
        "shared": {
            "objective": _text,
            "max_bin": _max_bin,
            "min_data_in_bin": _positive,
            "bin_construct_sample_cnt": _positive,
            "enable_bundle": _boolean,
            "is_enable_sparse": _boolean,
            "force_row_wise": _boolean,
            "force_col_wise": _boolean,
            "feature_pre_filter": _boolean,
            "use_missing": _boolean,
            "zero_as_missing": _boolean,
            "use_quantized_grad": _boolean,
            "linear_tree": _boolean,
            "forcedsplits_filename": _possibly_empty_text,
            "forcedbins_filename": _possibly_empty_text,
            "histogram_pool_size": _number,
        },
        "cpu": {"device_type": _text, "deterministic": _boolean, "num_threads": _positive},
    },
    "catboost": {
        "shared": {
            "loss_function": _text,
            "border_count": _positive,
            "feature_border_type": _text,
            "nan_mode": _text,
            "bootstrap_type": _text,
            "boosting_type": _text,
            "has_time": _boolean,
            "max_ctr_complexity": _positive,
            "one_hot_max_size": _non_negative,
            "counter_calc_method": _text,
            "grow_policy": _text,
            "score_function": _text,
            "random_score_type": _text,
            "leaf_estimation_method": _text,
            "leaf_estimation_backtracking": _text,
            "sampling_unit": _text,
        },
        "gpu": {
            "task_type": _text,
            "devices": _text,
            "simple_ctr": _string_list,
            "combinations_ctr": _string_list,
        },
        "cpu": {
            "task_type": _text,
            "thread_count": _positive,
            "simple_ctr": _string_list,
            "combinations_ctr": _string_list,
        },
    },
}


def _mapping(raw: object, what: str, keys: Sequence[str]) -> dict[str, Any]:
    """``raw`` as a mapping with exactly ``keys``: none missing, none unexpected."""
    if not isinstance(raw, dict):
        raise SpecificationError(f"{what} must be a mapping")
    if any(not isinstance(key, str) for key in raw):
        raise SpecificationError(f"{what} keys must be strings")
    missing = sorted(set(keys) - set(raw))
    unexpected = sorted(set(raw) - set(keys))
    if missing or unexpected:
        raise SpecificationError(f"{what}: missing keys {missing}; unexpected keys {unexpected}")
    return raw


def _parameters(
    raw: object, what: str, schema: Mapping[str, ParameterParser]
) -> dict[str, ParameterValue]:
    values = _mapping(raw, what, tuple(schema))
    return {key: parse(values[key], f"{what}.{key}") for key, parse in schema.items()}


@dataclass(frozen=True)
class LibrarySettings:
    """Parameters of one tree library: shared by its devices, and for each device it is declared
    for (DR-013); ``gpu`` is None for a library without GPU settings (LightGBM, §2).

    A shared key appears in no device section; a device key (the device selector, CTR lists) may
    appear in both device sections with device-specific values."""

    library: str
    shared: Mapping[str, ParameterValue]
    gpu: Mapping[str, ParameterValue] | None
    cpu: Mapping[str, ParameterValue]

    @property
    def devices(self) -> tuple[Device, ...]:
        """The devices this library has settings for."""
        return tuple(device for device in DEVICES if self._own(device) is not None)

    def _own(self, device: Device) -> Mapping[str, ParameterValue] | None:
        return self.gpu if device == "gpu" else self.cpu

    def params(self, device: Device) -> dict[str, Any]:
        """The parameters for ``device``: the shared ones and the device's own, with lists as
        Python lists (the form both libraries accept)."""
        if device not in DEVICES:
            raise ValueError(f"Unsupported device {device!r}; expected one of {DEVICES}")
        own = self._own(device)
        if own is None:
            raise SpecificationError(
                f"{self.library} has no {device} settings: it is declared for {self.devices} "
                "only (DR-013)"
            )
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in {**self.shared, **own}.items()
        }

    def validate(self) -> None:
        key, expected = _DEVICE_KEYS[self.library]
        if self.devices != tuple(expected):
            raise SpecificationError(
                f"{self.library} must have settings for {tuple(expected)}, not {self.devices}"
            )
        for device in self.devices:
            own = self._own(device) or {}
            overlap = sorted(set(self.shared) & set(own))
            if overlap:
                raise SpecificationError(
                    f"{self.library}: keys both shared and {device}-specific: {overlap}"
                )
            if own.get(key) != expected[device]:
                raise SpecificationError(
                    f"{self.library} {device} settings must set {key}={expected[device]!r}"
                )


@dataclass(frozen=True)
class LightGBMBuild:
    """The recorded LightGBM CUDA build (DR-013 §2)."""

    version: str
    library_sha256: str


@dataclass(frozen=True)
class NeuralSettings:
    """Precision and determinism settings of both neural models (DR-013 §4)."""

    matmul_fp32_precision: str
    allow_bf16_reduced_precision_reduction: bool
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    training_dtype: str
    prediction_dtype: str
    gpu_attention_backend: str


@dataclass(frozen=True)
class ModelSettings:
    """Device and numerical settings of the models (``config/models.yaml``, DR-013)."""

    main_seed: int
    training_devices: Mapping[str, Device]
    lightgbm_build: LightGBMBuild
    lightgbm: LibrarySettings
    catboost: LibrarySettings
    neural: NeuralSettings

    def training_device(self, model: str) -> Device:
        """The device ``model`` (a package name of ``models.registry``) trains on (DR-013 §1)."""
        if model not in self.training_devices:
            raise ValueError(f"Unknown model {model!r}; expected one of {TRAINED_MODELS}")
        return self.training_devices[model]


def _library(raw: Mapping[str, Any], library: str, extra: tuple[str, ...] = ()) -> LibrarySettings:
    schema = _SCHEMAS[library]
    section = _mapping(raw[library], library, (*schema, *extra))
    settings = LibrarySettings(
        library=library,
        shared=_parameters(section["shared"], f"{library}.shared", schema["shared"]),
        gpu=(
            _parameters(section["gpu"], f"{library}.gpu", schema["gpu"])
            if "gpu" in schema
            else None
        ),
        cpu=_parameters(section["cpu"], f"{library}.cpu", schema["cpu"]),
    )
    settings.validate()
    return settings


_NEURAL_KEYS = (
    "matmul_fp32_precision",
    "allow_bf16_reduced_precision_reduction",
    "deterministic_algorithms",
    "cudnn_benchmark",
    "training_dtype",
    "prediction_dtype",
    "gpu_attention_backend",
)


def _neural(raw: object) -> NeuralSettings:
    neural = _mapping(raw, "neural", _NEURAL_KEYS)
    return NeuralSettings(
        matmul_fp32_precision=_choice(
            neural["matmul_fp32_precision"], "neural.matmul_fp32_precision", FP32_PRECISIONS
        ),
        allow_bf16_reduced_precision_reduction=_boolean(
            neural["allow_bf16_reduced_precision_reduction"],
            "neural.allow_bf16_reduced_precision_reduction",
        ),
        deterministic_algorithms=_boolean(
            neural["deterministic_algorithms"], "neural.deterministic_algorithms"
        ),
        cudnn_benchmark=_boolean(neural["cudnn_benchmark"], "neural.cudnn_benchmark"),
        training_dtype=_choice(neural["training_dtype"], "neural.training_dtype", TRAINING_DTYPES),
        prediction_dtype=_choice(
            neural["prediction_dtype"], "neural.prediction_dtype", PREDICTION_DTYPES
        ),
        gpu_attention_backend=_choice(
            neural["gpu_attention_backend"], "neural.gpu_attention_backend", GPU_ATTENTION_BACKENDS
        ),
    )


def _training_devices(
    raw: object, lightgbm: LibrarySettings, catboost: LibrarySettings
) -> dict[str, Device]:
    """The training device of every model (DR-013 §1); a tree library trains on a device it has
    settings for."""
    section = _mapping(raw, "training_devices", TRAINED_MODELS)
    devices: dict[str, Device] = {}
    for model in TRAINED_MODELS:
        device = _choice(section[model], f"training_devices.{model}", DEVICES)
        devices[model] = "gpu" if device == "gpu" else "cpu"
    for library in (lightgbm, catboost):
        if devices[library.library] not in library.devices:
            raise SpecificationError(
                f"training_devices.{library.library}: {library.library} has settings for "
                f"{library.devices} only"
            )
    return devices


def load_model_settings(paths: ProjectPaths, *, content: bytes | None = None) -> ModelSettings:
    """Read ``config/models.yaml``: exact keys and value types in every section, the training
    device of every model and the device selectors of each library (DR-013). ``content``: the
    file's bytes, already read, so that the settings are parsed from exactly the bytes a run
    records."""
    path = paths.models_spec
    raw = _read_yaml(path, content=content)

    def build() -> ModelSettings:
        top = _mapping(
            raw,
            "models.yaml",
            ("reproducibility", "training_devices", "lightgbm", "catboost", "neural"),
        )
        reproducibility = _mapping(top["reproducibility"], "reproducibility", ("main_seed",))
        lightgbm = _mapping(top["lightgbm"], "lightgbm", ("build", "shared", "cpu"))
        build_entry = _mapping(lightgbm["build"], "lightgbm.build", ("version", "library_sha256"))
        lightgbm_build = LightGBMBuild(
            version=_text(build_entry["version"], "lightgbm.build.version"),
            library_sha256=_text(build_entry["library_sha256"], "lightgbm.build.library_sha256"),
        )
        if not _HEX_64.fullmatch(lightgbm_build.library_sha256):
            raise SpecificationError("lightgbm.build.library_sha256 must be 64 hex characters")
        lightgbm_settings = _library(top, "lightgbm", extra=("build",))
        catboost_settings = _library(top, "catboost")
        return ModelSettings(
            main_seed=_seed(reproducibility["main_seed"], "reproducibility.main_seed"),
            training_devices=_training_devices(
                top["training_devices"], lightgbm_settings, catboost_settings
            ),
            lightgbm_build=lightgbm_build,
            lightgbm=lightgbm_settings,
            catboost=catboost_settings,
            neural=_neural(top["neural"]),
        )

    return _parse(path, build)
