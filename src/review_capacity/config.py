"""Typed access to the machine-readable declarations in ``config/``.

* ``config/features.yaml``: column roles and derived inputs (DR-008).
* ``config/dataset.yaml``: raw files, temporal partitions and integrity anchors (DR-001).

Each loader reads values with their exact types (no silent coercion), rejects a key repeated in
one YAML mapping and a scalar where a list is required, checks the file's internal consistency and
raises ``SpecificationError`` for any malformed or inconsistent declaration.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            content = yaml.load(handle, Loader=_UniqueKeyLoader)
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error.strerror})") from error
    except yaml.YAMLError as error:
        raise SpecificationError(f"{path}: cannot be read as YAML") from error
    if not isinstance(content, dict):
        raise SpecificationError(f"{path} does not contain a mapping")
    return content


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
        if sum(p.rows for p in self.partitions) != self.total_rows:
            raise SpecificationError("Partition rows do not sum to total_rows")
        sealed = [p for p in self.partitions if p.sealed]
        if sealed and any(p.last_day > sealed[0].first_day for p in self.development):
            raise SpecificationError("A development partition lies after the sealed period")


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
        )
        spec.validate()
        return spec

    return _parse(path, build)
