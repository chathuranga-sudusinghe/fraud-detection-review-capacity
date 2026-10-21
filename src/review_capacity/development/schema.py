"""Declarative contracts for the JSON records of the development pipeline (Issue #117).

A contract lists the fields of a record: name, type, constraints and meaning. Contracts are
data, so one declaration validates a record, renders its reference documentation and resolves
the field paths that the generated comparison tables read (``development.tables``).

* ``RecordContract.validate`` reports every structural problem of a record at once; a record's
  cross-field rules run only on a structurally valid record, so that they can rely on its types.
* Records are strict JSON: no unknown or missing fields, no duplicate keys and no NaN or infinite
  numbers. ``dumps`` writes numbers with Python's shortest round-trip representation, so a
  float64 value (for example a selected threshold) is read back exactly by ``loads``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol

from review_capacity.errors import RecordContractError

type Check = Callable[[Mapping[str, Any]], list[str]]


class Kind(Protocol):
    """A value type: reports the problems of a value and describes itself for the reference."""

    def problems(self, value: object, where: str) -> list[str]: ...

    def describe(self) -> str: ...


def _show(value: object) -> str:
    text = repr(value)
    return text if len(text) <= 60 else f"{text[:57]}..."


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _range_text(minimum: float | None, maximum: float | None, exclusive: bool) -> str:
    low = "" if minimum is None else f"{'>' if exclusive else '≥'} {minimum:g}"
    high = "" if maximum is None else f"≤ {maximum:g}"
    return ", ".join(part for part in (low, high) if part)


def _out_of_range(
    value: float, minimum: float | None, maximum: float | None, exclusive: bool, where: str
) -> list[str]:
    below = minimum is not None and (value <= minimum if exclusive else value < minimum)
    above = maximum is not None and value > maximum
    if below or above:
        return [f"{where}: {_show(value)} is outside {_range_text(minimum, maximum, exclusive)}"]
    return []


@dataclass(frozen=True)
class Integer:
    """A JSON integer (never a Boolean or a float such as 3.0)."""

    minimum: int | None = None
    maximum: int | None = None

    def problems(self, value: object, where: str) -> list[str]:
        if isinstance(value, bool) or not isinstance(value, int):
            return [f"{where}: expected an integer, got {_show(value)}"]
        return _out_of_range(value, self.minimum, self.maximum, False, where)

    def describe(self) -> str:
        limits = _range_text(self.minimum, self.maximum, False)
        return f"integer ({limits})" if limits else "integer"


@dataclass(frozen=True)
class Number:
    """A finite JSON number (integer or float, never a Boolean)."""

    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False

    def problems(self, value: object, where: str) -> list[str]:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return [f"{where}: expected a number, got {_show(value)}"]
        if isinstance(value, float) and not math.isfinite(value):  # integers are always finite
            return [f"{where}: {_show(value)} is not a finite number"]
        return _out_of_range(value, self.minimum, self.maximum, self.exclusive_minimum, where)

    def describe(self) -> str:
        limits = _range_text(self.minimum, self.maximum, self.exclusive_minimum)
        return f"number ({limits})" if limits else "number"


@dataclass(frozen=True)
class Text:
    """A non-empty string, optionally matching ``pattern`` in full."""

    pattern: str | None = None
    label: str = "text"

    def problems(self, value: object, where: str) -> list[str]:
        if not isinstance(value, str) or not value:
            return [f"{where}: expected non-empty text, got {_show(value)}"]
        if self.pattern is not None and re.fullmatch(self.pattern, value) is None:
            return [f"{where}: {_show(value)} is not a valid {self.label}"]
        return []

    def describe(self) -> str:
        return self.label


@dataclass(frozen=True)
class RelativePath:
    """A POSIX path relative to a stated base directory, with no ``.`` or ``..`` segment, no
    empty segment and no backslash, so that it cannot leave its base directory."""

    base: str = "the run directory"

    def problems(self, value: object, where: str) -> list[str]:
        if not isinstance(value, str) or not value:
            return [f"{where}: expected a relative path, got {_show(value)}"]
        segments = value.split("/")
        if "\\" in value or any(part in ("", ".", "..") for part in segments):
            return [f"{where}: {_show(value)} is not a relative path inside {self.base}"]
        return []

    def describe(self) -> str:
        return f"path relative to {self.base}"


@dataclass(frozen=True)
class Boolean:
    """A JSON Boolean."""

    def problems(self, value: object, where: str) -> list[str]:
        if not isinstance(value, bool):
            return [f"{where}: expected true or false, got {_show(value)}"]
        return []

    def describe(self) -> str:
        return "Boolean"


@dataclass(frozen=True)
class Choice:
    """One of a fixed set of strings."""

    values: tuple[str, ...]

    def problems(self, value: object, where: str) -> list[str]:
        if not isinstance(value, str) or value not in self.values:
            return [f"{where}: {_show(value)} is not one of {', '.join(self.values)}"]
        return []

    def describe(self) -> str:
        return "one of " + ", ".join(f"`{value}`" for value in self.values)


@dataclass(frozen=True)
class AnyJSON:
    """Any JSON value (object keys are strings; numbers are finite)."""

    def problems(self, value: object, where: str) -> list[str]:
        if value is None or isinstance(value, bool | str) or _is_integer(value):
            return []
        if isinstance(value, float):
            return [] if math.isfinite(value) else [f"{where}: {value!r} is not finite"]
        if isinstance(value, list):
            return [p for i, item in enumerate(value) for p in self.problems(item, f"{where}[{i}]")]
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                return [f"{where}: object keys must be text"]
            return [p for key, item in value.items() for p in self.problems(item, f"{where}.{key}")]
        return [f"{where}: {type(value).__name__} is not a JSON value"]

    def describe(self) -> str:
        return "any JSON value"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class ListOf:
    """A JSON array whose items all have the type ``item``."""

    item: Kind
    min_items: int = 0
    max_items: int | None = None
    unique: bool = False

    def problems(self, value: object, where: str) -> list[str]:
        if not isinstance(value, list):
            return [f"{where}: expected a list, got {_show(value)}"]
        problems = [p for i, x in enumerate(value) for p in self.item.problems(x, f"{where}[{i}]")]
        if len(value) < self.min_items or (
            self.max_items is not None and len(value) > self.max_items
        ):
            problems.append(f"{where}: {len(value)} items; expected {self._count()}")
        if self.unique and not problems and len({_canonical(x) for x in value}) != len(value):
            problems.append(f"{where}: items must be unique")
        return problems

    def _count(self) -> str:
        if self.max_items == self.min_items:
            return f"exactly {self.min_items}"
        high = "" if self.max_items is None else f" and at most {self.max_items}"
        return f"at least {self.min_items}{high}"

    def describe(self) -> str:
        item = "objects" if isinstance(self.item, Object) else self.item.describe()
        count = "" if self.min_items == 0 and self.max_items is None else f", {self._count()}"
        unique = ", unique" if self.unique else ""
        return f"list of {item}{count}{unique}"


@dataclass(frozen=True)
class MapOf:
    """A JSON object with free keys of type ``keys`` and values of type ``value``."""

    value: Kind
    keys: Kind = Text()
    nullable_values: bool = False

    def problems(self, value: object, where: str) -> list[str]:
        if not isinstance(value, dict):
            return [f"{where}: expected an object, got {_show(value)}"]
        problems: list[str] = []
        for key, item in value.items():
            problems += self.keys.problems(key, f"{where} key")
            if item is None and self.nullable_values:
                continue
            problems += self.value.problems(item, f"{where}.{key}")
        return problems

    def describe(self) -> str:
        null = " or null" if self.nullable_values else ""
        return f"object: {self.keys.describe()} → {self.value.describe()}{null}"


@dataclass(frozen=True)
class Field:
    """A named field of an object; ``nullable`` allows JSON null."""

    name: str
    kind: Kind
    meaning: str
    nullable: bool = False


@dataclass(frozen=True)
class Object:
    """A JSON object with exactly the declared fields, in any order."""

    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate field names in {names}")

    def field(self, name: str) -> Field:
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def problems(self, value: object, where: str) -> list[str]:
        if not isinstance(value, dict):
            return [f"{where}: expected an object, got {_show(value)}"]
        declared = [f.name for f in self.fields]
        problems = [f"{where}: missing field {name!r}" for name in declared if name not in value]
        problems += [f"{where}: unknown field {key!r}" for key in value if key not in declared]
        for f in self.fields:
            if f.name not in value:
                continue
            item = value[f.name]
            if item is None:
                if not f.nullable:
                    problems.append(f"{where}.{f.name}: must not be null")
                continue
            problems += f.kind.problems(item, f"{where}.{f.name}")
        return problems

    def describe(self) -> str:
        return "object"


@dataclass(frozen=True)
class RecordContract:
    """The versioned contract of one kind of record: its fields and cross-field rules."""

    kind: str
    version: int
    meaning: str
    body: Object
    checks: tuple[Check, ...] = ()

    def problems(self, record: object) -> list[str]:
        problems = self.body.problems(record, self.kind)
        if problems or not isinstance(record, Mapping):
            return problems
        return [problem for check in self.checks for problem in check(record)]

    def validate(self, record: object) -> None:
        """Raise ``RecordContractError`` listing every problem of ``record``."""
        problems = self.problems(record)
        if problems:
            raise RecordContractError(
                f"{self.kind} record (version {self.version}) does not satisfy its contract:\n- "
                + "\n- ".join(problems)
            )

    def resolve(self, path: str) -> Field:
        """The field at a dotted ``path``: ``a.b`` for nested objects, ``a[].b`` for the objects
        of a list, and ``a.<key>`` for a value of a free-key object. ``KeyError`` if absent."""
        kind: Kind = self.body
        found: Field | None = None
        for segment in path.split("."):
            is_list = segment.endswith("[]")
            name = segment.removesuffix("[]")
            if isinstance(kind, Object):
                found = kind.field(name)
            elif isinstance(kind, MapOf):
                found = Field(name, kind.value, "value of a free-key object", kind.nullable_values)
            else:
                raise KeyError(path)
            kind = found.kind
            if is_list:
                if not isinstance(kind, ListOf):
                    raise KeyError(path)
                kind = kind.item
        if found is None:
            raise KeyError(path)
        return found

    def reference_rows(self) -> list[tuple[str, str, bool, str]]:
        """(path, type, nullable, meaning) for every field, nested fields flattened."""
        rows: list[tuple[str, str, bool, str]] = []

        def visit(obj: Object, prefix: str) -> None:
            for f in obj.fields:
                path = f"{prefix}{f.name}"
                rows.append((path, f.kind.describe(), f.nullable, f.meaning))
                if isinstance(f.kind, Object):
                    visit(f.kind, f"{path}.")
                elif isinstance(f.kind, ListOf) and isinstance(f.kind.item, Object):
                    visit(f.kind.item, f"{path}[].")

        visit(self.body, "")
        return rows


def _reject_constant(name: str) -> NoReturn:
    raise RecordContractError(f"Non-finite number {name} is not allowed in a record")


def _finite_float(text: str) -> float:
    """A JSON number with a fraction or exponent; one that overflows to infinity is rejected."""
    value = float(text)
    if not math.isfinite(value):
        raise RecordContractError(f"Number {text} is outside the float64 range")
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecordContractError(f"Duplicate key {key!r} in a record")
        result[key] = value
    return result


def dumps(record: Mapping[str, Any]) -> str:
    """``record`` as indented UTF-8 JSON text; a non-finite number raises ``ValueError``."""
    return json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def loads(text: str, source: str) -> dict[str, Any]:
    """Parse one record (a JSON object) strictly; ``source`` names it in error messages."""
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except json.JSONDecodeError as error:
        raise RecordContractError(f"{source}: not valid JSON ({error})") from error
    except RecordContractError as error:
        raise RecordContractError(f"{source}: {error}") from error
    if not isinstance(value, dict):
        raise RecordContractError(f"{source}: a record must be a JSON object")
    return value
