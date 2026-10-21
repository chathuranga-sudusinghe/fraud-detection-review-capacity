"""The contract language: value types, objects, record contracts and strict JSON (no data read)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from review_capacity.development.schema import (
    AnyJSON,
    Boolean,
    Choice,
    Field,
    Integer,
    ListOf,
    MapOf,
    Number,
    Object,
    RecordContract,
    RelativePath,
    Text,
    dumps,
    loads,
)
from review_capacity.errors import RecordContractError


def test_integer_rejects_booleans_floats_and_values_out_of_range() -> None:
    kind = Integer(1, 10)
    assert kind.problems(5, "x") == []
    for value in (True, 5.0, "5", None):
        assert kind.problems(value, "x") == [f"x: expected an integer, got {value!r}"]
    assert kind.problems(0, "x") == ["x: 0 is outside ≥ 1, ≤ 10"]
    assert kind.problems(11, "x") == ["x: 11 is outside ≥ 1, ≤ 10"]


def test_number_accepts_integers_and_rejects_non_finite_values() -> None:
    kind = Number(0, 1, exclusive_minimum=True)
    assert kind.problems(1, "x") == []
    assert kind.problems(0.5, "x") == []
    assert kind.problems(0, "x") == ["x: 0 is outside > 0, ≤ 1"]
    assert kind.problems(False, "x") == ["x: expected a number, got False"]
    for value in (math.nan, math.inf, -math.inf):
        assert kind.problems(value, "x") == [f"x: {value!r} is not a finite number"]


def test_text_requires_a_full_pattern_match_and_no_empty_string() -> None:
    kind = Text(r"[0-9a-f]{4}", "digest")
    assert kind.problems("ab12", "x") == []
    assert kind.problems("ab12x", "x") == ["x: 'ab12x' is not a valid digest"]
    assert kind.problems("", "x") == ["x: expected non-empty text, got ''"]


@pytest.mark.parametrize("path", ["/abs", "a/../b", "..", "a//b", "./a", "a\\b", "a/"])
def test_relative_path_cannot_leave_its_base_directory(path: str) -> None:
    assert RelativePath().problems(path, "x") != []


def test_relative_path_accepts_nested_files() -> None:
    assert RelativePath().problems("trials/LGB-u/trial-01.json", "x") == []


def test_choice_boolean_and_any_json() -> None:
    assert Choice(("a", "b")).problems("c", "x") == ["x: 'c' is not one of a, b"]
    assert Boolean().problems(1, "x") == ["x: expected true or false, got 1"]
    assert AnyJSON().problems({"a": [1, 2.5, None, {"b": "c"}]}, "x") == []
    assert AnyJSON().problems({"a": [math.nan]}, "x") == ["x.a[0]: nan is not finite"]
    assert AnyJSON().problems(np.float64(1.0), "x") == []  # a float subclass
    assert AnyJSON().problems({1: 2}, "x") == ["x: object keys must be text"]
    assert AnyJSON().problems((1, 2), "x") == ["x: tuple is not a JSON value"]


def test_list_counts_items_types_and_uniqueness() -> None:
    kind = ListOf(Integer(), min_items=2, max_items=2, unique=True)
    assert kind.problems([1, 2], "x") == []
    assert kind.problems([1], "x") == ["x: 1 items; expected exactly 2"]
    assert kind.problems([1, 1], "x") == ["x: items must be unique"]
    assert kind.problems([1, "2"], "x") == ["x[1]: expected an integer, got '2'"]


def test_free_key_object_checks_keys_and_values() -> None:
    kind = MapOf(Integer(), keys=Choice(("a",)), nullable_values=True)
    assert kind.problems({"a": None}, "x") == []
    assert kind.problems({"b": 1}, "x") == ["x key: 'b' is not one of a"]
    assert MapOf(Integer()).problems({"a": None}, "x") == ["x.a: expected an integer, got None"]


def test_object_reports_missing_unknown_and_null_fields_together() -> None:
    obj = Object((Field("a", Integer(), "A"), Field("b", Integer(), "B", nullable=True)))
    assert obj.problems({"a": 1, "b": None}, "x") == []
    assert obj.problems({"b": 1, "c": 2}, "x") == [
        "x: missing field 'a'",
        "x: unknown field 'c'",
    ]
    assert obj.problems({"a": None, "b": None}, "x") == ["x.a: must not be null"]
    with pytest.raises(ValueError, match="Duplicate field names"):
        Object((Field("a", Integer(), "A"), Field("a", Integer(), "A")))


def _contract() -> RecordContract:
    inner = Object((Field("c", Integer(), "C"),))
    body = Object(
        (
            Field("a", Integer(), "A"),
            Field("b", Integer(), "B"),
            Field("items", ListOf(inner), "Items"),
            Field("free", MapOf(Integer()), "Free"),
        )
    )

    def ordered(record: object) -> list[str]:
        assert isinstance(record, dict)  # cross-field rules see a structurally valid record
        return [] if record["a"] < record["b"] else ["toy: a must be below b"]

    return RecordContract("toy", 1, "A toy record", body, (ordered,))


def test_cross_field_rules_run_only_on_structurally_valid_records() -> None:
    contract = _contract()
    record = {"a": 1, "b": 2, "items": [], "free": {}}
    contract.validate(record)
    assert contract.problems(record | {"a": 3}) == ["toy: a must be below b"]
    assert contract.problems(record | {"a": "3"}) == ["toy.a: expected an integer, got '3'"]


def test_validate_lists_every_problem() -> None:
    with pytest.raises(RecordContractError) as caught:
        _contract().validate({"a": "x", "items": [{"c": None}], "free": {}})
    message = str(caught.value)
    assert "toy record (version 1)" in message
    assert "toy: missing field 'b'" in message
    assert "toy.a: expected an integer" in message
    assert "toy.items[0].c: must not be null" in message


def test_resolve_follows_objects_lists_and_free_keys() -> None:
    contract = _contract()
    assert contract.resolve("a").name == "a"
    assert contract.resolve("items[].c").name == "c"
    assert isinstance(contract.resolve("free.anything").kind, Integer)
    for path in ("missing", "a.b", "items.c", "a[]"):
        with pytest.raises(KeyError):
            contract.resolve(path)


def test_reference_rows_flatten_nested_fields() -> None:
    paths = [row[0] for row in _contract().reference_rows()]
    assert paths == ["a", "b", "items", "items[].c", "free"]


def test_float64_values_round_trip_exactly() -> None:
    values = [0.1 + 0.2, 1e-300, float(np.float32(0.7)), 0.7234567890123456, 5e-324]
    assert loads(dumps({"v": values}), "t")["v"] == values


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"a": 1, "a": 2}', "Duplicate key 'a'"),
        ('{"a": NaN}', "Non-finite number NaN"),
        ('{"a": -Infinity}', "Non-finite number -Infinity"),
        ("[1]", "must be a JSON object"),
        ("{", "not valid JSON"),
    ],
)
def test_loads_is_strict(text: str, message: str) -> None:
    with pytest.raises(RecordContractError, match=message):
        loads(text, "t")


def test_numbers_beyond_float64_are_rejected_by_loads() -> None:
    with pytest.raises(RecordContractError, match="1e400 is outside the float64 range"):
        loads('{"x": 1e400}', "t")
    assert loads('{"x": 1e308}', "t") == {"x": 1e308}


def test_very_large_integers_are_reported_not_raised() -> None:
    assert Number().problems(10**400, "x") == []
    (problem,) = Number(maximum=1).problems(10**400, "x")
    assert problem.startswith("x: 1000")
    assert problem.endswith("is outside ≤ 1")
    assert Integer(maximum=10).problems(10**400, "x") != []


def test_dumps_refuses_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="not JSON compliant"):
        dumps({"a": math.nan})
    assert json.loads(dumps({"a": "é"})) == {"a": "é"}
