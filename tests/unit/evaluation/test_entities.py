"""The fine pseudo-entity key of the feasibility audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.entities import pseudo_entity_keys, start_proxy

MISSING = "__MISSING__"


def frame(**columns: list[object]) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_the_key_joins_the_parts_with_the_implied_reference_day() -> None:
    features = frame(
        card1=["1001", "1001"],
        addr1=["315", MISSING],
        P_emaildomain=["gmail.com", "gmail.com"],
        D1=[2.0, 0.4],
    )
    keys = pseudo_entity_keys(features, np.array([10, 10], dtype=np.int64), MISSING)
    assert keys == ["1001|315|gmail.com|8.0", "1001|NA|gmail.com|10.0"]


def test_a_missing_offset_gives_a_missing_reference_day() -> None:
    assert start_proxy(np.array([5, 5], dtype=np.int64), np.array([np.nan, 1.0])).tolist() == [
        "NA",
        "4.0",
    ]


def test_rounding_is_half_to_even_like_the_audit() -> None:
    days = np.array([10, 10, 10, 10], dtype=np.int64)
    assert start_proxy(days, np.array([0.5, 1.5, 2.5, 3.5])).tolist() == [
        "10.0",
        "8.0",
        "8.0",
        "6.0",
    ]


def test_equal_inputs_give_equal_keys_and_any_difference_a_new_key() -> None:
    features = frame(
        card1=["1", "1", "2"],
        addr1=["3", "3", "3"],
        P_emaildomain=["a", "a", "a"],
        D1=[1.0, 1.0, 1.0],
    )
    keys = pseudo_entity_keys(features, np.array([4, 4, 4], dtype=np.int64), MISSING)
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    later = pseudo_entity_keys(features, np.array([4, 5, 4], dtype=np.int64), MISSING)
    assert later[0] != later[1]


def test_missing_inputs_and_row_counts_are_checked() -> None:
    features = frame(card1=["1"], addr1=["2"], P_emaildomain=["a"])
    with pytest.raises(DataValidationError, match="D1"):
        pseudo_entity_keys(features, np.array([1], dtype=np.int64), MISSING)
    complete = features.assign(D1=[1.0])
    with pytest.raises(DataValidationError, match="relative days"):
        pseudo_entity_keys(complete, np.array([1, 2], dtype=np.int64), MISSING)


def test_a_negative_zero_proxy_is_a_different_part_as_in_the_audit() -> None:
    # day 127 with D1 = 127.25 and 126.75: the audit's rounded floats are "-0.0" and "0.0"
    days = np.array([127, 127], dtype=np.int64)
    assert start_proxy(days, np.array([127.25, 126.75])).tolist() == ["-0.0", "0.0"]
