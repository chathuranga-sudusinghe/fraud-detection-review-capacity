"""The time-of-day inputs ``tod_sin`` and ``tod_cos`` (DR-008 §4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from review_capacity.errors import DataValidationError
from review_capacity.features.time_of_day import time_of_day

DAY = 86_400


def test_known_points_of_the_day() -> None:
    t = pd.Series([DAY, DAY + DAY // 4, DAY + DAY // 2, DAY + 3 * DAY // 4])
    tod = time_of_day(t)
    expected = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])
    np.testing.assert_allclose(tod[["tod_sin", "tod_cos"]].to_numpy(), expected, atol=1e-12)


def test_wraps_at_end_of_relative_day() -> None:
    tod = time_of_day(pd.Series([2 * DAY - 1, 2 * DAY])).to_numpy()
    assert np.linalg.norm(tod[0] - tod[1]) < 1e-4  # last second is next to the next day's start
    assert np.all(np.isfinite(tod))
    np.testing.assert_allclose((tod**2).sum(axis=1), 1.0)


def test_shift_of_offsets_rotates_the_pair() -> None:
    t = pd.Series(np.arange(DAY, 3 * DAY, 977))
    c = 12_345  # seconds added to every recorded offset: t' = t + c
    a = time_of_day(t).to_numpy()
    b = time_of_day(t + c).to_numpy()
    angle = np.arctan2(b[:, 0], b[:, 1]) - np.arctan2(a[:, 0], a[:, 1])
    np.testing.assert_allclose(np.mod(angle, 2 * np.pi), 2 * np.pi * c / DAY, atol=1e-9)


@pytest.mark.parametrize("bad", [DAY + 0.5, np.nan, np.inf])
def test_non_whole_or_non_finite_times_stop(bad: float) -> None:
    with pytest.raises(DataValidationError, match="finite whole numbers"):
        time_of_day(pd.Series([DAY, bad]))
