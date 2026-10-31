"""The Test features are read without the target and against the frozen anchor (Issue #203)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from review_capacity.confirmatory.inputs import read_test_inputs
from review_capacity.data import loading
from review_capacity.errors import DataValidationError
from review_capacity.hashing import id_set_sha256
from support.synthetic import Synthetic, build_synthetic


@pytest.fixture
def synthetic(tmp_path: Path) -> Synthetic:
    return build_synthetic(tmp_path)


def anchor(synthetic: Synthetic) -> str:
    test = synthetic.dataset.partition("test")
    days = synthetic.transaction["TransactionDT"] // synthetic.dataset.seconds_per_day
    rows = synthetic.transaction[(days >= test.first_day) & (days <= test.last_day)]
    return id_set_sha256(rows["TransactionID"])


def test_the_test_rows_become_the_canonical_inputs_without_the_target(
    synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the labels were read")

    monkeypatch.setattr(loading, "load_labels", refuse)
    inputs = read_test_inputs(synthetic.features, synthetic.dataset, anchor(synthetic))
    assert inputs.rows == synthetic.dataset.partition("test").rows
    assert list(inputs.features.columns) == list(synthetic.features.model_inputs)
    assert "isFraud" not in inputs.features.columns
    assert inputs.id_set_sha256 == anchor(synthetic)
    assert set(inputs.days.tolist()) == {5}
    assert np.array_equal(np.lexsort((inputs.ids, inputs.times)), np.arange(inputs.rows))
    assert len(inputs.ordered_ids_sha256) == 64


def test_the_labels_of_the_test_rows_are_never_parsed(synthetic: Synthetic) -> None:
    """The synthetic Test labels are the invalid value ``X``: reading them would fail."""
    assert set(synthetic.transaction["isFraud"].tail(4)) == {"X"}
    read_test_inputs(synthetic.features, synthetic.dataset, anchor(synthetic))


def test_a_test_identifier_set_other_than_the_frozen_anchor_is_refused(
    synthetic: Synthetic,
) -> None:
    with pytest.raises(DataValidationError, match="frozen anchor"):
        read_test_inputs(synthetic.features, synthetic.dataset, "0" * 64)


def test_a_raw_file_that_differs_from_its_checksum_is_refused(synthetic: Synthetic) -> None:
    path = synthetic.dataset.transaction.path
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="SHA-256"):
        read_test_inputs(synthetic.features, synthetic.dataset, anchor(synthetic))
