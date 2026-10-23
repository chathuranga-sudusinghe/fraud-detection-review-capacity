"""The input matrix of the logistic-regression reference (DR-011 §5; Issue #151), on synthetic
frames only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from review_capacity.features.logistic_inputs import (
    build_logistic_matrix,
    logistic_columns,
    matrix_identity,
)
from review_capacity.features.neural_preprocessing import fit_neural_preprocessing
from support.preprocessing import MISSING, fitted_artifact
from support.synthetic import FEATURES

NUMERIC = len(FEATURES.numeric_inputs)  # 5: three declared columns and the two time-of-day inputs
INDICATORS = 2  # TransactionAmt and id_01 have missing Train values
SIZES = (5, 4, 3, 3)  # vocabulary sizes of ProductCD, card1, DeviceType and id_13


def test_the_columns_are_numeric_then_indicators_then_every_category() -> None:
    artifact, _ = fitted_artifact()
    columns = logistic_columns(artifact)
    assert columns[:NUMERIC] == FEATURES.numeric_inputs
    assert columns[NUMERIC : NUMERIC + INDICATORS] == ("TransactionAmt__missing", "id_01__missing")
    assert columns[NUMERIC + INDICATORS] == f"ProductCD={MISSING}"
    assert f"card1={MISSING}" in columns
    assert len(columns) == NUMERIC + INDICATORS + sum(SIZES)
    assert len(set(columns)) == len(columns)


def test_the_matrix_is_float64_csr_with_the_artifacts_float64_blocks() -> None:
    artifact, (frame, _, _) = fitted_artifact()
    matrix = build_logistic_matrix(artifact, frame)
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.dtype == np.float64
    assert matrix.shape == (len(frame), len(logistic_columns(artifact)))
    assert matrix.has_sorted_indices
    assert (matrix.data != 0).all()  # no explicit zeros are stored
    dense = matrix.toarray()
    np.testing.assert_array_equal(dense[:, :NUMERIC], artifact.transform_numeric(frame))
    np.testing.assert_array_equal(
        dense[:, NUMERIC : NUMERIC + INDICATORS], artifact.missingness_indicators(frame)
    )


def test_each_categorical_block_has_exactly_one_one_per_row_at_its_vocabulary_index() -> None:
    artifact, (frame, _, _) = fitted_artifact()
    dense = build_logistic_matrix(artifact, frame).toarray()
    indices = artifact.category_indices(frame)
    start = NUMERIC + INDICATORS
    for j, size in enumerate(SIZES):
        block = dense[:, start : start + size]
        np.testing.assert_array_equal(block.sum(axis=1), 1.0)
        np.testing.assert_array_equal(block.argmax(axis=1), indices[:, j])
        start += size


def test_missing_and_unseen_values_fill_the_missing_column() -> None:
    artifact, (frame, _, _) = fitted_artifact()
    scored = frame.head(3).copy()
    scored["ProductCD"] = pd.Series(["ZZ", MISSING, "W"], dtype="str")
    dense = build_logistic_matrix(artifact, scored).toarray()
    start = NUMERIC + INDICATORS  # the ProductCD block; "__MISSING__" is its first column
    assert dense[0, start] == 1.0
    assert dense[1, start] == 1.0
    assert dense[2, start] == 0.0
    np.testing.assert_array_equal(dense[:, start : start + SIZES[0]].sum(axis=1), 1.0)


def test_the_matrix_does_not_depend_on_the_other_rows_or_their_order() -> None:
    artifact, (frame, _, _) = fitted_artifact()
    whole = build_logistic_matrix(artifact, frame).toarray()
    part = build_logistic_matrix(artifact, frame.iloc[::-1].head(40)).toarray()
    np.testing.assert_array_equal(part, whole[::-1][:40])


def test_the_identity_names_the_artifact_blocks_and_columns() -> None:
    artifact, (frame, ids, times) = fitted_artifact()
    identity = matrix_identity(artifact)
    assert identity["artifact_sha256"] == artifact.artifact_sha256
    assert identity["columns"] == NUMERIC + INDICATORS + sum(SIZES)
    assert identity["numeric"] == NUMERIC
    assert identity["indicators"] == INDICATORS
    assert list(identity["categories"].values()) == list(SIZES)
    assert identity == matrix_identity(artifact)
    other = fit_neural_preprocessing(frame, ids, times, FEATURES, 7)  # another artifact
    assert matrix_identity(other)["artifact_sha256"] != identity["artifact_sha256"]
    assert matrix_identity(other)["columns_sha256"] == identity["columns_sha256"]
