"""The input matrix of the logistic-regression reference (DR-011 §5; DR-015 §8; Issue #151).

The reference reads the shared Train-fitted preprocessing artifact of DR-012
(``features.neural_preprocessing``) and a canonical frame, and builds one float64 compressed
sparse row (CSR) matrix with these columns, in this order:

1. the transformed numeric inputs (every numeric input, missing values 0);
2. the missingness indicators (one per numeric input with a missing Train value);
3. one one-hot block per categorical input over every vocabulary category, including
   ``"__MISSING__"``, with no category dropped; a value unseen in Train fills the
   ``"__MISSING__"`` column.

There is no further scaling and no category dropout. The numeric blocks are the artifact's float64
values, never widened from float32. Nothing is fitted here; the matrix is a function of the
artifact and the frame.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from review_capacity.errors import DataValidationError
from review_capacity.features.neural_preprocessing import NeuralPreprocessing, indicator_name

CATEGORY_SEPARATOR = "="


def logistic_columns(artifact: NeuralPreprocessing) -> tuple[str, ...]:
    """The matrix columns in order: numeric inputs, indicators, then ``<input>=<label>`` of every
    category of every vocabulary."""
    names = [
        *artifact.numeric_inputs,
        *map(indicator_name, artifact.indicators),
        *(
            f"{name}{CATEGORY_SEPARATOR}{label}"
            for name, vocabulary in zip(
                artifact.categorical_inputs, artifact.vocabularies, strict=True
            )
            for label in vocabulary
        ),
    ]
    if len(set(names)) != len(names):
        raise DataValidationError("The logistic-regression columns are not unique")
    return tuple(names)


def matrix_identity(artifact: NeuralPreprocessing) -> dict[str, Any]:
    """What identifies the matrix a model was fitted on: the artifact's SHA-256, the block sizes
    and the SHA-256 of the ordered column names (UTF-8, newline-separated)."""
    columns = logistic_columns(artifact)
    return {
        "artifact_sha256": artifact.artifact_sha256,
        "columns": len(columns),
        "numeric": len(artifact.numeric_inputs),
        "indicators": len(artifact.indicators),
        "categories": {
            name: len(vocabulary)
            for name, vocabulary in zip(
                artifact.categorical_inputs, artifact.vocabularies, strict=True
            )
        },
        "columns_sha256": hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest(),
    }


def build_logistic_matrix(artifact: NeuralPreprocessing, frame: pd.DataFrame) -> Any:
    """The float64 CSR matrix of ``frame`` (canonical model inputs) for ``artifact``."""
    numeric = artifact.transform_numeric(frame)
    indicators = artifact.missingness_indicators(frame)
    indices = artifact.category_indices(frame)
    sizes = np.array([len(vocabulary) for vocabulary in artifact.vocabularies], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    rows, width = indices.shape
    one_hot = sparse.csr_matrix(
        (
            np.ones(rows * width, dtype=np.float64),
            (indices + offsets).reshape(-1),
            np.arange(0, rows * width + 1, width, dtype=np.int64),
        ),
        shape=(rows, int(sizes.sum())),
    )
    dense = np.concatenate([numeric, indicators], axis=1)
    matrix = sparse.hstack([sparse.csr_matrix(dense), one_hot], format="csr", dtype=np.float64)
    matrix.sort_indices()
    if matrix.shape != (rows, len(logistic_columns(artifact))):
        raise DataValidationError("The logistic-regression matrix has an unexpected shape")
    if not np.isfinite(matrix.data).all():
        raise DataValidationError("The logistic-regression matrix holds non-finite values")
    return matrix
