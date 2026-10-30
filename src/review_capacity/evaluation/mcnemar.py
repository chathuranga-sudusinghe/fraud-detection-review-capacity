"""The exact McNemar test of the cell contrasts (DR-005; Issue #191).

For two configurations scored on the same Test frauds, b frauds are captured only by the first and
c only by the second. Under the null the discordant frauds are equally likely to fall on either
side, so with n = b + c the two-sided exact p-value is ``min(1, 2 P(X <= min(b, c)))`` for
X ~ Binomial(n, 0.5); n = 0 gives p = 1. The test assumes independent frauds, which the clustering
evidence contradicts, so it is descriptive, labelled independence-based and outside the Holm
family (REF-035, REF-028, REF-071).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import binomtest

from review_capacity.errors import DataValidationError
from review_capacity.evaluation.declaration import EvaluationDeclaration


@dataclass(frozen=True)
class McNemar:
    """The discordant counts of a contrast a - b and the exact p-value."""

    first: str
    second: str
    only_first: int  # captured by the first configuration only (b)
    only_second: int  # captured by the second configuration only (c)
    p: float


def exact_p(only_first: int, only_second: int) -> float:
    """The two-sided exact McNemar p-value of the discordant counts."""
    if only_first < 0 or only_second < 0:
        raise DataValidationError("Discordant counts must not be negative")
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    return float(binomtest(min(only_first, only_second), discordant, 0.5).pvalue)


def cell_mcnemar(
    declaration: EvaluationDeclaration, captures: npt.ArrayLike
) -> tuple[McNemar, ...]:
    """The exact McNemar test of each declared cell contrast. ``captures`` is the boolean matrix
    of the daily top-K captures: one row per configuration in the declared order, one column per
    Test fraud."""
    matrix = np.asarray(captures)
    order = declaration.configurations.order
    if matrix.dtype != np.bool_ or matrix.ndim != 2 or matrix.shape[0] != len(order):
        raise DataValidationError(
            f"captures must be a boolean matrix with {len(order)} rows (one per configuration)"
        )
    results = []
    for first, second in declaration.contrasts:
        a, b = matrix[order.index(first)], matrix[order.index(second)]
        only_first, only_second = int((a & ~b).sum()), int((~a & b).sum())
        results.append(
            McNemar(first, second, only_first, only_second, exact_p(only_first, only_second))
        )
    return tuple(results)
