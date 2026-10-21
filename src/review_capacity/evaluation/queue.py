"""The shared daily review queue and the V1 objective (DR-015 §3 and §5; Issue #119).

One implementation orders and cuts the daily queue for every use of daily top-K review: the V1
objective that drives early stopping and trial selection, R3 at any budget K, and R4, which
reviews from R3's queue.

**Queue order.** Within each relative day, transactions are ordered by

1. model score, descending, compared exactly (no rounding, no tolerance);
2. for exactly equal scores, the SHA-256 hexadecimal digest of the decimal ``TransactionID``
   written as a string in UTF-8, ascending;
3. ``TransactionID`` ascending, only for an identical digest.

The order is fixed, reproducible and label-free. It is used only to cut review queues; it never
splits score ties in whole-ranking metrics (average precision, ROC-AUC), and neither
``TransactionID`` nor its digest is a model input.

**Cutoff ties.** A day's queue is cut after its K highest-ranked transactions. A tie crosses the
cutoff when the last transaction inside the queue and the first outside it have exactly equal
scores; the days with such a tie and the size of each tied group (all transactions of the day
with that score) are reported.

**V1 objective.** Recall@K under daily top-K review, unweighted: the frauds inside the daily
queues, summed over the days, divided by all frauds of the partition. Its integer numerator is
exposed beside the ratio, so that comparisons (strict improvement, ties between trials) can be
exact. The daily budget K (62 for V1, DR-015 §3) is an argument, set by the training declaration.

Scores must be finite real numbers. A float32 score is widened exactly to float64 before
comparison, and ``-0.0`` is compared as ``0.0``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from review_capacity.errors import DataValidationError
from review_capacity.logging_setup import log_step
from review_capacity.models.base import Keys, Labels, Scores

logger = logging.getLogger(__name__)

type Positions = npt.NDArray[np.intp]
type FloatScores = npt.NDArray[np.float32] | npt.NDArray[np.float64]


def id_digest(transaction_id: int | np.integer[Any]) -> bytes:
    """SHA-256 digest of the decimal ``TransactionID`` string in UTF-8 (DR-015 §5)."""
    return hashlib.sha256(str(int(transaction_id)).encode("utf-8")).digest()


def tie_ranks(ids: Keys) -> Positions:
    """Each row's rank under the tie order: digest ascending, then ``TransactionID`` ascending.

    The ranks are a permutation of ``0 .. len(ids) - 1``; a smaller rank is reviewed first among
    exactly equal scores.
    """
    order = sorted(range(len(ids)), key=lambda row: (id_digest(int(ids[row])), int(ids[row])))
    ranks = np.empty(len(ids), dtype=np.intp)
    ranks[np.asarray(order, dtype=np.intp)] = np.arange(len(ids), dtype=np.intp)
    return ranks


def _checked_scores(scores: object, rows: int) -> Scores:
    if not isinstance(scores, np.ndarray) or scores.ndim != 1 or scores.shape != (rows,):
        raise DataValidationError(f"Scores must be a one-dimensional array of {rows} rows")
    if scores.dtype.kind != "f":
        raise DataValidationError(f"Scores must be floating point, not {scores.dtype}")
    exact = scores.astype(np.float64)  # float32 widens exactly
    if not np.isfinite(exact).all():
        raise DataValidationError("Scores must be finite; the review order is undefined for NaN")
    return exact + 0.0  # -0.0 + 0.0 is 0.0: equal scores stay equal


def _check_labels(labels: object, rows: int) -> None:
    if (
        not isinstance(labels, np.ndarray)
        or labels.shape != (rows,)
        or labels.dtype != np.int8
        or not np.isin(labels, (0, 1)).all()
    ):
        raise DataValidationError(f"Labels must be an int8 array of 0 or 1 for {rows} rows")


@dataclass(frozen=True)
class DaySummary:
    """One relative day of a selection."""

    day: int
    transactions: int
    selected: int
    cutoff_tied_group: int | None  # transactions of the day sharing the score at the cutoff


@dataclass(frozen=True)
class Selection:
    """The transactions inside the daily queues for one budget K.

    ``mask`` marks the selected rows and ``day_rank`` gives every row's 0-based position in its
    day's queue (a row is selected when ``day_rank < k``), both in the queue's input row order.
    R4 reviews from R3's queue: its rows are ``day_rank < k`` where the calibrated probability
    reaches the cost threshold.
    """

    k: int
    mask: npt.NDArray[np.bool_]
    day_rank: Positions
    days: tuple[DaySummary, ...]

    @property
    def cutoff_tie_days(self) -> int:
        return sum(day.cutoff_tied_group is not None for day in self.days)

    @property
    def cutoff_tie_group_sizes(self) -> tuple[int, ...]:
        return tuple(d.cutoff_tied_group for d in self.days if d.cutoff_tied_group is not None)


@dataclass(frozen=True)
class DailyRecall:
    """Recall@K under daily top-K review: ``numerator`` frauds inside the daily queues of
    ``frauds`` frauds in all."""

    numerator: int
    frauds: int

    @property
    def value(self) -> float:
        return self.numerator / self.frauds


class DailyQueue:
    """The daily review queue of one partition: its rows' days and tie order, fixed once.

    Build it once per partition (``DailyQueue.build``) and apply it to the scores of any model,
    checkpoint or policy: only the scores change between uses, so each use is a single sort.
    """

    def __init__(self, days: Keys, tie_rank: Positions) -> None:
        self._days = days
        self._tie_rank = tie_rank

    @classmethod
    def build(cls, ids: Keys, days: Keys) -> DailyQueue:
        """The queue of the rows with ``TransactionID`` ``ids`` on relative days ``days``."""
        if ids.ndim != 1 or ids.shape != days.shape or ids.size == 0:
            raise DataValidationError("Identifiers and days must be non-empty arrays of one length")
        if ids.dtype != np.int64 or days.dtype != np.int64:
            raise DataValidationError("Identifiers and days must be int64 arrays")
        if len(np.unique(ids)) != len(ids):
            raise DataValidationError("TransactionIDs must be unique within a partition")
        if (days < 1).any():
            raise DataValidationError("Relative days start at 1")
        with log_step(logger, f"Building the daily review queue of {len(ids):,} transactions"):
            return cls(days.copy(), tie_ranks(ids))

    @property
    def rows(self) -> int:
        return len(self._days)

    def order(self, scores: FloatScores) -> Positions:
        """Row positions in queue order: by relative day, then the daily order of the module."""
        return self._order(_checked_scores(scores, self.rows))

    def _order(self, exact: Scores) -> Positions:
        return np.lexsort((self._tie_rank, -exact, self._days))

    def select(self, scores: FloatScores, k: int) -> Selection:
        """The first ``k`` transactions of every day's queue (all of a day with fewer)."""
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise DataValidationError(f"The daily budget K must be a positive integer, not {k!r}")
        exact = _checked_scores(scores, self.rows)
        order = self._order(exact)
        sorted_days, sorted_scores = self._days[order], exact[order]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_days)) + 1))
        ends = np.concatenate((starts[1:], [self.rows]))
        day_rank = np.empty(self.rows, dtype=np.intp)
        day_rank[order] = np.arange(self.rows, dtype=np.intp) - np.repeat(starts, ends - starts)
        summaries = []
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            size = end - start
            tied: int | None = None
            if size > k and sorted_scores[start + k - 1] == sorted_scores[start + k]:
                tied = int((sorted_scores[start:end] == sorted_scores[start + k - 1]).sum())
            summaries.append(DaySummary(int(sorted_days[start]), size, min(k, size), tied))
        return Selection(k, day_rank < k, day_rank, tuple(summaries))

    def recall(self, scores: FloatScores, labels: Labels, k: int) -> DailyRecall:
        """The unweighted Recall@K under daily top-K review (the V1 objective at ``k`` = 62)."""
        _check_labels(labels, self.rows)
        frauds = int(labels.sum())
        if frauds == 0:
            raise DataValidationError("Recall is undefined: the partition has no fraud")
        selection = self.select(scores, k)
        return DailyRecall(int(labels[selection.mask].sum()), frauds)
