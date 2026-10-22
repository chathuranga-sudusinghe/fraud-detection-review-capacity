"""The shared V1 objective bound to the tree libraries (DR-015 §2, §3 and §5; Issue #123).

``evaluation.queue`` computes the objective; this module lets LightGBM and CatBoost train and
early-stop on it:

* ``V1Objective``: the V1 partition's daily queue, labels and budget K, fixed once from the
  partition's rows. It is unweighted: the V1 sample weights play no part.
* ``ObjectiveRecorder``: one per fit. It scores the library's per-round predictions with the
  shared queue and keeps the integer numerator of every checkpoint, so the history, the strict
  improvement rule and the earliest-best checkpoint are decided on exact integers; the objective
  value is ``numerator / frauds``, and two values are equal exactly when their numerators are.
* ``lightgbm_evaluation`` and ``CatBoostObjective``: the custom evaluation function of LightGBM
  (``metric="None"``) and the user-defined evaluation metric of CatBoost. Both are given the
  fraud probability of every V1 row after each round; CatBoost reports the raw approximation,
  which is mapped through the logistic function in float64, as its prediction is.
* ``TrainingHistory``: the checkpoint history of a finished fit, with its stopping reason.

The scorer never sees ``TransactionID`` as a feature: it uses the identifiers only for the
queue's tie order (DR-015 §5).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from review_capacity.config import SECONDS_PER_DAY
from review_capacity.errors import DataValidationError, ModelError
from review_capacity.evaluation.queue import DailyQueue, DailyRecall, FloatScores
from review_capacity.models.base import Keys, Labels, ModelData

logger = logging.getLogger(__name__)

OBJECTIVE_NAME = "v1_objective"
VERSION = 1
IDENTITY_FIELDS = (
    "name",
    "version",
    "metric",
    "daily_k",
    "weighting",
    "tie_rule",
    "rows",
    "frauds",
)
STOPPING_REASONS = ("early_stopping", "maximum")
# DR-015 §2: a tree-based fit stops after ``patience`` consecutive rounds without improvement (its
# last round is best + patience); a neural fit stops at the (patience + 1)-th consecutive epoch
# without improvement (its last epoch is best + patience + 1).
STOP_OFFSET = {"round": 0, "epoch": 1}


class V1Objective:
    """Recall@K under daily top-K review on the V1 partition, unweighted (DR-015 §3).

    Built once from the V1 rows (their ``TransactionID``, ``TransactionDT`` and labels) and the
    daily budget ``k`` of the training declaration; the relative day is ``TransactionDT // 86400``
    (DR-001).
    """

    def __init__(self, valid: ModelData, k: int) -> None:
        if valid.ids is None or valid.times is None:
            raise DataValidationError(
                "The V1 objective needs the V1 rows' TransactionID and TransactionDT"
            )
        self._build(valid.ids, valid.times, valid.labels, k)

    @classmethod
    def from_rows(cls, ids: Keys, times: Keys, labels: Labels, k: int) -> V1Objective:
        """The objective of V1 rows given as arrays (the neural models' V1 inputs carry their
        identifiers and times as metadata, never as network features)."""
        objective = cls.__new__(cls)
        objective._build(ids, times, labels, k)
        return objective

    def _build(self, ids: Keys, times: Keys, labels: Labels, k: int) -> None:
        self.k = k
        self.labels = labels.copy()
        self.ids = ids.copy()
        self.days = times // SECONDS_PER_DAY
        self.queue = DailyQueue.build(self.ids, self.days)
        self.rows = len(self.labels)
        self.frauds = int(self.labels.sum())
        if self.frauds == 0:
            raise DataValidationError("The V1 objective is undefined: V1 has no fraud")
        self.queue.select(np.zeros(self.rows), k)  # rejects a budget that is not a positive int

    @property
    def identity(self) -> dict[str, Any]:
        """The recorded identity of the objective: what it computes, on how many rows."""
        return {
            "name": OBJECTIVE_NAME,
            "version": VERSION,
            "metric": "recall_at_daily_k",
            "daily_k": self.k,
            "weighting": "unweighted",
            "tie_rule": "sha256_transaction_id",
            "rows": self.rows,
            "frauds": self.frauds,
        }

    def check_partition(self, valid: ModelData) -> None:
        """The partition a fit early-stops on must be the one the objective was built from:
        the same rows, labels and relative days (the queues group by day)."""
        self.check_rows(valid.ids, valid.times, valid.labels)

    def check_rows(self, ids: Keys | None, times: Keys | None, labels: Labels) -> None:
        """``check_partition`` for V1 rows given as arrays: the same identifiers in the same
        order, labels and relative days."""
        if (
            ids is None
            or times is None
            or labels.shape != self.labels.shape
            or not np.array_equal(labels, self.labels)
            or not np.array_equal(ids, self.ids)
            or not np.array_equal(times // SECONDS_PER_DAY, self.days)
        ):
            raise DataValidationError("The V1 objective was built from different V1 rows")

    def recorder(self) -> ObjectiveRecorder:
        return ObjectiveRecorder(self)


def check_identity(identity: object) -> dict[str, Any]:
    """A recorded identity, validated: exactly the fields of ``V1Objective.identity``."""
    if not isinstance(identity, dict) or set(identity) != set(IDENTITY_FIELDS):
        raise ModelError("The recorded V1 objective identity is malformed")
    if (identity["name"], identity["version"]) != (OBJECTIVE_NAME, VERSION):
        raise ModelError("The recorded V1 objective is not this version of the shared objective")
    if (identity["metric"], identity["weighting"], identity["tie_rule"]) != (
        "recall_at_daily_k",
        "unweighted",
        "sha256_transaction_id",
    ):
        raise ModelError("The recorded V1 objective differs from DR-015 §3 and §5")
    for field in ("daily_k", "rows", "frauds"):
        value = identity[field]
        if type(value) is not int or value < 1:
            raise ModelError(f"The recorded V1 objective's {field} must be a positive integer")
    if identity["frauds"] > identity["rows"]:
        raise ModelError("The recorded V1 objective has more frauds than rows")
    return identity


@dataclass(frozen=True)
class Checkpoint:
    """One boosting round: the V1 objective after it (the fields of a trial's checkpoint row)."""

    checkpoint: int
    numerator: int
    objective: float
    improved: bool
    best_checkpoint: int
    seconds: float


@dataclass(frozen=True)
class TrainingHistory:
    """The finished fit's V1 history: every round, the earliest best, and why training stopped."""

    objective: dict[str, Any]
    checkpoints: tuple[Checkpoint, ...]
    best_checkpoint: int
    best_objective: float
    stopped_at: int
    stopping_reason: str
    patience: int
    maximum: int
    other_evaluations: int  # evaluations of other sets (not the V1 partition), not scored
    unit: str = "round"  # "round" (trees) or "epoch" (neural models)


class ObjectiveRecorder:
    """The V1 objective of one fit, evaluated after every round, with its history.

    ``observe`` scores one library evaluation. It returns ``None`` when the library evaluated
    something other than the V1 partition (a different row count or different labels, for
    example a learn-set or partial evaluation), which is counted in ``other_evaluations`` and
    never enters the history.
    """

    def __init__(self, objective: V1Objective) -> None:
        self.objective = objective
        self.other_evaluations = 0
        self._numerators: list[int] = []
        self._seconds: list[float] = []
        self._started = time.perf_counter()

    @property
    def evaluations(self) -> int:
        return len(self._numerators)

    def observe(self, probabilities: FloatScores, target: npt.ArrayLike) -> DailyRecall | None:
        objective = self.objective
        labels = np.asarray(target)
        if (
            probabilities.shape != (objective.rows,)
            or labels.shape != (objective.rows,)
            or not np.array_equal(labels, objective.labels)
        ):
            self.other_evaluations += 1
            return None
        recall = objective.queue.recall(probabilities, objective.labels, objective.k)
        self._numerators.append(recall.numerator)
        self._seconds.append(time.perf_counter() - self._started)
        return recall

    def history(self, *, patience: int, maximum: int, unit: str = "round") -> TrainingHistory:
        """The checkpoint history with its stopping reason.

        Improvement is strict and compared on the integer numerators; the earliest best is kept
        (DR-015 §5). Training must have stopped for a declared reason: ``patience`` rounds after
        the earliest best (``patience`` + 1 epochs for a neural fit, ``unit="epoch"``), or at
        ``maximum``. Any other length is a ``ModelError``.
        """
        if unit not in STOP_OFFSET:
            raise ModelError(f"The training unit must be one of {sorted(STOP_OFFSET)}")
        after_best = patience + STOP_OFFSET[unit]
        trained = len(self._numerators)
        if trained == 0:
            raise ModelError("No V1 objective evaluation was recorded")
        frauds = self.objective.frauds
        rows: list[Checkpoint] = []
        best_numerator, best = -1, 0
        for number, (numerator, seconds) in enumerate(
            zip(self._numerators, self._seconds, strict=True), start=1
        ):
            improved = numerator > best_numerator
            if improved:
                best_numerator, best = numerator, number
            rows.append(Checkpoint(number, numerator, numerator / frauds, improved, best, seconds))
        if trained == best + after_best:
            reason = "early_stopping"
        elif trained == maximum:
            reason = "maximum"
        else:
            raise ModelError(
                f"Training stopped after {trained} {unit}s, which is neither {after_best} {unit}s "
                f"after the earliest best ({best}) nor the maximum of {maximum}"
            )
        return TrainingHistory(
            self.objective.identity,
            tuple(rows),
            best,
            best_numerator / frauds,
            trained,
            reason,
            patience,
            maximum,
            self.other_evaluations,
            unit,
        )


def lightgbm_evaluation(
    recorder: ObjectiveRecorder,
) -> Callable[[Any, Any], tuple[str, float, bool]]:
    """LightGBM's custom evaluation function (``feval``) for ``metric="None"``.

    With a built-in objective LightGBM passes the fraud probabilities, the scores its prediction
    returns. Higher is better. Evaluating anything but the V1 partition is a defect.
    """

    def evaluate(predictions: Any, data: Any) -> tuple[str, float, bool]:
        recall = recorder.observe(np.asarray(predictions), data.get_label())
        if recall is None:
            raise ModelError("LightGBM evaluated rows other than the V1 partition")
        return OBJECTIVE_NAME, recall.value, True

    return evaluate


def logistic(raw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The fraud probability of CatBoost's raw approximation: the logistic function in float64,
    which is how the probabilities that ``predict_proba`` returns are formed. Very large
    magnitudes saturate to exactly 0 or 1 without warning (they then tie in the queue, as the
    scored probabilities do)."""
    with np.errstate(over="ignore", under="ignore"):
        probabilities: npt.NDArray[np.float64] = 1.0 / (1.0 + np.exp(-raw))
    return probabilities


class CatBoostObjective:
    """CatBoost's user-defined evaluation metric (``eval_metric``) for the V1 objective.

    CatBoost passes the raw approximation of the positive class; the fraud probability is its
    logistic transform in float64, as CatBoost's own probability prediction is. The metric
    returns the integer numerator as the error sum and the number of frauds as the weight sum,
    so CatBoost's final error is ``numerator / frauds``. An evaluation of anything but the whole
    V1 partition (for example the learn set) returns a neutral value and is counted by the
    recorder. The instance stands for itself when CatBoost copies its parameters, so the
    recorder it feeds is the fit's own.
    """

    def __init__(self, recorder: ObjectiveRecorder) -> None:
        self.recorder = recorder

    def __copy__(self) -> CatBoostObjective:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> CatBoostObjective:
        return self

    def is_max_optimal(self) -> bool:
        return True

    def evaluate(
        self,
        approxes: Any,
        target: Any,
        weight: Any,  # noqa: ARG002 (CatBoost's interface; the objective is unweighted)
    ) -> tuple[float, float]:
        raw = np.asarray(approxes[0], dtype=np.float64)
        recall = self.recorder.observe(logistic(raw), target)
        if recall is None:
            return 0.0, 1.0
        return float(recall.numerator), float(recall.frauds)

    def get_final_error(self, error: float, weight: float) -> float:
        return error / weight if weight else 0.0
