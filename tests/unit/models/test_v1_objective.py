"""The shared V1 objective bound to the tree libraries (synthetic rows; no research data)."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from review_capacity.errors import DataValidationError, ModelError
from review_capacity.evaluation.queue import tie_ranks
from review_capacity.models.base import ModelData
from review_capacity.models.v1_objective import (
    CatBoostObjective,
    ObjectiveRecorder,
    V1Objective,
    check_identity,
    lightgbm_evaluation,
    logistic,
)

DAY = 86_400


def _partition(
    labels: list[int], days: list[int], ids: list[int] | None = None, weight: float = 1.0
) -> ModelData:
    rows = len(labels)
    return ModelData(
        pd.DataFrame({"x": np.zeros(rows)}),
        np.array(labels, dtype=np.int8),
        np.full(rows, weight),
        times=np.array(days, dtype=np.int64) * DAY + 5,
        ids=np.array(ids if ids is not None else range(100, 100 + rows), dtype=np.int64),
    )


def _top(recorder: ObjectiveRecorder, rows: int, first: int) -> None:
    """One evaluation whose highest score is row ``first`` (the others tie below it)."""
    scores = np.zeros(rows)
    scores[first] = 0.9
    assert recorder.observe(scores, recorder.objective.labels) is not None


def test_daily_queues_not_the_global_ranking() -> None:
    # Day 1 holds the two highest scores, day 2 a fraud with a lower score: Recall@1 per day
    # counts the best row of each day, a global top-2 would not.
    data = _partition([0, 0, 1, 1], [1, 1, 2, 2])
    objective = V1Objective(data, 1)
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    recall = objective.queue.recall(scores, data.labels, 1)
    assert (recall.numerator, recall.frauds) == (1, 2)
    assert recall.value == 0.5


def test_v1_weights_play_no_part() -> None:
    plain = V1Objective(_partition([0, 1, 0, 1], [1, 1, 2, 2]), 1)
    heavy = V1Objective(_partition([0, 1, 0, 1], [1, 1, 2, 2], weight=100.0), 1)
    scores = np.array([0.2, 0.9, 0.5, 0.4])
    expected = plain.queue.recall(scores, plain.labels, 1)
    assert expected == heavy.queue.recall(scores, heavy.labels, 1)


def test_exactly_tied_scores_are_cut_by_the_digest_order() -> None:
    ids = [101, 102, 103, 104]
    data = _partition([1, 0, 0, 0], [1, 1, 1, 1], ids)
    objective = V1Objective(data, 1)
    recorder = objective.recorder()
    recall = recorder.observe(np.full(4, 0.5), data.labels)
    assert recall is not None
    first = int(np.argmin(tie_ranks(np.array(ids, dtype=np.int64))))
    assert recall.numerator == int(data.labels[first])


def test_identity_records_what_is_computed() -> None:
    objective = V1Objective(_partition([0, 1, 0, 1], [1, 1, 2, 2]), 62)
    assert objective.identity == {
        "name": "v1_objective",
        "version": 1,
        "metric": "recall_at_daily_k",
        "daily_k": 62,
        "weighting": "unweighted",
        "tie_rule": "sha256_transaction_id",
        "rows": 4,
        "frauds": 2,
    }
    assert check_identity(copy.deepcopy(objective.identity)) == objective.identity


@pytest.mark.parametrize(
    "change",
    [
        {"name": "other"},
        {"version": 2},
        {"weighting": "balanced"},
        {"tie_rule": "none"},
        {"daily_k": 0},
        {"daily_k": True},
        {"rows": 1.0},
        {"frauds": 9},
    ],
)
def test_a_malformed_identity_is_rejected(change: dict[str, Any]) -> None:
    identity = V1Objective(_partition([0, 1, 0, 1], [1, 1, 2, 2]), 1).identity
    with pytest.raises(ModelError):
        check_identity({**identity, **change})


def test_identity_needs_exactly_its_fields() -> None:
    identity = V1Objective(_partition([0, 1], [1, 1]), 1).identity
    with pytest.raises(ModelError, match="malformed"):
        check_identity({**identity, "extra": 1})
    with pytest.raises(ModelError, match="malformed"):
        check_identity(None)


def test_the_objective_needs_identifiers_fraud_and_a_budget() -> None:
    plain = ModelData(pd.DataFrame({"x": [0.0, 1.0]}), np.array([0, 1], np.int8), np.ones(2))
    with pytest.raises(DataValidationError, match="TransactionID and TransactionDT"):
        V1Objective(plain, 1)
    with pytest.raises(DataValidationError, match="no fraud"):
        V1Objective(_partition([0, 0], [1, 1]), 1)
    with pytest.raises(DataValidationError, match="positive integer"):
        V1Objective(_partition([0, 1], [1, 1]), 0)


def test_a_fit_must_use_the_partition_the_objective_was_built_from() -> None:
    data = _partition([0, 1, 0, 1], [1, 1, 2, 2])
    objective = V1Objective(data, 1)
    objective.check_partition(data)
    with pytest.raises(DataValidationError, match="different V1 rows"):
        objective.check_partition(_partition([1, 1, 0, 1], [1, 1, 2, 2]))
    with pytest.raises(DataValidationError, match="different V1 rows"):
        objective.check_partition(_partition([0, 1, 0, 1], [1, 1, 2, 2], ids=[1, 2, 3, 4]))


def test_history_applies_strict_improvement_and_keeps_the_earliest_best() -> None:
    # Two frauds (rows 1 and 2), K = 1: a round counts when its top row is a fraud.
    objective = V1Objective(_partition([0, 1, 1, 0], [1, 1, 1, 1]), 1)
    recorder = objective.recorder()
    for first in (0, 1, 2, 0, 1):  # numerators 0, 1, 1, 0, 1
        _top(recorder, 4, first)
    history = recorder.history(patience=3, maximum=60)
    assert [row.numerator for row in history.checkpoints] == [0, 1, 1, 0, 1]
    assert [row.improved for row in history.checkpoints] == [True, True, False, False, False]
    assert [row.best_checkpoint for row in history.checkpoints] == [1, 2, 2, 2, 2]
    assert [row.checkpoint for row in history.checkpoints] == [1, 2, 3, 4, 5]
    assert (history.best_checkpoint, history.best_objective) == (2, 0.5)
    assert (history.stopped_at, history.stopping_reason) == (5, "early_stopping")
    assert history.objective == objective.identity


def test_history_ends_at_the_maximum_when_patience_does_not_run_out() -> None:
    objective = V1Objective(_partition([0, 1, 1, 0], [1, 1, 1, 1]), 1)
    recorder = objective.recorder()
    for first in (0, 1, 2):
        _top(recorder, 4, first)
    history = recorder.history(patience=50, maximum=3)
    assert (history.stopped_at, history.stopping_reason) == (3, "maximum")


@pytest.mark.parametrize(("patience", "maximum"), [(5, 60), (50, 60)])
def test_a_stop_for_no_declared_reason_is_an_error(patience: int, maximum: int) -> None:
    objective = V1Objective(_partition([0, 1, 1, 0], [1, 1, 1, 1]), 1)
    recorder = objective.recorder()
    for first in (0, 1, 2):
        _top(recorder, 4, first)
    with pytest.raises(ModelError, match="neither"):
        recorder.history(patience=patience, maximum=maximum)


def test_an_empty_history_is_an_error() -> None:
    objective = V1Objective(_partition([0, 1], [1, 1]), 1)
    with pytest.raises(ModelError, match="No V1 objective evaluation"):
        objective.recorder().history(patience=1, maximum=2)


def test_other_row_sets_are_counted_and_never_enter_the_history() -> None:
    objective = V1Objective(_partition([0, 1, 1, 0], [1, 1, 1, 1]), 1)
    recorder = objective.recorder()
    assert recorder.observe(np.zeros(3), np.zeros(3)) is None  # another row count
    assert recorder.observe(np.zeros(4), np.array([1, 0, 0, 1])) is None  # other labels
    assert (recorder.evaluations, recorder.other_evaluations) == (0, 2)


def test_lightgbm_evaluation_reports_the_objective_as_higher_is_better() -> None:
    data = _partition([0, 1, 1, 0], [1, 1, 1, 1])
    objective = V1Objective(data, 1)
    recorder = objective.recorder()
    evaluate = lightgbm_evaluation(recorder)

    class Evaluated:
        def __init__(self, labels: Any) -> None:
            self.labels = labels

        def get_label(self) -> Any:
            return self.labels

    name, value, higher = evaluate(np.array([0.1, 0.9, 0.2, 0.3]), Evaluated(data.labels))
    assert (name, value, higher) == ("v1_objective", 0.5, True)
    with pytest.raises(ModelError, match="other than the V1 partition"):
        evaluate(np.zeros(3), Evaluated(np.zeros(3, dtype=np.float32)))


def test_catboost_metric_reports_numerator_and_frauds() -> None:
    data = _partition([0, 1, 1, 0], [1, 1, 1, 1])
    recorder = V1Objective(data, 1).recorder()
    metric = CatBoostObjective(recorder)
    assert metric.is_max_optimal() is True
    error, weight = metric.evaluate([np.array([-3.0, 2.0, 0.0, 1.0])], data.labels, None)
    assert (error, weight) == (1.0, 2.0)
    assert metric.get_final_error(error, weight) == 0.5
    assert metric.get_final_error(0.0, 0.0) == 0.0
    assert recorder.evaluations == 1


def test_catboost_metric_leaves_other_sets_neutral() -> None:
    data = _partition([0, 1, 1, 0], [1, 1, 1, 1])
    recorder = V1Objective(data, 1).recorder()
    metric = CatBoostObjective(recorder)
    assert metric.evaluate([np.zeros(7)], np.zeros(7), None) == (0.0, 1.0)
    assert (recorder.evaluations, recorder.other_evaluations) == (0, 1)


def test_catboost_metric_ranks_logistic_probabilities_and_keeps_saturated_ties() -> None:
    # Raw 40 and 50 are different approximations but the same float64 probability (1.0): the
    # rows tie and the queue's digest order decides, as it does for the scored probabilities.
    ids = [101, 102]
    data = _partition([1, 0], [1, 1], ids)
    recorder = V1Objective(data, 1).recorder()
    metric = CatBoostObjective(recorder)
    with np.errstate(all="raise"):  # the logistic of very large raw values must not warn
        error, _ = metric.evaluate([np.array([40.0, 50.0])], data.labels, None)
        metric.evaluate([np.array([-800.0, 800.0])], data.labels, None)
    first = int(np.argmin(tie_ranks(np.array(ids, dtype=np.int64))))
    assert error == float(data.labels[first])


def test_catboost_metric_survives_copying_with_its_recorder() -> None:
    recorder = V1Objective(_partition([0, 1], [1, 1]), 1).recorder()
    metric = CatBoostObjective(recorder)
    assert copy.deepcopy(metric) is metric
    assert copy.copy(metric) is metric
    assert copy.deepcopy([metric])[0].recorder is recorder


def test_the_partition_must_have_the_same_relative_days() -> None:
    data = _partition([0, 1, 0, 1], [1, 1, 2, 2])
    objective = V1Objective(data, 1)
    with pytest.raises(DataValidationError, match="different V1 rows"):
        objective.check_partition(_partition([0, 1, 0, 1], [1, 2, 2, 2]))  # same ids, labels


def test_an_identity_does_not_depend_on_its_key_order() -> None:
    identity = V1Objective(_partition([0, 1, 0, 1], [1, 1, 2, 2]), 1).identity
    reordered = dict(reversed(list(identity.items())))
    assert check_identity(reordered) == identity
    assert check_identity(json.loads(json.dumps(identity, sort_keys=True))) == identity


def test_logistic_matches_catboost_probabilities_on_its_own_raw_scores() -> None:
    """The probabilities the metric ranks agree with the ones ``predict_proba`` returns to within
    4 units in the last place (ulp), including a fit whose raw scores are far from zero
    (probabilities near 0 and 1). NumPy's and CatBoost's ``exp`` can differ in the last bits
    between CPUs and math libraries (the CI runner differs from the development machine), so bit
    equality is not guaranteed: the bound is an exponential, a rounded sum and a division, each
    within about one ulp. Exact saturation is covered by
    ``test_logistic_saturates_without_warnings``; ties on a real GPU fit are the subject of the
    acceptance check G3."""
    from catboost import CatBoostClassifier

    rng = np.random.default_rng(3)
    features = rng.normal(size=(600, 5))
    labels = (features[:, 0] + 0.1 * rng.normal(size=600) > 0).astype(int)
    for rate, rounds in ((0.1, 40), (1.0, 200)):  # the second drives raw scores far from 0
        model = CatBoostClassifier(
            iterations=rounds,
            learning_rate=rate,
            depth=4,
            random_seed=1,
            thread_count=1,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(features, labels)
        raw = np.asarray(model.predict(features, prediction_type="RawFormulaVal"), np.float64)
        native = model.predict_proba(features)[:, 1]
        ours = logistic(raw)
        assert np.all(np.abs(ours - native) <= 4 * np.spacing(native))
        far = float(np.abs(raw).max())
    assert far > 8  # the second fit: probabilities within 3.4e-4 of 0 or 1


def test_logistic_saturates_without_warnings() -> None:
    with np.errstate(all="raise"):
        values = logistic(np.array([-800.0, -40.0, 0.0, 40.0, 800.0]))
    assert values.tolist() == [0.0, 1.0 / (1.0 + float(np.exp(40.0))), 0.5, 1.0, 1.0]


def test_an_objective_built_from_arrays_equals_one_built_from_a_partition() -> None:
    partition = _partition([0, 1, 1, 0, 1], [1, 1, 2, 2, 2])
    assert partition.ids is not None
    assert partition.times is not None
    from_partition = V1Objective(partition, 2)
    from_rows = V1Objective.from_rows(partition.ids, partition.times, partition.labels, 2)
    assert from_rows.identity == from_partition.identity
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    assert from_rows.queue.recall(scores, from_rows.labels, 2) == from_partition.queue.recall(
        scores, from_partition.labels, 2
    )


def test_the_rows_of_an_objective_are_checked_as_arrays() -> None:
    partition = _partition([0, 1, 1, 0], [1, 1, 2, 2])
    assert partition.ids is not None
    assert partition.times is not None
    objective = V1Objective(partition, 1)
    objective.check_rows(partition.ids, partition.times, partition.labels)
    for ids, times, labels in (
        (None, partition.times, partition.labels),
        (partition.ids, None, partition.labels),
        (partition.ids + 1, partition.times, partition.labels),
        (partition.ids, partition.times + DAY, partition.labels),
        (partition.ids, partition.times, 1 - partition.labels),
        (partition.ids[:3], partition.times[:3], partition.labels[:3]),
    ):
        with pytest.raises(DataValidationError, match="different V1 rows"):
            objective.check_rows(ids, times, labels)


def test_a_neural_history_stops_one_epoch_later_than_a_tree_history() -> None:
    objective = V1Objective(_partition([1, 0, 0, 0], [1, 1, 1, 1]), 1)
    recorder = objective.recorder()
    for first in (0, 1, 2):  # the first top-1 pick is the fraud: the only gain
        _top(recorder, 4, first)
    assert recorder.history(patience=2, maximum=60).unit == "round"  # best 1 + patience 2
    assert recorder.history(patience=1, maximum=60, unit="epoch").unit == "epoch"  # 1 + 1 + 1
    with pytest.raises(ModelError, match="neither"):
        recorder.history(patience=2, maximum=60, unit="epoch")
    with pytest.raises(ModelError, match="training unit"):
        recorder.history(patience=1, maximum=60, unit="step")
