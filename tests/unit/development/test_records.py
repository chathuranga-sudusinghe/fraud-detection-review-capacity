"""Record contracts of the development pipeline: every kind round-trips, carries its version and
required fields, and rejects records that break its rules (synthetic records; no data read)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from review_capacity.config import load_dataset_spec
from review_capacity.development.records import (
    CONFIGURATION_NAMES,
    RECORD_CONTRACTS,
    DevelopmentPartitions,
    dump_record,
    read_record,
    record_problems,
    validate_record,
)
from review_capacity.development.schema import loads
from review_capacity.errors import FileAccessError, RecordContractError
from support.development import (
    PARTITIONS,
    V2_FRAUDS,
    V2_ROWS,
    bins,
    example,
    file,
    history,
    with_days,
)
from support.synthetic import REPO_PATHS

KINDS = list(RECORD_CONTRACTS)
FIELDS = [(kind, f.name) for kind, c in RECORD_CONTRACTS.items() for f in c.body.fields]
REQUIRED = [
    (kind, f.name) for kind, c in RECORD_CONTRACTS.items() for f in c.body.fields if not f.nullable
]


def validate(record: dict[str, Any], kind: str | None = None) -> None:
    validate_record(record, PARTITIONS, kind)


def dump(record: dict[str, Any]) -> str:
    return dump_record(record, PARTITIONS)


def read(path: Path, kind: str | None = None) -> dict[str, Any]:
    return read_record(path, PARTITIONS, kind)


def problems(record: dict[str, Any]) -> list[str]:
    return record_problems(record, PARTITIONS)


def assert_rejected(record: dict[str, Any], fragment: str) -> None:
    found = problems(record)
    assert any(fragment in problem for problem in found), found


def test_every_kind_has_a_valid_example_in_production_order() -> None:
    assert KINDS == [
        "candidates",
        "cost_prediction",
        "run",
        "preprocessing",
        "preflight",
        "fit",
        "checkpoint",
        "trial",
        "configuration",
        "refit",
        "score",
        "evaluation",
        "timing",
        "device_comparison",
        "calibration",
        "cost_analysis",
        "r1_selection",
        "h3_selection",
        "run_end",
    ]
    for kind in KINDS:
        validate(example(kind), kind)


def test_a_cost_analysis_keeps_its_arithmetic_and_cases_consistent() -> None:
    record = example("cost_analysis")
    assert problems(record) == []
    grid = record["grid"]

    def with_row(index: int, **changes: Any) -> dict[str, Any]:
        return record | {"grid": [*grid[:index], grid[index] | changes, *grid[index + 1 :]]}

    assert_rejected(with_row(0, unused=69), "reviews + unused")
    assert_rejected(with_row(0, cost_difference=-57.0), "cost_difference")
    assert_rejected(with_row(0, fixed_status="identical"), "fixed_status")
    assert_rejected(with_row(0, declined_fraud_rate=None), "null exactly when")
    assert_rejected(with_row(0, declined_fraud_rate=0.9), "frauds_forgone / unused")
    assert_rejected(with_row(0, utilisation=0.0), "reviews / (k × days)")
    assert_rejected(with_row(0, declined_mean_probability=0.6), "lie below p*")
    assert_rejected(with_row(0, fraud_value_exposed_r4=1.0), "plus the value forgone")
    assert_rejected(with_row(0, amount_weighted_recall_r4=0.9), "above R3's")
    assert_rejected(record | {"grid": grid[1:]}, "the ratios must be")
    assert_rejected(record | {"grid": [grid[1], grid[0], *grid[2:]]}, "the ratios must be")
    case = record["break_even"]
    assert_rejected(record | {"break_even": case | {"classification": "none"}}, "none exactly")
    assert_rejected(record | {"break_even": case | {"ratio": 9.0}}, "end of the last interval")
    assert_rejected(record | {"break_even": case | {"single_interval": False}}, "single_interval")
    assert_rejected(
        record | {"break_even": case | {"classification": "crossing"}}, "crossing needs ratio <"
    )
    assert_rejected(record | {"friction": record["friction"][:1]}, "the values phi must be")
    assert_rejected(
        record | {"friction": [{"phi": 1.0, "ratio": 20.0}, record["friction"][1]]}, "(1 + phi)"
    )
    assert_rejected(record | {"saturation": {"ratio": 10.0, "infinite": True}}, "null exactly")
    assert_rejected(record | {"period": "latency"}, "held-out range")
    no_ratio = [{"phi": 1.0, "ratio": None}, {"phi": 5.0, "ratio": None}]
    reach = {"start": 1.0, "end": 1000.0, "end_closed": True}
    censored = case | {"classification": "right_censored", "ratio": None, "intervals": [reach]}
    infinite = {"ratio": None, "infinite": True}
    censored_record = record | {
        "break_even": censored,
        "saturation": infinite,
        "friction": no_ratio,
    }
    assert problems(censored_record) == []
    # right-censoring needs ΔJ(1000) < 0 and a saturation ratio above 1000 (or infinite)
    assert_rejected(
        censored_record | {"saturation": {"ratio": 10.0, "infinite": False}},
        "saturation above 1000",
    )
    short = case | {
        "classification": "right_censored",
        "ratio": None,
        "intervals": [{"start": 1.0, "end": 5.0, "end_closed": True}],
    }
    assert_rejected(
        record | {"break_even": short, "saturation": infinite, "friction": no_ratio},
        "only the range end 1000 is closed",
    )
    # a single negative ratio, exactly 1000, is a closed point interval
    point = {"start": 1000.0, "end": 1000.0, "end_closed": True}
    single = case | {"classification": "right_censored", "ratio": None, "intervals": [point]}
    assert (
        problems(record | {"break_even": single, "saturation": infinite, "friction": no_ratio})
        == []
    )
    open_point = {"start": 5.0, "end": 5.0, "end_closed": False}
    assert_rejected(
        record | {"break_even": case | {"intervals": [open_point], "ratio": 5.0}}, "start < end"
    )


def test_a_device_comparison_is_a_catboost_pair_over_all_of_v2() -> None:
    record = example("device_comparison")
    assert problems(record) == []
    assert_rejected(record | {"configuration": "LGB-u"}, "only CatBoost")
    bootstrap = record["bootstrap"]
    assert_rejected(record | {"bootstrap": bootstrap | {"lower": 0.5}}, "lower is above upper")
    assert_rejected(record | {"bootstrap": bootstrap | {"days": 27}}, "V2 has 28 relative days")
    assert_rejected(
        record | {"bootstrap": bootstrap | {"undefined_replicates": 10_000}}, "no replicate"
    )
    assert_rejected(record | {"bootstrap": bootstrap | {"k": 57}}, "one of the capacities")
    assert_rejected(record | {"capacities": record["capacities"] * 2}, "listed once")
    assert_rejected(record | {"seed": 43}, "device_comparison.seed: 43 must be the main seed 42")
    first = record["capacities"][0] | {"differences": {"bogus": 1.0}}
    assert_rejected(record | {"capacities": [first]}, "bogus")


@pytest.mark.parametrize("kind", KINDS)
def test_records_round_trip_through_json(kind: str, tmp_path: Path) -> None:
    record = example(kind)
    text = dump(record)
    assert loads(text, kind) == record
    path = tmp_path / f"{kind}.json"
    path.write_text(text, encoding="utf-8")
    assert read(path, kind) == record


@pytest.mark.parametrize("kind", KINDS)
def test_version_field_must_match_the_contract(kind: str) -> None:
    record = example(kind) | {"schema_version": RECORD_CONTRACTS[kind].version + 1}
    assert_rejected(record, f"{kind}.schema_version")


@pytest.mark.parametrize(("kind", "name"), FIELDS)
def test_every_field_is_required(kind: str, name: str) -> None:
    record = example(kind)
    del record[name]
    assert f"{kind}: missing field {name!r}" in RECORD_CONTRACTS[kind].problems(record)


@pytest.mark.parametrize(("kind", "name"), REQUIRED)
def test_non_nullable_fields_reject_null(kind: str, name: str) -> None:
    record = example(kind) | {name: None}
    assert f"{kind}.{name}: must not be null" in RECORD_CONTRACTS[kind].problems(record)


@pytest.mark.parametrize("kind", KINDS)
def test_unknown_fields_are_rejected(kind: str) -> None:
    assert_rejected(example(kind) | {"note": "x"}, f"{kind}: unknown field 'note'")


def test_unknown_and_unexpected_kinds_are_rejected() -> None:
    with pytest.raises(RecordContractError, match="Unknown record kind 'summary'"):
        validate(example("fit") | {"record": "summary"})
    with pytest.raises(RecordContractError, match="Expected a trial record, got a fit record"):
        validate(example("fit"), "trial")
    with pytest.raises(RecordContractError, match="does not satisfy its contract"):
        dump(example("fit") | {"seed": -1})


def test_read_record_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(FileAccessError, match="cannot be read"):
        read(tmp_path / "missing.json")
    path = tmp_path / "fit.json"
    path.write_text(dump(example("fit")).replace('"seed": 42', '"seed": 7'), "utf-8")
    with pytest.raises(RecordContractError, match=r"fit\.json: fit record"):
        read(path)


def test_identity_rejects_malformed_values() -> None:
    record = example("score")
    record["identity"]["run_id"] = "2026-10-20 09:15"
    record["identity"]["code_commit"] = "abc"
    assert_rejected(record, "score.identity.run_id")
    assert_rejected(record, "score.identity.code_commit")


def test_development_records_cannot_name_the_test_period() -> None:
    assert_rejected(example("score") | {"partition": "Test"}, "score.partition")
    assert_rejected(example("evaluation") | {"partition": "Test"}, "evaluation.partition")


def test_partitions_come_from_the_dataset_specification() -> None:
    spec = load_dataset_spec(REPO_PATHS)
    assert DevelopmentPartitions.from_spec(spec, REPO_PATHS) == PARTITIONS
    assert dict(PARTITIONS.processed) == {
        "development_joined": "data/processed/development_joined.parquet",
        "train": "data/processed/train.parquet",
        "v1": "data/processed/v1.parquet",
        "v2": "data/processed/v2.parquet",
    }
    assert (PARTITIONS.v2.first_day, PARTITIONS.v2.last_day) == (127, 154)
    assert PARTITIONS.named("V1") is PARTITIONS.v1


def test_sealed_days_cannot_be_labelled_as_development_days() -> None:
    """Test days (155–182) under a V2 label are rejected by every day-bearing record."""
    test_days = {"first": 155, "last": 182}
    score = example("score") | {"first_day": 155, "last_day": 182}
    assert_rejected(score, "days 155–182 lie outside Validation 2 (days 127–154)")
    evaluation = with_days(example("evaluation"), 155, 182)
    assert_rejected(evaluation, "evaluation.days: days 155–182 lie outside")
    assert_rejected(example("r1_selection") | {"selection_days": test_days}, "lie outside")
    assert_rejected(example("h3_selection") | {"selection_days": test_days}, "lie outside")
    calibration = example("calibration") | {"fitting_days": test_days}
    calibration["diagnostics"][0]["days"] = test_days
    assert_rejected(calibration, "calibration.fitting_days: days 155–182 lie outside")
    held_out = _latency_calibration()
    held_out["diagnostics"][1]["days"] = {"first": 148, "last": 160}
    assert_rejected(held_out, "calibration.diagnostics[1].days: days 148–160 lie outside")


def test_partition_rows_and_full_ranges_are_checked() -> None:
    assert_rejected(example("score") | {"rows": 77_821}, "not the 77822 of Validation 2")
    assert_rejected(example("score") | {"first_day": 128}, "must cover all of Validation 2")
    record = example("fit") | {"configuration": "FTT-u", "positive_weight": 1}
    record["train"]["rows"] = 368_000
    assert_rejected(record, "fit.train.rows: 368000, not the 368632 Train rows")
    evaluation = with_days(example("evaluation"), 128, 154)
    assert_rejected(evaluation, "must cover all of Validation 2")
    evaluation["counts"]["days"] = 28
    assert_rejected(evaluation, "counts.days")


def test_latency_evaluations_use_a_fitting_or_held_out_part_of_v2() -> None:
    evaluation = with_days(example("evaluation") | {"period": "latency"}, 148, 154)
    validate(evaluation)
    middle = with_days(evaluation, 130, 136)
    assert_rejected(middle, "a label-latency range is a fitting or a held-out part")
    assert_rejected(evaluation | {"partition": "V1"}, "uses V2 only")


# --- run -----------------------------------------------------------------------------------------


def test_candidate_lists_are_ordered_and_start_at_the_reference() -> None:
    record = example("candidates")
    record["trials"][1]["reference"] = True
    assert_rejected(record, "only trial 1 is the reference configuration")
    record = example("candidates")
    record["trials"][2]["trial"] = 4
    assert_rejected(record, "numbered 1, 2, ... in trial order")
    record = example("candidates")
    del record["trials"][2]["parameters"]["num_leaves"]
    assert_rejected(record, "same parameters in order")
    record = example("candidates")
    record["trials"][2]["parameters"] = dict(reversed(record["trials"][2]["parameters"].items()))
    assert_rejected(record, "same parameters in order")


def test_candidate_lists_belong_to_their_stream_and_seed() -> None:
    assert_rejected(example("candidates") | {"sampler_seed": 7}, "declared sampler seed")
    assert_rejected(
        example("candidates") | {"stream": {"index": 1, "spawn_key": [0]}},
        "the spawn key is the stream index",
    )
    assert_rejected(
        example("candidates") | {"configurations": ["LGB-u", "Cat-b"]},
        "Cat-b does not belong to lightgbm",
    )
    assert_rejected(example("candidates") | {"model": "keras"}, "candidates.model")
    short = example("candidates")
    short["trials"] = short["trials"][:1]
    assert_rejected(short, "at least 2")


def test_each_model_uses_its_own_stream() -> None:
    wrong = example("candidates") | {"stream": {"index": 4, "spawn_key": [4]}}
    assert_rejected(wrong, "candidates.stream.index: lightgbm uses stream 0")
    reference = example("candidates") | {
        "model": "logistic_regression",
        "configurations": ["LR"],
        "stream": {"index": 4, "spawn_key": [4]},
    }
    validate(reference)


def test_candidates_name_the_sampling_software() -> None:
    record = example("candidates")
    del record["sampler"]["numpy"]
    assert_rejected(record, "candidates.sampler: missing field 'numpy'")


def test_cost_estimates_follow_whether_a_cost_model_exists() -> None:
    record = example("cost_prediction")
    record["estimates"][0]["search_hours_low"] = 1.0
    assert_rejected(record, "estimates[lightgbm].search_hours_low: must be null")
    record = example("cost_prediction")
    record["estimates"][2]["refit_hours"] = None
    assert_rejected(record, "estimates[resnet].refit_hours: required when estimated")
    record = example("cost_prediction")
    record["estimates"][2]["candidate_epoch_minutes"] = []
    assert_rejected(record, "candidate_epoch_minutes: required when estimated")
    record = example("cost_prediction")
    record["estimates"][3]["search_hours_low"] = 99.0
    assert_rejected(record, "search_hours_low must not exceed search_hours_high")
    record = example("cost_prediction")
    record["estimates"][1]["candidate_epoch_minutes"] = [1.0]
    assert_rejected(record, "estimates[catboost].candidate_epoch_minutes: empty")


def test_cost_totals_and_models_are_checked() -> None:
    record = example("cost_prediction")
    record["totals"]["search_hours_high"] = 9.0
    assert_rejected(record, "totals.search_hours_high: must be the sum 8.0")
    record = example("cost_prediction")
    record["estimates"].reverse()
    assert_rejected(record, "one per model, in DR-011 order then LR")
    record = example("cost_prediction")
    del record["candidates"]["resnet"]
    assert_rejected(record, "one candidate record per model")
    approved = example("cost_prediction") | {"decision": "approved"}
    assert_rejected(approved, "cost_prediction.decision")


def test_plan_records_carry_the_search_plan_run_type() -> None:
    record = example("candidates")
    record["identity"]["run_type"] = "search_plan"
    validate(record)
    record["identity"]["run_type"] = "training"
    assert_rejected(record, "candidates.identity.run_type")


def test_run_identity_must_match_the_run_provenance() -> None:
    record = example("run")
    record["identity"]["settings_sha256"] = "0" * 64
    assert_rejected(record, "run.identity.settings_sha256")


def test_runs_read_processed_development_files_only() -> None:
    record = example("run")
    record["dataset"]["processed"]["test"] = file("data/processed/test.parquet")
    assert_rejected(record, "run.dataset.processed: 'test' is not a development file")
    moved = example("run")
    moved["dataset"]["processed"]["train"] = file("data/other/train.parquet")
    assert_rejected(moved, "run.dataset.processed.train.path: must be data/processed/train.parquet")


def test_run_lists_the_current_contract_versions() -> None:
    record = example("run")
    record["contracts"]["fit"] = 9
    assert_rejected(record, "run.contracts")


# --- preprocessing and preflight -----------------------------------------------------------------


def test_preprocessing_uses_the_declared_seed() -> None:
    assert_rejected(example("preprocessing") | {"seed": 1}, "declared preprocessing seed")


def test_score_seeds_follow_the_model_role() -> None:
    refit = example("score") | {"model_role": "seed_refit"}
    assert_rejected(refit, "score.seed: 42 must be one of the additional seeds")
    validate(refit | {"seed": 44})


def test_preprocessing_input_counts_add_up() -> None:
    record = example("preprocessing")
    record["inputs"]["total"] = 758
    assert_rejected(record, "preprocessing.inputs.total")


def test_preflight_selects_the_largest_passing_micro_batch() -> None:
    assert_rejected(example("preflight") | {"micro_batch": 128}, "largest passing micro-batch")
    record = example("preflight")
    record["attempts"][0], record["attempts"][1] = record["attempts"][1], record["attempts"][0]
    assert_rejected(record, "largest first")


def test_preflight_outcomes_follow_the_memory_margin() -> None:
    record = example("preflight")
    record["attempts"][1]["outcome"] = "pass"  # 651 MiB free is below the 1,024 MiB margin
    assert_rejected(record, "contradicts the free-memory margin")


def test_resnet_preflight_must_pass_at_the_effective_batch() -> None:
    record = example("preflight") | {"model": "resnet"}
    assert_rejected(record, "runs the effective batch")


def test_failed_preflight_records_an_error_and_no_plan() -> None:
    record = example("preflight") | {"status": "fail", "error": "no micro-batch fits"}
    assert_rejected(record, "preflight.micro_batch: must be null")
    validate(record | {"micro_batch": None})


# --- fit -----------------------------------------------------------------------------------------


def test_fit_class_weight_follows_the_weighting() -> None:
    assert_rejected(example("fit") | {"positive_weight": 28.0}, "n_neg / n_pos")
    unweighted = example("fit") | {"configuration": "FTT-u"}
    assert_rejected(unweighted, "must be 1 for an unweighted configuration")
    validate(unweighted | {"positive_weight": 1})


def test_fit_inputs_follow_the_family() -> None:
    tree = example("fit") | {"configuration": "Cat-u", "positive_weight": 1}
    assert_rejected(tree, "fit.preprocessing: must be null when the model is tree-based")
    validate(tree | {"preprocessing": None, "preflight": None})
    assert_rejected(example("fit") | {"preflight": None}, "fit.preflight: required")


def test_refit_fits_run_at_a_fixed_length_with_their_seeds() -> None:
    refit = example("fit") | {"purpose": "seed_refit", "early_stopping": False, "length": 7}
    assert_rejected(refit, "fit.seed: 42 must be one of the additional seeds")
    validate(refit | {"seed": 43})
    assert_rejected(refit | {"seed": 43, "length": 8}, "a refit grows exactly its length")
    assert_rejected(refit | {"seed": 43, "early_stopping": True}, "fixed length")
    assert_rejected(refit | {"seed": 43, "length": None}, "fit.length: required")
    repeat = refit | {"purpose": "same_seed_repeat"}
    assert_rejected(repeat, "a same-seed repeat runs in a fresh process")
    validate(repeat | {"fresh_process": True})


def test_cpu_refits_are_catboost_on_the_cpu() -> None:
    record = example("fit") | {
        "purpose": "cpu_refit",
        "early_stopping": False,
        "length": 7,
    }
    assert_rejected(record, "a CPU refit is a CatBoost refit")
    assert_rejected(record, "a CPU refit runs on the cpu")


def test_reference_fits_converge_in_its_search() -> None:
    record = example("fit") | {
        "configuration": "LR",
        "positive_weight": 1,
        "preflight": None,
        "device": "cpu",
        "early_stopping": False,
    }
    validate(record)
    assert_rejected(record | {"early_stopping": True}, "fitted to convergence")


def test_failed_fit_records_its_error() -> None:
    record = example("fit") | {"status": "failed"}
    assert_rejected(record, "fit.error: required when status is failed")
    assert_rejected(example("fit") | {"model": None}, "fit.model: required when status")


# --- trial and checkpoint ------------------------------------------------------------------------


def test_tree_trials_stop_patience_rounds_after_the_best() -> None:
    assert_rejected(example("trial") | {"stopped_at": 171}, "early stopping after 120 ends at 170")


def test_neural_trials_stop_at_the_epoch_after_patience() -> None:
    record = example("trial") | {
        "configuration": "FTT-u",
        "device": "gpu",
        "unit": "epoch",
        "patience": 5,
        "maximum": 15,
        "best_checkpoint": 4,
        "stopped_at": 10,
        "checkpoints": history(4, 10),
    }
    validate(record)
    assert_rejected(record | {"stopped_at": 9}, "early stopping after 4 ends at 10")


def test_maximum_applies_only_when_patience_did_not_run_out() -> None:
    record = example("trial") | {"stopping_reason": "maximum", "stopped_at": 2000}
    validate(record | {"best_checkpoint": 1990, "checkpoints": history(1990, 2000)})
    assert_rejected(record | {"best_checkpoint": 1900}, "patience did not run out")
    assert_rejected(record | {"best_checkpoint": 1990, "stopped_at": 1999}, "maximum applies")


def test_trial_units_follow_the_model() -> None:
    assert_rejected(example("trial") | {"unit": "epoch"}, "trial.unit: must be round")
    assert_rejected(example("trial") | {"stopping_reason": "converged"}, "only the reference")


def test_reference_trials_converge_without_checkpoints() -> None:
    record = example("trial") | {
        "configuration": "LR",
        "unit": "convergence",
        "patience": None,
        "maximum": None,
        "best_checkpoint": None,
        "stopped_at": None,
        "stopping_reason": "converged",
        "checkpoint": None,
        "checkpoints": [],
    }
    validate(record)
    assert_rejected(record | {"checkpoints": history(1, 1)}, "must be empty")
    assert_rejected(record | {"stopped_at": 30}, "trial.stopped_at: must be null")


def test_checkpoint_history_follows_strict_improvement() -> None:
    record = example("trial")
    record["checkpoints"][120]["improved"] = True  # below the best: no improvement
    assert_rejected(record, "trial.checkpoints[120]: improvement is strict")
    equal = example("trial")
    equal["checkpoints"][120] |= {"objective": 0.41}  # an equal value is no improvement
    validate(equal)
    gap = example("trial")
    del gap["checkpoints"][5]
    assert_rejected(gap, "trial.checkpoints: numbered 1, 2, ... in order")


def test_checkpoint_history_ends_when_patience_first_runs_out() -> None:
    objectives = (0.4, 0.3, 0.3, 0.5, 0.2, 0.2)
    improved = (True, False, False, True, False, False)
    best = (1, 1, 1, 4, 4, 4)
    rows = [
        {"checkpoint": i + 1, "objective": o, "improved": u, "best_checkpoint": b, "seconds": 1.0}
        for i, (o, u, b) in enumerate(zip(objectives, improved, best, strict=True))
    ]
    late = example("trial") | {
        "patience": 2,
        "best_checkpoint": 4,
        "best_objective": 0.5,
        "stopped_at": 6,
        "checkpoints": rows,
    }
    assert_rejected(late, "training stops when patience runs out after checkpoint 1")


def test_checkpoint_history_matches_the_trial_result() -> None:
    assert_rejected(example("trial") | {"checkpoints": history(120, 169)}, "one entry per round")
    other = example("trial") | {"checkpoints": history(120, 170, 0.42)}
    assert_rejected(other, "the history's best differs from the trial's result")


def test_trials_use_the_main_seed() -> None:
    assert_rejected(example("trial") | {"seed": 43}, "must be the main seed 42")


def test_failed_trials_keep_their_error() -> None:
    record = example("trial") | {
        "status": "failed",
        "error": "out of memory",
        "best_checkpoint": None,
        "best_objective": None,
        "stopped_at": None,
        "stopping_reason": None,
        "runtime_seconds": None,
        "fit": None,
        "checkpoint": None,
    }
    validate(record)
    assert_rejected(record | {"error": None}, "trial.error: required")


def test_checkpoint_unit_follows_the_model() -> None:
    assert_rejected(example("checkpoint") | {"unit": "epoch"}, "checkpoints per round")
    assert_rejected(example("checkpoint") | {"configuration": "LR"}, "no checkpoints")


# --- configuration (selection) -------------------------------------------------------------------


def test_selection_takes_the_lowest_of_tied_trials() -> None:
    assert_rejected(example("configuration") | {"selected_trial": 3}, "selects trial 2")
    assert_rejected(example("configuration") | {"tied_trials": [2]}, "best value are [2, 3]")
    assert_rejected(example("configuration") | {"selected_objective": 0.41}, "best trial")


def test_selection_lists_trials_in_order() -> None:
    record = example("configuration")
    record["trials"].reverse()
    assert_rejected(record, "numbered 1, 2, ...")


def test_reference_selection_has_no_length_or_checkpoint() -> None:
    assert_rejected(example("configuration") | {"configuration": "LR"}, "must be null")


# --- refit and score -----------------------------------------------------------------------------


def test_refit_comparison_is_consistent() -> None:
    record = example("refit")
    record["comparison"]["differing_rows"] = 0
    assert_rejected(record, "identical exactly when no row differs")
    assert_rejected(example("refit") | {"comparison": None}, "refit.comparison: required")
    seed = example("refit") | {"kind": "seed_refit", "seed": 44}
    assert_rejected(seed, "refit.comparison: must be null when the seed differs")
    validate(seed | {"comparison": None})


def test_refits_exist_only_for_the_eight_configurations() -> None:
    assert_rejected(example("refit") | {"configuration": "LR"}, "refit.configuration")


def test_score_summary_is_consistent() -> None:
    assert_rejected(example("score") | {"distinct_scores": 80_000}, "exceeds the rows")
    assert_rejected(example("score") | {"distinct_scores": 1}, "one distinct score")
    assert_rejected(example("score") | {"first_day": 155}, "first_day after last_day")
    assert_rejected(example("score") | {"min_score": 1.5}, "score.min_score")


# --- evaluation ----------------------------------------------------------------------------------


def test_policies_carry_exactly_their_settings() -> None:
    assert_rejected(example("evaluation") | {"k": None}, "evaluation.k: required")
    assert_rejected(example("evaluation") | {"threshold": 0.5}, "threshold: must be null")
    ranking = example("evaluation") | {"policy": "ranking", "capacity": None, "k": None}
    assert_rejected(ranking, "evaluation.counts: must be null")
    assert_rejected(ranking, "evaluation.daily: must be empty")
    validate(with_days(ranking, 127, 154))
    r1 = example("evaluation") | {"policy": "R1", "threshold": 0.7, "cutoff_ties": None}
    validate(with_days(r1, 127, 154))
    validate(with_days(r1 | {"capacity": None, "k": None}, 127, 154))
    assert_rejected(r1 | {"k": None}, "capacity and k are given together")


def test_r4_reviews_at_the_derived_threshold() -> None:
    record = example("evaluation") | {
        "policy": "R4",
        "threshold": 1 / 50,
        "cost_ratio": 50,
        "calibrator": "platt",
        "calibration": file("calibration/LGB-u/platt.json"),
    }
    validate(record)
    assert_rejected(record | {"threshold": 0.03}, "p* = 1/rho")


def test_evaluation_counts_add_up() -> None:
    record = example("evaluation")
    record["counts"]["false_negatives"] = 1
    assert_rejected(record, "true_positives + false_negatives ≠ frauds")
    record = example("evaluation")
    record["counts"] |= {"selected": 1569, "false_positives": 669}
    assert_rejected(record, "reviews at most K a day")


def test_daily_values_add_up_to_the_pooled_counts() -> None:
    record = example("evaluation")
    record["daily"][3]["frauds"] += 1
    assert_rejected(record, "evaluation.daily: frauds do not sum to counts.frauds")
    shuffled = example("evaluation")
    shuffled["daily"].reverse()
    assert_rejected(shuffled, "one entry per evaluated day, in day order")


def test_r3_fills_its_daily_budget() -> None:
    record = example("evaluation")
    record["daily"][1] |= {"selected": 55, "unused": 1, "load_ratio": 55 / 56}
    record["counts"]["selected"] -= 1
    record["counts"]["false_positives"] -= 1
    assert_rejected(record, "evaluation.daily[1]: R3 reviews min(K, N_d) transactions")


def test_full_period_evaluations_hold_the_partition_volume() -> None:
    record = example("evaluation")
    record["daily"][0]["transactions"] += 1
    record["counts"]["transactions"] += 1
    assert_rejected(record, "evaluation.counts.transactions: 77823, not the 77822 of Validation 2")


def test_daily_workload_measures_follow_the_budget() -> None:
    record = example("evaluation")
    record["daily"][0]["unused"] = 1
    assert_rejected(record, "evaluation.daily[0].unused: must be 0")
    unbudgeted = with_days(
        example("evaluation") | {"policy": "R1", "threshold": 0.7, "cutoff_ties": None},
        127,
        154,
    )
    unbudgeted["daily"][0]["overflow"] = 4
    unbudgeted |= {"capacity": None, "k": None}
    assert_rejected(unbudgeted, "evaluation.daily[0].overflow: must be null when there is no")


def test_cutoff_ties_summarise_the_daily_tied_groups() -> None:
    record = example("evaluation")
    record["daily"][5]["cutoff_tied_group"] = 4
    assert_rejected(record, "evaluation.cutoff_ties: must summarise the daily tied groups")
    r2 = example("evaluation") | {"policy": "R2", "threshold": 0.7, "cutoff_ties": None}
    assert_rejected(r2, "cutoff ties are recorded for capped policies only")


def test_evaluation_metrics_use_the_declared_names() -> None:
    record = example("evaluation")
    record["metrics"]["accuracy"] = 0.9
    assert_rejected(record, "'accuracy' is not one of")


def test_reference_and_refit_evaluations_are_limited() -> None:
    r1 = {"policy": "R1", "threshold": 0.7, "cutoff_ties": None}
    assert_rejected(example("evaluation") | r1 | {"configuration": "LR"}, "ranking and R3 only")
    refit = example("evaluation") | {"model_role": "seed_refit", "seed": 43}
    validate(refit)
    assert_rejected(refit | {"partition": "V1"}, "under R3 on all of V2 only")
    assert_rejected(refit | {"period": "latency"}, "under R3 on all of V2 only")
    assert_rejected(refit | {"seed": 42}, "evaluation.seed: 42 must be one of the additional")


# --- timing and calibration ----------------------------------------------------------------------


def test_latency_percentiles_are_ordered() -> None:
    record = example("timing")
    record["latency_ms"]["p95"] = 1.0
    assert_rejected(record, "median ≤ p95 ≤ p99")


def test_timing_records_its_procedure() -> None:
    timing = example("timing")
    assert_rejected(timing | {"rows_sha256": None}, "timing.rows_sha256: required")
    assert_rejected(timing | {"row_selection": None}, "timing.row_selection: required")
    assert_rejected(timing | {"memory_scope": None}, "timing.memory_scope: required")
    assert_rejected(timing | {"model_role": None}, "timing.model_role: required")
    assert_rejected(timing | {"attention_backend": "math"}, "the model has no attention")
    assert_rejected(timing | {"batch_rows": 16}, "a batch of 1")
    transformer = timing | {"configuration": "FTT-u"}
    assert_rejected(transformer, "timing.attention_backend: required")
    validate(transformer | {"attention_backend": "efficient"})
    assert_rejected(transformer | {"attention_backend": "math"}, "uses the efficient backend")


def test_gpu_timing_is_synchronised() -> None:
    assert_rejected(example("timing") | {"synchronised": False}, "GPU work is synchronised")


def test_search_timing_names_no_single_model() -> None:
    search = example("timing") | {
        "configuration": "LGB-u",
        "measurement": "search",
        "model_role": None,
        "latency_ms": None,
        "boundary": None,
        "seconds": 3600.0,
        "device": "cpu",
        "threads": 24,
        "cpu_affinity": [0, 1],
        "batch_rows": None,
        "warmup": None,
        "repetitions": None,
        "rows": None,
        "rows_sha256": None,
        "row_selection": None,
        "settings": {
            "procedure_version": 1,
            "session": {"load_average_1m": 0.25},
            "memory_problems": [],
            "trials_completed": 10,
            "trials_declared": 10,
            "interrupted": False,
            "host_memory_method": "clear_refs",
        },
    }
    validate(search)
    assert_rejected(search | {"model_role": "selected"}, "timing.model_role: must be null")
    for missing in ("trials_completed", "trials_declared", "interrupted"):
        broken = {k: v for k, v in search["settings"].items() if k != missing}
        assert_rejected(search | {"settings": broken}, f"timing.settings.{missing}: required")


FIT: dict[str, Any] = {
    "measurement": "fit",
    "boundary": None,
    "latency_ms": None,
    "seconds": 12.5,
    "batch_rows": None,
    "warmup": None,
    "repetitions": None,
    "rows": None,
    "rows_sha256": None,
    "row_selection": None,
    "settings": {
        "procedure_version": 1,
        "session": {"load_average_1m": 0.25},
        "memory_problems": [],
        "completed_length": 170,
        "retained_length": 120,
        "startup_seconds": 1.5,
        "startup_scope": "library import, once per process",
        "startup_status": "measured",
        "preprocessing_seconds": None,
        "persistence_seconds": 0.4,
        "host_memory_method": "clear_refs",
    },
}


@pytest.mark.parametrize(
    ("configuration", "role", "device"),
    [
        ("LGB-u", "selected", "cpu"),
        ("Cat-b", "selected", "gpu"),
        ("Cat-b", "cpu_refit", "cpu"),
        ("RES-b", "selected", "gpu"),
        ("FTT-u", "seed_refit", "gpu"),
        ("LR", "selected", "cpu"),
    ],
)
def test_a_fit_runs_on_its_study_device(configuration: str, role: str, device: str) -> None:
    other = "gpu" if device == "cpu" else "cpu"
    record = example("timing") | FIT | {"configuration": configuration, "model_role": role}
    if configuration.startswith("FTT"):
        record["attention_backend"] = "efficient"
    cpu = {"threads": 24, "cpu_affinity": [0, 1], "synchronised": False}
    gpu = {"threads": None, "cpu_affinity": None, "synchronised": True}
    validate(record | {"device": device} | (cpu if device == "cpu" else gpu))
    wrong = record | {"device": other} | (cpu if other == "cpu" else gpu)
    assert_rejected(wrong, f"runs on the {device}")


@pytest.mark.parametrize(
    ("configuration", "device"),
    [("LGB-u", "cpu"), ("Cat-u", "cpu"), ("RES-u", "gpu"), ("FTT-b", "gpu"), ("LR", "cpu")],
)
def test_scoring_runs_on_the_prediction_device(configuration: str, device: str) -> None:
    other = "gpu" if device == "cpu" else "cpu"
    cpu = {"threads": 24, "cpu_affinity": [0, 1], "synchronised": False}
    gpu = {"threads": None, "cpu_affinity": None, "synchronised": True}
    base = example("timing") | {"configuration": configuration, "measurement": "scoring"}
    base |= {"boundary": "end_to_end", "partition": "V2", "batch_rows": 4096, "latency_ms": None}
    base |= {"rows": V2_ROWS, "seconds": 2.0, "rows_per_second": V2_ROWS / 2.0, "warmup": 1}
    base |= {"repetitions": 3, "rows_sha256": None, "row_selection": None}
    base["settings"] = {
        "procedure_version": 1,
        "session": {"load_average_1m": 0.25},
        "memory_problems": [],
        "pass_seconds": [2.5, 2.0, 1.9],
        "pass_min_seconds": 1.9,
        "pass_max_seconds": 2.5,
        "host_memory_method": "clear_refs",
    }
    if configuration.startswith("FTT"):
        base["attention_backend"] = "efficient"
    validate(base | {"device": device} | (cpu if device == "cpu" else gpu))
    wrong = base | {"device": other} | (cpu if other == "cpu" else gpu)
    assert_rejected(wrong, f"runs on the {device}")


def test_scoring_seconds_are_the_median_of_the_passes() -> None:
    base = example("timing") | {"measurement": "scoring", "boundary": "end_to_end"}
    base |= {"partition": "V2", "batch_rows": 4096, "latency_ms": None, "rows": V2_ROWS}
    base |= {"seconds": 2.0, "rows_per_second": V2_ROWS / 2.0, "warmup": 3, "repetitions": 3}
    base |= {"rows_sha256": None, "row_selection": None}
    base["settings"] = {
        "procedure_version": 1,
        "session": {"load_average_1m": 0.25},
        "memory_problems": [],
        "pass_seconds": [2.5, 2.0, 1.9],
        "pass_min_seconds": 1.9,
        "pass_max_seconds": 2.5,
        "host_memory_method": "clear_refs",
    }
    validate(base)
    assert_rejected(base | {"seconds": 2.5}, "median of settings.pass_seconds")
    short = {**base["settings"], "pass_seconds": [2.5, 2.0]}
    assert_rejected(base | {"settings": short}, "one entry per repetition")
    assert_rejected(base | {"warmup": None}, "timing.warmup: required")
    assert_rejected(base | {"repetitions": None}, "timing.repetitions: required")


def test_latency_counts_agree_with_the_rounds() -> None:
    timing = example("timing")
    assert_rejected(timing | {"rows": 9_999}, "rows x repetitions")
    settings = timing["settings"]
    assert_rejected(timing | {"repetitions": 2}, "one entry per round")
    unordered = [{"median": 0.9, "p95": 0.6, "p99": 0.4}] * 3
    assert_rejected(timing | {"settings": {**settings, "round_latency_ms": unordered}}, "in order")
    assert_rejected(
        timing | {"settings": {**settings, "percentile_method": "lower"}},
        "percentile_method: must be linear",
    )


def test_every_timing_records_the_procedure_and_the_session() -> None:
    timing = example("timing")
    for name in ("procedure_version", "session"):
        broken = {k: v for k, v in timing["settings"].items() if k != name}
        assert_rejected(timing | {"settings": broken}, f"timing.settings.{name}: required")
    assert_rejected(
        timing | {"settings": {**timing["settings"], "procedure_version": 2}},
        "procedure_version: must be 1",
    )
    assert_rejected(
        timing | {"settings": {**timing["settings"], "session": "ac power"}},
        "session: must be the session conditions",
    )


def test_the_host_memory_method_is_named() -> None:
    timing = example("timing")
    settings = {k: v for k, v in timing["settings"].items() if k != "host_memory_method"}
    assert_rejected(timing | {"settings": settings}, "host_memory_method")
    validate(timing | {"settings": {**settings, "host_memory_method": "clear_refs"}})


def test_a_fit_records_its_lengths() -> None:
    record = example("timing") | FIT | {"device": "gpu"}
    validate(record)
    longer = {**record["settings"], "retained_length": 171}
    assert_rejected(record | {"settings": longer}, "at most the completed length")


def test_platt_records_its_smoothed_targets_and_a_positive_slope() -> None:
    record = example("calibration")
    record["targets"]["positive"] = 1.0
    assert_rejected(record, "(N_pos + 1)/(N_pos + 2)")
    record = example("calibration")
    record["parameters"]["slope"] = 0.0
    assert_rejected(record, "kappa_1 must be positive")
    record = example("calibration")
    record["solver"]["converged"] = False
    assert_rejected(record, "a completed fit converged")
    assert_rejected(example("calibration") | {"reload_verified": False}, "after reloading")


def test_isotonic_records_knots_and_no_platt_fields() -> None:
    isotonic = example("calibration") | {"method": "isotonic", "knots": 4000}
    assert_rejected(isotonic, "calibration.targets: must be null")
    platt = dict.fromkeys(("clipping", "targets", "parameters", "solver"))
    validate(isotonic | platt)
    assert_rejected(isotonic | platt | {"knots": None}, "calibration.knots: required")


def _latency_calibration() -> dict[str, Any]:
    """A valid label-latency Platt calibrator: fitted on days 127–147, held out on 148–154."""
    rows, frauds = 60_000, 2_000
    record = example("calibration") | {
        "period": "latency",
        "fitting_days": {"first": 127, "last": 147},
        "rows": rows,
        "frauds": frauds,
        "legitimate": rows - frauds,
        "targets": {"positive": (frauds + 1) / (frauds + 2), "negative": 1 / (rows - frauds + 2)},
    }
    fitting = record["diagnostics"][0] | {
        "days": {"first": 127, "last": 147},
        "rows": rows,
        "frauds": frauds,
        "reliability": bins(rows, frauds),
    }
    held_out = fitting | {
        "rows_used": "held_out",
        "days": {"first": 148, "last": 154},
        "rows": V2_ROWS - rows,
        "frauds": V2_FRAUDS - frauds,
        "reliability": bins(V2_ROWS - rows, V2_FRAUDS - frauds),
    }
    record["diagnostics"] = [fitting, held_out]
    return record


def test_latency_calibrators_keep_fitting_and_held_out_diagnostics_apart() -> None:
    validate(_latency_calibration())
    fitting_only = _latency_calibration()
    fitting_only["diagnostics"].pop()
    assert_rejected(fitting_only, "expected ['fitting', 'held_out']")
    gap = _latency_calibration()
    gap["diagnostics"][1]["days"] = {"first": 150, "last": 154}
    assert_rejected(gap, "held-out days start after the fitting days")
    late_fit = _latency_calibration() | {"fitting_days": {"first": 127, "last": 154}}
    late_fit["diagnostics"] = late_fit["diagnostics"][:1]
    late_fit["diagnostics"][0]["days"] = {"first": 127, "last": 154}
    late_fit["period"] = "latency"
    assert_rejected(late_fit, "expected ['fitting', 'held_out']")
    primary = example("calibration")
    primary["diagnostics"][0]["rows"] = 1000
    assert_rejected(primary, "in-sample diagnostics use exactly the fitting rows")
    full_days = {"first": 127, "last": 154}
    partial = _latency_calibration() | {"period": "primary", "fitting_days": full_days}
    partial["diagnostics"] = partial["diagnostics"][:1]
    partial["diagnostics"][0]["days"] = full_days
    assert_rejected(partial, "a full-V2 calibrator fits all 77822 V2 rows")


def test_reliability_bins_are_ordered_separate_and_complete() -> None:
    record = example("calibration")
    reliability = record["diagnostics"][0]["reliability"]
    reliability[1]["min_probability"] = 0.01  # an equal probability would straddle two bins
    assert_rejected(record, "equal probabilities share one bin; bins do not overlap")
    record = example("calibration")
    record["diagnostics"][0]["reliability"][0]["observed_rate"] = 0.5
    assert_rejected(record, "bin 1: observed_rate ≠ frauds/transactions")
    record = example("calibration")
    record["diagnostics"][0]["reliability"].pop()
    assert_rejected(record, "bin transactions do not sum to the diagnostic's rows")
    record = example("calibration")
    record["diagnostics"][0]["reliability"][1]["bin"] = 3
    assert_rejected(record, "reliability: bins numbered 1, 2, ... in order")


def test_latency_diagnostics_cover_v2_and_fitting_counts_match() -> None:
    short = _latency_calibration()
    held_out = short["diagnostics"][1]
    held_out |= {"rows": held_out["rows"] - 1, "reliability": bins(held_out["rows"] - 1, 638)}
    assert_rejected(short, "fitting and held-out rows cover 77822")
    record = example("calibration")
    record["diagnostics"][0] |= {"frauds": 2639, "reliability": bins(V2_ROWS, 2639)}
    assert_rejected(record, "in-sample diagnostics use exactly the fitting rows")


def test_calibration_needs_both_classes() -> None:
    record = example("calibration") | {"frauds": 0, "legitimate": 77_822}
    assert_rejected(record, "both classes are required")


# --- R1 and H3 selections ------------------------------------------------------------------------


def test_r1_counts_and_f1_terms_are_consistent() -> None:
    assert_rejected(example("r1_selection") | {"f1_numerator": 2401}, "must be 2 TP")
    assert_rejected(example("r1_selection") | {"f1_denominator": 4000}, "must be P + A")
    assert_rejected(example("r1_selection") | {"alerts": 2101}, "≠ alerts")
    assert_rejected(example("r1_selection") | {"false_negatives": 1}, "≠ frauds")


def test_r1_threshold_round_trips_at_full_precision() -> None:
    record = example("r1_selection") | {"threshold": 0.1 + 0.2}
    assert loads(dump(record), "r1")["threshold"] == 0.1 + 0.2


def test_r1_errors_are_recorded_without_a_threshold() -> None:
    selection = dict.fromkeys(
        (
            "distinct_scores",
            "threshold",
            "true_positives",
            "false_positives",
            "false_negatives",
            "alerts",
            "f1_numerator",
            "f1_denominator",
        )
    )
    no_fraud = example("r1_selection") | selection | {"status": "error", "frauds": 0}
    validate(no_fraud | {"error": "no_fraud"})
    assert_rejected(no_fraud, "r1_selection.error: required")
    assert_rejected(no_fraud | {"error": "empty_selection_set"}, "is not empty")
    empty = no_fraud | {"error": "empty_selection_set", "transactions": 0}
    assert_rejected(empty, f"a full-V2 selection uses all {V2_ROWS} V2 rows")
    validate(empty | {"period": "latency", "selection_days": {"first": 127, "last": 147}})
    assert_rejected(example("r1_selection") | {"status": "error", "error": "no_fraud"}, "null")


def _h3(captured: dict[str, int], ap: dict[str, float] | None = None) -> dict[str, Any]:
    record = example("h3_selection")
    for candidate in record["candidates"]:
        name = candidate["configuration"]
        if name in captured:
            candidate["frauds_captured"] = captured[name]
            candidate["recall"] = captured[name] / V2_FRAUDS
        if ap is not None and name in ap:
            candidate["average_precision"] = ap[name]
    return record


def test_h3_selects_the_most_frauds_captured() -> None:
    assert_rejected(example("h3_selection") | {"selected": "LGB-b"}, "the rule selects Cat-u")


def test_h3_ties_go_to_average_precision_then_the_fixed_order() -> None:
    by_ap = _h3({"LGB-b": 920}, {"LGB-b": 0.6})
    validate(by_ap | {"selected": "LGB-b", "tie_break_steps": ["average_precision"]})
    assert_rejected(by_ap | {"selected": "LGB-b"}, "the rule used ['average_precision']")
    fixed = _h3({"LGB-b": 920})  # equal Average Precision too: Cat-u precedes LGB-b
    steps = ["average_precision", "fixed_order"]
    validate(fixed | {"tie_break_steps": steps})
    assert_rejected(fixed | {"selected": "LGB-b", "tie_break_steps": steps}, "selects Cat-u")


def test_h3_candidates_are_the_eight_in_dr011_order() -> None:
    record = example("h3_selection")
    record["candidates"].reverse()
    assert_rejected(record, "are not the eight in DR-011 order")
    assert [c["configuration"] for c in example("h3_selection")["candidates"]] == list(
        CONFIGURATION_NAMES
    )


def test_only_the_primary_h3_selection_is_binding() -> None:
    assert_rejected(example("h3_selection") | {"period": "latency"}, "only the primary")
    latency = example("h3_selection") | {"period": "latency", "binding": False}
    assert_rejected(latency, "a label-latency fit starts on day 127 and ends early")
    validate(latency | {"selection_days": {"first": 127, "last": 147}})


def test_h3_recall_matches_the_counts() -> None:
    record = example("h3_selection")
    record["candidates"][0]["recall"] = 0.5
    assert_rejected(record, "LGB-u recall ≠ captured / frauds")


def test_run_end_error_follows_the_outcome() -> None:
    assert_rejected(example("run_end") | {"error": "boom"}, "run_end.error: must be null")
    assert_rejected(example("run_end") | {"outcome": "failed"}, "run_end.error: required")
    validate(example("run_end") | {"outcome": "interrupted", "error": "KeyboardInterrupt"})


def test_run_end_lists_records_in_path_order() -> None:
    records = [file("records/b.json"), file("records/a.json")]
    assert_rejected(example("run_end") | {"records": records}, "increasing path order")


def test_run_end_paths_are_unique_and_name_other_records() -> None:
    first, second = file("records/a.json"), file("records/a.json") | {"sha256": "b" * 64}
    assert_rejected(example("run_end") | {"records": [first, second]}, "listed once")
    assert_rejected(
        example("run_end") | {"records": [file("records/run_end.json")]}, "not another record"
    )
    assert_rejected(example("run_end") | {"records": [file("scores/a.parquet")]}, "not another")


@pytest.mark.parametrize(
    "path",
    ["records/scores.parquet", "records/nested/a.json", "records/.hidden", "records/a b.json"],
)
def test_run_end_names_flat_json_record_files_only(path: str) -> None:
    assert_rejected(example("run_end") | {"records": [file(path)]}, "not another record")


def scoring_example() -> dict[str, Any]:
    base = example("timing") | {
        "measurement": "scoring",
        "boundary": "end_to_end",
        "partition": "V2",
        "batch_rows": 4096,
        "latency_ms": None,
        "rows": V2_ROWS,
        "seconds": 2.0,
        "rows_per_second": V2_ROWS / 2.0,
        "warmup": 3,
        "repetitions": 3,
        "rows_sha256": None,
        "row_selection": None,
    }
    base["settings"] = {
        "procedure_version": 1,
        "session": {"load_average_1m": 0.25},
        "memory_problems": [],
        "pass_seconds": [2.5, 2.0, 1.9],
        "pass_min_seconds": 1.9,
        "pass_max_seconds": 2.5,
        "host_memory_method": "clear_refs",
    }
    return base


def search_example() -> dict[str, Any]:
    search = example("timing") | {
        "configuration": "LGB-u",
        "measurement": "search",
        "model_role": None,
        "latency_ms": None,
        "boundary": None,
        "seconds": 3600.0,
        "device": "cpu",
        "threads": 24,
        "cpu_affinity": [0, 1],
        "batch_rows": None,
        "warmup": None,
        "repetitions": None,
        "rows": None,
        "rows_sha256": None,
        "row_selection": None,
        "peak_host_mib": None,
        "memory_scope": None,
    }
    search["settings"] = {
        "procedure_version": 1,
        "session": {"load_average_1m": 0.25},
        "memory_problems": [],
        "trials_completed": 10,
        "trials_declared": 10,
        "interrupted": False,
    }
    return search


def test_a_scoring_time_covers_the_whole_partition() -> None:
    base = scoring_example()
    validate(base)
    assert_rejected(base | {"rows": 63_000, "rows_per_second": 63_000 / 2.0}, "covers all")
    v1_rows = PARTITIONS.v1.rows
    v1 = base | {"partition": "V1", "rows": v1_rows, "rows_per_second": v1_rows / 2.0}
    validate(v1)
    assert_rejected(v1 | {"rows": V2_ROWS, "rows_per_second": V2_ROWS / 2.0}, "covers all")


def test_throughput_is_the_rows_over_the_seconds() -> None:
    base = scoring_example()
    assert_rejected(base | {"rows_per_second": 1.0}, "rows divided by the seconds")
    assert_rejected(base | {"seconds": 0.0, "rows_per_second": 5.0}, "rows divided by the seconds")


def test_a_search_is_complete_or_marked_partial() -> None:
    search = search_example()
    validate(search)
    settings = search["settings"]
    short = search | {"settings": {**settings, "trials_completed": 4}}
    assert_rejected(short, "completed every declared trial")
    partial = {**settings, "trials_completed": 4, "interrupted": True}
    validate(search | {"settings": partial})
    overdone = {**settings, "trials_completed": 10, "interrupted": True}
    assert_rejected(search | {"settings": overdone}, "an interrupted search is partial")


def test_the_session_conditions_are_recorded_as_context() -> None:
    timing = example("timing")
    assert "session_confirmed" not in timing["settings"]
    missing = {k: v for k, v in timing["settings"].items() if k != "session"}
    assert_rejected(timing | {"settings": missing}, "timing.settings.session: required")


GPU_MEMORY = {
    "sampled_max_mib": 5200,
    "baseline_mib": 1200,
    "increase_above_baseline_mib": 4000,
    "samples": 40,
    "interval_ms": 500,
}


def test_reported_gpu_memory_carries_its_baseline_and_sampling() -> None:
    timing = example("timing") | {"peak_gpu_mib": 5200}
    with_memory = {**timing["settings"], "gpu_memory": GPU_MEMORY}
    validate(timing | {"settings": with_memory})
    assert_rejected(timing, "timing.settings.gpu_memory: required")
    wrong_max = {**with_memory, "gpu_memory": {**GPU_MEMORY, "sampled_max_mib": 5100}}
    assert_rejected(timing | {"settings": wrong_max}, "must equal peak_gpu_mib")
    wrong_gain = {**with_memory, "gpu_memory": {**GPU_MEMORY, "increase_above_baseline_mib": 1}}
    assert_rejected(timing | {"settings": wrong_gain}, "the increase is the maximum minus baseline")
    no_samples = {**with_memory, "gpu_memory": {**GPU_MEMORY, "samples": 0}}
    assert_rejected(timing | {"settings": no_samples}, "gpu_memory.samples: required")
    unknown_base = {**with_memory, "gpu_memory": {**GPU_MEMORY, "baseline_mib": None}}
    validate(timing | {"settings": unknown_base})


def test_sampled_host_memory_names_its_interval() -> None:
    timing = example("timing")
    sampled = {**timing["settings"], "host_memory_method": "sampled"}
    assert_rejected(timing | {"settings": sampled}, "host_sample_interval_ms: required")
    validate(timing | {"settings": {**sampled, "host_sample_interval_ms": 100}})


def test_timing_names_the_memory_components_it_could_not_measure() -> None:
    timing = example("timing")
    validate(timing)
    broken = {k: v for k, v in timing["settings"].items() if k != "memory_problems"}
    assert_rejected(timing | {"settings": broken}, "timing.settings.memory_problems: required")
    assert_rejected(
        timing | {"settings": {**timing["settings"], "memory_problems": ["", 3]}},
        "memory_problems: must be a list",
    )
    listed = {**timing["settings"], "memory_problems": ["GPU memory not measured: stalled"]}
    validate(timing | {"settings": listed})


def _scoring_timing() -> dict[str, Any]:
    base = example("timing") | {"measurement": "scoring", "boundary": "end_to_end"}
    base |= {"partition": "V2", "batch_rows": 4096, "latency_ms": None, "rows": V2_ROWS}
    base |= {"seconds": 2.0, "rows_per_second": V2_ROWS / 2.0, "warmup": 3, "repetitions": 3}
    base |= {"rows_sha256": None, "row_selection": None}
    base["settings"] = {
        "procedure_version": 1,
        "session": {"load_average_1m": 0.25},
        "memory_problems": [],
        "pass_seconds": [2.5, 2.0, 1.9],
        "pass_min_seconds": 1.9,
        "pass_max_seconds": 2.5,
        "host_memory_method": "clear_refs",
    }
    return base


def test_a_scoring_timing_states_the_range_of_its_passes() -> None:
    scoring = _scoring_timing()
    validate(scoring)
    for name in ("pass_min_seconds", "pass_max_seconds"):
        broken = {k: v for k, v in scoring["settings"].items() if k != name}
        assert_rejected(scoring | {"settings": broken}, f"timing.settings.{name}: required")
    wrong = {**scoring["settings"], "pass_min_seconds": 1.0}
    assert_rejected(scoring | {"settings": wrong}, "pass_min_seconds: must be the min")


def test_a_boolean_is_not_a_pass_range_in_seconds() -> None:
    scoring = _scoring_timing()
    scoring["settings"] = {
        **scoring["settings"],
        "pass_seconds": [1.0, 2.0, 3.0],
        "pass_min_seconds": True,
    }
    assert_rejected(scoring | {"seconds": 2.0}, "pass_min_seconds: must be the min")


def test_the_startup_status_is_known_and_consistent() -> None:
    fit = example("timing") | FIT | {"device": "gpu"}
    validate(fit)
    settings = fit["settings"]
    for bad in ("later", 1):
        record = fit | {"settings": {**settings, "startup_status": bad}}
        assert_rejected(record, "startup_status: must be")
    measured = {**settings, "startup_seconds": None, "startup_status": "measured"}
    assert_rejected(fit | {"settings": measured}, "required when measured")
    unmeasured = {**settings, "startup_seconds": None, "startup_status": "not_measured"}
    validate(fit | {"settings": unmeasured})
