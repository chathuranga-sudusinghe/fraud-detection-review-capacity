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
        "calibration",
        "r1_selection",
        "h3_selection",
    ]
    for kind in KINDS:
        validate(example(kind), kind)


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
    assert_rejected(timing | {"cpu_affinity": None}, "timing.cpu_affinity: required")
    assert_rejected(timing | {"memory_scope": None}, "timing.memory_scope: required")
    assert_rejected(timing | {"model_role": None}, "timing.model_role: required")
    assert_rejected(timing | {"attention_backend": "math"}, "the model has no attention")
    transformer = timing | {"configuration": "FTT-u"}
    assert_rejected(transformer, "timing.attention_backend: required")
    validate(transformer | {"attention_backend": "math"})


def test_search_timing_names_no_single_model() -> None:
    search = example("timing") | {
        "measurement": "search",
        "model_role": None,
        "latency_ms": None,
        "boundary": None,
        "seconds": 3600.0,
        "study_device": True,
    }
    validate(search)
    assert_rejected(search | {"model_role": "selected"}, "timing.model_role: must be null")


def test_additional_cpu_timing_is_for_neural_scoring_only() -> None:
    assert_rejected(example("timing") | {"configuration": "LGB-u"}, "only neural models")
    fit = example("timing") | {"measurement": "fit", "boundary": None, "latency_ms": None}
    assert_rejected(fit | {"seconds": 1.0}, "covers scoring only")
    on_gpu = fit | {"seconds": 1.0, "study_device": True, "device": "gpu", "cpu_affinity": None}
    assert_rejected(on_gpu, "GPU work is synchronised")
    validate(on_gpu | {"synchronised": True})


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
