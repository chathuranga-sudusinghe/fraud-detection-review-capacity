"""Components of the GPU acceptance probe (``docs/gpu-acceptance-probe.md``): synthetic inputs,
score files and comparisons, the GPU memory sampler, records and the generated summary. No
research data; CPU only."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from review_capacity.config import load_feature_spec
from review_capacity.data.catalog import read_catalog
from review_capacity.errors import EnvironmentMismatchError, SpecificationError
from review_capacity.models.registry import Weighting
from review_capacity.probe import diagnostics
from review_capacity.probe.records import (
    CHECKS,
    CheckBook,
    check_output,
    summary_markdown,
    write_records,
)
from review_capacity.probe.runner import catboost_combinations
from review_capacity.probe.synthetic import (
    ProbeSizes,
    category_levels,
    data_generator,
    neural_data,
    neural_layout,
    top_k,
    tree_data,
)
from support.synthetic import REPO_PATHS

SIZES = ProbeSizes(
    train=400, valid=200, score=200, neural_train=256, neural_valid=128, neural_score=128
)
FEATURES = load_feature_spec(REPO_PATHS.features_spec)
LEVELS = category_levels(read_catalog(REPO_PATHS.feature_catalog), FEATURES)


# --- Synthetic inputs -----------------------------------------------------------------------------


def test_category_levels_are_the_documented_train_levels() -> None:
    assert len(LEVELS) == 49
    assert sum(LEVELS.values()) == 16_797
    assert (LEVELS["card1"], LEVELS["DeviceInfo"], LEVELS["ProductCD"]) == (11_870, 1_470, 5)


def test_tree_inputs_take_the_shape_of_the_study_inputs() -> None:
    data = tree_data(FEATURES, LEVELS, SIZES)
    train = data.train[Weighting.UNWEIGHTED]
    assert tuple(train.features.columns) == FEATURES.model_inputs
    assert len(train.features) == 400
    assert (len(data.valid[Weighting.UNWEIGHTED].features), len(data.score)) == (200, 200)
    assert train.in_time_order()
    assert set(np.unique(train.labels).tolist()) == {0, 1}
    assert bool((train.weights == 1).all())
    assert len(np.unique(data.train[Weighting.BALANCED].weights)) == 2
    for name in FEATURES.categorical:
        observed = set(train.features[name].tolist())
        unseen = {label for label in data.score[name].tolist() if label.startswith("unseen-")}
        assert not unseen & observed
        assert len(observed - {FEATURES.missing_label}) <= LEVELS[name]
    assert sum(data.unseen.values()) > 0
    assert data.diagnostics["score_unseen"] == data.unseen
    assert set(data.diagnostics["train_missing"]) == set(FEATURES.categorical)
    assert data.diagnostics["fraud_rows"]["train"] == int(train.labels.sum())
    assert tree_data(FEATURES, LEVELS, SIZES).sha256 == data.sha256  # the data seed fixes them
    assert tree_data(FEATURES, LEVELS, SIZES, data_generator(1)).sha256 != data.sha256


def test_one_generator_feeds_both_builders() -> None:
    """The neural inputs drawn after the tree inputs from one generator differ from those of a
    fresh generator: the stream continues (one data generator, Section 1)."""
    layout = neural_layout(FEATURES, LEVELS)
    rng = data_generator()
    tree_data(FEATURES, LEVELS, SIZES, rng)
    continued = neural_data(layout, SIZES, rng)
    assert continued.sha256 != neural_data(layout, SIZES).sha256


def test_neural_inputs_take_the_form_of_dr012() -> None:
    layout = neural_layout(FEATURES, LEVELS)
    assert (len(layout.numeric), len(layout.categorical)) == (708, 49)
    assert sum(layout.cardinalities) == 16_846
    assert set(layout.missing_indices) == {0}
    data = neural_data(layout, SIZES)
    train = data.train[Weighting.BALANCED]
    train.inputs.check(layout)
    data.score.check(layout)
    indicators = train.inputs.numeric[:, 384:]
    assert set(np.unique(indicators).tolist()) <= {0.0, 1.0}
    assert bool((data.score.categorical == 0).any())  # missing and unseen map to "__MISSING__"
    assert data.diagnostics["layout"] == layout.to_json()
    assert neural_data(layout, SIZES).sha256 == data.sha256


def test_top_k_rounds_up() -> None:
    assert (top_k(4096), top_k(100_000), top_k(200)) == (82, 2000, 4)


# --- Score files and comparisons ------------------------------------------------------------------


def test_bitwise_equality_compares_bytes() -> None:
    assert diagnostics.bitwise_equal(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert not diagnostics.bitwise_equal(np.array([0.0]), np.array([-0.0]))  # equal values
    assert not diagnostics.bitwise_equal(np.array([1.0]), np.array([1.0], dtype=np.float32))
    assert not diagnostics.bitwise_equal(np.zeros(2), np.zeros(3))
    first, second = np.array([0.5, 0.0]), np.array([0.5, -0.0])
    assert diagnostics.compare_scores(first, second, np.arange(2), 1)["bitwise_identical"] is False


def test_files_are_compared_by_their_bytes(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_bytes(b"x" * 3000)
    b.write_bytes(b"x" * 3000)
    c.write_bytes(b"x" * 2999 + b"y")
    assert diagnostics.files_identical(a, b, chunk=1024)
    assert not diagnostics.files_identical(a, c, chunk=1024)
    assert diagnostics.files_identical(a, a, chunk=7)


def test_score_files_round_trip_and_are_checked(tmp_path: Path) -> None:
    scores = np.array([0.1, 0.5, 0.5, 0.9])
    ids = np.array([4, 3, 2, 1], dtype=np.int64)
    record = diagnostics.save_scores(tmp_path, "LGB-u.default.gpu.fit1.score", scores, ids)
    assert record["path"] == "LGB-u.default.gpu.fit1.score.npz"
    assert (record["dtype"], record["shape"]) == ("<f8", [4])
    assert record["sha256"] == diagnostics.canonical_sha256(scores, "<f8")
    loaded, loaded_ids = diagnostics.load_scores(tmp_path, record)
    assert diagnostics.bitwise_equal(loaded, scores)
    assert diagnostics.bitwise_equal(loaded_ids, ids)
    with pytest.raises(EnvironmentMismatchError, match="does not match"):
        diagnostics.load_scores(tmp_path, {**record, "sha256": "0" * 64})


def test_ranking_breaks_ties_by_ascending_row_identifier() -> None:
    scores = np.array([0.5, 0.9, 0.5, 0.1])
    ids = np.array([30, 20, 10, 40], dtype=np.int64)
    assert ids[diagnostics.ranking(scores, ids)].tolist() == [20, 10, 30, 40]
    summary = diagnostics.top_k_summary(scores, ids, 2)
    assert summary["top_k_rows_with_tied_scores"] == 1  # row 10 shares 0.5 with row 30
    assert summary["cutoff_margin"] == 0.0


def test_score_comparison() -> None:
    ids = np.arange(5, dtype=np.int64)
    first = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    second = np.array([0.9, 0.6, 0.75, 0.2, 0.1])
    comparison = diagnostics.compare_scores(first, second, ids, 2)
    assert comparison["bitwise_identical"] is False
    assert comparison["max_abs_difference"] == pytest.approx(0.2)
    assert comparison["top_k_membership_changes"] == 1
    assert diagnostics.compare_scores(first, first.copy(), ids, 2)["bitwise_identical"] is True


def test_logit_comparison_applies_tolerance_t1() -> None:
    reference = np.array([0.0, 2.0, -3.0], dtype=np.float32)
    close = reference + np.float32(5e-5)
    far = reference + np.float32(2e-4)
    assert diagnostics.logit_comparison(close, reference, 1e-4)["passed"] is True
    result = diagnostics.logit_comparison(far, reference, 1e-4)
    assert result["passed"] is False
    assert result["max_abs_score_difference"] <= 0.25 * result["max_abs_logit_difference"] + 1e-12


def test_timing_synchronises_at_every_boundary() -> None:
    events: list[str] = []
    timing = diagnostics.timed_calls(lambda: events.append("call"), lambda: events.append("sync"))
    warm = ["call"] * diagnostics.WARM_UP_CALLS + ["sync"]
    assert events == warm + ["call", "sync"] * diagnostics.TIMED_REPETITIONS
    assert len(timing["seconds"]) == diagnostics.TIMED_REPETITIONS
    latency = diagnostics.latency_summary([0.001] * 99 + [0.1])
    assert latency["median_ms"] == pytest.approx(1.0)
    assert latency["p99_ms"] > latency["p95_ms"]
    assert diagnostics.host_peak_bytes() > 0


# --- GPU memory sampler (a stand-in program prints nvidia-smi's output format) --------------------


def _sampler(script: str, timeout: float = 5.0) -> diagnostics.GpuMemorySampler:
    return diagnostics.GpuMemorySampler((sys.executable, "-c", script), timeout)


LOOP = "import sys, time\nwhile True:\n    print('{}', flush=True)\n    time.sleep(0.05)\n"


def test_sampler_reads_valid_samples() -> None:
    sampler = _sampler(LOOP.format("512, 8151"))
    sampler.start()
    sampler.stop()
    snapshot = sampler.snapshot()
    assert snapshot["samples"] >= 1
    assert (snapshot["sampled_max_used_mib"], snapshot["total_mib"]) == (512, 8151)
    assert snapshot["baseline_used_mib"] == 512
    assert snapshot["problems"] == []


def test_sampler_stops_the_run_without_a_valid_sample() -> None:
    sampler = _sampler(LOOP.format("[N/A], [N/A]"), timeout=1.0)
    with pytest.raises(EnvironmentMismatchError, match="no valid sample"):
        sampler.start()


def test_sampler_records_an_early_end_and_error_output() -> None:
    script = "import sys\nprint('100, 8151', flush=True)\nprint('lost GPU', file=sys.stderr)\n"
    sampler = _sampler(script)
    sampler.start()
    sampler._thread.join(timeout=5)  # type: ignore[union-attr]
    sampler._process.wait(timeout=5)  # type: ignore[union-attr]
    sampler.stop()
    problems = sampler.snapshot()["problems"]
    assert any("ended early" in problem for problem in problems)
    assert any("lost GPU" in problem for problem in problems)


def test_sampler_records_a_stall() -> None:
    # Two samples, then silence while the program keeps running: monitoring that stalls during the
    # run is a problem, so C13 cannot pass on it.
    script = "import time\nfor _ in range(2):\n    print('100, 8151', flush=True)\ntime.sleep(60)\n"
    sampler = diagnostics.GpuMemorySampler((sys.executable, "-c", script), 5.0, stall_timeout=0.5)
    sampler.start()
    deadline = time.monotonic() + 5
    while sampler.snapshot()["samples"] < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(1.0)
    sampler.stop()
    snapshot = sampler.snapshot()
    assert snapshot["samples"] == 2
    assert snapshot["final_sample_age_seconds"] > 0.5
    assert any("sampling stalled" in problem for problem in snapshot["problems"])


def test_gpu_memory_sampling_can_be_disabled() -> None:
    with diagnostics.sampled_gpu_memory(False) as sampler:
        assert sampler is None


# --- Records --------------------------------------------------------------------------------------


def test_check_statuses_and_overall_result() -> None:
    book = CheckBook()
    assert book.status("C1") == "not run"
    book.add("C1", "a", True)
    book.add("C9", "a", None, k=4)
    book.add("C2", "a", "review", text="a warning")
    statuses = (book.status("C1"), book.status("C9"), book.status("C2"))
    assert statuses == ("pass", "descriptive", "review")
    assert book.overall(compared=True) == "incomplete"
    for check in CHECKS:
        book.add(check, "all", True)
    assert book.overall(compared=False) == "pending second run"  # never "pass" alone
    assert book.overall(compared=True) == "review"
    book.add("C4", "b", False)
    assert book.status("C4") == "fail"
    assert book.overall(compared=True) == "fail"
    with pytest.raises(ValueError, match="Unknown check"):
        book.add("C14", "a", True)
    with pytest.raises(ValueError, match="Unknown outcome"):
        book.add("C1", "a", "maybe")


def _record(book: CheckBook) -> dict[str, Any]:
    return {
        "run": {"started": "s", "finished": "f", "data_seed": 77, "model_seed": 42},
        "code": {"git_commit": "abc", "git_uncommitted_changes": False},
        "environment": {
            "gpu": {"name": "GPU", "driver": "1", "torch_cuda": "12.8"},
            "packages": {"torch": "2.11.0"},
        },
        "specification": {"sha256": "1" * 64},
        "settings": {"sha256": "2" * 64},
        "checks": book.to_json(),
        "fits": [{"subject": "LGB-u gpu", "device": "gpu", "kept": 10, "seconds": 1.5}],
        "gpu_memory": {
            "sampled_max_used_mib": 5000,
            "total_mib": 8151,
            "baseline_used_mib": 500,
            "samples": 9,
            "interval_ms": 500,
            "problems": [],
        },
        "comparison": None,
        "overall": book.overall(compared=True),
    }


def test_summary_is_generated_from_the_record(tmp_path: Path) -> None:
    book = CheckBook()
    for check in CHECKS:
        book.add(check, "all", True)
    book.add(
        "C8",
        "RES-u small",
        True,
        max_abs_logit_difference=3e-6,
        tolerance=1e-4,
        max_abs_score_difference=7e-7,
    )
    book.add("C2", "LGB-u gpu", "review", text="some warning")
    record = _record(book)
    output = tmp_path / "evidence" / "gpu_acceptance_probe" / "run-1"
    output.mkdir(parents=True)
    write_records(output, record, tmp_path / "docs" / "gpu-acceptance-probe.md")
    text = (output / "probe.md").read_text(encoding="utf-8")
    assert text.startswith("# Graphics processing unit (GPU) acceptance probe: run summary")
    assert "(../../../docs/gpu-acceptance-probe.md)" in text  # relative to the output directory
    assert "**Overall result: review.**" in text
    assert "| C8 | float32 evaluation path | pass |" in text
    assert "some warning" in text
    assert "| RES-u small | 3e-06 | 0.0001 | 7e-07 |" in text
    assert "sampled maximum 5000 MiB of 8151 MiB" in text
    assert text.rstrip().endswith("| SHA-256 | Secure Hash Algorithm, 256-bit |")
    assert text == summary_markdown(record, "../../../docs/gpu-acceptance-probe.md")
    assert json.loads((output / "probe.json").read_text(encoding="utf-8"))["overall"] == "review"


def test_catboost_combinations_are_counted_from_the_json_model() -> None:
    def element(index: int) -> dict[str, object]:
        return {"cat_feature_index": index, "combination_element": "cat_feature_value"}

    model = {
        "features_info": {
            "ctrs": [{"elements": [element(1)]}, {"elements": [element(1), element(4)]}]
        }
    }
    assert catboost_combinations(model) == 1
    assert catboost_combinations({}) == 0


def test_an_earlier_runs_log_marks_the_directory_as_used(tmp_path: Path) -> None:
    check_output(tmp_path, None)  # empty: free
    (tmp_path / "probe.log").write_text("earlier\n", encoding="utf-8")
    with pytest.raises(SpecificationError, match="already holds a probe run"):
        check_output(tmp_path, None)
    check_output(tmp_path, None, log_opened=True)  # this run's own log, opened by the command
