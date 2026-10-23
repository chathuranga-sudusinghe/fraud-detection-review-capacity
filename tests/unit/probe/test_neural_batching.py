"""The neural batching acceptance (probe specification, Section 9) end to end on the CPU with small
synthetic inputs, plus its comparison functions and its command. A CPU run exercises the code
paths; it is never acceptance evidence. No research data."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from review_capacity.cli import _batching_output, _log_file, build_parser
from review_capacity.config import ModelSettings, load_model_settings
from review_capacity.errors import ModelError, SpecificationError
from review_capacity.models import neural_training
from review_capacity.models.registry import Weighting
from review_capacity.probe import neural_batching
from review_capacity.probe.neural_batching import (
    CHECKS,
    GradientCase,
    Workload,
    classify_warnings,
    corner,
    corner_candidates,
    evidence_blockers,
    gradient_case,
    gradient_difference,
    run_acceptance,
    upper,
)
from review_capacity.probe.records import CheckBook
from review_capacity.probe.runner import WarningCapture
from review_capacity.training.declaration import ModelSpace, TrainingDeclaration, load_declaration
from review_capacity.training.preflight import Reading, Watch, synthetic_data
from support.neural import LAYOUT, preserved_torch_state
from support.synthetic import REPO_PATHS

ROOT = REPO_PATHS.root
PACKAGE = (ROOT / "src" / "review_capacity").as_posix()
CLEAN = {
    "git_uncommitted_changes": False,
    "git_commit": "0" * 40,
    "package": {"location": PACKAGE},
}
TOTAL_MIB = 8_000
AMPLE_MIB = 3_000
TINY = Workload(
    cases=(
        GradientCase("equal micro-batches", 1, 24, 32, 8, Weighting.UNWEIGHTED),
        GradientCase("unequal micro-batches, final batch", 2, 24, 29, 12, Weighting.BALANCED),
    ),
    heads=(3,),
)


@pytest.fixture(autouse=True)
def restore_torch_state() -> Iterator[None]:
    with preserved_torch_state():
        yield


@pytest.fixture(scope="module")
def settings() -> ModelSettings:
    return load_model_settings(REPO_PATHS)


@pytest.fixture(scope="module")
def real_declaration() -> TrainingDeclaration:
    declaration, _ = load_declaration(REPO_PATHS)
    return declaration


@pytest.fixture(scope="module")
def declaration(real_declaration: TrainingDeclaration) -> TrainingDeclaration:
    """The real declaration with small batches and small upper bounds, so that a CPU run takes
    seconds."""
    small = replace(
        real_declaration.neural,
        effective_batch=32,
        micro_batch_candidates=(32, 16, 8),
        evaluation_batch={"resnet": 64, "ft_transformer": 8},
    )
    smaller: dict[str, ModelSpace] = {}
    caps: dict[str, dict[str, int | float]] = {
        "resnet": {"n_layers": 2, "d": 32, "d_hidden_factor": 2.0, "d_embedding": 8},
        "ft_transformer": {"n_layers": 2, "d_token": 4},  # scaled by 8: token width 32
    }
    for model, space in real_declaration.models.items():
        high = caps.get(model, {})
        smaller[model] = replace(
            space,
            parameters=tuple(
                replace(p, high=high[p.name]) if p.name in high else p for p in space.parameters
            ),
        )
    return replace(real_declaration, neural=small, models=smaller)


class Ample:
    """A memory meter that always reports ample free memory (or ``free`` MiB)."""

    def __init__(self, free: int = AMPLE_MIB) -> None:
        self.free = free

    @contextmanager
    def watch(self) -> Iterator[Watch]:
        watch = Watch()
        yield watch
        watch.reading = Reading(TOTAL_MIB, 1_000, TOTAL_MIB - self.free, (), 10)


# --- comparison and corners --------------------------------------------------------------------


def test_identical_gradients_have_no_difference() -> None:
    gradients = {"a": np.array([1.0, -2.0]), "b": np.array([[0.5, 0.25]])}
    result = gradient_difference(gradients, {k: v.copy() for k, v in gradients.items()})
    assert result["relative_l2"] == 0.0
    assert result["largest_tensor_relative_l2"] == 0.0
    assert result["cosine"] == pytest.approx(1.0)
    assert result["finite"] is True


def test_a_scaled_gradient_is_detected_by_its_relative_norm() -> None:
    reference = {"a": np.array([3.0, 4.0])}
    result = gradient_difference(reference, {"a": 2.0 * reference["a"]})
    assert result["relative_l2"] == pytest.approx(1.0)
    assert result["cosine"] == pytest.approx(1.0)


def test_gradients_over_other_parameters_are_not_comparable_and_a_zero_reference_is_refused() -> (
    None
):
    assert gradient_difference({"a": np.ones(2)}, {"b": np.ones(2)}) == {"same_parameters": False}
    with pytest.raises(ModelError, match="zero"):
        gradient_difference({"a": np.zeros(2)}, {"a": np.ones(2)})


def test_the_corners_are_the_largest_models_and_the_padded_widths(
    real_declaration: TrainingDeclaration,
) -> None:
    corners = corner_candidates(real_declaration, Workload())
    assert [model for model, _, _ in corners] == ["resnet", *["ft_transformer"] * 5]
    resnet = corners[0][2]
    assert (resnet["n_layers"], resnet["d"], resnet["d_hidden_factor"]) == (16, 1024, 4.0)
    assert resnet["d_embedding"] == 512
    widths = [candidate["d_token"] for model, _, candidate in corners[1:]]
    assert widths == [72, 200, 456, 504, 512]
    assert {candidate["n_layers"] for _, _, candidate in corners[1:]} == {6}
    space = real_declaration.space("ft_transformer")
    assert upper(space, "d_token") == 512
    with pytest.raises(SpecificationError, match="not searched"):
        corner(space, n_heads=8)


def test_only_the_specified_gpu_run_from_a_clean_tree_is_acceptance_evidence(
    real_declaration: TrainingDeclaration,
) -> None:
    assert evidence_blockers("gpu", Workload(), real_declaration, CLEAN, ROOT) == []
    assert evidence_blockers("cpu", Workload(), real_declaration, CLEAN, ROOT) == [
        "device cpu, not gpu"
    ]
    assert evidence_blockers("gpu", TINY, real_declaration, CLEAN, ROOT)
    dirty = {**CLEAN, "git_uncommitted_changes": True}
    assert evidence_blockers("gpu", Workload(), real_declaration, dirty, ROOT)
    assert evidence_blockers("gpu", Workload(), real_declaration, None, ROOT)
    elsewhere = {**CLEAN, "package": {"location": "/elsewhere/review_capacity"}}
    assert evidence_blockers("gpu", Workload(), real_declaration, elsewhere, ROOT)
    small = replace(real_declaration, neural=replace(real_declaration.neural, effective_batch=512))
    assert evidence_blockers("gpu", Workload(), small, CLEAN, ROOT)


# --- B1 ------------------------------------------------------------------------------------------


def test_accumulated_gradients_equal_the_unsplit_gradient(
    declaration: TrainingDeclaration, settings: ModelSettings
) -> None:
    data = synthetic_data(LAYOUT, 32, np.random.default_rng(0))
    for case in TINY.cases:
        details = gradient_case(case, declaration, settings, LAYOUT, data, "cpu")
        assert details["passes"], details
        assert details["unsplit_repeats_bit_for_bit"] is True
        assert details["micro_sizes"] == _micro_sizes(case)
        assert details["relative_l2"] <= neural_batching.GRADIENT_TOLERANCE
        assert details["tensors"] > 0


def _micro_sizes(case: GradientCase) -> list[int]:
    full, rest = divmod(case.rows, case.micro)
    return [case.micro] * full + ([rest] if rest else [])


def test_a_wrong_loss_reduction_is_detected(
    declaration: TrainingDeclaration, settings: ModelSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mean per micro-batch instead of the sum over the effective batch's rows changes the
    accumulated gradient by a factor of the number of micro-batches."""
    original = neural_training.weighted_bce

    def per_micro_mean(logits: Any, labels: Any, weights: Any, **_ignored: Any) -> Any:
        return original(logits, labels, weights)

    monkeypatch.setattr(neural_training, "weighted_bce", per_micro_mean)
    data = synthetic_data(LAYOUT, 32, np.random.default_rng(0))
    details = gradient_case(TINY.cases[0], declaration, settings, LAYOUT, data, "cpu")
    assert details["passes"] is False
    assert details["relative_l2"] > neural_batching.GRADIENT_TOLERANCE


def test_a_loss_that_ignores_the_weights_is_caught_by_the_independent_oracle(
    declaration: TrainingDeclaration, settings: ModelSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both accumulation and the unsplit reference use the same loss function, so a shared defect
    passes the comparison between them; the oracle does not share it."""
    original = neural_training.weighted_bce

    def unit_weights(logits: Any, labels: Any, weights: Any, **kwargs: Any) -> Any:
        return original(logits, labels, weights.new_ones(weights.shape), **kwargs)

    monkeypatch.setattr(neural_training, "weighted_bce", unit_weights)
    data = synthetic_data(LAYOUT, 32, np.random.default_rng(0))
    details = gradient_case(TINY.cases[1], declaration, settings, LAYOUT, data, "cpu")
    assert details["passes"] is False
    assert details["oracle_relative_l2"] > neural_batching.ORACLE_TOLERANCE


def test_a_wrong_tail_denominator_is_caught_by_the_oracle(
    declaration: TrainingDeclaration, settings: ModelSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A denominator one row too large is a shared, small (3 %) error."""
    original = neural_training.weighted_bce

    def one_row_too_many(logits: Any, labels: Any, weights: Any, rows: int | None = None) -> Any:
        return original(logits, labels, weights, rows=None if rows is None else rows + 1)

    monkeypatch.setattr(neural_training, "weighted_bce", one_row_too_many)
    data = synthetic_data(LAYOUT, 32, np.random.default_rng(0))
    details = gradient_case(TINY.cases[1], declaration, settings, LAYOUT, data, "cpu")
    assert details["passes"] is False
    assert details["oracle_relative_l2"] > neural_batching.ORACLE_TOLERANCE


# --- the run -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def acceptance(
    tmp_path_factory: pytest.TempPathFactory,
    declaration: TrainingDeclaration,
) -> tuple[dict[str, Any], Path]:
    output = tmp_path_factory.mktemp("batching") / "run"
    with preserved_torch_state():
        record = run_acceptance(
            REPO_PATHS,
            output,
            device="cpu",
            workload=TINY,
            declaration=declaration,
            layout=LAYOUT,
            meter=Ample(),
        )
    return record, output


def test_every_check_runs_and_the_deterministic_ones_pass(
    acceptance: tuple[dict[str, Any], Path],
) -> None:
    record, _ = acceptance
    assert list(record["checks"]) == list(CHECKS)
    for check in ("B1", "B2", "B3", "B5"):
        assert record["checks"][check]["status"] == "pass", record["checks"][check]
    assert record["checks"]["B4"]["status"] in ("pass", "review")
    assert record["overall"] in ("pass", "review")
    assert record["run"]["stage"] == "finished"
    assert record["run"]["error"] is None


def test_a_cpu_run_is_not_acceptance_evidence(acceptance: tuple[dict[str, Any], Path]) -> None:
    record, output = acceptance
    assert record["run"]["acceptance_evidence"] is False
    assert any("device cpu" in blocker for blocker in record["run"]["evidence_blockers"])
    summary = (output / "batching.md").read_text(encoding="utf-8")
    assert "This run is not acceptance evidence: " in summary
    assert "device cpu, not gpu" in summary


def test_the_record_holds_every_item_of_every_check(
    acceptance: tuple[dict[str, Any], Path],
) -> None:
    record, output = acceptance
    checks = record["checks"]
    assert [item["subject"] for item in checks["B1"]["items"]] == [c.name for c in TINY.cases]
    assert [item["subject"] for item in checks["B2"]["items"]] == ["resnet", "ft_transformer"]
    assert len(checks["B3"]["items"]) == 1 + 1 + 1  # ResNet, one padded width, the largest width
    assert (output / "batching.json").is_file()
    summary = (output / "batching.md").read_text(encoding="utf-8")
    for item in checks["B1"]["items"]:
        assert item["subject"] in summary
    assert "## Symbols and notation" in summary


def test_a_corner_that_fits_no_micro_batch_is_a_recorded_failure(
    declaration: TrainingDeclaration, settings: ModelSettings
) -> None:
    book = CheckBook(CHECKS)
    neural_batching._corner_item(book, TINY, declaration, settings, LAYOUT, "cpu", Ample(free=100))
    items = book.items["B3"]
    assert items
    assert all(item["outcome"] is False for item in items)
    assert all("passes at no listed micro-batch" in item["error"] for item in items)
    assert book.status("B3") == "fail"


def test_warnings_are_classified_like_the_other_acceptances() -> None:
    capture = WarningCapture()
    capture.records = [
        {"logger": "x", "phase": "B1", "text": "option ignored"},
        {"logger": "x", "phase": "B3", "text": "something else"},
    ]
    book = CheckBook(CHECKS)
    classify_warnings(capture, book)
    assert [item["outcome"] for item in book.items["B4"]] == [False, "review"]
    empty = CheckBook(CHECKS)
    classify_warnings(WarningCapture(), empty)
    assert empty.status("B4") == "pass"


def test_an_out_of_memory_attempt_failure_line_is_accepted_once_per_recorded_attempt() -> None:
    line = "Fit FTT-u on gpu (2048 Train rows): failed after 0.3 s"
    capture = WarningCapture()
    capture.records = [{"logger": "x", "phase": "B3", "text": line}] * 2
    book = CheckBook(CHECKS)
    book.add("B3", "corner", True, attempts=[{"completed": False}, {"completed": True}])
    classify_warnings(capture, book)
    assert [item["outcome"] for item in book.items["B4"]] == [True, "review"]


# --- the command ---------------------------------------------------------------------------------


def test_command_is_registered_and_logs_in_its_output_directory() -> None:
    args = build_parser().parse_args(["probe-neural-batching", "--output", "evidence/x"])
    assert args.output == Path("evidence/x")
    assert _log_file(args) == ROOT / "evidence" / "x" / "batching.log"
    args = build_parser().parse_args(["probe-neural-batching"])
    log_file = _log_file(args)
    assert log_file is not None
    assert log_file.parent.parent == ROOT / "outputs" / "neural_batching_acceptance"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{6}Z", log_file.parent.name)
    assert _batching_output(args, REPO_PATHS) == log_file.parent  # one directory per run
    args = build_parser().parse_args(["--log-file", "a.log", "probe-neural-batching"])
    with pytest.raises(SpecificationError, match="omit --log-file"):
        _log_file(args)


def test_the_command_refuses_a_directory_with_an_earlier_run(tmp_path: Path) -> None:
    (tmp_path / "batching.log").write_text("earlier", encoding="utf-8")
    args = build_parser().parse_args(["probe-neural-batching", "--output", str(tmp_path)])
    with pytest.raises(SpecificationError, match="is not empty"):
        _log_file(args)
