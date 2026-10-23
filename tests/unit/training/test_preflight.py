"""The micro-batch preflight of the neural models (DR-015 §7; Issue #151, Part 2), on small
synthetic tensors on the CPU with a scripted memory meter (no GPU, no project data)."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from review_capacity.config import ModelSettings, load_model_settings
from review_capacity.errors import (
    DataValidationError,
    FileAccessError,
    ModelError,
    SpecificationError,
)
from review_capacity.models.ft_transformer_model import FTTransformerModel
from review_capacity.models.neural_training import BatchPlan
from review_capacity.training.declaration import TrainingDeclaration, load_declaration
from review_capacity.training.preflight import (
    FILE,
    Reading,
    Watch,
    _phase,
    build_record,
    candidate_sha256,
    micro_batches,
    neural_hyperparameters,
    read_preflight,
    run_preflight,
    synthetic_data,
    write_preflight,
)
from review_capacity.training.sampler import Candidate
from support.neural import LAYOUT, preserved_torch_state
from support.synthetic import REPO_PATHS

TOTAL_MIB = 8_000
AMPLE_MIB = 3_000  # free memory of an ordinary attempt
MARGIN = 1_024
ARTIFACT = "a" * 64
RESNET: Candidate = {
    "n_layers": 1,
    "d": 16,
    "d_hidden_factor": 2.0,
    "hidden_dropout": 0.1,
    "residual_dropout": 0.05,
    "d_embedding": 4,
    "lr": 1e-3,
    "weight_decay": 0.0,
}
FTT: Candidate = {
    "n_layers": 1,
    "d_token": 16,
    "attention_dropout": 0.1,
    "ffn_dropout": 0.1,
    "lr": 1e-4,
    "weight_decay": 1e-5,
}
CANDIDATES: dict[str, list[Candidate]] = {
    "resnet": [RESNET, {**RESNET, "d": 24}],
    "ft_transformer": [FTT, {**FTT, "n_layers": 2}],
}


@pytest.fixture(autouse=True)
def restore_torch_state() -> Iterator[None]:
    with preserved_torch_state():
        yield


@pytest.fixture(scope="module")
def settings() -> ModelSettings:
    return load_model_settings(REPO_PATHS)


@pytest.fixture(scope="module")
def declaration() -> TrainingDeclaration:
    """The real declaration with small batches, so that a CPU run takes seconds."""
    real, _ = load_declaration(REPO_PATHS)
    small = replace(
        real.neural,
        effective_batch=32,
        micro_batch_candidates=(32, 16, 8),
        evaluation_batch={"resnet": 64, "ft_transformer": 8},
    )
    return replace(real, neural=small)


class Scripted:
    """A memory meter that reports the next scripted free memory (MiB) for each watched phase and
    ``AMPLE_MIB`` once the script is used up."""

    def __init__(self, *free: int, problems: tuple[str, ...] = ()) -> None:
        self.free = list(free)
        self.problems = problems
        self.phases = 0

    @contextmanager
    def watch(self) -> Iterator[Watch]:
        watch = Watch()
        yield watch
        free = self.free.pop(0) if self.free else AMPLE_MIB
        self.phases += 1
        watch.reading = Reading(TOTAL_MIB, 1_000, TOTAL_MIB - free, self.problems)


def run(
    declaration: TrainingDeclaration,
    settings: ModelSettings,
    meter: Scripted,
    models: tuple[str, ...] = ("resnet", "ft_transformer"),
) -> dict[str, Any]:
    return run_preflight(
        declaration,
        settings,
        LAYOUT,
        CANDIDATES,
        device="cpu",
        meter=meter,
        seed=42,
        models=models,
    )


# --- hyperparameters and micro-batches ---------------------------------------------------------


def test_the_hyperparameters_are_the_candidate_the_declared_fixed_values_and_the_batches(
    declaration: TrainingDeclaration,
) -> None:
    resnet = neural_hyperparameters(declaration, "resnet", RESNET)
    assert resnet == {**RESNET, "batch_size": 32, "eval_batch_size": 64}
    ftt = neural_hyperparameters(declaration, "ft_transformer", FTT)
    assert ftt["residual_dropout"] == 0.0  # fixed values the model takes as hyperparameters
    assert ftt["d_ffn_factor"] == 4 / 3
    assert (ftt["batch_size"], ftt["eval_batch_size"]) == (32, 8)
    assert "activation" not in ftt  # built into the model, checked by the declaration tests
    with pytest.raises(SpecificationError, match="fixed declaration values"):
        neural_hyperparameters(declaration, "ft_transformer", {**FTT, "d_ffn_factor": 2.0})


def test_the_resnet_is_tried_at_the_effective_batch_only() -> None:
    assert micro_batches("resnet", 1024, (1024, 512, 256, 128, 64)) == (1024,)
    assert micro_batches("ft_transformer", 1024, (1024, 512, 256, 128, 64)) == (
        1024,
        512,
        256,
        128,
        64,
    )
    assert micro_batches("ft_transformer", 32, (1024, 32, 16)) == (32, 16)


def test_the_synthetic_data_follow_the_layout_and_have_both_classes() -> None:
    data = synthetic_data(LAYOUT, 40, np.random.default_rng(0))
    data.inputs.check(LAYOUT)
    assert data.inputs.rows == 40
    assert set(data.labels.tolist()) == {0, 1}


# --- the procedure --------------------------------------------------------------------------------


def test_every_candidate_passes_at_the_largest_micro_batch_when_memory_is_ample(
    declaration: TrainingDeclaration, settings: ModelSettings
) -> None:
    models = run(declaration, settings, Scripted())
    for model in ("resnet", "ft_transformer"):
        assert models[model]["effective_batch"] == 32
        assert [item["trial"] for item in models[model]["candidates"]] == [1, 2]
        for item in models[model]["candidates"]:
            assert item["micro_batch"] == 32
            assert len(item["attempts"]) == 1
            assert item["attempts"][0]["training"]["passes"]
            assert item["evaluation"]["passes"]
    assert models["resnet"]["evaluation_batch"] == 64
    assert models["ft_transformer"]["evaluation_batch"] == 8


def test_a_smaller_micro_batch_is_used_when_the_margin_is_missed(
    declaration: TrainingDeclaration, settings: ModelSettings
) -> None:
    # candidate 1: 32 rows leave 500 MiB (fails), 16 leave 2,000 (passes), then the evaluation
    meter = Scripted(500, 2_000, AMPLE_MIB)
    models = run(declaration, settings, meter, models=("ft_transformer",))
    first = models["ft_transformer"]["candidates"][0]
    assert first["micro_batch"] == 16
    assert [a["micro_batch"] for a in first["attempts"]] == [32, 16]
    failed = first["attempts"][0]["training"]
    assert failed["completed"]
    assert failed["free_at_peak_mib"] == 500
    assert not failed["passes"]
    assert models["ft_transformer"]["candidates"][1]["micro_batch"] == 32


def test_the_resnet_does_not_fall_back_to_a_smaller_micro_batch(
    declaration: TrainingDeclaration, settings: ModelSettings
) -> None:
    with pytest.raises(ModelError, match="resnet candidate 1 passes at no listed micro-batch"):
        run(declaration, settings, Scripted(500), models=("resnet",))


def test_a_candidate_that_passes_nowhere_stops_the_search(
    declaration: TrainingDeclaration, settings: ModelSettings
) -> None:
    with pytest.raises(ModelError, match="passes at no listed micro-batch"):
        run(declaration, settings, Scripted(1, 1, 1), models=("ft_transformer",))


def test_a_failed_evaluation_stops_the_search(
    declaration: TrainingDeclaration, settings: ModelSettings
) -> None:
    # training passes, then the evaluation phase leaves only 100 MiB
    with pytest.raises(ModelError, match="float32 evaluation"):
        run(declaration, settings, Scripted(AMPLE_MIB, 100), models=("ft_transformer",))


def test_sampler_problems_make_an_attempt_fail() -> None:
    reading = Reading(TOTAL_MIB, 1_000, 2_000, ("nvidia-smi ended early",))
    phase = _phase(reading, None, MARGIN)
    assert phase["completed"]
    assert not phase["passes"]
    assert phase["problems"] == ["nvidia-smi ended early"]


def test_out_of_memory_is_a_failed_attempt_and_other_errors_propagate(
    monkeypatch: pytest.MonkeyPatch, declaration: TrainingDeclaration, settings: ModelSettings
) -> None:
    original: Callable[..., None] = FTTransformerModel.fit
    calls: list[int] = []

    def fit(self: FTTransformerModel, *args: Any, plan: BatchPlan, **kwargs: Any) -> None:
        calls.append(plan.micro)
        if plan.micro == 32:
            raise ModelError("out of GPU memory during training") from torch.cuda.OutOfMemoryError()
        original(self, *args, plan=plan, **kwargs)

    monkeypatch.setattr(FTTransformerModel, "fit", fit)
    models = run(declaration, settings, Scripted(), models=("ft_transformer",))
    first = models["ft_transformer"]["candidates"][0]
    assert first["micro_batch"] == 16
    assert first["attempts"][0]["training"]["error"] == "out of GPU memory during training"
    assert not first["attempts"][0]["training"]["completed"]

    def broken(_model: FTTransformerModel, *_args: Any, **_kwargs: Any) -> None:
        raise ModelError("the loss is not finite")

    monkeypatch.setattr(FTTransformerModel, "fit", broken)
    with pytest.raises(ModelError, match="loss is not finite"):
        run(declaration, settings, Scripted(), models=("ft_transformer",))


def test_the_random_states_are_unchanged(
    declaration: TrainingDeclaration, settings: ModelSettings
) -> None:
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    run(declaration, settings, Scripted())
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_the_preflight_is_repeatable(
    declaration: TrainingDeclaration, settings: ModelSettings
) -> None:
    first = run(declaration, settings, Scripted())
    second = run(declaration, settings, Scripted())
    plan = [[c["micro_batch"] for c in first[m]["candidates"]] for m in first]
    assert plan == [[c["micro_batch"] for c in second[m]["candidates"]] for m in second]


# --- the record and its reading ---------------------------------------------------------------


def recorded(
    tmp_path: Path, declaration: TrainingDeclaration, settings: ModelSettings
) -> tuple[Path, dict[str, Any]]:
    # two ResNet candidates (training and evaluation each), then FT-Transformer candidate 1:
    # 32 rows leave 500 MiB (fails), 16 leave 2,000 (passes)
    meter = Scripted(*[AMPLE_MIB] * 4, 500, 2_000, AMPLE_MIB)
    models = run(declaration, settings, meter)
    record = build_record(declaration, {"artifact_sha256": ARTIFACT}, models)
    return write_preflight(tmp_path, record), record


def read(path: Path, declaration: TrainingDeclaration, **changes: Any) -> Any:
    options: dict[str, Any] = {"artifact_sha256": ARTIFACT}
    options.update(changes)
    return read_preflight(path, declaration, CANDIDATES, **options)


def test_a_recorded_plan_is_read_back_as_batch_plans(
    tmp_path: Path, declaration: TrainingDeclaration, settings: ModelSettings
) -> None:
    path, _ = recorded(tmp_path, declaration, settings)
    assert path.name == FILE
    plans = read(path, declaration)
    assert plans["resnet"] == {1: BatchPlan(32, 32), 2: BatchPlan(32, 32)}
    assert plans["ft_transformer"] == {1: BatchPlan(32, 16), 2: BatchPlan(32, 32)}
    with pytest.raises(FileAccessError, match="never replaced"):
        write_preflight(tmp_path, json.loads(path.read_text(encoding="utf-8")))


def test_a_plan_for_other_inputs_is_refused(
    tmp_path: Path, declaration: TrainingDeclaration, settings: ModelSettings
) -> None:
    path, _ = recorded(tmp_path, declaration, settings)
    with pytest.raises(SpecificationError, match="another preprocessing artifact"):
        read(path, declaration, artifact_sha256="b" * 64)
    other = replace(declaration, neural=replace(declaration.neural, preflight_free_margin_mib=1))
    with pytest.raises(SpecificationError, match="margin or micro-batch"):
        read(path, other)
    changed = {**CANDIDATES, "resnet": [RESNET, {**RESNET, "d": 25}]}
    with pytest.raises(SpecificationError, match="other candidates"):
        read_preflight(path, declaration, changed, artifact_sha256=ARTIFACT)
    wider = replace(declaration, neural=replace(declaration.neural, effective_batch=64))
    with pytest.raises(SpecificationError, match="other batch sizes"):
        read(path, wider)


@pytest.mark.parametrize(
    "tamper",
    ["unlisted", "resnet_split", "not_passing", "evaluation", "missing_model", "format"],
)
def test_a_tampered_plan_is_refused(
    tmp_path: Path, declaration: TrainingDeclaration, settings: ModelSettings, tamper: str
) -> None:
    path, record = recorded(tmp_path, declaration, settings)
    ftt = record["models"]["ft_transformer"]["candidates"][0]
    resnet = record["models"]["resnet"]["candidates"][0]
    if tamper == "unlisted":
        ftt["micro_batch"] = 12
    elif tamper == "resnet_split":
        resnet["micro_batch"] = 16
    elif tamper == "not_passing":
        ftt["attempts"][-1]["training"]["passes"] = False
    elif tamper == "evaluation":
        ftt["evaluation"]["passes"] = False
    elif tamper == "missing_model":
        del record["models"]["ft_transformer"]
    else:
        record["format_version"] = 99
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises((SpecificationError, DataValidationError)):
        read(path, declaration)


def test_a_foreign_or_unreadable_file_is_a_project_error(
    tmp_path: Path, declaration: TrainingDeclaration
) -> None:
    with pytest.raises(FileAccessError):
        read(tmp_path / "absent.json", declaration)
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(DataValidationError, match="JSON"):
        read(bad, declaration)
    bad.write_text('{"record": "other"}', encoding="utf-8")
    with pytest.raises(DataValidationError, match="expected keys"):
        read(bad, declaration)


def test_the_candidate_hash_is_stable_and_distinguishes_candidates() -> None:
    assert candidate_sha256(RESNET) == candidate_sha256(dict(reversed(list(RESNET.items()))))
    assert candidate_sha256(RESNET) != candidate_sha256({**RESNET, "d": 24})


def test_the_command_line_needs_the_preprocessing_artifact() -> None:
    from review_capacity.cli import build_parser

    args = build_parser().parse_args(["preflight-neural", "--preprocessing", "outputs/x"])
    assert args.command == "preflight-neural"
    assert args.preprocessing == Path("outputs/x")
    assert args.output is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["preflight-neural"])
