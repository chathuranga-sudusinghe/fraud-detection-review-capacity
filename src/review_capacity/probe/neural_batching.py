"""Neural batching acceptance: checks B1–B5 of ``docs/gpu-acceptance-probe.md``, Section 9
(Issue #151, Part 2).

The micro-batch preflight (``training.preflight``) fixes a batch plan for each candidate; this
acceptance confirms, on synthetic inputs of the study's shape and before any neural development
run, that the plan machinery of DR-015 §7 does what it must:

* B1: the gradient accumulated over micro-batches equals the gradient of the unsplit effective
  batch (dropout disabled), for equal and unequal micro-batches, non-uniform weights, the final
  batch of 1,016 rows and padded attention heads;
* B2: a fit under a recorded plan repeats bit for bit with the same seed;
* B3: the corners of the search space (the largest ResNet-style MLP with its largest category
  embedding, and FT-Transformers of representative widths with padded heads) pass the preflight
  at some listed micro-batch, in training and in float32 evaluation, with the declared free
  memory.

``run_acceptance`` writes ``batching.json`` and the generated ``batching.md`` before the work,
after each step and on any failure. A GPU run of the default workload on a clean committed tree is
the acceptance run; any other run (the unit tests) runs the same code on small inputs and is marked
as not being acceptance evidence. There is no fallback: nothing is moved to the CPU, reduced or
skipped.
"""

from __future__ import annotations

import gc
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from review_capacity.config import Device, load_feature_spec
from review_capacity.data.catalog import read_catalog
from review_capacity.errors import FileAccessError, ModelError, SpecificationError
from review_capacity.files import write_atomically, write_json
from review_capacity.hashing import file_sha256
from review_capacity.logging_setup import log_step
from review_capacity.models.base import class_weights, sample_weights
from review_capacity.models.device import configure_torch, environment_record, seed_everything
from review_capacity.models.ft_transformer_model import FTTransformerModel
from review_capacity.models.neural_training import (
    BatchPlan,
    InputLayout,
    NeuralData,
    NeuralModel,
    accumulate_gradients,
    float32_default,
    torch_device,
)
from review_capacity.models.registry import Weighting, configuration
from review_capacity.models.resnet_model import ResNetModel
from review_capacity.models.run_record import jsonable, load_settings_with_snapshot
from review_capacity.paths import ProjectPaths
from review_capacity.probe import diagnostics
from review_capacity.probe.catboost_objective import _cell, _record_section, _scalar
from review_capacity.probe.records import (
    BATCHING_ENTRIES,
    BATCHING_LOG_NAME,
    CheckBook,
    check_acceptance_output,
)
from review_capacity.probe.runner import FAILING_WARNING, MODEL_SEED, captured_warnings
from review_capacity.probe.synthetic import DATA_SEED, category_levels, neural_layout
from review_capacity.provenance import code_provenance
from review_capacity.training.declaration import ModelSpace, TrainingDeclaration, load_declaration
from review_capacity.training.preflight import (
    GpuMeter,
    Meter,
    neural_hyperparameters,
    run_preflight,
    synthetic_data,
)
from review_capacity.training.sampler import Candidate

logger = logging.getLogger(__name__)

SPECIFICATION = "docs/gpu-acceptance-probe.md"
RECORD_NAME, SUMMARY_NAME = BATCHING_ENTRIES
SCHEMA_VERSION = 1
GRADIENT_TOLERANCE = 1e-2  # T5: relative L2 difference of the gradients (training precision)
ORACLE_TOLERANCE = 1e-4  # T6: the same against the independent float32 oracle
CHECKS: dict[str, str] = {
    "B1": "Gradient accumulation against the unsplit batch",
    "B2": "Same-seed repeatability under a micro-batch plan",
    "B3": "Memory at the corners of the search space",
    "B4": "Warnings",
    "B5": "Resources",
}


@dataclass(frozen=True)
class GradientCase:
    """One comparison of B1: an FT-Transformer of ``n_layers`` layers and token width ``d_token``,
    one effective batch of ``rows`` rows split into micro-batches of ``micro`` rows, with unit or
    balanced (non-uniform) weights."""

    name: str
    n_layers: int
    d_token: int
    rows: int
    micro: int
    weighting: Weighting


@dataclass(frozen=True)
class Workload:
    """The probe values of Section 9 [I]. The gradient cases use batches that the unsplit
    reference can process in 8 GB (the default width at 256 rows; the light architecture at the
    1,016 rows of the final batch of Train)."""

    cases: tuple[GradientCase, ...] = (
        GradientCase(
            "default width, equal micro-batches, unit weights",
            3,
            192,
            256,
            64,
            Weighting.UNWEIGHTED,
        ),
        GradientCase(
            "default width, unequal micro-batches, non-uniform weights",
            3,
            192,
            256,
            96,
            Weighting.BALANCED,
        ),
        GradientCase(
            "padded heads (25 to 32), unequal micro-batches",
            2,
            200,
            256,
            96,
            Weighting.BALANCED,
        ),
        GradientCase(
            "final batch of 1,016 rows, padded heads (9 to 16)",
            1,
            72,
            1016,
            256,
            Weighting.BALANCED,
        ),
        GradientCase(
            "final batch of 1,016 rows, unequal micro-batches",
            1,
            72,
            1016,
            384,
            Weighting.BALANCED,
        ),
    )
    repeat_micro: dict[str, int] | None = None  # micro-batch of B2 per model (None: the default)
    repeat_epochs: int = 2
    heads: tuple[int, ...] = (9, 25, 57, 63)  # head widths padded to 16, 32, 64 and 64 (B3)

    def micro_for(self, model: str, effective: int) -> int:
        """The micro-batch of B2: the effective batch for the ResNet-style MLP (it cannot
        accumulate), a quarter of it for the FT-Transformer."""
        if self.repeat_micro is not None:
            return self.repeat_micro[model]
        return effective if model == "resnet" else max(1, effective // 4)


def evidence_blockers(
    device: Device,
    workload: Workload,
    declaration: TrainingDeclaration,
    code: dict[str, Any] | None,
    root: Path,
) -> list[str]:
    """Why a run is not the acceptance run of Section 9 (empty for the acceptance run): the GPU,
    the default workload and effective batch, and a clean committed tree whose code is the code
    that runs (the running package lies inside the inspected checkout ``root``)."""
    blockers = []
    if device != "gpu":
        blockers.append(f"device {device}, not gpu")
    if workload != Workload():
        blockers.append("a workload other than the default of Section 9")
    if declaration.neural.effective_batch != 1024:
        blockers.append(f"effective batch {declaration.neural.effective_batch}, not 1024")
    code = code or {}
    if code.get("git_uncommitted_changes") is not False:
        blockers.append("the working tree is not shown to be clean and committed")
    if not code.get("git_commit"):
        blockers.append("the commit is not recorded")
    location = (code.get("package") or {}).get("location")
    if not location or not Path(location).resolve().is_relative_to(root.resolve()):
        blockers.append("the running package is not the inspected checkout's")
    return blockers


# --- B1: gradients -------------------------------------------------------------------------------


def corner(space: ModelSpace, **values: int | float) -> Candidate:
    """A candidate at the reference configuration of ``space`` with ``values`` overriding it."""
    candidate: Candidate = {p.name: p.reference for p in space.parameters}
    unknown = sorted(set(values) - set(candidate))
    if unknown:
        raise SpecificationError(f"{space.name}: {unknown} are not searched parameters")
    return {**candidate, **values}


def upper(space: ModelSpace, name: str) -> int | float:
    """The largest value of a searched parameter (scaled integers in model units)."""
    parameter = next(p for p in space.parameters if p.name == name)
    return parameter.high * (parameter.factor or 1)


def gradient_difference(
    reference: Mapping[str, npt.NDArray[np.float64]], other: Mapping[str, npt.NDArray[np.float64]]
) -> dict[str, Any]:
    """The difference of two gradient sets over the same parameters: the relative L2 norm of the
    whole difference (T5), the largest relative L2 difference of a single tensor, the cosine of
    the two gradients and whether every value is finite."""
    if set(reference) != set(other):
        return {"same_parameters": False}
    names = sorted(reference)
    first = np.concatenate([reference[name].ravel() for name in names])
    second = np.concatenate([other[name].ravel() for name in names])
    norm = float(np.linalg.norm(first))
    if norm == 0.0:
        raise ModelError("The reference gradient is zero: the comparison would be vacuous")
    tensors = [
        float(np.linalg.norm(other[name] - reference[name]) / np.linalg.norm(reference[name]))
        for name in names
        if np.linalg.norm(reference[name]) > 0.0
    ]
    return {
        "same_parameters": True,
        "tensors": len(names),
        "values": int(first.size),
        "finite": bool(np.isfinite(first).all() and np.isfinite(second).all()),
        "relative_l2": float(np.linalg.norm(second - first)) / norm,
        "largest_tensor_relative_l2": max(tensors) if tensors else None,
        "cosine": float(first @ second / (norm * np.linalg.norm(second))),
    }


def _weighted(data: NeuralData, weighting: Weighting) -> NeuralData:
    weights = sample_weights(data.labels, class_weights(data.labels, weighting))
    return replace(data, weights=weights)


def _release(device: Device) -> None:
    gc.collect()
    if device == "gpu":
        import torch

        torch.cuda.empty_cache()


def _case_gradients(
    case: GradientCase,
    declaration: TrainingDeclaration,
    settings: Any,
    layout: InputLayout,
    data: NeuralData,
    device: Device,
) -> tuple[Any, Any, Any, float, dict[str, Any]]:
    """The unsplit gradient twice and the accumulated gradient of one case, each with its loss,
    the weight of a fraud row, and the comparison of the float32 accumulated gradient with the
    independent float32 oracle. The network does not outlive the call."""
    import torch

    space = declaration.space("ft_transformer")
    candidate = corner(
        space,
        n_layers=case.n_layers,
        d_token=case.d_token,
        attention_dropout=0.0,
        ffn_dropout=0.0,
    )
    hyperparameters = neural_hyperparameters(declaration, "ft_transformer", candidate)
    model = FTTransformerModel(
        configuration("FTT-u"), MODEL_SEED, settings=settings, layout=layout, device=device
    )
    _, architecture = model._split(hyperparameters, early_stopping=False)
    configure_torch(settings.neural)
    seed_everything(MODEL_SEED)
    target = torch_device(device)
    network = model._new_network(architecture).to(target)
    network.train()
    rows = case.rows
    weighted = _weighted(data, case.weighting)
    index = np.arange(rows)
    categorical = torch.from_numpy(weighted.inputs.categorical[index])
    labels = weighted.labels.astype(np.float32)
    weights = weighted.weights.astype(np.float32)

    def collect() -> dict[str, npt.NDArray[np.float64]]:
        return {
            name: parameter.grad.detach().double().cpu().numpy()
            for name, parameter in network.named_parameters()
            if parameter.grad is not None
        }

    def oracle() -> dict[str, npt.NDArray[np.float64]]:
        """The gradient of the whole effective batch in float32, written independently of
        ``accumulate_gradients``: one forward pass over all rows, the weighted binary
        cross-entropy summed and divided by the rows of the batch (DR-015 §7)."""
        network.zero_grad(set_to_none=True)
        with float32_default():
            logits = network(
                torch.from_numpy(weighted.inputs.numeric[index]).to(target),
                categorical.to(target),
            )
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), torch.from_numpy(labels[index]).to(target), reduction="none"
            )
            total = (losses * torch.from_numpy(weights[index]).to(target)).sum() / rows
            total.backward()  # type: ignore[no-untyped-call]
        return collect()

    def gradients(
        micro: int, autocast: bool = True
    ) -> tuple[float, dict[str, npt.NDArray[np.float64]]]:
        network.zero_grad(set_to_none=True)
        with float32_default():
            loss = accumulate_gradients(
                network,
                weighted.inputs.numeric,
                categorical,
                labels,
                weights,
                index,
                BatchPlan(rows, micro),
                device=target,
                name="FTT-u",
                where=f"gradient case {case.name!r}",
                logits_dtypes=set(),
                autocast=autocast,
            )
        return float(loss.item()), collect()

    exact = gradients(case.micro, autocast=False)[1]
    return (
        gradients(rows),
        gradients(rows),
        gradients(case.micro),
        float(weights[labels == 1].max()),
        gradient_difference(oracle(), exact),
    )


def gradient_case(
    case: GradientCase,
    declaration: TrainingDeclaration,
    settings: Any,
    layout: InputLayout,
    data: NeuralData,
    device: Device,
) -> dict[str, Any]:
    """B1 for one case: the gradient of the unsplit effective batch (twice, to show that the
    reference repeats bit for bit) and of the same batch accumulated over micro-batches, on one
    network with dropout disabled."""
    rows = case.rows
    try:
        (
            (unsplit_loss, unsplit),
            (repeat_loss, repeat),
            (accumulated_loss, accumulated),
            fraud,
            exact,
        ) = _case_gradients(case, declaration, settings, layout, data, device)
    finally:
        _release(device)
    difference = gradient_difference(unsplit, accumulated)
    repeated = (
        unsplit_loss == repeat_loss
        and gradient_difference(unsplit, repeat).get("relative_l2") == 0.0
    )
    sizes = BatchPlan(rows, case.micro).micro_sizes(rows)
    passes = bool(
        difference.get("same_parameters")
        and difference.get("finite")
        and difference["relative_l2"] <= GRADIENT_TOLERANCE
        and repeated
        and exact.get("same_parameters")
        and exact.get("finite")
        and exact["relative_l2"] <= ORACLE_TOLERANCE
    )
    return {
        "passes": passes,
        "rows": rows,
        "micro_batch": case.micro,
        "micro_sizes": sizes,
        "weighting": case.weighting.value,
        "fraud_row_weight": fraud,
        "n_layers": case.n_layers,
        "d_token": case.d_token,
        "head_width": case.d_token // 8,
        "unsplit_repeats_bit_for_bit": repeated,
        "loss_unsplit": unsplit_loss,
        "loss_accumulated": accumulated_loss,
        "loss_relative_difference": abs(accumulated_loss - unsplit_loss) / abs(unsplit_loss),
        "tolerance": GRADIENT_TOLERANCE,
        "oracle_tolerance": ORACLE_TOLERANCE,
        "oracle_relative_l2": exact.get("relative_l2"),
        "oracle_largest_tensor_relative_l2": exact.get("largest_tensor_relative_l2"),
        **difference,
    }


def _gradient_item(
    book: CheckBook,
    workload: Workload,
    declaration: TrainingDeclaration,
    settings: Any,
    layout: InputLayout,
    device: Device,
) -> None:
    import torch

    rows = max(case.rows for case in workload.cases)
    data = synthetic_data(layout, rows, np.random.default_rng(DATA_SEED))
    for case in workload.cases:
        with log_step(logger, f"Gradient accumulation: {case.name}"):
            try:
                details = gradient_case(case, declaration, settings, layout, data, device)
            except torch.cuda.OutOfMemoryError as error:
                _release(device)
                book.add("B1", case.name, False, error=f"out of GPU memory: {error}")
                continue
        outcome = details.pop("passes")
        book.add("B1", case.name, outcome, **details)


# --- B2: repeatability ---------------------------------------------------------------------------


def _reference_candidate(declaration: TrainingDeclaration, model: str) -> Candidate:
    return {p.name: p.reference for p in declaration.space(model).parameters}


def _fit_once(
    model_name: str,
    declaration: TrainingDeclaration,
    settings: Any,
    layout: InputLayout,
    data: NeuralData,
    plan: BatchPlan,
    epochs: int,
    device: Device,
) -> tuple[NeuralModel, npt.NDArray[np.float64]]:
    cls: type[NeuralModel] = ResNetModel if model_name == "resnet" else FTTransformerModel
    name = "RES-b" if model_name == "resnet" else "FTT-b"
    model = cls(configuration(name), MODEL_SEED, settings=settings, layout=layout, device=device)
    hyperparameters = neural_hyperparameters(
        declaration, model_name, _reference_candidate(declaration, model_name)
    )
    model.fit(data, None, hyperparameters, iterations=epochs, plan=plan)
    return model, model.predict(data.inputs)


def _repeat_item(
    book: CheckBook,
    workload: Workload,
    declaration: TrainingDeclaration,
    settings: Any,
    layout: InputLayout,
    device: Device,
) -> None:
    """B2: each model, at the reference candidate of DR-015 §6 and §7, with balanced weights, is
    fitted twice with the same seed under one plan on Train rows that end in a final batch of
    1,016 rows; the scores and the per-epoch Train losses must repeat bit for bit (T3)."""
    effective = declaration.neural.effective_batch
    final = min(1016, effective)
    rows = 2 * effective + final
    data = _weighted(
        synthetic_data(layout, rows, np.random.default_rng(DATA_SEED + 1)), Weighting.BALANCED
    )
    for model_name in ("resnet", "ft_transformer"):
        plan = BatchPlan(effective, workload.micro_for(model_name, effective))
        with log_step(logger, f"Repeatability of {model_name} under {plan}"):
            first, scores = _fit_once(
                model_name,
                declaration,
                settings,
                layout,
                data,
                plan,
                workload.repeat_epochs,
                device,
            )
            losses = [row["train_loss"] for row in first.parameters["resolved"]["history"]]
            del first
            _release(device)
            second, again = _fit_once(
                model_name,
                declaration,
                settings,
                layout,
                data,
                plan,
                workload.repeat_epochs,
                device,
            )
            repeat = [row["train_loss"] for row in second.parameters["resolved"]["history"]]
            del second
            _release(device)
        identical = diagnostics.bitwise_equal(scores, again) and losses == repeat
        book.add(
            "B2",
            model_name,
            identical,
            effective_batch=plan.effective,
            micro_batch=plan.micro,
            train_rows=rows,
            final_batch_rows=rows % plan.effective or plan.effective,
            epochs=workload.repeat_epochs,
            scores_identical=diagnostics.bitwise_equal(scores, again),
            train_losses_identical=losses == repeat,
            train_losses=losses,
        )


# --- B3: corners ---------------------------------------------------------------------------------


def corner_candidates(
    declaration: TrainingDeclaration, workload: Workload
) -> list[tuple[str, str, Candidate]]:
    """The corners of the search space: the largest ResNet-style MLP (most layers, widest, largest
    hidden factor and largest category embedding), and FT-Transformers with the most layers at the
    widest token width and at the widths whose heads (``workload.heads`` units per head) the
    attention kernel pads."""
    resnet = declaration.space("resnet")
    transformer = declaration.space("ft_transformer")
    layers = upper(transformer, "n_layers")
    corners: list[tuple[str, str, Candidate]] = [
        (
            "resnet",
            "largest ResNet-style MLP",
            corner(
                resnet,
                **{
                    name: upper(resnet, name)
                    for name in ("n_layers", "d", "d_hidden_factor", "d_embedding")
                },
            ),
        )
    ]
    for head in workload.heads:
        padded = ((head + 7) // 8) * 8
        corners.append(
            (
                "ft_transformer",
                f"FT-Transformer, token width {8 * head} (heads of {head} padded to {padded})",
                corner(transformer, n_layers=layers, d_token=8 * head),
            )
        )
    corners.append(
        (
            "ft_transformer",
            f"largest FT-Transformer, token width {upper(transformer, 'd_token')}",
            corner(transformer, n_layers=layers, d_token=upper(transformer, "d_token")),
        )
    )
    return corners


def _corner_item(
    book: CheckBook,
    workload: Workload,
    declaration: TrainingDeclaration,
    settings: Any,
    layout: InputLayout,
    device: Device,
    meter: Meter,
) -> None:
    """B3: each corner through the preflight of DR-015 §7; it passes when a listed micro-batch
    completes the training phase and the float32 evaluation phase with the declared free memory."""
    for model, name, candidate in corner_candidates(declaration, workload):
        with log_step(logger, f"Corner: {name}"):
            try:
                entry = run_preflight(
                    declaration,
                    settings,
                    layout,
                    {model: [candidate]},
                    device=device,
                    meter=meter,
                    seed=MODEL_SEED,
                    models=(model,),
                )[model]
            except ModelError as error:
                book.add("B3", name, False, model=model, candidate=candidate, error=str(error))
                continue
        chosen = entry["candidates"][0]
        book.add(
            "B3",
            name,
            True,
            model=model,
            candidate=chosen["candidate"],
            micro_batch=chosen["micro_batch"],
            attempts=[
                {
                    "micro_batch": a["micro_batch"],
                    "completed": a["training"]["completed"],
                    "free_at_peak_mib": a["training"]["free_at_peak_mib"],
                }
                for a in chosen["attempts"]
            ],
            evaluation_free_at_peak_mib=chosen["evaluation"]["free_at_peak_mib"],
        )


# --- Warnings, resources and the run ------------------------------------------------------------


ATTEMPT_FAILURE = re.compile(r"^Fit \S+ on gpu \(\d+ Train rows\): failed after [\d.]+ s$")


def classify_warnings(capture: Any, book: CheckBook) -> None:
    """B4: any warning of an ignored setting, an unsupported capability or a fallback fails. The
    step-failure line that ``log_step`` writes for a preflight attempt that ran out of GPU memory
    is accepted by its exact form, as many times as B3 recorded such attempts (a micro-batch too
    large, which the preflight is built to try and abandon). Any other warning needs a recorded
    review."""
    expected = sum(
        not attempt["completed"]
        for item in book.items["B3"]
        for attempt in item.get("attempts", [])
    )
    for item in capture.records:
        text = item["text"]
        outcome: bool | str
        if ATTEMPT_FAILURE.match(text) and expected > 0:
            outcome, expected = True, expected - 1
        elif FAILING_WARNING.search(text):
            outcome = False
        else:
            outcome = "review"
        book.add("B4", item["phase"], outcome, logger=item["logger"], text=text)
    if not capture.records:
        book.add("B4", "whole run", True, warnings=0)


def _resources_item(book: CheckBook, sampler: diagnostics.GpuMemorySampler | None) -> None:
    """B5: no out-of-memory error outside the checks that record one, and, on the GPU, valid
    whole-device memory monitoring over the run; every measurement is recorded."""
    monitored: dict[str, Any] = {"monitored": False}
    valid = True
    if sampler is not None:
        snapshot = sampler.snapshot()
        valid = bool(
            not snapshot["problems"]
            and snapshot["samples"] > 1
            and snapshot["total_mib"] is not None
            and snapshot["baseline_used_mib"] is not None
        )
        monitored = {"monitored": True, **snapshot}
    book.add("B5", "whole run", valid, host_peak_bytes=diagnostics.host_peak_bytes(), **monitored)


def run_acceptance(
    paths: ProjectPaths,
    output: Path,
    *,
    device: Device = "gpu",
    workload: Workload | None = None,
    declaration: TrainingDeclaration | None = None,
    layout: InputLayout | None = None,
    meter: Meter | None = None,
    log_opened: bool = False,
) -> dict[str, Any]:
    """Run the acceptance, write its records to ``output`` at every stage and on any failure, and
    return the final record. ``declaration``, ``layout`` and ``meter`` are those of the study
    unless given (the unit tests give small ones)."""
    workload = workload or Workload()
    output = output.resolve()
    check_acceptance_output(output, log_name=BATCHING_LOG_NAME, log_opened=log_opened)
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FileAccessError(f"{output}: cannot be created ({error.strerror})") from error
    specification = paths.root / SPECIFICATION
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "specification": {"path": SPECIFICATION, "sha256": None},
        "settings": None,
        "declaration": None,
        "run": {
            "started": datetime.now(UTC).isoformat(timespec="seconds"),
            "finished": None,
            "stage": "start",
            "error": None,
            "device": device,
            "acceptance_evidence": False,
            "evidence_blockers": ["start-up not finished"],
            "pid": os.getpid(),
            "data_seed": DATA_SEED,
            "model_seed": MODEL_SEED,
            "tolerance_gradient": GRADIENT_TOLERANCE,
            "tolerance_oracle": ORACLE_TOLERANCE,
            "workload": asdict(workload),
        },
        "code": None,
        "environment": None,
        "gpu_memory": None,
        "host_peak_bytes": None,
        "checks": CheckBook(CHECKS).to_json(),
        "overall": "in progress",
    }
    book = CheckBook(CHECKS)
    sampler: diagnostics.GpuMemorySampler | None = None
    classified = False

    def write(stage: str) -> dict[str, Any]:
        record["run"]["stage"] = stage
        record["checks"] = book.to_json()
        record["gpu_memory"] = None if sampler is None else sampler.snapshot()
        record["host_peak_bytes"] = diagnostics.host_peak_bytes()
        written: dict[str, Any] = jsonable(record)  # type: ignore[assignment]
        write_json(output / RECORD_NAME, written)
        link = Path(os.path.relpath(specification, output)).as_posix()
        text = summary_markdown(written, link)
        write_atomically(output / SUMMARY_NAME, lambda target: target.write_text(text, "utf-8"))
        return written

    with captured_warnings() as capture:
        try:
            record["specification"]["sha256"] = file_sha256(specification)
            settings, snapshot = load_settings_with_snapshot(paths)
            record["settings"] = {"path": snapshot.path, "sha256": snapshot.sha256}
            if declaration is None:
                declaration, declared = load_declaration(paths)
                record["declaration"] = {"sha256": declared.sha256}
            record["code"] = code_provenance(paths.root)
            blockers = evidence_blockers(device, workload, declaration, record["code"], paths.root)
            record["run"]["evidence_blockers"] = blockers
            record["run"]["acceptance_evidence"] = not blockers
            record["environment"] = environment_record()
            if layout is None:
                features = load_feature_spec(paths.features_spec)
                levels = category_levels(read_catalog(paths.feature_catalog), features)
                layout = neural_layout(features, levels)
            with diagnostics.sampled_gpu_memory(device == "gpu") as running:
                sampler = running
                meter = meter or GpuMeter()
                capture.phase = "B1 gradients"
                write("B1 gradients")
                _gradient_item(book, workload, declaration, settings, layout, device)
                capture.phase = "B2 repeatability"
                write("B2 repeatability")
                _repeat_item(book, workload, declaration, settings, layout, device)
                capture.phase = "B3 corners"
                write("B3 corners")
                _corner_item(book, workload, declaration, settings, layout, device, meter)
            capture.phase = "checks"
            _resources_item(book, sampler)
            classify_warnings(capture, book)
            classified = True
            record["run"]["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
            record["overall"] = book.overall(compared=True)
            final = write("finished")
        except BaseException as error:
            if not classified:
                classify_warnings(capture, book)
            interrupted = isinstance(error, KeyboardInterrupt)
            record["run"]["error"] = f"{type(error).__name__}: {error}"
            record["overall"] = "interrupted" if interrupted else "failed"
            try:
                write("interrupted" if interrupted else "failed")
            except Exception:  # the original error is the one that is raised
                logger.exception("The batching record could not be written after the failure")
            raise
    logger.info("Neural batching acceptance: %s (records in %s)", final["overall"], output)
    return final


def summary_markdown(record: dict[str, Any], specification_link: str) -> str:
    """The run summary generated from ``batching.json``, never written by hand: the checks, every
    item with all its recorded details, and every other recorded field."""
    run = record["run"]
    code: dict[str, Any] = record.get("code") or {}
    settings: dict[str, Any] = record.get("settings") or {}
    blockers = run.get("evidence_blockers") or []
    verdict = (
        ""
        if run["acceptance_evidence"]
        else " This run is not acceptance evidence: " + "; ".join(blockers) + "."
    )
    lines = [
        "# Neural batching acceptance: run summary",
        "",
        "Generated by `review-capacity probe-neural-batching` from `batching.json`; not edited by "
        "hand. The checks (B1–B5), criteria and tolerance (T5) are those of Section 9 of "
        f"[`docs/gpu-acceptance-probe.md`]({specification_link}).",
        "",
        f"**Overall result: {record['overall']}.**{verdict}",
        "",
        "## Run identity",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Record schema version | {record['schema_version']} |",
        f"| Code commit | `{_cell(code.get('git_commit'))}` |",
        f"| Uncommitted changes | {_cell(code.get('git_uncommitted_changes'))} |",
        f"| Specification SHA-256 | `{_cell(record['specification'].get('sha256'))}` |",
        f"| Settings SHA-256 | `{_cell(settings.get('sha256'))}` |",
        f"| Declaration SHA-256 | `{_cell((record.get('declaration') or {}).get('sha256'))}` |",
        f"| Data seed; model seed | {run['data_seed']}; {run['model_seed']} |",
        f"| Device | {run['device']} |",
        f"| Gradient tolerance T5 (relative L2) | {_scalar(run['tolerance_gradient'])} |",
        f"| Oracle tolerance T6 (relative L2, float32) | {_scalar(run['tolerance_oracle'])} |",
        f"| Started; finished | {run['started']}; {_cell(run.get('finished'))} |",
        f"| Error | {_cell(run.get('error'))} |",
        "",
        "## Checks",
        "",
        "| Check | Title | Status | Items | Failed | For review |",
        "|---|---|---|---|---|---|",
    ]
    for check, entry in record["checks"].items():
        items = entry["items"]
        failed = sum(item["outcome"] is False for item in items)
        review = sum(item["outcome"] == "review" for item in items)
        lines.append(
            f"| {check} | {entry['title']} | {entry['status']} | {len(items)} | {failed} | "
            f"{review} |"
        )
    lines += ["", "## Items of every check", ""]
    for check, entry in record["checks"].items():
        for item in entry["items"]:
            outcome = "descriptive" if item["outcome"] is None else item["outcome"]
            details = {k: v for k, v in item.items() if k not in ("subject", "outcome")}
            lines += _record_section(f"{check}: {item['subject']} ({outcome})", details)[1:]
    lines += _record_section("Run record", {k: v for k, v in run.items() if k != "workload"})
    lines += _record_section("Workload", run["workload"])
    software = {"code": record.get("code"), "environment": record.get("environment")}
    lines += _record_section("Code, packages and environment", software)
    lines += _record_section(
        "GPU memory and host memory",
        {"gpu_memory": record.get("gpu_memory"), "host_peak_bytes": record.get("host_peak_bytes")},
    )
    lines += [
        "",
        "## Symbols and notation",
        "",
        "| Symbol / abbreviation | Meaning |",
        "|---|---|",
        "| B1–B5 | Checks of the neural batching acceptance (Section 9 of the specification) |",
        "| T5 | Tolerance of B1: relative L2 difference of the accumulated and unsplit gradients |",
        "| T6 | Tolerance of B1: relative L2 difference of the float32 accumulated gradient "
        "and the independent float32 oracle |",
        "| Relative L2 | Euclidean norm of the difference divided by the norm of the unsplit "
        "gradient |",
        "| Micro-batch | Rows of an effective batch processed together; gradients are added |",
        "| MiB | Mebibyte (2^20 bytes) |",
        "| GPU | Graphics processing unit |",
        "| SHA-256 | Secure Hash Algorithm, 256-bit |",
        "",
    ]
    return "\n".join(lines)
