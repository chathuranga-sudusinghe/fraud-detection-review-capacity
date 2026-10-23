"""Records of the GPU acceptance probe (``docs/gpu-acceptance-probe.md``, Section 5).

* ``CheckBook`` collects the evidence of every check C1–C13 and decides its status:
  ``fail`` if any pass/fail item failed; ``review`` if an item needs a recorded review (warnings
  of check C2); ``pass`` if every pass/fail item passed; ``descriptive`` if the check holds only
  measurements.
* The overall result is ``fail``, ``incomplete`` (a check not run), ``pending second run`` (no
  comparison with a first run in a fresh process yet, check C11), ``review`` or ``pass``.
* ``probe.json`` is the record; ``probe.md`` is generated from it by ``summary_markdown`` and never
  written by hand.
* ``check_output`` keeps every run in a directory of its own; it has no side effects, so that the
  command can call it before it opens the run's log file ``probe.log``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from review_capacity.errors import FileAccessError, SpecificationError
from review_capacity.files import write_atomically, write_json

SCHEMA_VERSION = 1
LOG_NAME = "probe.log"
RECORD_ENTRIES = ("probe.json", "probe.md", "scores")
ACCEPTANCE_ROOT = Path("outputs") / "catboost_objective_acceptance"  # Section 8
ACCEPTANCE_LOG_NAME = "acceptance.log"
ACCEPTANCE_ENTRIES = ("acceptance.json", "acceptance.md")
BATCHING_ROOT = Path("outputs") / "neural_batching_acceptance"  # Section 9
BATCHING_LOG_NAME = "batching.log"
BATCHING_ENTRIES = ("batching.json", "batching.md")
CHECKS: dict[str, str] = {
    "C1": "Backend identity",
    "C2": "Ignored-option and unsupported-option warnings",
    "C3": "Resolved settings",
    "C4": "LightGBM Dataset reproducibility",
    "C5": "Categorical behaviour",
    "C6": "All eight configurations, non-uniform weights",
    "C7": "Neural computation under strict determinism",
    "C8": "float32 evaluation path",
    "C9": "Numerical behaviour",
    "C10": "Prediction repeatability and backend",
    "C11": "Same-seed repeats",
    "C12": "Persistence",
    "C13": "Resources and timing",
}


def check_output(output: Path, previous: Path | None, *, log_opened: bool = False) -> None:
    """A run writes to a directory of its own: never one holding another run's records or log, and
    never the first run's directory (or a directory inside or around it). ``log_opened``: the
    command has already opened this run's ``probe.log`` in ``output``."""
    output = output.resolve()
    entries = RECORD_ENTRIES if log_opened else (*RECORD_ENTRIES, LOG_NAME)
    if any((output / entry).exists() for entry in entries):
        raise SpecificationError(f"{output} already holds a probe run; use a new directory")
    if previous is not None:
        first = previous.resolve().parent
        if first == output or first in output.parents or output in first.parents:
            raise SpecificationError("The output directory must be separate from the first run's")


def check_acceptance_output(
    output: Path, *, log_name: str = ACCEPTANCE_LOG_NAME, log_opened: bool = False
) -> None:
    """An acceptance run (Section 8: the CatBoost tree objective; Section 9: neural batching)
    writes to a directory of its own: a directory that does not exist or is empty. ``log_opened``:
    the command has already opened this run's log ``log_name`` in ``output``, which is then the
    only entry allowed."""
    output = output.resolve()
    allowed = {log_name} if log_opened else set()
    try:
        occupied = sorted(entry.name for entry in output.iterdir() if entry.name not in allowed)
    except FileNotFoundError:
        return
    except OSError as error:
        raise FileAccessError(f"{output}: cannot be inspected ({error.strerror})") from error
    if occupied:
        raise SpecificationError(
            f"{output} is not empty ({', '.join(occupied)}); use a new directory for the run"
        )


class CheckBook:
    """The evidence of every check: items with a subject, an outcome and their details."""

    def __init__(self, checks: Mapping[str, str] = CHECKS) -> None:
        """``checks``: the identifiers and titles of the checks (those of the probe by default)."""
        self.checks = dict(checks)
        self.items: dict[str, list[dict[str, Any]]] = {check: [] for check in self.checks}

    def add(self, check: str, subject: str, outcome: bool | str | None, **details: Any) -> None:
        """Record an item: ``outcome`` True (passed), False (failed), ``"review"`` (needs a
        recorded review) or None (descriptive)."""
        if check not in self.items:
            raise ValueError(f"Unknown check {check!r}")
        if outcome not in (True, False, None, "review"):
            raise ValueError(f"Unknown outcome {outcome!r}")
        self.items[check].append({"subject": subject, "outcome": outcome, **details})

    def status(self, check: str) -> str:
        outcomes = [item["outcome"] for item in self.items[check]]
        if not outcomes:
            return "not run"
        if any(outcome is False for outcome in outcomes):
            return "fail"
        if "review" in outcomes:
            return "review"
        if any(outcome is True for outcome in outcomes):
            return "pass"
        return "descriptive"

    def overall(self, *, compared: bool) -> str:
        """The overall result; without a comparison with a first run (``compared``), a run that
        has not failed is ``pending second run``, never ``pass``."""
        statuses = [self.status(check) for check in self.checks]
        if "fail" in statuses:
            return "fail"
        if "not run" in statuses:
            return "incomplete"
        if not compared:
            return "pending second run"
        if "review" in statuses:
            return "review"
        return "pass"

    def to_json(self) -> dict[str, Any]:
        return {
            check: {"title": title, "status": self.status(check), "items": self.items[check]}
            for check, title in self.checks.items()
        }


def _cell(value: object) -> str:
    text = "—" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return _cell(value)


def summary_markdown(record: dict[str, Any], specification_link: str) -> str:
    """The run summary, generated from the record (``probe.md``); ``specification_link`` is the
    path of the specification relative to the summary. A run that failed during start-up has no
    code, environment or settings record yet: those cells are shown as absent."""
    run = record["run"]
    code: dict[str, Any] = record.get("code") or {}
    environment: dict[str, Any] = record.get("environment") or {}
    gpu: dict[str, Any] = environment.get("gpu") or {}
    packages: dict[str, Any] = environment.get("packages") or {}
    settings: dict[str, Any] = record.get("settings") or {}
    lines = [
        "# Graphics processing unit (GPU) acceptance probe: run summary",
        "",
        "Generated by `review-capacity probe-gpu` from `probe.json`; not edited by hand. The "
        "checks (C1–C13), criteria and tolerances (T1–T4) are those of "
        f"[`docs/gpu-acceptance-probe.md`]({specification_link}).",
        "",
        "Terms: CPU, central processing unit; CUDA, NVIDIA's GPU computing platform; float32, the "
        "32-bit floating-point format; SHA-256, Secure Hash Algorithm, 256-bit; MiB, mebibyte "
        "(2^20 bytes); ms, millisecond. Configurations are named model-weighting: LGB "
        "(LightGBM), Cat (CatBoost), RES (ResNet-style multilayer perceptron) and FTT "
        "(FT-Transformer, Feature Tokenizer + Transformer), each unweighted (u) or with balanced "
        "class weighting (b).",
        "",
        f"**Overall result: {record['overall']}.**",
        "",
        "## Run identity",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Code commit | `{_cell(code.get('git_commit'))}` |",
        f"| Uncommitted changes | {_cell(code.get('git_uncommitted_changes'))} |",
        f"| Specification SHA-256 | `{_cell(record['specification'].get('sha256'))}` |",
        f"| Settings SHA-256 | `{_cell(settings.get('sha256'))}` |",
        f"| Data seed; model seed | {run['data_seed']}; {run['model_seed']} |",
        f"| Device | {_cell(gpu.get('name'))}, driver {_cell(gpu.get('driver'))} |",
        f"| PyTorch; CUDA | {_cell(packages.get('torch'))}; {_cell(gpu.get('torch_cuda'))} |",
        f"| Started; finished | {run['started']}; {_cell(run.get('finished'))} |",
        f"| Compared with a first run | {'yes' if record.get('comparison') else 'no'} |",
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
    problems = [
        (check, item)
        for check, entry in record["checks"].items()
        for item in entry["items"]
        if item["outcome"] is False or item["outcome"] == "review"
    ]
    lines += ["", "## Failed items and items for review", ""]
    if not problems:
        lines.append("None.")
    else:
        lines += ["| Check | Subject | Outcome | Details |", "|---|---|---|---|"]
        for check, item in problems:
            details = {k: v for k, v in item.items() if k not in ("subject", "outcome")}
            lines.append(
                f"| {check} | {_cell(item['subject'])} | {_cell(item['outcome'])} | "
                f"{_cell(details)} |"
            )
    lines += [
        "",
        "## Tolerance T1: float32 logits against the reference (C8)",
        "",
        "| Model | Max. abs. logit difference | Tolerance | Max. abs. score difference |",
        "|---|---|---|---|",
    ]
    for item in record["checks"].get("C8", {}).get("items", []):
        if "max_abs_logit_difference" in item:
            lines.append(
                f"| {_cell(item['subject'])} | {_number(item['max_abs_logit_difference'])} | "
                f"{_number(item['tolerance'])} | {_number(item['max_abs_score_difference'])} |"
            )
    lines += [
        "",
        "## Fits",
        "",
        "| Fit | Device | Kept trees or epochs | Fit seconds | Peak PyTorch allocated (MiB) |",
        "|---|---|---|---|---|",
    ]
    for fit in record["fits"]:
        lines.append(
            f"| {_cell(fit['subject'])} | {_cell(fit['device'])} | {_cell(fit['kept'])} | "
            f"{_number(fit['seconds'])} | {_number(fit.get('peak_allocated_mib'))} |"
        )
    memory = record.get("gpu_memory")
    if memory:
        lines += [
            "",
            f"Whole-device GPU memory: sampled maximum {memory['sampled_max_used_mib']} MiB of "
            f"{memory['total_mib']} MiB (baseline {memory['baseline_used_mib']} MiB; "
            f"{memory['samples']} samples every {memory['interval_ms']} ms; a sampled maximum, "
            "not the true peak).",
        ]
        if memory.get("problems"):
            lines.append(f"Monitoring problems: {_cell('; '.join(memory['problems']))}.")
    lines += [
        "",
        "## Symbols and notation",
        "",
        "| Symbol / abbreviation | Meaning |",
        "|---|---|",
        "| C1–C13, T1–T4 | Checks and tolerances of the probe specification |",
        "| CPU, GPU | Central processing unit; graphics processing unit |",
        "| CUDA | NVIDIA's GPU computing platform |",
        "| float32 | 32-bit floating-point format |",
        "| LGB, Cat, RES, FTT | LightGBM; CatBoost; ResNet-style multilayer perceptron; "
        "FT-Transformer |",
        "| u, b | Unweighted; balanced class weighting |",
        "| MiB, ms | Mebibyte (2^20 bytes); millisecond |",
        "| SHA-256 | Secure Hash Algorithm, 256-bit |",
        "",
    ]
    return "\n".join(lines)


def write_records(directory: Path, record: dict[str, Any], specification: Path) -> None:
    """Write ``probe.json`` and the generated ``probe.md``, atomically; ``specification`` is the
    specification file, linked relative to ``directory``."""
    write_json(directory / "probe.json", record)
    link = Path(os.path.relpath(specification, directory)).as_posix()
    text = summary_markdown(record, link)
    write_atomically(directory / "probe.md", lambda target: target.write_text(text, "utf-8"))
