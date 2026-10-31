"""The frozen thresholds and calibrators of the Test evaluation (Issue #205; protocol, Authorised
Test evaluation).

Everything here comes from the records and files that the binding names (``test_run``'s
``frozen_inputs``), each hash-checked again when it is used:

- the R1 threshold of a configuration is the ``threshold`` of its primary ``r1_selection`` record;
- the R2 threshold of a configuration and a capacity is the ``threshold`` of its primary V2
  ``evaluation`` record of policy R2 at that capacity;
- a calibrator is the saved file that its primary ``calibration`` record names.

The Test data play no part: a missing or ambiguous item stops the evaluation before any outcome is
computed. The R0 threshold is the default 0.5.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from review_capacity.development.records import CONFIGURATION_NAMES
from review_capacity.errors import SpecificationError
from review_capacity.evaluation.calibration import Calibrator, load_calibrator_bytes
from review_capacity.evaluation.outcomes import Thresholds

logger = logging.getLogger(__name__)

R0_THRESHOLD = 0.5
METHODS = ("platt", "isotonic")


@dataclass(frozen=True)
class FrozenPolicies:
    """The thresholds of the eight configurations and their calibrators."""

    thresholds: Mapping[str, Thresholds]
    calibrators: Mapping[tuple[str, str], Calibrator]
    calibrator_files: Mapping[tuple[str, str], Mapping[str, str]]  # the frozen file references


def _read(root: Path, reference: Mapping[str, str]) -> bytes:
    """The bytes of a frozen file, hashed as read: the bytes returned are the bytes verified."""
    path = root / reference["path"]
    try:
        content = path.read_bytes()
    except OSError as error:
        raise SpecificationError(
            f"{reference['path']}: cannot be read ({error.strerror})"
        ) from error
    if hashlib.sha256(content).hexdigest() != reference["sha256"]:
        raise SpecificationError(f"{reference['path']} differs from the frozen file")
    return content


def _records(root: Path, references: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """The JSON records among ``references`` (a saved calibrator is JSON without ``record``)."""
    records = []
    for reference in references:
        content = _read(root, reference)  # a changed file is refused here, not skipped
        try:
            value = json.loads(content)
        except ValueError:
            continue
        if isinstance(value, dict) and "record" in value:
            records.append(value)
    return records


def _single(found: list[dict[str, Any]], what: str) -> dict[str, Any]:
    if len(found) != 1:
        raise SpecificationError(f"The frozen inputs hold {len(found)} records for {what}, not one")
    return found[0]


def load_policies(
    root: Path,
    frozen_inputs: Mapping[str, Any],
    capacities: Sequence[float],
    configurations: Sequence[str] = CONFIGURATION_NAMES,
) -> FrozenPolicies:
    """The frozen thresholds and calibrators of ``configurations``, each from its one primary
    record; every capacity needs an R2 threshold."""
    threshold_records = _records(root, frozen_inputs["thresholds"])
    calibration_records = [
        record
        for record in _records(root, frozen_inputs["calibrators"])
        if record["record"] == "calibration"
    ]
    saved = {(ref["path"], ref["sha256"]): ref for ref in frozen_inputs["calibrators"]}
    thresholds: dict[str, Thresholds] = {}
    calibrators: dict[tuple[str, str], Calibrator] = {}
    files: dict[tuple[str, str], Mapping[str, str]] = {}
    for name in configurations:
        selection = _single(
            [
                r
                for r in threshold_records
                if r["record"] == "r1_selection"
                and r["period"] == "primary"
                and r["configuration"] == name
            ],
            f"the primary R1 selection of {name}",
        )
        if selection["status"] != "selected" or selection["threshold"] is None:
            raise SpecificationError(f"{name}: the frozen R1 selection has no threshold")
        r2: dict[float, float] = {}
        for capacity in capacities:
            record = _single(
                [
                    r
                    for r in threshold_records
                    if r["record"] == "evaluation"
                    and r["policy"] == "R2"
                    and r["partition"] == "V2"
                    and r["period"] == "primary"
                    and r["model_role"] == "selected"
                    and r["configuration"] == name
                    and r["capacity"] == capacity
                ],
                f"the primary V2 R2 evaluation of {name} at capacity {capacity}",
            )
            r2[capacity] = float(record["threshold"])
        thresholds[name] = Thresholds(R0_THRESHOLD, float(selection["threshold"]), r2)
        for method in METHODS:
            calibration = _single(
                [
                    r
                    for r in calibration_records
                    if r["period"] == "primary"
                    and r["configuration"] == name
                    and r["method"] == method
                ],
                f"the primary {method} calibration of {name}",
            )
            artifact = calibration["artifact"]
            if calibration["status"] != "completed" or artifact is None:
                raise SpecificationError(f"{name}: the frozen {method} calibration is not saved")
            match = [
                ref
                for (path, sha), ref in saved.items()
                if path.endswith(artifact["path"]) and sha == artifact["sha256"]
            ]
            if len(match) != 1:
                raise SpecificationError(
                    f"{name}: the saved {method} calibrator is not among the frozen inputs"
                )
            content = _read(
                root, match[0]
            )  # hashed as read; the calibrator is built from these bytes
            calibrators[(name, method)] = load_calibrator_bytes(method, content, match[0]["path"])
            files[(name, method)] = dict(match[0])
    logger.info("Frozen policies: %d thresholds, %d calibrators", len(thresholds), len(calibrators))
    return FrozenPolicies(thresholds, calibrators, files)
