"""The calibrators of a configuration's selected main-seed model, fitted on its saved V2 scores
(protocol Calibration; DR-007 §8; Issue #158).

For one configuration this reads the saved V2 scores through ``read_scores`` (which checks the
file against its record and the partition's identifier sequence) and, for the Platt (primary) and
the isotonic (secondary) method, fits

* the **primary** calibrator on all of V2 (days 127–154), diagnosed in-sample;
* the **label-latency** calibrator on V2 days 127–147 only, with its smoothing counts recomputed
  (the selected model unchanged), diagnosed in-sample and on the held-out days 148–154. It does
  not replace the primary calibrator.

Each calibrator is saved under ``calibrators/`` in the output run directory, reloaded, and its
predictions are verified to be equal before its ``calibration`` record is written. A fit that
fails (one class only, no two distinct log-odds, non-convergence, a non-positive Platt slope)
writes its ``calibration`` record with the error and then stops the analysis with a project
exception: there is no substitute method and no forced slope.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from review_capacity.config import SECONDS_PER_DAY
from review_capacity.development.records import DevelopmentPartitions
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.scoring import (
    HoldsRunFiles,
    ScoredPartition,
    check_selected,
    read_scores,
)
from review_capacity.development.trial import run_file
from review_capacity.errors import ModelError, ReviewCapacityError
from review_capacity.evaluation.calibration import (
    EPSILON,
    GRADIENT_TOLERANCE,
    INITIAL,
    MAX_ITERATIONS,
    SOLVER,
    Calibrator,
    PlattCalibrator,
    PlattFit,
    diagnostics,
    fit_isotonic,
    fit_platt,
    load_calibrator,
    software,
)
from review_capacity.logging_setup import log_step

logger = logging.getLogger(__name__)

CALIBRATORS_DIR = "calibrators"
METHODS = ("platt", "isotonic")
PERIODS = ("primary", "latency")


@dataclass(frozen=True)
class FittedCalibrator:
    """A fitted calibrator with the reference of its ``calibration`` record and the V2 days it
    was fitted on."""

    method: str
    period: str
    calibrator: Calibrator
    record: dict[str, str]
    fitting_days: dict[str, int]


def artifact_path(run: RunDirectory, configuration: str, method: str, period: str) -> Path:
    suffix = "json" if method == "platt" else "npz"
    return run.path / CALIBRATORS_DIR / f"{configuration}-{method}-{period}.{suffix}"


def _base_record(
    identity: Mapping[str, Any],
    configuration: str,
    method: str,
    period: str,
    score: Mapping[str, str],
    days: Mapping[str, int],
    labels: np.ndarray,
) -> dict[str, Any]:
    frauds = int(labels.sum())
    return {
        "record": "calibration",
        "schema_version": 1,
        "identity": dict(identity),
        "configuration": configuration,
        "method": method,
        "period": period,
        "score": dict(score),
        "fitting_days": dict(days),
        "rows": len(labels),
        "frauds": frauds,
        "legitimate": len(labels) - frauds,
        "clipping": None,
        "targets": None,
        "parameters": None,
        "solver": None,
        "knots": None,
        "artifact": None,
        "software": software(),
        "reload_verified": None,
        "diagnostics": [],
        "status": "failed",
        "error": None,
    }


def _platt_fields(calibrator: PlattCalibrator, fit: PlattFit) -> dict[str, Any]:
    return {
        "clipping": {
            "epsilon": EPSILON,
            "clipped_low": fit.clipped_low,
            "clipped_high": fit.clipped_high,
        },
        "targets": {"positive": fit.target_positive, "negative": fit.target_negative},
        "parameters": {"intercept": calibrator.intercept, "slope": calibrator.slope},
        "solver": {
            "name": SOLVER,
            "settings": {
                "initial": list(INITIAL),
                "gradient_tolerance": GRADIENT_TOLERANCE,
                "max_iterations": MAX_ITERATIONS,
                "step_halving": True,
            },
            "converged": fit.converged,
            "iterations": fit.iterations,
        },
    }


def _fit(method: str, scores: np.ndarray, labels: np.ndarray) -> tuple[Calibrator, dict[str, Any]]:
    if method == "platt":
        platt, fit = fit_platt(scores, labels)
        return platt, _platt_fields(platt, fit)
    isotonic = fit_isotonic(scores, labels)
    return isotonic, {"knots": isotonic.knots}


def fit_calibrators(
    output: RunDirectory,
    source: HoldsRunFiles,
    partitions: DevelopmentPartitions,
    identity: Mapping[str, Any],
    v2: ScoredPartition,
) -> list[FittedCalibrator]:
    """Fit, save, reload and record the calibrators of ``v2``'s configuration (see the module
    description); returns them in the order Platt then isotonic, primary then label-latency."""
    name = v2.configuration
    if v2.label != "V2":
        raise ModelError("Calibrators are fitted on the V2 scores")
    check_selected(v2, identity["seeds"]["main"])
    scores = read_scores(source, v2.file, v2.rows)
    days = v2.rows.times // SECONDS_PER_DAY
    labels = v2.rows.labels
    fitting, held_out = partitions.latency_ranges()
    fitted: list[FittedCalibrator] = []
    with log_step(logger, f"Calibrating the selected {name} on V2"):
        for period in PERIODS:
            first, last = (partitions.v2.first_day, partitions.v2.last_day)
            if period == "latency":
                first, last = fitting
            fit_rows = (days >= first) & (days <= last)
            fit_days = {"first": first, "last": last}
            held = None if period == "primary" else _held_out(scores, labels, days, held_out)
            fitted.extend(
                _calibrate_one(
                    output,
                    identity,
                    v2,
                    method,
                    period,
                    fit_days,
                    (scores[fit_rows], labels[fit_rows]),
                    held,
                )
                for method in METHODS
            )
    return fitted


def _held_out(
    scores: np.ndarray, labels: np.ndarray, days: np.ndarray, held_out: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rows = (days >= held_out[0]) & (days <= held_out[1])
    return scores[rows], labels[rows], {"first": held_out[0], "last": held_out[1]}


def _calibrate_one(
    output: RunDirectory,
    identity: Mapping[str, Any],
    v2: ScoredPartition,
    method: str,
    period: str,
    fit_days: dict[str, int],
    fit_rows: tuple[np.ndarray, np.ndarray],
    held: tuple[np.ndarray, np.ndarray, dict[str, int]] | None,
) -> FittedCalibrator:
    name = v2.configuration
    fit_scores, fit_labels = fit_rows
    record = _base_record(identity, name, method, period, v2.record, fit_days, fit_labels)
    with log_step(logger, f"{method} calibrator of {name} ({period})"):
        try:
            calibrator, fields = _fit(method, fit_scores, fit_labels)
            path = artifact_path(output, name, method, period)
            path.parent.mkdir(exist_ok=True)
            calibrator.save(path)
            again = load_calibrator(method, path)
            if not np.array_equal(again.predict(fit_scores), calibrator.predict(fit_scores)):
                raise ModelError(
                    f"{name}: the {method} calibrator predicts differently after reloading"
                )
            record |= fields
            record["artifact"] = run_file(output, path)
            record["reload_verified"] = True
            record["diagnostics"] = [
                diagnostics(calibrator, fit_scores, fit_labels, rows_used="fitting", days=fit_days)
            ]
            if held is not None:
                held_scores, held_labels, held_days = held
                record["diagnostics"].append(
                    diagnostics(
                        calibrator, held_scores, held_labels, rows_used="held_out", days=held_days
                    )
                )
            record["status"] = "completed"
        except ReviewCapacityError as error:
            record["error"] = f"{type(error).__name__}: {error}"
            output.write_record(record, name, method, period)
            raise
    written = output.write_record(record, name, method, period)
    logger.info("%s %s calibrator (%s) written", name, method, period)
    return FittedCalibrator(method, period, calibrator, run_file(output, written), fit_days)
