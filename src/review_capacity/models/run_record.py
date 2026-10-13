"""Model-run records: what a fitted model used, written as JSON beside it (DR-013 §6).

A record (schema version 1) holds the configuration, device and seed; the requested
hyperparameters; the model's parameters (requested and resolved, with the seeds, thread settings
and backends it actually used); the trees or epochs kept; the exact contents and SHA-256 of the
loaded ``config/models.yaml``; the verified LightGBM build (required for LightGBM); the run
environment; and the code provenance. The settings are parsed from the same bytes that are hashed
and recorded, so the record always shows the settings in use. Warnings are kept in the run's log
file, which the run evidence identifies.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from review_capacity.config import Device, ModelSettings, load_model_settings
from review_capacity.errors import FileAccessError
from review_capacity.models.device import LightGBMIdentity
from review_capacity.models.registry import Configuration
from review_capacity.paths import ProjectPaths
from review_capacity.provenance import write_record

type JSON = bool | int | float | str | list[JSON] | dict[str, JSON] | None

RECORD_SCHEMA_VERSION = 1


class RecordedModel(Protocol):
    """What a record reads from a fitted model: a ``models.base.ScoringModel`` (tree models) or a
    ``models.neural_training.NeuralModel`` (neural models)."""

    configuration: Configuration
    seed: int
    device: Device

    @property
    def parameters(self) -> dict[str, Any]: ...

    @property
    def fitted_iterations(self) -> int: ...


@dataclass(frozen=True)
class SettingsSnapshot:
    """The loaded ``config/models.yaml``: its root-relative path, SHA-256 and exact contents."""

    path: str
    sha256: str
    content: str


def load_settings_with_snapshot(paths: ProjectPaths) -> tuple[ModelSettings, SettingsSnapshot]:
    """Load the model settings once, with a snapshot of the bytes they were parsed from.

    The file is read once; the same bytes are parsed, hashed and recorded, so a later edit of
    the file cannot make the record differ from the settings in use.
    """
    path = paths.models_spec
    try:
        content = path.read_bytes()
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error.strerror})") from error
    settings = load_model_settings(paths, content=content)
    snapshot = SettingsSnapshot(
        path=path.relative_to(paths.root).as_posix(),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content.decode("utf-8"),
    )
    return settings, snapshot


def jsonable(value: object) -> JSON:
    """``value`` as plain JSON data: NumPy scalars and arrays, tuples and paths are converted;
    a non-finite number or an unsupported type is a defect and raises."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufUO":
            raise TypeError(f"NumPy dtype {value.dtype} cannot be recorded as JSON")
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biufU":
            raise TypeError(f"NumPy dtype {value.dtype} cannot be recorded as JSON")
        if isinstance(value, np.floating) and not np.isfinite(value):
            raise ValueError(f"Non-finite number {value!r} cannot be recorded as JSON")
        converted = value.item()
        if isinstance(converted, np.generic):  # e.g. extended precision: no Python equivalent
            raise TypeError(f"{type(value).__name__} has no lossless Python scalar for JSON")
        return jsonable(converted)
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite number {value!r} cannot be recorded as JSON")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: jsonable(item) for key, item in value.items()}
    raise TypeError(f"{type(value).__name__} cannot be recorded as JSON")


def model_run_record(
    *,
    model: RecordedModel,
    settings: SettingsSnapshot,
    hyperparameters: Mapping[str, Any],
    lightgbm_build: LightGBMIdentity | None,
    environment: Mapping[str, Any],
    code: Mapping[str, Any],
) -> dict[str, JSON]:
    """The record of one fitted model. ``environment`` and ``code`` come from
    ``device.environment_record()`` and ``provenance.code_provenance()``. A LightGBM record must
    carry the verified build identity (DR-013 §2)."""
    configuration = model.configuration
    if configuration.model.name == "lightgbm" and lightgbm_build is None:
        raise ValueError("A LightGBM run record requires its verified build identity")
    build: JSON = None
    if lightgbm_build is not None:
        build = {
            "version": lightgbm_build.version,
            "library": lightgbm_build.library.as_posix(),
            "sha256": lightgbm_build.sha256,
        }
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "configuration": configuration.name,
        "model": configuration.model.name,
        "family": configuration.family.value,
        "weighting": configuration.weighting.value,
        "device": model.device,
        "seed": model.seed,
        "hyperparameters": jsonable(hyperparameters),
        "parameters": jsonable(model.parameters),
        "fitted_iterations": model.fitted_iterations,
        "settings": {"path": settings.path, "sha256": settings.sha256, "content": settings.content},
        "lightgbm_build": build,
        "environment": jsonable(environment),
        "code": jsonable(code),
    }


def write_model_run_record(path: Path, record: dict[str, JSON]) -> None:
    """Write a model-run record as indented UTF-8 JSON, atomically."""
    write_record(path, record)
