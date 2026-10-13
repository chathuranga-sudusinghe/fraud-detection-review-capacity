"""A minimal ``ScoringModel`` for testing the common interface and the model-run record."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

from review_capacity.config import Device, ModelSettings
from review_capacity.errors import ModelError
from review_capacity.models.base import (
    ModelData,
    Scores,
    ScoringModel,
    check_fit_arguments,
    checked_scores,
)
from review_capacity.models.registry import Configuration

FORMAT_VERSION = 1


class MeanModel(ScoringModel):
    """Scores every row with the weighted fraud rate of Train (no library involved)."""

    rate: float | None = None
    rows: int = 0
    iterations: int = 0
    hyperparameters: dict[str, Any]

    def fit(
        self,
        train: ModelData,
        valid: ModelData | None,
        hyperparameters: Mapping[str, Any],
        *,
        iterations: int | None = None,
    ) -> None:
        check_fit_arguments(valid, iterations)
        self.rate = float(np.average(train.labels, weights=train.weights))
        self.rows = len(train.labels)
        self.iterations = 1 if iterations is None else iterations
        self.hyperparameters = dict(hyperparameters)

    def _fitted_rate(self) -> float:
        if self.rate is None:
            raise ModelError("MeanModel is not fitted")
        return self.rate

    def predict(self, features: pd.DataFrame) -> Scores:
        rate = self._fitted_rate()
        return checked_scores(np.full(len(features), rate), len(features), "MeanModel")

    def save(self, directory: Path) -> None:
        payload = {
            "format_version": FORMAT_VERSION,
            "configuration": self.configuration.name,
            "seed": self.seed,
            "rate": self._fitted_rate(),
            "rows": self.rows,
            "iterations": self.iterations,
            "hyperparameters": self.hyperparameters,
        }
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(
        cls,
        directory: Path,
        configuration: Configuration,
        seed: int,
        *,
        settings: ModelSettings,
        device: Device,
    ) -> Self:
        payload = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if payload["format_version"] != FORMAT_VERSION:
            raise ModelError(f"Unsupported format version {payload['format_version']}")
        if (payload["configuration"], payload["seed"]) != (configuration.name, seed):
            raise ModelError("Saved model does not match the requested configuration and seed")
        model = cls(configuration, seed, settings=settings, device=device)
        model.rate = payload["rate"]
        model.rows = payload["rows"]
        model.iterations = payload["iterations"]
        model.hyperparameters = payload["hyperparameters"]
        return model

    @property
    def parameters(self) -> dict[str, Any]:
        rate = self._fitted_rate()
        resolved = {"rate": rate, "rows": np.int64(self.rows)}  # a NumPy value, as libraries return
        return {"requested": {"seed": self.seed, **self.hyperparameters}, "resolved": resolved}

    @property
    def fitted_iterations(self) -> int:
        self._fitted_rate()
        return self.iterations
