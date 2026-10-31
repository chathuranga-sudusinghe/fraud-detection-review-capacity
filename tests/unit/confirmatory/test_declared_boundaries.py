"""The false-negative slice boundaries are computed from development data, written for entry into
the declaration, and required frozen before the Test run (Issue #211)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from review_capacity.confirmatory import command
from review_capacity.confirmatory.boundaries import render_yaml
from review_capacity.confirmatory.command import TestRunOptions, run_test
from review_capacity.errors import SpecificationError
from review_capacity.evaluation.declaration import parse_declaration
from review_capacity.evaluation.false_negatives import (
    Boundaries,
    check_frozen,
    compute_boundaries,
    require_frozen,
)
from support import command_stubs as stubs
from support import confirmatory as c
from support.synthetic import REPO_PATHS


def test_the_rendered_block_parses_back_to_the_same_boundaries() -> None:
    boundaries = Boundaries(
        (0.1, 12.5, 59.0, 117.5),
        {name: {"W": 400, "H": 120} for name in ("ProductCD", "card4", "card6", "DeviceType")},
    )
    block = yaml.safe_load("false_negative:\n" + render_yaml(boundaries))
    data = yaml.safe_load(REPO_PATHS.evaluation_spec.read_text(encoding="utf-8"))
    data["false_negative"]["boundaries"] = block["false_negative"]["boundaries"]
    frozen = parse_declaration(data).false_negative.boundaries
    assert frozen is not None
    assert frozen.amount_edges == boundaries.amount_edges  # exact, not rounded
    assert {k: list(v.items()) for k, v in frozen.kept.items()} == {
        k: list(v.items()) for k, v in boundaries.kept.items()
    }
    base = c.declaration()
    declared = dataclasses.replace(
        base, false_negative=dataclasses.replace(base.false_negative, boundaries=frozen)
    )
    check_frozen(boundaries, declared)


def test_boundaries_come_from_the_train_amounts_and_the_v2_fraud_levels() -> None:
    declaration = c.declaration()
    amounts = np.arange(1.0, 1001.0)
    levels = np.array(["W"] * 300 + ["H"] * 150 + ["R"] * 5, dtype=np.str_)
    computed = compute_boundaries(declaration, amounts, dict.fromkeys(c.POOLED_NAMES, levels))
    assert len(computed.amount_edges) == declaration.false_negative.amount_quantiles - 1
    assert list(computed.kept["card4"].items()) == [("W", 300), ("H", 150)]  # R is pooled


def _unfrozen() -> Any:
    base = c.declaration()
    return dataclasses.replace(
        base, false_negative=dataclasses.replace(base.false_negative, boundaries=None)
    )


def test_an_unfrozen_declaration_is_refused() -> None:
    with pytest.raises(SpecificationError, match="not frozen"):
        require_frozen(_unfrozen())
    require_frozen(c.outcome_declaration())


def test_the_test_run_does_not_start_without_frozen_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = stubs.install_evaluation_stubs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        command,
        "load_evaluation_declaration",
        lambda _paths: (_unfrozen(), SimpleNamespace(sha256=c.sha("evaluation"))),
    )
    options = TestRunOptions(
        tag=stubs.TAG,
        development=Path("outputs/development/x"),
        analyses=(Path("outputs/analysis/y"),),
        started=c.RUN_ID,
        phase="all",
    )
    with pytest.raises(SpecificationError, match="not frozen"):
        run_test(project["paths"], options, release=lambda: None)
