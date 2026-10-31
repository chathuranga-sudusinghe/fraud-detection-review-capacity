"""The frozen thresholds and calibrators come from the bound records (Issue #205)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from review_capacity.confirmatory.frozen_policies import R0_THRESHOLD, load_policies
from review_capacity.development.records import CONFIGURATION_NAMES
from review_capacity.errors import SpecificationError
from review_capacity.evaluation.calibration import fit_isotonic, fit_platt
from review_capacity.hashing import file_sha256

CAPACITIES = [0.01, 0.02, 0.05, 0.1]
NAMES = ("LGB-b", "Cat-u")


def write(root: Path, relative: str, value: Any) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": relative, "sha256": file_sha256(path)}


@pytest.fixture
def frozen(tmp_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(3)
    scores = np.clip(rng.random(500), 0.001, 0.999)
    labels = (rng.random(500) < 0.7 * scores).astype(np.int8)
    thresholds: list[dict[str, str]] = []
    calibrators: list[dict[str, str]] = []
    for index, name in enumerate(NAMES):
        thresholds.append(
            write(
                tmp_path,
                f"analysis/records/r1_selection-{name}-V2-primary.json",
                {
                    "record": "r1_selection",
                    "configuration": name,
                    "period": "primary",
                    "status": "selected",
                    "threshold": 0.3 + 0.01 * index,
                },
            )
        )
        thresholds += [
            write(
                tmp_path,
                f"analysis/records/evaluation-{name}-R2-{capacity}.json",
                {
                    "record": "evaluation",
                    "configuration": name,
                    "policy": "R2",
                    "partition": "V2",
                    "period": "primary",
                    "model_role": "selected",
                    "capacity": capacity,
                    "threshold": 0.5 - capacity,
                },
            )
            for capacity in CAPACITIES
        ]
        for method in ("platt", "isotonic"):
            artifact = f"analysis/calibrators/{name}-{method}-primary.json"
            target = tmp_path / artifact
            target.parent.mkdir(parents=True, exist_ok=True)
            (
                fit_platt(scores, labels)[0] if method == "platt" else fit_isotonic(scores, labels)
            ).save(target)
            saved = {"path": artifact, "sha256": file_sha256(target)}
            calibrators.append(saved)
            calibrators.append(
                write(
                    tmp_path,
                    f"analysis/records/calibration-{name}-{method}-primary.json",
                    {
                        "record": "calibration",
                        "configuration": name,
                        "method": method,
                        "period": "primary",
                        "status": "completed",
                        "artifact": {
                            "path": f"calibrators/{name}-{method}-primary.json",
                            "sha256": saved["sha256"],
                        },
                    },
                )
            )
    return {"thresholds": thresholds, "calibrators": calibrators}


def test_the_thresholds_and_calibrators_come_from_the_bound_records(
    tmp_path: Path, frozen: dict[str, Any]
) -> None:
    policies = load_policies(tmp_path, frozen, CAPACITIES, NAMES)
    lgb = policies.thresholds["LGB-b"]
    assert (lgb.r0, lgb.r1) == (R0_THRESHOLD, 0.3)
    assert lgb.r2 == {c: 0.5 - c for c in CAPACITIES}
    assert policies.thresholds["Cat-u"].r1 == pytest.approx(0.31)
    assert set(policies.calibrators) == {(n, m) for n in NAMES for m in ("platt", "isotonic")}
    assert policies.calibrators[("LGB-b", "platt")].predict(np.array([0.2, 0.8])).shape == (2,)
    ref = policies.calibrator_files[("LGB-b", "isotonic")]
    assert ref["path"].endswith("LGB-b-isotonic-primary.json")


def test_all_eight_configurations_are_named_by_default(
    tmp_path: Path, frozen: dict[str, Any]
) -> None:
    with pytest.raises(SpecificationError, match="not one"):
        load_policies(tmp_path, frozen, CAPACITIES)  # the other six configurations are absent
    assert len(CONFIGURATION_NAMES) == 8


def test_a_changed_frozen_file_is_refused(tmp_path: Path, frozen: dict[str, Any]) -> None:
    (tmp_path / frozen["thresholds"][1]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(SpecificationError, match="differs from the frozen file"):
        load_policies(tmp_path, frozen, CAPACITIES, NAMES)


def test_a_changed_calibrator_file_is_refused(tmp_path: Path, frozen: dict[str, Any]) -> None:
    saved = next(
        r
        for r in frozen["calibrators"]
        if r["path"].endswith("LGB-b-platt-primary.json") and "calibrators/" in r["path"]
    )
    (tmp_path / saved["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(SpecificationError, match="differs from the frozen file"):
        load_policies(tmp_path, frozen, CAPACITIES, NAMES)


def test_a_missing_or_ambiguous_item_stops_the_load(tmp_path: Path, frozen: dict[str, Any]) -> None:
    fewer = {**frozen, "thresholds": frozen["thresholds"][1:]}  # no R1 selection of LGB-b
    with pytest.raises(SpecificationError, match="primary R1 selection of LGB-b"):
        load_policies(tmp_path, fewer, CAPACITIES, NAMES)
    twice = {**frozen, "thresholds": [*frozen["thresholds"], frozen["thresholds"][0]]}
    with pytest.raises(SpecificationError, match="2 records"):
        load_policies(tmp_path, twice, CAPACITIES, NAMES)
    with pytest.raises(SpecificationError, match=r"R2 evaluation of LGB-b at capacity 0\.03"):
        load_policies(tmp_path, frozen, [*CAPACITIES, 0.03], NAMES)
    unsaved = {
        **frozen,
        "calibrators": [r for r in frozen["calibrators"] if "calibrators/" not in r["path"]],
    }
    with pytest.raises(SpecificationError, match="not among the frozen inputs"):
        load_policies(tmp_path, unsaved, CAPACITIES, NAMES)


def test_a_calibrator_built_from_the_verified_bytes_equals_the_one_loaded_from_the_file(
    tmp_path: Path, frozen: dict[str, Any]
) -> None:
    from review_capacity.evaluation.calibration import load_calibrator, load_calibrator_bytes

    scores = np.array([0.05, 0.3, 0.6, 0.95])
    for method in ("platt", "isotonic"):
        ref = next(
            r
            for r in frozen["calibrators"]
            if r["path"].endswith(f"LGB-b-{method}-primary.json") and "calibrators/" in r["path"]
        )
        path = tmp_path / ref["path"]
        from_file = load_calibrator(method, path)
        from_bytes = load_calibrator_bytes(method, path.read_bytes(), ref["path"])
        assert from_bytes.predict(scores).tolist() == from_file.predict(scores).tolist()
    with pytest.raises(Exception, match="cannot be read"):
        load_calibrator_bytes("platt", b"not json", "x.json")
    with pytest.raises(Exception, match="cannot be read"):
        load_calibrator_bytes("isotonic", b"not an archive", "x.npz")
