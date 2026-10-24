"""The declarations in ``config/`` load and agree with the documented counts (no data read)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from review_capacity.config import (
    FeatureSpec,
    features_fingerprint,
    load_dataset_spec,
    load_feature_spec,
)
from review_capacity.errors import FileAccessError, SpecificationError
from review_capacity.paths import ProjectPaths
from support.synthetic import REPO_PATHS


def test_feature_declaration_counts() -> None:
    spec = load_feature_spec(REPO_PATHS.features_spec)
    assert len(spec.joined_columns) == 434
    assert len(spec.categorical) == 49
    assert len(spec.numeric) == 382
    assert spec.derived == ("tod_sin", "tod_cos")
    assert len(spec.model_inputs) == 433
    assert len(spec.numeric_inputs) == 384
    assert spec.target == "isFraud"
    assert set(spec.excluded) == {"TransactionID", "TransactionDT"}
    assert len(spec.numeric_coded) == 18


def test_dataset_declaration_matches_dr001() -> None:
    spec = load_dataset_spec(REPO_PATHS)
    assert [(p.name, p.first_day, p.last_day, p.rows) for p in spec.partitions] == [
        ("train", 1, 105, 368_632),
        ("v1", 106, 126, 65_544),
        ("v2", 127, 154, 77_822),
        ("test", 155, 182, 78_542),
    ]
    assert spec.total_rows == 590_540
    assert [p.name for p in spec.development] == ["train", "v1", "v2"]
    assert spec.partition("test").sealed
    assert spec.last_development_day == 154
    assert spec.label_latency_fitting_last_day == 147  # fit on V2 days 127–147, hold out 148–154
    rolling = spec.rolling_origin
    assert rolling is not None
    assert (rolling.origins, rolling.gap_days, rolling.window_days, rolling.transfer_days) == (
        (71, 85, 99),
        21,
        28,
        28,
    )
    # training days 1–49, 1–63, 1–77; windows s..s+27; transfers s+28..s+55 (the last is V2)
    assert [rolling.training_last_day(o) for o in rolling.origins] == [49, 63, 77]
    assert [rolling.window_last_day(o) for o in rolling.origins] == [98, 112, 126]
    assert [rolling.transfer_first_day(o) for o in rolling.origins] == [99, 113, 127]
    assert [rolling.transfer_last_day(o) for o in rolling.origins] == [126, 140, 154]


def test_repeated_input_rejected() -> None:
    spec = load_feature_spec(REPO_PATHS.features_spec)
    broken = replace(spec, numeric=(*spec.numeric, spec.numeric[0]))
    with pytest.raises(SpecificationError, match="Repeated"):
        broken.validate()


def test_non_contiguous_partitions_rejected() -> None:
    spec = load_dataset_spec(REPO_PATHS)
    v1 = replace(spec.partitions[1], first_day=107)
    broken = replace(spec, partitions=(spec.partitions[0], v1, *spec.partitions[2:]))
    with pytest.raises(SpecificationError, match="not contiguous"):
        broken.validate()


# Malformed declarations (finding: a malformed configuration must never pass silently).


def _write_dataset(root: Path, edit: Callable[[dict[str, Any]], None]) -> ProjectPaths:
    content = yaml.safe_load(REPO_PATHS.dataset_spec.read_text(encoding="utf-8"))
    edit(content)
    (root / "config").mkdir()
    (root / "config" / "dataset.yaml").write_text(yaml.safe_dump(content), encoding="utf-8")
    return ProjectPaths(root)


def _test_partition(content: dict[str, Any]) -> dict[str, Any]:
    part: dict[str, Any] = content["partitions"][-1]
    return part


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: _test_partition(c).update(sealed=None), "true or false"),
        (lambda c: _test_partition(c).update(sealed=0), "true or false"),
        (lambda c: _test_partition(c).update(sealed=False), "must be sealed"),
        (lambda c: _test_partition(c).update(transaction_id_sha256_prefix=""), "non-empty"),
        (lambda c: _test_partition(c).update(transaction_id_sha256_prefix="xyz"), "16 hex"),
        (lambda c: c["raw_files"]["identity"].update(sha256="abc"), "64 hex"),
        (lambda c: c.update(seconds_per_day=3600), "must be 86400"),
        (lambda c: c["partitions"][0].update(rows="368632"), "must be an integer"),
        (lambda c: c.pop("key"), "malformed declaration"),
        (lambda c: c.pop("label_latency"), "malformed declaration"),
        (lambda c: c["label_latency"].update(fitting_last_day=154), "held-out part"),
        (lambda c: c["label_latency"].update(fitting_last_day=126), "held-out part"),
        (lambda c: c.pop("rolling_origin"), "malformed declaration"),
        (lambda c: c["rolling_origin"].update(origins=[85, 71]), "increasing and distinct"),
        (lambda c: c["rolling_origin"].update(origins=[]), "increasing and distinct"),
        (lambda c: c["rolling_origin"].update(gap_days=0), "are positive"),
        (lambda c: c["rolling_origin"].update(origins=[71, 100]), "ends after day 154"),
        (lambda c: c["rolling_origin"].update(origins=[60]), "fewer than 42 days"),
        (lambda c: c["rolling_origin"].update(gap_days=-5, origins=[71]), "are positive"),
        (lambda c: c["rolling_origin"].update(window_days="28"), "must be an integer"),
    ],
)
def test_malformed_dataset_declaration_rejected(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None], message: str
) -> None:
    paths = _write_dataset(tmp_path, edit)
    with pytest.raises(SpecificationError, match=message):
        load_dataset_spec(paths)


def test_unreadable_yaml_rejected(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "dataset.yaml").write_text("key: [unclosed", encoding="utf-8")
    with pytest.raises(SpecificationError, match="cannot be read as YAML"):
        load_dataset_spec(ProjectPaths(tmp_path))


def test_missing_label_comes_from_the_declaration(tmp_path: Path) -> None:
    content = yaml.safe_load(REPO_PATHS.features_spec.read_text(encoding="utf-8"))
    assert load_feature_spec(REPO_PATHS.features_spec).missing_label == "__MISSING__"
    del content["encoding"]["categorical_canonical_form"]["missing_token"]
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    with pytest.raises(SpecificationError, match="malformed declaration"):
        load_feature_spec(path)


def test_missing_declaration_is_a_file_access_error(tmp_path: Path) -> None:
    with pytest.raises(FileAccessError):
        load_dataset_spec(ProjectPaths(tmp_path))


def test_repeated_yaml_key_rejected(tmp_path: Path) -> None:
    text = REPO_PATHS.dataset_spec.read_text(encoding="utf-8")
    assert "\ntotal_rows: 590540\n" in text
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "dataset.yaml").write_text(
        text.replace("\ntotal_rows: 590540\n", "\ntotal_rows: 590540\ntotal_rows: 1\n"),
        encoding="utf-8",
    )
    with pytest.raises(SpecificationError, match="cannot be read as YAML"):
        load_dataset_spec(ProjectPaths(tmp_path))


def _write_features(tmp_path: Path, edit: Callable[[dict[str, Any]], None]) -> Path:
    content = yaml.safe_load(REPO_PATHS.features_spec.read_text(encoding="utf-8"))
    edit(content)
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c["numeric"].update(columns="TransactionAmt"), "must be a list"),
        (lambda c: c["categorical"].update(identity={"DeviceType": 1}), "must be a list"),
        (lambda c: c["categorical"]["numeric_coded"].append("TransactionAmt"), "not categorical"),
        (lambda c: c["numeric"]["columns"].append("isFraud"), "Repeated joined columns"),
        (lambda c: c["derived"]["numeric"][0].update(name="TransactionID"), "reuse joined"),
        (lambda c: c["source"].update(model_input_count=432), "432 expected"),
        (lambda c: c["source"].update(joined_column_count="434"), "must be an integer"),
    ],
)
def test_malformed_feature_declaration_rejected(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None], message: str
) -> None:
    with pytest.raises(SpecificationError, match=message):
        load_feature_spec(_write_features(tmp_path, edit))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c["raw_files"]["identity"]["columns_besides_key"].pop(), "40 expected"),
        (
            lambda c: c["raw_files"]["identity"]["columns_besides_key"].append("TransactionID"),
            "must not list the key",
        ),
        (lambda c: c["partitions"][1].update(transaction_dt=[9158441, 9158440]), "reversed"),
        (lambda c: c["partitions"][0].update(sealed=True), "lies after the sealed period"),
    ],
)
def test_malformed_split_or_identity_declaration_rejected(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None], message: str
) -> None:
    with pytest.raises(SpecificationError, match=message):
        load_dataset_spec(_write_dataset(tmp_path, edit))


def test_identity_columns_give_table_membership() -> None:
    spec = load_dataset_spec(REPO_PATHS)
    assert spec.source_table("DeviceInfo") == "identity"
    assert spec.source_table("id_38") == "identity"
    assert spec.source_table("V339") == "transaction"


def _fingerprint_of(edit: Callable[[FeatureSpec], FeatureSpec]) -> str:
    return features_fingerprint(edit(load_feature_spec(REPO_PATHS.features_spec)))


def test_the_fingerprint_is_the_same_for_the_same_specification() -> None:
    assert _fingerprint_of(lambda spec: spec) == _fingerprint_of(lambda spec: spec)
    assert re.fullmatch(r"[0-9a-f]{64}", _fingerprint_of(lambda spec: spec))


def test_the_fingerprint_ignores_comments_and_library_handling(tmp_path: Path) -> None:
    text = REPO_PATHS.features_spec.read_text(encoding="utf-8")
    edited = tmp_path / "features.yaml"
    edited.write_text(
        "# note\n"
        + text.replace(
            'unseen_categories: "handled natively by CatBoost"',
            'unseen_categories: "handled natively"',
        ),
        encoding="utf-8",
    )
    assert features_fingerprint(load_feature_spec(edited)) == _fingerprint_of(lambda spec: spec)


@pytest.mark.parametrize(
    "edit",
    [
        lambda spec: replace(spec, target="label"),
        lambda spec: replace(spec, key="Identifier"),
        lambda spec: replace(spec, time="Time"),
        lambda spec: replace(spec, excluded=spec.excluded[:1]),
        lambda spec: replace(spec, categorical=spec.categorical[:-1]),
        lambda spec: replace(spec, categorical=spec.categorical[::-1]),
        lambda spec: replace(spec, numeric_coded=frozenset(sorted(spec.numeric_coded)[1:])),
        lambda spec: replace(spec, numeric=spec.numeric[1:]),
        lambda spec: replace(spec, derived=spec.derived[::-1]),
        lambda spec: replace(spec, missing_label="NA"),
    ],
)
def test_the_fingerprint_follows_every_role_the_build_uses(
    edit: Callable[[FeatureSpec], FeatureSpec],
) -> None:
    assert _fingerprint_of(edit) != _fingerprint_of(lambda spec: spec)
