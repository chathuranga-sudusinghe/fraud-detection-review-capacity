"""The shared Train-only neural preprocessing (DR-012; Issue #149), on synthetic frames only."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from sklearn.preprocessing import QuantileTransformer

from review_capacity.errors import (
    DataValidationError,
    EnvironmentMismatchError,
    FileAccessError,
    SpecificationError,
)
from review_capacity.features import neural_preprocessing as module
from review_capacity.features.neural_preprocessing import (
    ARRAY_NAMES,
    ARRAYS_FILE,
    CATEGORY_DROPOUT,
    MANIFEST_FILE,
    NeuralPreprocessing,
    fit_neural_preprocessing,
    load_neural_preprocessing,
    ordered_ids_sha256,
    requested_quantiles,
)
from review_capacity.training.timing import rows_identity
from support.preprocessing import MISSING, canonical_frame
from support.synthetic import FEATURES

SEED = 0


def fitted(
    rows: int = 400, seed: int = 1, **options: Any
) -> tuple[NeuralPreprocessing, pd.DataFrame, Any, Any]:
    frame, ids, times = canonical_frame(rows, seed, **options)
    return fit_neural_preprocessing(frame, ids, times, FEATURES, SEED), frame, ids, times


def reference(frame: pd.DataFrame, seed: int) -> list[Any]:
    """The DR-012 procedure written out plainly: one generator, inputs in declared order,
    non-missing values in row order, one standard-normal vector per input."""
    rng = np.random.default_rng(seed)
    count = min(max(min(len(frame) // 30, 1000), 10), len(frame))
    transformers = []
    for name in FEATURES.numeric_inputs:
        column = frame[name].to_numpy()
        present = ~np.isnan(column)
        scale = 1e-3 / max(np.std(column[present]), 1e-3)
        noisy = column.copy()
        noisy[present] = column[present] + scale * rng.standard_normal(int(present.sum()))
        transformers.append(
            QuantileTransformer(
                n_quantiles=count, output_distribution="normal", subsample=None, random_state=seed
            ).fit(noisy.reshape(-1, 1))
        )
    return transformers


# --- fitting -----------------------------------------------------------------------------------


def test_the_requested_quantile_count_follows_the_formula_not_a_constant() -> None:
    assert requested_quantiles(368_632) == 1_000
    assert requested_quantiles(30_000) == 1_000
    assert requested_quantiles(600) == 20
    assert requested_quantiles(300) == 10
    assert requested_quantiles(5) == 10
    small, *_ = fitted(rows=300)
    assert (small.requested_quantiles, small.actual_quantiles) == (10, 10)
    assert small.quantiles.shape == (10, len(FEATURES.numeric_inputs))
    # scikit-learn cannot use more quantiles than rows
    tiny, *_ = fitted(rows=6, missing={"TransactionAmt": 0.0, "id_01": 0.0})
    assert (tiny.requested_quantiles, tiny.actual_quantiles) == (10, 6)


def test_an_input_with_fewer_observed_values_than_quantiles_keeps_the_common_grid() -> None:
    frame, ids, times = canonical_frame(rows=300)  # n_q = 10
    frame.loc[5:, "V1"] = np.nan  # five observed values, fewer than n_q
    artifact = fit_neural_preprocessing(frame, ids, times, FEATURES, SEED)
    assert frame["V1"].notna().sum() == 5
    assert artifact.quantiles.shape == (10, len(FEATURES.numeric_inputs))
    whole_column = reference(frame, SEED)[1]  # scikit-learn fitted on the column containing NaN
    np.testing.assert_array_equal(artifact.quantiles[:, 1], whole_column.quantiles_[:, 0])


def test_noise_draws_follow_the_declared_procedure_with_irregular_missingness() -> None:
    artifact, frame, *_ = fitted(missing={"TransactionAmt": 0.2, "V1": 0.05, "id_01": 0.6})
    for j, transformer in enumerate(reference(frame, SEED)):
        np.testing.assert_array_equal(artifact.quantiles[:, j], transformer.quantiles_[:, 0])
    column = frame["id_01"].to_numpy()
    present = ~np.isnan(column)
    assert artifact.std[2] == np.std(column[present])  # ddof 0, non-missing values only
    assert artifact.noise_scale[2] == 1e-3 / max(np.std(column[present]), 1e-3)


def test_the_fit_is_repeatable_and_leaves_the_frame_and_random_states_unchanged() -> None:
    frame, ids, times = canonical_frame()
    before = frame.copy(deep=True)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    first = fit_neural_preprocessing(frame, ids, times, FEATURES, SEED)
    second = fit_neural_preprocessing(frame, ids, times, FEATURES, SEED)
    pd.testing.assert_frame_equal(frame, before)
    assert first.artifact_sha256 == second.artifact_sha256
    np.testing.assert_array_equal(first.quantiles, second.quantiles)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    assert after[2:] == numpy_state[2:]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert fit_neural_preprocessing(frame, ids, times, FEATURES, 7).artifact_sha256 != (
        first.artifact_sha256
    )


def test_the_transformation_is_applied_to_the_original_noise_free_values() -> None:
    artifact, frame, *_ = fitted()
    transformed = artifact.transform_numeric(frame)
    for j, transformer in enumerate(reference(frame, SEED)):
        column = frame.iloc[:, len(FEATURES.categorical) + j].to_numpy()
        expected = transformer.transform(column.reshape(-1, 1))[:, 0]
        expected[np.isnan(column)] = 0.0
        np.testing.assert_array_equal(transformed[:, j], expected)


def test_a_dominant_repeated_value_is_not_pushed_to_an_extreme() -> None:
    artifact, frame, *_ = fitted()
    zeros = (frame["V1"] == 0.0).to_numpy()
    assert zeros.sum() > 150
    assert (np.abs(artifact.transform_numeric(frame)[zeros, 1]) < 2.0).all()


def test_an_input_constant_in_train_is_fitted_with_noise_scale_one() -> None:
    frame, ids, times = canonical_frame()
    frame["id_01"] = frame["id_01"].where(frame["id_01"].isna(), 0.0)
    artifact = fit_neural_preprocessing(frame, ids, times, FEATURES, SEED)
    assert artifact.std[2] == 0.0
    assert artifact.noise_scale[2] == 1.0
    transformed = artifact.transform_numeric(frame)
    assert np.isfinite(transformed).all()
    assert np.abs(transformed[:, 2]).max() < 5.2


def test_an_input_entirely_missing_in_train_stops_before_any_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, ids, times = canonical_frame()
    frame["V1"] = np.nan

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("a generator was created before the check")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    with pytest.raises(DataValidationError, match="entirely missing in Train"):
        fit_neural_preprocessing(frame, ids, times, FEATURES, SEED)


def test_infinite_numeric_values_and_unlabelled_categorical_missing_values_are_refused() -> None:
    frame, ids, times = canonical_frame()
    infinite = frame.copy()
    infinite.loc[3, "TransactionAmt"] = np.inf
    with pytest.raises(DataValidationError, match="infinite"):
        fit_neural_preprocessing(infinite, ids, times, FEATURES, SEED)
    unlabelled = frame.copy()
    unlabelled["card1"] = pd.Series([None] * len(frame), dtype="str")
    with pytest.raises(DataValidationError, match="label"):
        fit_neural_preprocessing(unlabelled, ids, times, FEATURES, SEED)
    with pytest.raises(DataValidationError, match="columns"):
        fit_neural_preprocessing(frame.drop(columns="V1"), ids, times, FEATURES, SEED)


def test_rows_must_be_in_canonical_order_with_ties_broken_by_identifier() -> None:
    frame, ids, times = canonical_frame(rows=40)
    ties = np.repeat(times[::2], 2)  # equal times in pairs, ascending identifiers: valid
    assert fit_neural_preprocessing(frame, ids, ties, FEATURES, SEED).train_rows == 40
    descending = ids.copy()
    descending[0], descending[1] = ids[1], ids[0]  # a tie with descending identifiers
    with pytest.raises(DataValidationError, match="canonical order"):
        fit_neural_preprocessing(frame, descending, ties, FEATURES, SEED)
    with pytest.raises(DataValidationError, match="canonical order"):
        fit_neural_preprocessing(frame, ids, times[::-1].copy(), FEATURES, SEED)


def test_row_metadata_must_match_the_fitted_rows() -> None:
    frame, ids, times = canonical_frame(rows=40)
    wrong: list[tuple[Any, Any]] = [
        (ids[:0], times[:0]),
        (ids[:-1], times[:-1]),
        (ids.astype(np.int32), times),
        (ids, times.astype(np.float64)),
        (ids.reshape(-1, 1), times),
    ]
    for bad_ids, bad_times in wrong:
        with pytest.raises(DataValidationError, match="int64 value for each of 40 rows"):
            fit_neural_preprocessing(frame, bad_ids, bad_times, FEATURES, SEED)


def test_the_identity_of_the_train_rows_is_the_ordered_identifier_hash() -> None:
    artifact, _, ids, _ = fitted()
    assert artifact.train_ids_sha256 == ordered_ids_sha256(ids) == rows_identity(ids)
    assert artifact.train_ids_sha256 != ordered_ids_sha256(ids[::-1].copy())


# --- vocabularies, indicators and the layout ---------------------------------------------------


def test_missing_has_index_zero_in_every_vocabulary_exactly_once() -> None:
    artifact, *_ = fitted()  # card1 and DeviceType already hold "__MISSING__" in Train
    assert artifact.categorical_inputs == FEATURES.categorical
    for name, vocabulary in zip(artifact.categorical_inputs, artifact.vocabularies, strict=True):
        assert vocabulary[0] == MISSING, name
        assert vocabulary.count(MISSING) == 1, name
        assert list(vocabulary[1:]) == sorted(set(vocabulary[1:]))
    by_name = dict(zip(artifact.categorical_inputs, artifact.vocabularies, strict=True))
    assert by_name["ProductCD"] == (MISSING, "C", "H", "R", "W")  # none missing in Train
    assert by_name["id_13"] == (MISSING, "49", "52")
    assert artifact.layout.cardinalities == (5, 4, 3, 3)
    assert artifact.layout.missing_indices == (0, 0, 0, 0)
    assert artifact.layout.category_dropout == CATEGORY_DROPOUT == 0.01


def test_missing_and_unseen_values_map_to_index_zero() -> None:
    artifact, frame, *_ = fitted()
    scored = frame.head(4).copy()
    scored["ProductCD"] = pd.Series(["ZZ", "W", "C", MISSING], dtype="str")
    scored["card1"] = pd.Series([MISSING, "1000", "9999", "2000"], dtype="str")
    indices = artifact.category_indices(scored)
    assert indices.dtype == np.int64
    assert indices[:, 0].tolist() == [0, 4, 1, 0]
    assert indices[:, 1].tolist() == [0, 1, 0, 2]


def test_indicators_follow_the_train_missingness_in_numeric_input_order() -> None:
    artifact, frame, *_ = fitted()
    assert artifact.indicators == ("TransactionAmt", "id_01")  # not contiguous: V1 sits between
    block = artifact.missingness_indicators(frame)
    assert block.dtype == np.float64
    assert block.shape == (len(frame), 2)
    np.testing.assert_array_equal(block[:, 0], frame["TransactionAmt"].isna().to_numpy())
    assert artifact.layout.numeric == (
        *FEATURES.numeric_inputs,
        "TransactionAmt__missing",
        "id_01__missing",
    )


def test_missingness_that_first_appears_after_train_has_no_indicator_and_is_reported() -> None:
    artifact, frame, *_ = fitted()
    later = frame.head(50).copy()
    later.loc[later.index[:3], "V1"] = np.nan
    assert "V1" not in artifact.indicators
    transformed = artifact.transform_numeric(later)
    assert (transformed[:3, 1] == 0.0).all()
    report = artifact.diagnostics("v1", later)
    assert {"name": "V1", "missing": 3} in report["missing_without_indicator"]
    assert report["missing_without_indicator_total"] >= 3


def test_the_neural_inputs_follow_the_layout_and_are_the_only_float32_conversion() -> None:
    artifact, frame, *_ = fitted()
    inputs = artifact.to_neural_inputs(frame)
    inputs.check(artifact.layout)
    n = len(FEATURES.numeric_inputs)
    assert inputs.numeric.dtype == np.float32
    assert inputs.categorical.dtype == np.int64
    assert inputs.numeric.shape == (len(frame), n + len(artifact.indicators))
    wide = artifact.transform_numeric(frame)
    assert wide.dtype == np.float64
    assert not np.array_equal(wide.astype(np.float32).astype(np.float64), wide)  # float32 rounds
    np.testing.assert_array_equal(inputs.numeric[:, :n], wide.astype(np.float32))
    np.testing.assert_array_equal(
        inputs.numeric[:, n:], artifact.missingness_indicators(frame).astype(np.float32)
    )


def test_time_of_day_over_the_whole_cycle_stays_finite_and_distinct() -> None:
    artifact, frame, *_ = fitted(rows=2_000)
    transformed = artifact.transform_numeric(frame)[:, 3:5]
    assert np.isfinite(transformed).all()
    first, last = transformed[0], transformed[-1]  # second 0 and the last second of the cycle
    assert not np.array_equal(first, last)
    angles = np.linspace(0, len(frame) - 1, 8).astype(int)
    assert len({tuple(pair) for pair in transformed[angles].tolist()}) == 8


# --- applying the artifact -----------------------------------------------------------------------


def test_values_outside_and_at_the_fitted_range_map_to_finite_monotone_bounds() -> None:
    artifact, frame, *_ = fitted()
    low, high = artifact.quantiles[0, 0], artifact.quantiles[-1, 0]
    probe = frame.head(9).copy()
    probe["TransactionAmt"] = [
        low - 1e3, low - 1e-3, low - 1e-8, low, low + 1e-8,
        high - 1e-8, high, high + 1e-3, high + 1e9,
    ]  # fmt: skip
    transformed = artifact.transform_numeric(probe)[:, 0]
    assert np.isfinite(transformed).all()
    assert (np.diff(transformed) >= 0).all()
    assert transformed[0] < -5.19
    assert transformed[-1] > 5.19
    report = artifact.diagnostics("v1", probe)
    amount = report["numeric"][0]
    assert (amount["below"], amount["above"], amount["non_missing"]) == (3, 2, 9)
    assert amount["share_outside"] == 5 / 9


def test_diagnostics_report_unseen_values_and_undefined_shares() -> None:
    artifact, frame, *_ = fitted()
    scored = frame.head(10).copy()
    scored["ProductCD"] = pd.Series(["ZZ"] * 4 + ["W"] * 6, dtype="str")
    scored["card1"] = pd.Series([MISSING] * 10, dtype="str")
    scored["V1"] = np.nan
    report = artifact.diagnostics("v2", scored)
    by_name = {entry["name"]: entry for entry in report["categorical"]}
    assert (by_name["ProductCD"]["unseen"], by_name["ProductCD"]["share_unseen"]) == (4, 0.4)
    assert by_name["card1"]["non_missing"] == 0
    assert by_name["card1"]["share_unseen"] is None
    numeric = {entry["name"]: entry for entry in report["numeric"]}
    assert numeric["V1"]["non_missing"] == 0
    assert numeric["V1"]["share_outside"] is None
    assert report["rows"] == 10
    assert report["partition"] == "v2"
    json.dumps(report)  # plain data


def test_applying_the_artifact_neither_refits_nor_draws_and_ignores_the_scored_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, frame, *_ = fitted()
    digest = artifact.artifact_sha256
    quantiles = artifact.quantiles.copy()
    other, *_ = canonical_frame(rows=300, seed=9, missing={"V1": 0.4})
    other.loc[other.index[:5], "ProductCD"] = "NEW"  # values, categories, missingness all differ

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("scoring drew random numbers")

    numpy_state, python_state = np.random.get_state(), random.getstate()
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(QuantileTransformer, "fit", forbidden)
    artifact.to_neural_inputs(other)
    artifact.to_neural_inputs(frame)
    artifact.diagnostics("v1", other)
    assert artifact.artifact_sha256 == digest
    np.testing.assert_array_equal(artifact.quantiles, quantiles)
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    with pytest.raises(ValueError, match="read-only"):
        artifact.quantiles[0, 0] = 0.0


def test_a_frame_without_the_declared_inputs_or_with_missing_text_is_refused() -> None:
    artifact, frame, *_ = fitted()
    with pytest.raises(DataValidationError, match="columns"):
        artifact.to_neural_inputs(frame.drop(columns="id_01"))
    with pytest.raises(DataValidationError, match="empty"):
        artifact.to_neural_inputs(frame.head(0))
    unlabelled = frame.copy()
    unlabelled["card1"] = pd.Series([None] * len(frame), dtype="str")
    with pytest.raises(DataValidationError, match="label"):
        artifact.to_neural_inputs(unlabelled)
    integers = frame.copy()
    integers["V1"] = integers["V1"].fillna(0).astype("int64")
    with pytest.raises(DataValidationError, match="float64"):
        artifact.to_neural_inputs(integers)


# --- persistence ---------------------------------------------------------------------------------


def saved(tmp_path: Path, name: str = "artifact") -> tuple[NeuralPreprocessing, Path, str]:
    artifact, *_ = fitted()
    directory = tmp_path / name
    return artifact, directory, artifact.save(directory)


def test_save_and_reload_are_bitwise_identical(tmp_path: Path) -> None:
    artifact, directory, digest = saved(tmp_path)
    _, frame, ids, _ = fitted()
    assert sorted(path.name for path in directory.iterdir()) == [ARRAYS_FILE, MANIFEST_FILE]
    loaded = load_neural_preprocessing(
        directory, FEATURES, SEED, train_ids_sha256=ordered_ids_sha256(ids)
    )
    assert loaded.artifact_sha256 == digest == artifact.artifact_sha256
    wide = artifact.transform_numeric(frame)  # ties, missing values and the float64 values
    assert wide.tobytes() == loaded.transform_numeric(frame).tobytes()
    expected, actual = artifact.to_neural_inputs(frame), loaded.to_neural_inputs(frame)
    assert expected.numeric.tobytes() == actual.numeric.tobytes()
    np.testing.assert_array_equal(expected.categorical, actual.categorical)
    assert loaded.layout == artifact.layout
    _, _, again = saved(tmp_path, "again")  # the digest does not depend on file times
    assert again == digest
    with pytest.raises(FileAccessError, match="never replaced"):
        artifact.save(directory)


def rewrite(
    directory: Path,
    *,
    arrays: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    recompute: bool = True,
) -> None:
    """Change the stored state; with ``recompute`` both digests are made valid again, so that
    only the semantic validation can refuse the change."""
    text = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    with np.load(directory / ARRAYS_FILE, allow_pickle=False) as archive:
        stored: dict[str, npt.NDArray[np.float64]] = {name: archive[name] for name in ARRAY_NAMES}
    stored.update(arrays or {})
    text.update(manifest or {})
    if recompute:
        text["arrays_sha256"] = module.arrays_sha256(stored)
        text["artifact_sha256"] = module._digest_of(text)
    with (directory / ARRAYS_FILE).open("wb") as handle:
        np.savez(
            handle,
            quantiles=stored["quantiles"],
            references=stored["references"],
            std=stored["std"],
            noise_scale=stored["noise_scale"],
        )
    (directory / MANIFEST_FILE).write_text(json.dumps(text), encoding="utf-8")


def test_changed_bytes_are_detected_by_the_digests(tmp_path: Path) -> None:
    _, directory, _ = saved(tmp_path)
    with np.load(directory / ARRAYS_FILE, allow_pickle=False) as archive:
        quantiles = archive["quantiles"].copy()
    quantiles[0, 0] -= 1.0
    rewrite(directory, arrays={"quantiles": quantiles}, recompute=False)
    with pytest.raises(DataValidationError, match="arrays differ"):
        load_neural_preprocessing(directory, FEATURES, SEED)
    _, other, _ = saved(tmp_path, "second")
    rewrite(other, manifest={"seed": 5}, recompute=False)
    with pytest.raises(DataValidationError, match="manifest does not match"):
        load_neural_preprocessing(other, FEATURES, SEED)


def test_a_different_environment_declaration_seed_or_train_rows_is_refused(tmp_path: Path) -> None:
    _, directory, _ = saved(tmp_path)
    with pytest.raises(SpecificationError, match="seed"):
        load_neural_preprocessing(directory, FEATURES, 1)
    with pytest.raises(SpecificationError, match="different Train rows"):
        load_neural_preprocessing(directory, FEATURES, SEED, train_ids_sha256="0" * 64)
    reordered = replace(FEATURES, numeric=("TransactionAmt", "id_01", "V1"))
    with pytest.raises(SpecificationError, match="different feature declaration"):
        load_neural_preprocessing(directory, reordered, SEED)
    older = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))["versions"]
    rewrite(directory, manifest={"versions": older | {"scikit-learn": "0.0.1"}})
    with pytest.raises(EnvironmentMismatchError, match="installed"):
        load_neural_preprocessing(directory, FEATURES, SEED)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("decreasing", "nondecreasing"),
        ("grid", "grid"),
        ("scale", "noise_scale"),
        ("shape", "shape"),
        ("dtype", "float64"),
        ("nan", "finite"),
        ("vocabulary", "index 0"),
        ("duplicate", "sorted"),
        ("indicators", "numeric-input order"),
        ("counts", "quantile count"),
        ("train_null", "malformed entries"),
        ("numeric_labels", "malformed entries"),
    ],
)
def test_malformed_state_with_valid_digests_is_refused(
    tmp_path: Path, change: str, message: str
) -> None:
    _, directory, _ = saved(tmp_path)
    stored = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    with np.load(directory / ARRAYS_FILE, allow_pickle=False) as archive:
        quantiles = archive["quantiles"].copy()
        references = archive["references"].copy()
        scale = archive["noise_scale"].copy()
    vocabularies = stored["vocabularies"]
    arrays: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    if change == "decreasing":
        arrays["quantiles"] = quantiles[::-1].copy()
    elif change == "grid":
        arrays["references"] = references[::-1].copy()
    elif change == "scale":
        arrays["noise_scale"] = scale * 2
    elif change == "shape":
        arrays["quantiles"] = quantiles[:-1]
    elif change == "dtype":
        arrays["quantiles"] = quantiles.astype(np.float32)
    elif change == "nan":
        quantiles[0, 0] = np.nan
        arrays["quantiles"] = quantiles
    elif change == "vocabulary":
        vocabularies["ProductCD"] = [*vocabularies["ProductCD"][1:], MISSING]
        manifest["vocabularies"] = vocabularies
    elif change == "duplicate":
        vocabularies["ProductCD"] = [MISSING, "W", "W", "C"]
        manifest["vocabularies"] = vocabularies
    elif change == "indicators":
        manifest["indicators"] = ["id_01", "TransactionAmt"]
    elif change == "train_null":
        manifest["train"] = None
    elif change == "numeric_labels":
        vocabularies["ProductCD"] = [MISSING, 1, 2]
        manifest["vocabularies"] = vocabularies
    else:
        manifest["quantiles"] = {"requested": 1_000, "actual": 400}
    rewrite(directory, arrays=arrays, manifest=manifest)
    with pytest.raises(DataValidationError, match=message):
        load_neural_preprocessing(directory, FEATURES, SEED)


def test_a_reordered_schema_with_valid_digests_is_refused(tmp_path: Path) -> None:
    _, directory, _ = saved(tmp_path)
    names = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))["numeric_inputs"]
    rewrite(directory, manifest={"numeric_inputs": [names[1], names[0], *names[2:]]})
    with pytest.raises(SpecificationError, match="differ from the feature declaration"):
        load_neural_preprocessing(directory, FEATURES, SEED)


def test_missing_or_unreadable_files_are_project_errors(tmp_path: Path) -> None:
    with pytest.raises(FileAccessError):
        load_neural_preprocessing(tmp_path / "absent", FEATURES, SEED)
    _, directory, _ = saved(tmp_path)
    (directory / MANIFEST_FILE).write_text("not json", encoding="utf-8")
    with pytest.raises(DataValidationError, match="JSON"):
        load_neural_preprocessing(directory, FEATURES, SEED)
    (directory / MANIFEST_FILE).write_text('{"kind": "other"}', encoding="utf-8")
    with pytest.raises(DataValidationError, match="expected keys"):
        load_neural_preprocessing(directory, FEATURES, SEED)
