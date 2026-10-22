"""Migration of provenance records to the feature-declaration fingerprint (Issue #125)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from review_capacity.config import features_fingerprint
from review_capacity.data.migration import migrate_provenance
from review_capacity.data.processed import FINGERPRINT, stems, verify_processed
from review_capacity.errors import DataValidationError
from review_capacity.hashing import file_sha256
from support.synthetic import Synthetic, build_processed


@pytest.fixture
def processed(tmp_path: Path) -> Synthetic:
    return build_processed(tmp_path)


def _record(data: Synthetic, stem: str) -> dict[str, Any]:
    record = json.loads(data.paths.provenance_file(stem).read_text(encoding="utf-8"))
    assert isinstance(record, dict)
    return record


def _strip(data: Synthetic, stems_to_strip: tuple[str, ...] | None = None) -> None:
    """Records as the first build wrote them: the whole-file hash and no fingerprint."""
    for stem in stems_to_strip or stems(data.dataset):
        record = _record(data, stem)
        del record["configuration"][FINGERPRINT]
        data.paths.provenance_file(stem).write_text(json.dumps(record), encoding="utf-8")


def _migrate(data: Synthetic, **options: bool) -> Any:
    return migrate_provenance(data.paths, data.features, data.dataset, **options)


def test_a_new_build_records_the_fingerprint(processed: Synthetic) -> None:
    expected = features_fingerprint(processed.features)
    for stem in stems(processed.dataset):
        assert _record(processed, stem)["configuration"][FINGERPRINT] == expected


def test_migration_rerecords_without_touching_the_data(processed: Synthetic) -> None:
    _strip(processed)
    for stem in stems(processed.dataset):
        with pytest.raises(DataValidationError, match="run migrate-provenance"):
            verify_processed(processed.paths, stem, processed.features, processed.dataset)
    names = stems(processed.dataset)
    data_hashes = {s: file_sha256(processed.paths.processed_file(s)) for s in names}
    old = {s: file_sha256(processed.paths.provenance_file(s)) for s in names}
    original_records = {s: processed.paths.provenance_file(s).read_text("utf-8") for s in names}
    result = _migrate(processed)
    assert result.migrated == stems(processed.dataset)
    assert result.current == ()
    for stem in stems(processed.dataset):
        verify_processed(processed.paths, stem, processed.features, processed.dataset)
        record = _record(processed, stem)
        assert file_sha256(processed.paths.processed_file(stem)) == data_hashes[stem]
        assert record["sha256"] == data_hashes[stem]
        assert record["migration"]["previous_record_sha256"] == old[stem]
        assert record["migration"]["data_file_unchanged_sha256"] == data_hashes[stem]
        assert record["migration"]["issue"] == 125
        original = json.loads(original_records[stem])
        for field in ("sources", "code", "rows", "columns", "scope", "transaction_id_sha256"):
            assert record[field] == original[field]
        declaration_hash = original["configuration"]["features.yaml"]
        assert record["configuration"]["features.yaml"] == declaration_hash


def test_check_only_verifies_and_writes_nothing(processed: Synthetic) -> None:
    _strip(processed)
    before = {s: processed.paths.provenance_file(s).read_bytes() for s in stems(processed.dataset)}
    result = _migrate(processed, check_only=True)
    assert result.check_only is True
    assert result.migrated == stems(processed.dataset)
    for stem, content in before.items():
        assert processed.paths.provenance_file(stem).read_bytes() == content


def test_a_second_migration_changes_nothing(processed: Synthetic) -> None:
    _strip(processed)
    _migrate(processed)
    after = {s: processed.paths.provenance_file(s).read_bytes() for s in stems(processed.dataset)}
    result = _migrate(processed)
    assert result.migrated == ()
    assert result.current == stems(processed.dataset)
    for stem, content in after.items():
        assert processed.paths.provenance_file(stem).read_bytes() == content


def test_an_interrupted_migration_is_completed_by_running_it_again(processed: Synthetic) -> None:
    _strip(processed, ("v1", "v2"))  # the joined file and train were migrated before the stop
    result = _migrate(processed)
    assert result.migrated == ("v1", "v2")
    assert result.current == ("development_joined", "train")


def test_an_altered_data_file_stops_the_migration_before_any_record_is_written(
    processed: Synthetic,
) -> None:
    _strip(processed)
    path = processed.paths.processed_file("v2")
    frame = pd.read_parquet(path)
    frame.loc[0, "TransactionAmt"] = 999.0
    frame.to_parquet(path, index=False)
    before = {s: processed.paths.provenance_file(s).read_bytes() for s in stems(processed.dataset)}
    with pytest.raises(DataValidationError, match=r"SHA-256 .* differs from its record"):
        _migrate(processed)
    for stem, content in before.items():
        assert processed.paths.provenance_file(stem).read_bytes() == content


def test_a_declaration_changed_since_the_build_cannot_be_migrated(processed: Synthetic) -> None:
    _strip(processed)
    with processed.paths.features_spec.open("a", encoding="utf-8") as handle:
        handle.write("# edited after the build\n")
    with pytest.raises(
        DataValidationError, match=r"Migrate with the features\.yaml the files were"
    ):
        _migrate(processed)


def test_a_record_of_another_declaration_cannot_be_migrated(processed: Synthetic) -> None:
    record = _record(processed, "train")
    record["configuration"][FINGERPRINT] = "0" * 64
    processed.paths.provenance_file("train").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(DataValidationError, match="built under another declaration"):
        _migrate(processed)
