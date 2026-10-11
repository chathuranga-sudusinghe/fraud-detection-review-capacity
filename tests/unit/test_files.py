"""Atomic writes and their file-system errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from review_capacity.errors import FileAccessError
from review_capacity.files import write_atomically, write_json


def test_write_json_replaces_the_file_and_leaves_no_temporary(tmp_path: Path) -> None:
    path = tmp_path / "out" / "record.json"
    write_json(path, {"a": 1})
    write_json(path, {"a": 2})
    assert path.read_text(encoding="utf-8") == '{\n  "a": 2\n}\n'
    assert [p.name for p in path.parent.iterdir()] == ["record.json"]


def test_unwritable_destination_is_a_file_access_error(tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("", encoding="utf-8")
    with pytest.raises(FileAccessError, match="cannot be written"):
        write_json(blocker / "record.json", {"a": 1})


def test_other_errors_from_the_writer_propagate_unchanged(tmp_path: Path) -> None:
    def failing(_target: Path) -> None:
        raise ValueError("serialisation failed")

    with pytest.raises(ValueError, match="serialisation failed"):
        write_atomically(tmp_path / "x.bin", failing)
    assert list(tmp_path.iterdir()) == []


def test_failed_write_keeps_the_previous_file(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("old", encoding="utf-8")

    def partial(target: Path) -> None:
        target.write_text("partial", encoding="utf-8")
        raise ValueError("writer failed")

    with pytest.raises(ValueError, match="writer failed"):
        write_atomically(path, partial)
    assert path.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["x.txt"]


def test_failed_replacement_is_a_file_access_error_with_its_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_replace(_self: Path, _target: Path) -> Path:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(FileAccessError, match="cannot be written") as caught:
        write_atomically(tmp_path / "x.txt", lambda t: t.write_text("new", encoding="utf-8"))
    assert isinstance(caught.value.__cause__, PermissionError)


def test_clean_up_failure_never_hides_the_writer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing_unlink(_self: Path, **_options: object) -> None:
        raise PermissionError(13, "Permission denied")

    def partial(target: Path) -> None:
        target.write_text("partial", encoding="utf-8")
        raise ValueError("writer failed")

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with (
        caplog.at_level("WARNING", logger="review_capacity"),
        pytest.raises(ValueError, match="writer failed"),
    ):
        write_atomically(tmp_path / "x.txt", partial)
    assert "could not be removed" in caplog.text
