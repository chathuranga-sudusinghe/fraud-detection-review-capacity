"""Atomic writes and their file-system errors."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from review_capacity.errors import FileAccessError
from review_capacity.files import write_atomically, write_json, write_new_atomically


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


def test_a_new_file_is_written_whole_and_never_replaced(tmp_path: Path) -> None:
    path = tmp_path / "out" / "record.json"
    write_new_atomically(path, "first\n")
    assert path.read_text(encoding="utf-8") == "first\n"
    with pytest.raises(FileAccessError, match="already exists"):
        write_new_atomically(path, "second\n")
    assert path.read_text(encoding="utf-8") == "first\n"
    assert [p.name for p in path.parent.iterdir()] == ["record.json"]


def test_an_unwritable_new_file_is_a_file_access_error(tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("", encoding="utf-8")
    with pytest.raises(FileAccessError, match="cannot be written"):
        write_new_atomically(blocker / "record.json", "x")


def test_a_leftover_hard_link_never_changes_a_published_file(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    write_new_atomically(path, "published\n")
    leftover = tmp_path / f".record.json.{os.getpid()}.tmp"  # the name an older writer used
    os.link(path, leftover)
    with pytest.raises(FileAccessError, match="already exists"):
        write_new_atomically(path, "other\n")
    other = tmp_path / "other.json"
    write_new_atomically(other, "other\n")
    assert path.read_text(encoding="utf-8") == "published\n"
    assert leftover.read_text(encoding="utf-8") == "published\n"


def test_every_call_uses_its_own_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names: list[str] = []
    real = os.link

    def recording(source: Path, target: Path) -> None:
        names.append(Path(source).name)
        real(source, target)

    monkeypatch.setattr(os, "link", recording)
    write_new_atomically(tmp_path / "a.json", "a")
    write_new_atomically(tmp_path / "b.json", "b")
    assert len(set(names)) == 2
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_a_file_system_without_hard_links_is_an_error_not_an_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(_source: Path, _target: Path) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", refused)
    with pytest.raises(FileAccessError, match="cannot be written"):
        write_new_atomically(tmp_path / "record.json", "x")
    assert list(tmp_path.iterdir()) == []


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The order of synchronisations and publications, by what they act on."""
    events: list[str] = []
    real_fsync, real_link, real_replace = os.fsync, os.link, Path.replace

    def fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(f"fsync {kind}")
        real_fsync(descriptor)

    def link(source: Any, target: Any) -> None:
        events.append("publish")
        real_link(source, target)

    def replace(self: Path, target: Any) -> Path:
        events.append("publish")
        return real_replace(self, target)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "link", link)
    monkeypatch.setattr(Path, "replace", replace)
    return events


def test_a_new_file_is_synchronised_before_it_is_published_and_its_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _spy(monkeypatch)
    write_new_atomically(tmp_path / "marker.json", "{}\n")
    assert events == ["fsync file", "publish", "fsync directory"]


def test_a_replaced_file_is_synchronised_before_it_is_published_and_its_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _spy(monkeypatch)
    write_atomically(tmp_path / "scores.txt", lambda out: out.write_text("x", encoding="utf-8"))
    assert events == ["fsync file", "publish", "fsync directory"]
