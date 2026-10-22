"""Evidence promotion: aggregate files of a completed run only, with identity and hashes, never
automatic and never replacing (synthetic records; no data read)."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from review_capacity.development.promotion import (
    EVIDENCE_ROOT,
    MANIFEST_NAME,
    promote,
    verify_promoted,
)
from review_capacity.development.rendering import write_summary
from review_capacity.errors import FileAccessError, RecordContractError, SpecificationError
from support.development import IDENTITY, PARTITIONS
from support.runs import RUN_ID, finished_run, new_run, write_records
from support.synthetic import REPO_PATHS


def promoted_files(destination: Path) -> list[str]:
    found = (p for p in destination.rglob("*") if p.is_file())
    return sorted(p.relative_to(destination).as_posix() for p in found)


def test_a_completed_run_is_promoted_with_its_identity_and_hashes(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    destination = promote(run.path, tmp_path, PARTITIONS)
    assert destination == tmp_path / "evidence" / "development" / RUN_ID
    manifest = verify_promoted(destination)
    assert manifest["identity"] == IDENTITY
    assert manifest["source"] == f"outputs/development/{RUN_ID}"
    listed = sorted(e["path"] for e in manifest["files"])
    assert listed == [n for n in promoted_files(destination) if n != MANIFEST_NAME]
    for entry in manifest["files"]:
        assert (destination / entry["path"]).read_bytes() == (run.path / entry["path"]).read_bytes()


def test_only_records_summary_and_log_are_promoted(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    (run.path / "scores").mkdir()
    (run.path / "scores" / "LGB-u.parquet").write_bytes(b"row-level scores")
    (run.path / "preprocessing.npz").write_bytes(b"fitted artifact")
    (run.path / "models").mkdir()
    (run.path / "models" / "LGB-u.txt").write_text("model", encoding="utf-8")
    destination = promote(run.path, tmp_path, PARTITIONS)
    names = promoted_files(destination)
    assert MANIFEST_NAME in names
    assert {n.split("/")[0] for n in names} == {"records", "summary.md", "run.log", MANIFEST_NAME}
    assert not any(n.endswith((".parquet", ".npz", ".txt")) for n in names)
    assert all(n.endswith(".json") for n in names if n.startswith("records/"))


def test_promotion_never_replaces_existing_evidence(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    promote(run.path, tmp_path, PARTITIONS)
    with pytest.raises(SpecificationError, match="never replaced"):
        promote(run.path, tmp_path, PARTITIONS)


def test_nothing_is_created_when_a_needed_file_is_missing(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    (run.path / "run.log").unlink()
    with pytest.raises(SpecificationError, match=r"run\.log"):
        promote(run.path, tmp_path, PARTITIONS)
    assert not (tmp_path / "evidence").exists()


def test_an_unclosed_run_is_not_promoted(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    write_records(run)
    run.log_file.write_text("log\n", encoding="utf-8")
    write_summary(run.path, PARTITIONS)
    with pytest.raises(SpecificationError, match="never closed"):
        promote(run.path, tmp_path, PARTITIONS)


@pytest.mark.parametrize("outcome", ["failed", "interrupted"])
def test_only_a_completed_run_is_promoted(tmp_path: Path, outcome: str) -> None:
    run = new_run(tmp_path)
    write_records(run)
    run.log_file.write_text("log\n", encoding="utf-8")
    run.finish(outcome, "stopped")
    write_summary(run.path, PARTITIONS)
    with pytest.raises(SpecificationError, match=outcome):
        promote(run.path, tmp_path, PARTITIONS)


def test_a_summary_that_differs_from_the_records_is_refused(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    summary = run.path / "summary.md"
    summary.write_text(summary.read_text(encoding="utf-8") + "edited by hand\n", encoding="utf-8")
    with pytest.raises(SpecificationError, match="regenerate"):
        promote(run.path, tmp_path, PARTITIONS)


def test_records_changed_after_closing_are_refused(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    trial = run.path / "records" / "trial-LGB-u-1.json"
    text = trial.read_text(encoding="utf-8")
    changed = text.replace('"runtime_seconds": 12.5', '"runtime_seconds": 13.5')
    trial.write_text(changed, encoding="utf-8")
    with pytest.raises(RecordContractError, match="changed"):
        promote(run.path, tmp_path, PARTITIONS)


def test_the_directories_must_lie_inside_the_project(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    run = finished_run(root)
    with pytest.raises(SpecificationError, match="inside the project root"):
        promote(run.path, root, PARTITIONS, evidence_root=tmp_path / "elsewhere")
    with pytest.raises(SpecificationError, match="inside the project root"):
        promote(tmp_path, root, PARTITIONS)


def test_a_symbolic_link_is_not_promoted(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    log = run.path / "run.log"
    target = tmp_path / "outside.log"
    target.write_text("elsewhere\n", encoding="utf-8")
    log.unlink()
    log.symlink_to(target)
    with pytest.raises(SpecificationError, match="regular file"):
        promote(run.path, tmp_path, PARTITIONS)


def test_a_promoted_directory_is_checked_against_its_manifest(tmp_path: Path) -> None:
    destination = promote(finished_run(tmp_path).path, tmp_path, PARTITIONS)
    log = destination / "run.log"
    log.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RecordContractError, match="differs from its manifest"):
        verify_promoted(destination)
    log.write_text("log of the run\n", encoding="utf-8")
    (destination / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RecordContractError, match="differs from its manifest"):
        verify_promoted(destination)
    (destination / "extra.txt").unlink()
    manifest = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["run_type"] == "development"


def evidence_entries(root: Path) -> list[str]:
    """What lies under ``evidence/development``: the run directory only, no staging leftovers."""
    base = root / "evidence" / "development"
    return sorted(p.name for p in base.iterdir()) if base.exists() else []


def patch_copy(monkeypatch: pytest.MonkeyPatch, after: Callable[[int, Path, Path], None]) -> None:
    """Run ``after(call number, source, target)`` after each file copy of the promotion."""
    calls: list[Path] = []
    real = shutil.copyfile

    def copyfile(source: Path, target: Path) -> Path:
        calls.append(target)
        result = real(source, target)
        after(len(calls), Path(source), Path(target))
        return Path(result)

    monkeypatch.setattr("review_capacity.development.promotion.shutil.copyfile", copyfile)


def test_a_stray_file_in_the_staging_area_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = finished_run(tmp_path)

    def stray(_call: int, _source: Path, target: Path) -> None:
        (target.parent / "scores.parquet").write_bytes(b"row-level scores")

    patch_copy(monkeypatch, stray)
    with pytest.raises(SpecificationError, match="differs from the allowlist"):
        promote(run.path, tmp_path, PARTITIONS)
    assert evidence_entries(tmp_path) == []


def test_a_failed_copy_leaves_no_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = finished_run(tmp_path)

    def fail_second(call: int, _source: Path, _target: Path) -> None:
        if call == 2:
            raise OSError(28, "No space left on device")

    patch_copy(monkeypatch, fail_second)
    with pytest.raises(FileAccessError, match="cannot be written"):
        promote(run.path, tmp_path, PARTITIONS)
    assert evidence_entries(tmp_path) == []


def test_a_run_changing_during_the_copy_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = finished_run(tmp_path)
    trial = run.path / "records" / "trial-LGB-u-1.json"

    def change(call: int, _source: Path, _target: Path) -> None:
        if call == 1:
            text = trial.read_text(encoding="utf-8")
            trial.write_text(text.replace('"runtime_seconds": 12.5', '"runtime_seconds": 13.5'))

    patch_copy(monkeypatch, change)
    with pytest.raises(RecordContractError, match="changed"):
        promote(run.path, tmp_path, PARTITIONS)
    assert evidence_entries(tmp_path) == []


def test_an_evidence_directory_that_is_a_link_is_refused(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "development").symlink_to(outside)
    with pytest.raises(SpecificationError, match="not a link"):
        promote(run.path, tmp_path, PARTITIONS)
    assert list(outside.iterdir()) == []


def test_a_records_directory_that_is_a_link_is_refused(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    real = run.path / "elsewhere"
    (run.path / "records").rename(real)
    (run.path / "records").symlink_to(real)
    with pytest.raises(RecordContractError, match="not a link"):
        promote(run.path, tmp_path, PARTITIONS)
    assert evidence_entries(tmp_path) == []


def test_publication_leaves_only_the_run_directory(tmp_path: Path) -> None:
    promote(finished_run(tmp_path).path, tmp_path, PARTITIONS)
    assert evidence_entries(tmp_path) == [RUN_ID]


def test_verification_exempts_only_the_root_manifest(tmp_path: Path) -> None:
    destination = promote(finished_run(tmp_path).path, tmp_path, PARTITIONS)
    nested = destination / "models"
    nested.mkdir()
    (nested / MANIFEST_NAME).write_text("a fitted artifact", encoding="utf-8")
    with pytest.raises(RecordContractError, match="differs from its manifest"):
        verify_promoted(destination)


def test_verification_refuses_links_and_malformed_manifests(tmp_path: Path) -> None:
    destination = promote(finished_run(tmp_path).path, tmp_path, PARTITIONS)
    link = destination / "records" / "scores.json"
    link.symlink_to(tmp_path / "outside.json")
    with pytest.raises(RecordContractError, match="not a regular file"):
        verify_promoted(destination)
    link.unlink()
    manifest = destination / MANIFEST_NAME
    good = manifest.read_text(encoding="utf-8")
    for broken in ("{}", "[]", "not json", json.dumps({**json.loads(good), "files": [1]})):
        manifest.write_text(broken, encoding="utf-8")
        with pytest.raises(RecordContractError):
            verify_promoted(destination)
    listed = json.loads(good)
    listed["files"].append({"path": "models/model.txt", "sha256": "0" * 64})
    manifest.write_text(json.dumps(listed), encoding="utf-8")
    with pytest.raises(RecordContractError, match="outside the allowlist"):
        verify_promoted(destination)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
@pytest.mark.parametrize(
    "name", ["run.log", "summary.md", MANIFEST_NAME, "records/run.json", "records/run_end.json"]
)
def test_promoted_files_are_not_ignored_by_git(name: str) -> None:
    assert (REPO_PATHS.root / ".git").exists()
    path = EVIDENCE_ROOT / "development" / RUN_ID / name  # the default destination
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", str(path)],
        cwd=REPO_PATHS.root,
        check=False,
    )
    assert result.returncode == 1, f"{path} is ignored by Git"


def test_a_summary_with_other_line_endings_is_not_the_generated_one(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    summary = run.path / "summary.md"
    summary.write_bytes(summary.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(SpecificationError, match="differs from the records"):
        promote(run.path, tmp_path, PARTITIONS)


def test_a_staged_summary_must_match_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = finished_run(tmp_path)

    def crlf(_call: int, _source: Path, target: Path) -> None:
        if target.name == "summary.md":
            target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))

    patch_copy(monkeypatch, crlf)
    with pytest.raises(SpecificationError, match="summary changed"):
        promote(run.path, tmp_path, PARTITIONS)
    assert evidence_entries(tmp_path) == []


def test_a_linked_summary_is_refused_before_it_is_read(tmp_path: Path) -> None:
    run = finished_run(tmp_path)
    summary = run.path / "summary.md"
    summary.unlink()
    summary.symlink_to(tmp_path / "missing.md")  # reading it would be a different error
    with pytest.raises(SpecificationError, match="regular file"):
        promote(run.path, tmp_path, PARTITIONS)


def test_a_linked_manifest_is_refused_before_it_is_read(tmp_path: Path) -> None:
    destination = promote(finished_run(tmp_path).path, tmp_path, PARTITIONS)
    manifest = destination / MANIFEST_NAME
    real = tmp_path / "real-manifest.json"
    real.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(real)
    with pytest.raises(RecordContractError, match="manifest must be a regular file"):
        verify_promoted(destination)
