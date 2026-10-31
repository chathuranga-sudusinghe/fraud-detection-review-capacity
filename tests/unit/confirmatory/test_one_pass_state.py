"""The lock and the consumption marker of the one-pass guard (Issue #203)."""

from __future__ import annotations

from pathlib import Path

import pytest

from review_capacity.confirmatory.state import (
    begin_scoring,
    consume,
    is_consumed,
    lock_path,
    locked,
    marker_path,
    record_scoring,
    refuse_if_consumed,
    refuse_if_scored,
    refuse_if_started,
    scored_path,
    started_path,
    state_directory,
)
from review_capacity.errors import SpecificationError

STATE = "a" * 64
OTHER = "b" * 64


def test_the_state_files_are_named_by_the_frozen_state(tmp_path: Path) -> None:
    assert lock_path(tmp_path, STATE).parent == state_directory(tmp_path)
    assert lock_path(tmp_path, STATE).name == STATE + ".lock"
    assert marker_path(tmp_path, STATE).name == STATE + ".consumed.json"
    with pytest.raises(SpecificationError, match="SHA-256"):
        lock_path(tmp_path, "../escape")


def test_a_lock_is_exclusive_and_released_whatever_the_way(tmp_path: Path) -> None:
    with locked(tmp_path, STATE, run_id="r1") as path:
        assert path.exists()
        assert "r1" in path.read_text(encoding="utf-8")
        with (
            pytest.raises(SpecificationError, match="never removes a lock"),
            locked(tmp_path, STATE, run_id="r2"),
        ):
            pass
        with locked(tmp_path, OTHER, run_id="r3"):  # another frozen state is independent
            pass
    assert not lock_path(tmp_path, STATE).exists()
    with pytest.raises(RuntimeError, match="boom"), locked(tmp_path, STATE, run_id="r4"):
        raise RuntimeError("boom")
    assert not lock_path(tmp_path, STATE).exists()
    with locked(tmp_path, STATE, run_id="r5"):  # free again
        pass


def test_a_stale_lock_is_never_removed_by_the_code(tmp_path: Path) -> None:
    lock_path(tmp_path, STATE).parent.mkdir(parents=True)
    lock_path(tmp_path, STATE).write_text("killed\n", encoding="utf-8")
    with pytest.raises(SpecificationError, match=STATE), locked(tmp_path, STATE, run_id="r"):
        pass
    assert lock_path(tmp_path, STATE).read_text(encoding="utf-8") == "killed\n"


def test_the_marker_is_written_once_and_refuses_a_new_scoring(tmp_path: Path) -> None:
    assert not is_consumed(tmp_path, STATE)
    refuse_if_consumed(tmp_path, STATE)  # nothing consumed yet
    path = consume(tmp_path, STATE, run_id="r1", scores_sha256="c" * 64)
    assert path == marker_path(tmp_path, STATE)
    assert is_consumed(tmp_path, STATE)
    with pytest.raises(SpecificationError, match="consumed"):
        refuse_if_consumed(tmp_path, STATE)
    with pytest.raises(SpecificationError, match="consumed"):
        consume(tmp_path, STATE, run_id="r2", scores_sha256="d" * 64)
    assert "r1" in path.read_text(encoding="utf-8")  # the first marker stands
    refuse_if_consumed(tmp_path, OTHER)  # another frozen state is not consumed


def test_the_scoring_marker_is_written_once_and_refuses_a_new_scoring(tmp_path: Path) -> None:
    refuse_if_scored(tmp_path, STATE)  # nothing scored yet
    path = record_scoring(tmp_path, STATE, run_id="r1", scores_sha256="c" * 64)
    assert path == scored_path(tmp_path, STATE)
    with pytest.raises(SpecificationError, match="completed"):
        refuse_if_scored(tmp_path, STATE)
    with pytest.raises(SpecificationError, match="already exists"):
        record_scoring(tmp_path, STATE, run_id="r2", scores_sha256="d" * 64)
    assert "r1" in path.read_text(encoding="utf-8")
    refuse_if_scored(tmp_path, OTHER)  # another frozen state is independent
    assert not is_consumed(tmp_path, STATE)  # scoring is not consumption


def test_a_resumed_evaluation_may_reuse_its_own_marker_and_no_other(tmp_path: Path) -> None:
    from review_capacity.confirmatory.state import begin_consumption

    path = begin_consumption(tmp_path, STATE, run_id="r1", scores_sha256="c" * 64)
    assert path == marker_path(tmp_path, STATE)
    assert begin_consumption(tmp_path, STATE, run_id="r1", scores_sha256="c" * 64) == path
    with pytest.raises(SpecificationError, match="another run or other scores"):
        begin_consumption(tmp_path, STATE, run_id="r2", scores_sha256="c" * 64)
    with pytest.raises(SpecificationError, match="another run or other scores"):
        begin_consumption(tmp_path, STATE, run_id="r1", scores_sha256="d" * 64)


def test_the_started_marker_claims_the_state_for_one_run_in_one_directory(
    tmp_path: Path,
) -> None:
    assert started_path(tmp_path, STATE).name == STATE + ".started.json"
    refuse_if_started(tmp_path, STATE)  # nothing started yet
    begin_scoring(tmp_path, STATE, run_id="run-1", directory="outputs/confirmatory/run-1")
    begin_scoring(  # a repeated attempt of the same run in the same directory
        tmp_path, STATE, run_id="run-1", directory="outputs/confirmatory/run-1"
    )
    refuse_if_started(
        tmp_path, STATE, except_run="run-1", except_directory="outputs/confirmatory/run-1"
    )
    with pytest.raises(SpecificationError, match="owns the state until its run ends"):
        begin_scoring(tmp_path, STATE, run_id="run-2", directory="outputs/confirmatory/run-2")
    with pytest.raises(SpecificationError, match="run-1"):
        refuse_if_started(tmp_path, STATE)
    refuse_if_started(tmp_path, OTHER)  # another frozen state is not claimed


def test_the_same_run_identifier_in_another_directory_is_not_the_owner(tmp_path: Path) -> None:
    begin_scoring(tmp_path, STATE, run_id="run-1", directory="outputs/confirmatory/a")
    with pytest.raises(SpecificationError, match="owns the state until its run ends"):
        begin_scoring(tmp_path, STATE, run_id="run-1", directory="outputs/confirmatory/b")
    with pytest.raises(SpecificationError, match="owns the state until its run ends"):
        refuse_if_started(tmp_path, STATE, except_run="run-1")  # no directory named


def test_a_started_state_refuses_a_scoring_phase_of_any_other_run(tmp_path: Path) -> None:
    begin_scoring(tmp_path, STATE, run_id="run-1", directory="outputs/confirmatory/run-1")
    with pytest.raises(SpecificationError, match="owns the state"):
        refuse_if_scored(tmp_path, STATE)
    refuse_if_scored(
        tmp_path, STATE, except_run="run-1", except_directory="outputs/confirmatory/run-1"
    )
