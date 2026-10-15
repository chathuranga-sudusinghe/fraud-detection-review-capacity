"""Run logging: format, log file, repeated setup, warnings, step timing and loop progress."""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from review_capacity.errors import FileAccessError
from review_capacity.logging_setup import ProgressLogger, configure_logging, log_step

pytestmark = pytest.mark.usefixtures("restore_logging")
TIMESTAMP = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"


def test_log_file_has_timestamped_lines_and_keeps_non_ascii(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "run.log"
    configure_logging(logging.INFO, log_file)
    logging.getLogger("review_capacity.example").info("hello %d θ 360°", 42)
    logging.getLogger("review_capacity.example").debug("not shown at INFO")
    text = log_file.read_text(encoding="utf-8")
    pattern = rf"^{TIMESTAMP} INFO +review_capacity\.example \| hello 42 θ 360°$"
    assert re.search(pattern, text, re.MULTILINE)
    assert "not shown" not in text


def test_quiet_level_drops_info(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    configure_logging(logging.WARNING, log_file)
    logging.getLogger("review_capacity.example").info("progress")
    logging.getLogger("review_capacity.example").warning("attention")
    text = log_file.read_text(encoding="utf-8")
    assert "progress" not in text
    assert "attention" in text


def test_a_new_log_file_is_never_appended_to(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    log_file.write_text("an earlier run\n", encoding="utf-8")
    with pytest.raises(FileAccessError, match="cannot be opened"):
        configure_logging(logging.INFO, log_file, new_file=True)
    logging.getLogger("review_capacity.example").warning("after the refusal")
    assert log_file.read_text(encoding="utf-8") == "an earlier run\n"
    fresh = tmp_path / "fresh.log"
    configure_logging(logging.INFO, fresh, new_file=True)
    logging.getLogger("review_capacity.example").info("first line")
    assert "first line" in fresh.read_text(encoding="utf-8")


def test_a_quiet_terminal_can_keep_a_complete_log_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_file = tmp_path / "run.log"
    configure_logging(logging.WARNING, log_file, file_level=logging.INFO)
    logging.getLogger("review_capacity.example").info("progress for the evidence log")
    logging.getLogger("review_capacity.example").warning("a warning for both")
    text = log_file.read_text(encoding="utf-8")
    assert "progress for the evidence log" in text
    assert "a warning for both" in text
    terminal = capsys.readouterr().err
    assert "progress for the evidence log" not in terminal
    assert "a warning for both" in terminal


def test_repeated_configuration_does_not_duplicate_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_file = tmp_path / "run.log"
    configure_logging(logging.INFO, log_file)
    configure_logging(logging.INFO, log_file)
    logging.getLogger("review_capacity.example").info("once")
    assert log_file.read_text(encoding="utf-8").count("once") == 1
    assert capsys.readouterr().err.count("once") == 1


@pytest.mark.parametrize("terminal_tracebacks", [False, True])
def test_traceback_always_in_the_file_and_on_the_terminal_only_when_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], terminal_tracebacks: bool
) -> None:
    log_file = tmp_path / "run.log"
    configure_logging(logging.INFO, log_file, terminal_tracebacks=terminal_tracebacks)
    logger = logging.getLogger("review_capacity.example")
    try:
        raise ValueError("cause")
    except ValueError:
        logger.exception("failed")
    terminal = capsys.readouterr().err
    assert "failed" in terminal
    assert ("Traceback" in terminal) is terminal_tracebacks
    assert "ValueError: cause" in log_file.read_text(encoding="utf-8")


def test_python_warnings_reach_the_log_file(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    configure_logging(logging.INFO, log_file)
    # pytest records warnings itself, so the hook installed by configure_logging is called directly.
    warnings.showwarning("deprecated thing", DeprecationWarning, "module.py", 1)
    assert "deprecated thing" in log_file.read_text(encoding="utf-8")


def test_unopenable_log_file_is_a_file_access_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("", encoding="utf-8")
    with pytest.raises(FileAccessError, match="Log file"):
        configure_logging(logging.INFO, blocker / "run.log")
    logging.getLogger("review_capacity.example").info("still logged")
    assert "still logged" in capsys.readouterr().err  # terminal logging kept


def test_log_step_reports_start_and_duration(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("review_capacity.step")
    with caplog.at_level(logging.INFO, logger="review_capacity.step"), log_step(logger, "Load"):
        pass
    messages = [r.getMessage() for r in caplog.records]
    assert messages[0] == "Load ..."
    assert re.fullmatch(r"Load: done in \d+\.\d s", messages[1])


@pytest.mark.parametrize(
    ("error", "word"), [(ValueError("boom"), "failed"), (KeyboardInterrupt(), "interrupted")]
)
def test_log_step_reports_failure_or_interruption_and_reraises(
    caplog: pytest.LogCaptureFixture, error: BaseException, word: str
) -> None:
    logger = logging.getLogger("review_capacity.step")
    with (
        caplog.at_level(logging.INFO, logger="review_capacity.step"),
        pytest.raises(type(error)),
        log_step(logger, "Load"),
    ):
        raise error
    messages = [r.getMessage() for r in caplog.records]
    assert caplog.records[-1].levelno == logging.ERROR
    assert messages[-1].startswith(f"Load: {word} after")
    assert not any("done in" in m for m in messages)


def _clock(times: list[float]) -> Iterator[float]:
    yield from times


def test_progress_is_periodic_and_always_reports_the_end(caplog: pytest.LogCaptureFixture) -> None:
    ticks = _clock([0.0, 1.0, 2.0, 31.0, 32.0, 33.0])
    progress = ProgressLogger(
        logging.getLogger("review_capacity.loop"), "Columns", 5,
        every_seconds=30.0, clock=lambda: next(ticks),
    )  # fmt: skip
    with caplog.at_level(logging.INFO, logger="review_capacity.loop"):
        for _ in range(5):
            progress.advance()
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2  # at 31 s (30 s since the start) and at the last item
    assert messages[0].startswith("Columns: 3/5 (60 %), elapsed 31.0 s, about 20.7 s left")
    assert messages[1].startswith("Columns: 5/5 (100 %), elapsed 33.0 s, about 0.0 s left")
