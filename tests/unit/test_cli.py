"""Command-line parsing (no command is executed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from review_capacity.cli import build_parser


def test_prepare_data_options() -> None:
    args = build_parser().parse_args(["prepare-data", "--check-only", "--root", "/tmp/project"])
    assert args.command == "prepare-data"
    assert args.check_only is True
    assert args.root == Path("/tmp/project")


def test_prepare_data_defaults() -> None:
    args = build_parser().parse_args(["prepare-data"])
    assert args.check_only is False
    assert args.root is None


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
