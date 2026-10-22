"""The ``fit-neural-preprocessing`` run on synthetic processed files (no raw file, no project
data), and its command-line form."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from review_capacity import cli
from review_capacity.cli import build_parser, main
from review_capacity.development.preprocessing_run import (
    LOG_NAME,
    RESULT_FILE,
    output_directory,
    run_fit,
)
from review_capacity.errors import SpecificationError
from review_capacity.features.neural_preprocessing import (
    ARRAYS_FILE,
    MANIFEST_FILE,
    load_neural_preprocessing,
)
from review_capacity.paths import ProjectPaths
from support.synthetic import REPO_PATHS, Synthetic, build_processed

pytestmark = pytest.mark.usefixtures("restore_logging")


@pytest.fixture
def processed(tmp_path: Path) -> Synthetic:
    data = build_processed(tmp_path)
    shutil.copy(REPO_PATHS.training_spec, data.paths.training_spec)  # the declared seed
    return data


def test_the_run_writes_the_artifact_diagnostics_and_a_result(processed: Synthetic) -> None:
    directory = output_directory(processed.paths, "2026-10-22T120000Z", None)
    assert (
        directory
        == (
            processed.paths.root / "outputs" / "neural_preprocessing" / "2026-10-22T120000Z"
        ).resolve()
    )
    result = run_fit(processed.paths, processed.features, processed.dataset, directory, "x")
    names = {path.name for path in directory.iterdir()}
    assert names == {
        MANIFEST_FILE,
        ARRAYS_FILE,
        RESULT_FILE,
        "diagnostics_v1.json",
        "diagnostics_v2.json",
    }
    stored = json.loads((directory / RESULT_FILE).read_text(encoding="utf-8"))
    assert stored["artifact_sha256"] == result["artifact_sha256"]
    assert stored["seed"] == 0
    assert stored["train"]["rows"] == 8
    assert stored["inputs"]["numeric"] == len(processed.features.numeric_inputs)
    assert stored["inputs"]["categorical"] == len(processed.features.categorical)
    inputs = stored["inputs"]
    assert inputs["total"] == inputs["numeric"] + inputs["indicators"] + inputs["categorical"]
    assert set(stored["diagnostics"]) == {"v1", "v2"}
    report = json.loads((directory / "diagnostics_v1.json").read_text(encoding="utf-8"))
    assert report["partition"] == "v1"
    assert report["rows"] == 4
    loaded = load_neural_preprocessing(directory, processed.features, 0)
    assert loaded.artifact_sha256 == result["artifact_sha256"]


def test_two_runs_give_the_same_artifact_and_a_used_directory_is_refused(
    processed: Synthetic,
) -> None:
    first = output_directory(processed.paths, "a", None)
    second = output_directory(processed.paths, "b", None)
    one = run_fit(processed.paths, processed.features, processed.dataset, first, "a")
    two = run_fit(processed.paths, processed.features, processed.dataset, second, "b")
    assert one["artifact_sha256"] == two["artifact_sha256"]
    with pytest.raises(SpecificationError, match="already exists"):
        output_directory(processed.paths, "a", None)
    with pytest.raises(SpecificationError, match="outside the project root"):
        output_directory(processed.paths, "c", Path("/elsewhere"))


def test_the_command_runs_through_main_with_its_own_log_in_its_new_directory(
    processed: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ProjectPaths, "discover", lambda _root=None: processed.paths)
    monkeypatch.setattr(cli, "load_feature_spec", lambda _path: processed.features)
    monkeypatch.setattr(cli, "load_dataset_spec", lambda _paths: processed.dataset)
    assert main(["fit-neural-preprocessing"]) == 0
    (run,) = list((processed.paths.root / "outputs" / "neural_preprocessing").iterdir())
    assert (run / LOG_NAME).is_file()
    assert (run / MANIFEST_FILE).is_file()
    assert (run / RESULT_FILE).is_file()


def test_the_command_line_takes_an_optional_output_and_no_log_file() -> None:
    parser = build_parser()
    args = parser.parse_args(["fit-neural-preprocessing"])
    assert args.command == "fit-neural-preprocessing"
    assert args.output is None
    chosen = parser.parse_args(["fit-neural-preprocessing", "--output", "outputs/x"])
    assert chosen.output == Path("outputs/x")
    assert LOG_NAME == "run.log"
