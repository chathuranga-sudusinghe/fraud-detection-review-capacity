"""The fresh process of the same-seed repeat: the job document, the child's work and the parent's
handshake (no process is started; LightGBM on synthetic data on the CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_capacity.cli import build_parser
from review_capacity.development import refit_process as refit_process_module
from review_capacity.development.refit_process import (
    COMMAND,
    JOBS_DIR,
    FreshProcess,
    check_identity,
    job_document,
    read_result,
    run_job,
    run_job_file,
)
from review_capacity.development.refits import (
    SAME_SEED_REPEAT,
    RefitJob,
    RefitSpec,
    refit_directory,
    run_refit,
)
from review_capacity.development.scoring import load_selected, score_partition
from review_capacity.development.search import search_configuration
from review_capacity.development.trial import SearchContext, TrialOutcome
from review_capacity.errors import FileAccessError, ModelError, SpecificationError
from review_capacity.models.registry import configuration
from support.search import candidate_file, candidate_list, scoring_context, search_context

LGB_U = configuration("LGB-u")
SPEC = RefitSpec(SAME_SEED_REPEAT, 42, "cpu", fresh_process=True)

pytestmark = pytest.mark.usefixtures("restore_lightgbm_logger")


@pytest.fixture
def searched_context(tmp_path: Path) -> tuple[SearchContext, TrialOutcome, RefitJob]:
    search = search_context(tmp_path)
    result = search_configuration(search, LGB_U, candidate_list(), candidate_file(search.run))
    selected = result.selected
    job = RefitJob(
        "LGB-u", SPEC, selected.trial, selected.hyperparameters, selected.fitted_iterations
    )
    return search, selected, job


def test_the_job_document_describes_the_refit_for_the_child(
    searched_context: tuple[SearchContext, TrialOutcome, RefitJob], tmp_path: Path
) -> None:
    search, _, job = searched_context
    directory = refit_directory(search.run.path, "LGB-u", SPEC)
    document = job_document(
        tmp_path, search.run, search.identity, job, directory, tmp_path / "r.json", None
    )
    assert document["configuration"] == "LGB-u"
    assert (document["kind"], document["seed"], document["device"]) == (SAME_SEED_REPEAT, 42, "cpu")
    assert (document["trial"], document["length"]) == (job.trial, job.length)
    assert document["hyperparameters"] == job.hyperparameters
    assert document["directory"] == str(directory)
    assert document["preprocessing"] is None
    assert document["batch_plan"] is None
    assert json.loads(json.dumps(document)) == document  # plain JSON


def test_the_child_fits_the_selected_length_saves_the_model_and_leaves_its_seconds(
    searched_context: tuple[SearchContext, TrialOutcome, RefitJob], tmp_path: Path
) -> None:
    search, selected, job = searched_context
    directory = refit_directory(search.run.path, "LGB-u", SPEC)
    result = tmp_path / "result.json"
    document = job_document(tmp_path, search.run, search.identity, job, directory, result, None)
    seconds = run_job(document, load=lambda _job: search)
    assert seconds >= 0
    assert read_result(result) == seconds
    assert any(directory.iterdir())
    scoring = scoring_context(search)
    # the parent reloads the child's model like any saved model
    model = load_selected(scoring, LGB_U, selected, directory=directory, seed=42, device="cpu")
    assert model.fitted_iterations == job.length


def test_a_job_file_is_read_and_run(
    searched_context: tuple[SearchContext, TrialOutcome, RefitJob],
    tmp_path: Path,
) -> None:
    search, _, job = searched_context
    directory = refit_directory(search.run.path, "LGB-u", SPEC)
    result = tmp_path / "result.json"
    document = job_document(tmp_path, search.run, search.identity, job, directory, result, None)
    path = tmp_path / "job.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert run_job_file(path, load=lambda _job: search) == read_result(result)
    with pytest.raises(FileAccessError, match="job file cannot be read"):
        run_job_file(tmp_path / "missing.json")


def test_the_parent_starts_the_command_and_reads_the_result(
    searched_context: tuple[SearchContext, TrialOutcome, RefitJob], tmp_path: Path
) -> None:
    search, _, job = searched_context
    directory = refit_directory(search.run.path, "LGB-u", SPEC)
    commands: list[list[str]] = []

    def launch(command: list[str]) -> int:
        commands.append(command)
        result = search.run.path / JOBS_DIR / "LGB-u-same_seed_repeat.result.json"
        result.write_text('{"seconds": 1.5}', encoding="utf-8")
        return 0

    fresh = FreshProcess(tmp_path, search.run, search.identity, launch=launch)
    assert fresh(job, directory) == 1.5
    (command,) = commands
    assert command[1:3] == ["-m", "review_capacity"]
    assert command[command.index(COMMAND) + 1 :][:1] == ["--root"]
    log = Path(command[command.index("--log-file") + 1])
    assert log.name == "refit-LGB-u-same_seed_repeat.log"
    assert log.parent == search.run.path
    job_file = Path(command[command.index("--job") + 1])
    assert json.loads(job_file.read_text(encoding="utf-8"))["configuration"] == "LGB-u"


def test_the_parent_releases_its_gpu_memory_before_it_starts_the_child(
    searched_context: tuple[SearchContext, TrialOutcome, RefitJob], tmp_path: Path
) -> None:
    search, _, job = searched_context
    directory = refit_directory(search.run.path, "LGB-u", SPEC)
    events: list[str] = []

    def launch(_command: list[str]) -> int:
        events.append("launch")
        result = search.run.path / JOBS_DIR / "LGB-u-same_seed_repeat.result.json"
        result.write_text('{"seconds": 1.0}', encoding="utf-8")
        return 0

    fresh = FreshProcess(
        tmp_path,
        search.run,
        search.identity,
        launch=launch,
        release=lambda: events.append("release"),
    )
    fresh(job, directory)
    assert events == ["release", "launch"]


@pytest.mark.parametrize(
    ("name", "device", "capped"),
    [
        ("RES-u", "gpu", True),  # a neural fit on the GPU: the new process has no allocator cap
        ("FTT-b", "gpu", True),
        ("RES-u", "cpu", False),
        ("LGB-u", "gpu", False),  # a tree-based library has its own allocator, not PyTorch's
    ],
)
def test_the_child_caps_the_gpu_allocator_before_a_neural_fit_on_the_gpu(
    searched_context: tuple[SearchContext, TrialOutcome, RefitJob],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    device: str,
    capped: bool,
) -> None:
    search, _, job = searched_context
    directory = refit_directory(search.run.path, "LGB-u", SPEC)
    result = tmp_path / "result.json"
    document = job_document(tmp_path, search.run, search.identity, job, directory, result, None)
    document |= {"configuration": name, "device": device}
    events: list[object] = []

    def fake_fit(*_args: object) -> tuple[float, None]:
        events.append("fit")
        return 0.5, None

    def cap(margin: int) -> None:
        events.append(margin)

    monkeypatch.setattr(refit_process_module, "fit_refit", fake_fit)
    run_job(document, load=lambda _job: search, cap=cap)
    margin = search.declaration.neural.preflight_free_margin_mib
    assert events == ([margin, "fit"] if capped else ["fit"])  # capped before the fit starts


def test_a_failing_child_is_a_model_error_naming_its_log(
    searched_context: tuple[SearchContext, TrialOutcome, RefitJob], tmp_path: Path
) -> None:
    search, _, job = searched_context
    directory = refit_directory(search.run.path, "LGB-u", SPEC)
    fresh = FreshProcess(tmp_path, search.run, search.identity, launch=lambda _command: 3)
    with pytest.raises(ModelError, match=r"exit code 3.*refit-LGB-u-same_seed_repeat\.log"):
        fresh(job, directory)


def test_a_child_without_a_result_is_refused(
    searched_context: tuple[SearchContext, TrialOutcome, RefitJob], tmp_path: Path
) -> None:
    search, _, job = searched_context
    directory = refit_directory(search.run.path, "LGB-u", SPEC)
    fresh = FreshProcess(tmp_path, search.run, search.identity, launch=lambda _command: 0)
    with pytest.raises(FileAccessError, match="no readable result"):
        fresh(job, directory)
    bad = tmp_path / "bad.json"
    bad.write_text('{"seconds": -1}', encoding="utf-8")
    with pytest.raises(ModelError, match="non-negative"):
        read_result(bad)


def test_the_run_reaches_the_repeat_through_the_fresh_process(
    searched_context: tuple[SearchContext, TrialOutcome, RefitJob], tmp_path: Path
) -> None:
    search, selected, job = searched_context
    scoring = scoring_context(search)
    model = load_selected(scoring, LGB_U, selected)
    reference = score_partition(scoring, LGB_U, selected, model, "V2")

    def launch(_command: list[str]) -> int:
        directory = refit_directory(search.run.path, "LGB-u", SPEC)
        document = job_document(
            tmp_path, search.run, search.identity, job, directory,
            search.run.path / JOBS_DIR / "LGB-u-same_seed_repeat.result.json", None,
        )  # fmt: skip
        run_job(document, load=lambda _job: search)
        return 0

    fresh = FreshProcess(tmp_path, search.run, search.identity, launch=launch)
    scored = run_refit(search, scoring, LGB_U, selected, SPEC, reference, fresh=fresh)
    assert scored.tag == "same_seed_repeat"


def test_the_refit_fresh_command_takes_a_job_file() -> None:
    args = build_parser().parse_args(["refit-fresh", "--job", "j.json"])
    assert args.command == COMMAND
    assert args.job == Path("j.json")


def test_the_child_refuses_other_source_settings_or_declaration() -> None:
    identity = {
        "source_sha256": "a" * 64,
        "settings_sha256": "b" * 64,
        "declaration_sha256": "c" * 64,
    }
    check_identity(identity, "a" * 64, "b" * 64, "c" * 64)
    with pytest.raises(SpecificationError, match="package source"):
        check_identity(identity, "0" * 64, "b" * 64, "c" * 64)
    with pytest.raises(SpecificationError, match=r"models\.yaml"):
        check_identity(identity, "a" * 64, "0" * 64, "c" * 64)
    with pytest.raises(SpecificationError, match=r"training\.yaml"):
        check_identity(identity, "a" * 64, "b" * 64, "0" * 64)
