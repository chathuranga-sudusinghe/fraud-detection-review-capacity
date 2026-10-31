"""The binding of the frozen artifacts: every check fails, before anything could read a Test row, on
a synthetic closed development run and analysis runs with an altered hash, identity, outcome or
selection record, and a clean set passes. The runs are plain containers of hand-made records and
files in a temporary project root, shaped like the real ones (see
``test_binding_on_evidence.py``); no research data is read."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from review_capacity.confirmatory.frozen import (
    EvaluatorState,
    FrozenBinding,
    bind_frozen_inputs,
    check_evaluator,
    evaluator_state,
)
from review_capacity.confirmatory.identity import (
    artifact_hashes,
    declaration_hashes,
    frozen_state_id,
)
from review_capacity.confirmatory.records import FROZEN_INPUTS
from review_capacity.development.records import ALL_CONFIGURATIONS, CONFIGURATION_NAMES
from review_capacity.development.run_output import LoadedRun
from review_capacity.errors import SpecificationError
from review_capacity.hashing import file_sha256

SETTINGS, TRAINING, EVALUATION = "s" * 64, "t" * 64, "e" * 64
ANCHOR = "9f266b2bdf2da9f1" + "0" * 48
CAPACITIES = (0.01, 0.02, 0.05, 0.1)
CLEAN = EvaluatorState(commit="a" * 40, tag_commit="a" * 40, clean=True, annotated=True)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return file_sha256(path)


def identity(run_type: str, run_id: str, manifest_sha256: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_type": run_type,
        "code_commit": "c" * 40,
        "source_sha256": sha("source"),
        "settings_sha256": SETTINGS,
        "declaration_sha256": TRAINING,
        "dataset_manifest_sha256": manifest_sha256,
    }


def closing(directory: Path) -> dict[str, Any]:
    """The closing record of a run: every other record file with its hash."""
    records = sorted(p for p in (directory / "records").iterdir() if p.name != "run_end.json")
    return {
        "outcome": "completed",
        "records": [{"path": f"records/{p.name}", "sha256": file_sha256(p)} for p in records],
    }


@dataclass
class World:
    root: Path
    development: LoadedRun
    analyses: list[LoadedRun]
    manifest_sha256: str

    @property
    def manifest(self) -> Path:
        return self.root / "data" / "datapackage.json"

    def bind(self, evaluator: EvaluatorState = CLEAN, **changes: Any) -> FrozenBinding:
        arguments: dict[str, Any] = {
            "root": self.root,
            "evaluator": evaluator,
            "development": self.development,
            "analyses": self.analyses,
            "manifest_path": self.manifest,
            "capacities": CAPACITIES,
            "settings_sha256": SETTINGS,
            "training_declaration_sha256": TRAINING,
            "evaluation_declaration_sha256": EVALUATION,
            "dataset_manifest_sha256": self.manifest_sha256,
        }
        return bind_frozen_inputs(**{**arguments, **changes})


def selection(selected: str = "LGB-b", *, binding: bool = True, rule: str = "1") -> dict[str, Any]:
    return {
        "record": "h3_selection",
        "period": "primary",
        "binding": binding,
        "rule_version": rule,
        "candidates": [{"configuration": name} for name in CONFIGURATION_NAMES],
        "selected": selected,
    }


def make_development(root: Path, manifest_sha256: str) -> LoadedRun:
    directory = root / "outputs" / "development" / "2026-10-25T134341Z"
    records: list[tuple[str, dict[str, Any]]] = [
        (
            "run.json",
            {
                "record": "run",
                "identity": identity("development", "2026-10-25T134341Z", manifest_sha256),
            },
        )
    ]
    write(directory / "records" / "run.json", "run")
    write(directory / "records" / "run_end.json", "development end")
    for name in ALL_CONFIGURATIONS:
        folder = directory / "models" / name / "trial-01"
        model_hash = write(folder / "model.bin", f"model {name}")
        manifest = {"model_sha256": model_hash} if name != "LR" else {"arrays_sha256": sha(name)}
        write(folder / "manifest.json", json.dumps(manifest))
        run_hash = write(folder / "model_run.json", f"model run {name}")
        stem = f"models/{name}/trial-01"
        fit_hash = write(directory / "records" / f"fit-{name}.json", f"fit {name}")
        records.append(
            (
                f"configuration-{name}.json",
                {
                    "record": "configuration",
                    "configuration": name,
                    "selected_fit": {"path": f"records/fit-{name}.json", "sha256": fit_hash},
                },
            )
        )
        records.append(
            (
                f"fit-{name}.json",
                {
                    "record": "fit",
                    "model": {"path": f"{stem}/model.bin", "sha256": model_hash},
                    "model_record": {"path": f"{stem}/model_run.json", "sha256": run_hash},
                },
            )
        )
    folder = root / "outputs" / "neural_preprocessing" / "x"
    artifact_hash = write(folder / "manifest.json", "preprocessing manifest")
    write(folder / "arrays.npz", "preprocessing arrays")
    records.append(
        (
            "preprocessing.json",
            {
                "record": "preprocessing",
                "artifact": {
                    "path": "outputs/neural_preprocessing/x/manifest.json",
                    "sha256": artifact_hash,
                },
            },
        )
    )
    return LoadedRun(directory, records, closing(directory))


def evaluation_record(name: str, policy: str, capacity: float) -> tuple[str, dict[str, Any]]:
    record = {
        "record": "evaluation",
        "partition": "V2",
        "period": "primary",
        "model_role": "selected",
        "configuration": name,
        "policy": policy,
        "capacity": capacity,
    }
    return f"evaluation-{name}-V2-{policy}-c{capacity}.json", record


def calibration_record(name: str, method: str, artifact: str) -> tuple[str, dict[str, Any]]:
    record = {
        "record": "calibration",
        "period": "primary",
        "configuration": name,
        "method": method,
        "artifact": {"path": f"calibrators/{name}-{method}.bin", "sha256": artifact},
    }
    return f"calibration-{name}-{method}-primary.json", record


def make_analysis(
    root: Path, run_id: str, development: LoadedRun, manifest_sha256: str, *, coverage: bool
) -> LoadedRun:
    directory = root / "outputs" / "analysis" / run_id
    write(directory / "records" / "run_end.json", f"end {run_id}")
    recs: list[tuple[str, dict[str, Any]]] = [
        (
            "run.json",
            {
                "record": "run",
                "identity": identity("analysis", run_id, manifest_sha256),
                "source": {
                    "run_id": "2026-10-25T134341Z",
                    "run_end": {
                        "path": "x",
                        "sha256": file_sha256(development.path / "records" / "run_end.json"),
                    },
                },
            },
        ),
        ("h3_selection-primary.json", selection()),
    ]
    if coverage:
        for name in CONFIGURATION_NAMES:
            recs.append(
                (
                    f"r1_selection-{name}-primary.json",
                    {"record": "r1_selection", "period": "primary", "configuration": name},
                )
            )
            recs.extend(
                evaluation_record(name, policy, capacity)
                for policy in ("R1", "R2", "R3")
                for capacity in CAPACITIES
            )
            for method in ("platt", "isotonic"):
                artifact = write(
                    directory / "calibrators" / f"{name}-{method}.bin", f"cal {name} {method}"
                )
                recs.append(calibration_record(name, method, artifact))
    for name, record in recs:
        write(directory / "records" / name, f"{run_id} {name} {record['record']}")
    return LoadedRun(directory, recs, closing(directory))


def make_world(tmp_path: Path) -> World:
    manifest = {
        "_split": {
            "partitions": [
                {"name": name, "transaction_id_sha256": sha(name)} for name in ("train", "v1", "v2")
            ]
            + [{"name": "test", "transaction_id_sha256": ANCHOR}]
        }
    }
    manifest_sha256 = write(tmp_path / "data" / "datapackage.json", json.dumps(manifest))
    development = make_development(tmp_path, manifest_sha256)
    analyses = [
        make_analysis(tmp_path, "2026-10-29T013124Z", development, manifest_sha256, coverage=True),
        make_analysis(tmp_path, "2026-10-29T013414Z", development, manifest_sha256, coverage=False),
    ]
    return World(tmp_path, development, analyses, manifest_sha256)


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path)


# --- the clean set -------------------------------------------------------------------------------


def test_a_clean_set_binds_and_gives_a_valid_frozen_inputs_block(world: World) -> None:
    binding = world.bind()
    inputs = binding.frozen_inputs
    assert FROZEN_INPUTS.problems(inputs, "frozen_inputs") == []
    assert binding.h3_configuration == "LGB-b"
    assert binding.test_anchor_sha256 == ANCHOR
    for group in ("models", "model_records", "model_manifests"):
        assert set(inputs[group]) == set(ALL_CONFIGURATIONS)
    assert all(not entry["path"].startswith("/") for entry in inputs["models"].values())
    assert inputs["preprocessing_arrays"]["path"].endswith("arrays.npz")
    assert [run["run_id"] for run in inputs["analysis_runs"]] == [
        "2026-10-29T013124Z",
        "2026-10-29T013414Z",
    ]
    assert len(inputs["thresholds"]) == 8 + 8 * 2 * 4  # R1 selections, R1 and R2 evaluations
    assert len(inputs["calibrators"]) == 2 * (8 * 2)  # calibration records and saved calibrators
    assert binding.training["run_id"] == "2026-10-25T134341Z"
    expected = frozen_state_id(
        commit=CLEAN.tag_commit,
        declarations={
            "settings": SETTINGS,
            "training": TRAINING,
            "evaluation": EVALUATION,
            "dataset_manifest": world.manifest_sha256,
        },
        artifacts=artifact_hashes(inputs),
    )
    assert binding.frozen_state_sha256 == expected


def test_binding_twice_gives_the_same_identity(world: World) -> None:
    assert world.bind().frozen_state_sha256 == world.bind().frozen_state_sha256


def test_a_single_analysis_run_that_holds_everything_is_enough(world: World) -> None:
    assert world.bind(analyses=world.analyses[:1]).h3_configuration == "LGB-b"


def test_at_least_one_analysis_run_is_needed(world: World) -> None:
    with pytest.raises(SpecificationError, match="at least one analysis run"):
        world.bind(analyses=[])


# --- failures ------------------------------------------------------------------------------------


def tamper(path: Path) -> None:
    path.write_text(path.read_text(encoding="utf-8") + " changed", encoding="utf-8")


def drop(run: LoadedRun, keep: Callable[[str, dict[str, Any]], bool]) -> None:
    run.records[:] = [(n, r) for n, r in run.records if keep(n, r)]


def find(run: LoadedRun, kind: str, **fields: Any) -> tuple[str, dict[str, Any]]:
    return next(
        (n, r)
        for n, r in run.records
        if r["record"] == kind and all(r.get(k) == v for k, v in fields.items())
    )


def case_dev_not_closed(world: World) -> None:
    world.development.end = None


def case_dev_failed(world: World) -> None:
    world.development.end = {"outcome": "failed", "records": []}


def case_dev_wrong_type(world: World) -> None:
    world.development.records[0][1]["identity"]["run_type"] = "analysis"


def case_analysis_open(world: World) -> None:
    world.analyses[0].end = None


def case_analysis_other_source(world: World) -> None:
    world.analyses[0].records[0][1]["source"]["run_end"]["sha256"] = sha("another run")


def case_analysis_no_source(world: World) -> None:
    world.analyses[1].records[0][1]["source"] = None


def case_analysis_declarations(world: World) -> None:
    world.analyses[0].records[0][1]["identity"]["settings_sha256"] = "x" * 64


def case_no_calibration(world: World) -> None:
    for run in world.analyses:
        drop(run, lambda _, r: r["record"] != "calibration")


def case_no_binding_selection(world: World) -> None:
    for run in world.analyses:
        for _, record in run.records:
            if record["record"] == "h3_selection":
                record["binding"] = False


def case_two_binding_selections(world: World) -> None:
    world.analyses[0].records.append(("h3_selection-copy.json", selection()))


def case_old_rule(world: World) -> None:
    world.analyses[0].records[1][1]["rule_version"] = "2"


def case_selections_disagree(world: World) -> None:
    world.analyses[1].records[1][1]["selected"] = "LGB-u"


def case_missing_r2_evaluation(world: World) -> None:
    name, _ = find(
        world.analyses[0], "evaluation", configuration="Cat-b", policy="R2", capacity=0.05
    )
    drop(world.analyses[0], lambda n, _: n != name)


def case_missing_r1_selection(world: World) -> None:
    drop(world.analyses[0], lambda n, _: n != "r1_selection-RES-u-primary.json")


def case_missing_calibration_method(world: World) -> None:
    drop(world.analyses[0], lambda n, _: n != "calibration-FTT-b-isotonic-primary.json")


def case_conflicting_duplicate(world: World) -> None:
    extra = {
        "record": "r1_selection",
        "period": "primary",
        "configuration": "LGB-b",
        "threshold": 0.5,
    }
    world.analyses[1].records.append(("r1_selection-LGB-b-primary-again.json", extra))


def case_model_tampered(world: World) -> None:
    tamper(world.development.path / "models" / "LGB-b" / "trial-01" / "model.bin")


def case_model_missing(world: World) -> None:
    (world.development.path / "models" / "Cat-u" / "trial-01" / "model.bin").unlink()


def case_model_is_a_link(world: World) -> None:
    folder = world.development.path / "models" / "Cat-u" / "trial-01"
    copy_path = folder.parent / "copy.bin"
    copy_path.write_bytes((folder / "model.bin").read_bytes())
    (folder / "model.bin").unlink()
    (folder / "model.bin").symlink_to(copy_path)


def case_model_outside_the_run(world: World) -> None:
    source = world.development.path / "models" / "RES-u" / "trial-01" / "model.bin"
    (world.root / "outside.bin").write_bytes(source.read_bytes())
    for _, record in world.development.records:
        if record["record"] == "fit" and "RES-u" in record["model"]["path"]:
            record["model"]["path"] = "../../../outside.bin"


def case_model_run_tampered(world: World) -> None:
    tamper(world.development.path / "models" / "FTT-u" / "trial-01" / "model_run.json")


def case_model_manifest_states_another_model(world: World) -> None:
    path = world.development.path / "models" / "Cat-b" / "trial-01" / "manifest.json"
    path.write_text(json.dumps({"model_sha256": sha("another")}), encoding="utf-8")


def case_model_manifest_missing(world: World) -> None:
    (world.development.path / "models" / "RES-b" / "trial-01" / "manifest.json").unlink()


def case_model_manifest_unreadable(world: World) -> None:
    path = world.development.path / "models" / "RES-b" / "trial-01" / "manifest.json"
    path.write_text("not json", encoding="utf-8")


def case_fit_tampered(world: World) -> None:
    tamper(world.development.path / "records" / "fit-RES-b.json")


def case_configuration_missing(world: World) -> None:
    drop(world.development, lambda _, r: r.get("configuration") != "LR")


def case_fit_without_model(world: World) -> None:
    for _, record in world.development.records:
        if record["record"] == "fit":
            record["model"] = None
            break


def case_fit_without_model_run(world: World) -> None:
    for _, record in world.development.records:
        if record["record"] == "fit":
            record["model_record"] = None
            break


def case_preprocessing_tampered(world: World) -> None:
    tamper(world.root / "outputs" / "neural_preprocessing" / "x" / "manifest.json")


def case_preprocessing_arrays_missing(world: World) -> None:
    (world.root / "outputs" / "neural_preprocessing" / "x" / "arrays.npz").unlink()


def case_preprocessing_record_missing(world: World) -> None:
    drop(world.development, lambda _, r: r["record"] != "preprocessing")


def case_calibrator_tampered(world: World) -> None:
    tamper(world.analyses[0].path / "calibrators" / "LGB-b-platt.bin")


def case_calibrator_unrecorded(world: World) -> None:
    for _, record in world.analyses[0].records:
        if record["record"] == "calibration":
            record["artifact"] = None
            break


def case_threshold_record_changed_after_closing(world: World) -> None:
    name, _ = find(
        world.analyses[0], "evaluation", configuration="LGB-b", policy="R2", capacity=0.02
    )
    tamper(world.analyses[0].path / "records" / name)


def case_h3_selection_changed_after_closing(world: World) -> None:
    tamper(world.analyses[0].path / "records" / "h3_selection-primary.json")


def case_record_not_listed(world: World) -> None:
    end = world.analyses[0].end
    assert end is not None
    end["records"] = [e for e in end["records"] if e["path"] != "records/h3_selection-primary.json"]


def case_manifest_changed(world: World) -> None:
    tamper(world.manifest)


def case_manifest_without_test_partition(world: World) -> None:
    manifest = json.loads(world.manifest.read_text(encoding="utf-8"))
    manifest["_split"]["partitions"] = manifest["_split"]["partitions"][:3]
    new_hash = write(world.manifest, json.dumps(manifest))
    world.manifest_sha256 = new_hash
    world.development.records[0][1]["identity"]["dataset_manifest_sha256"] = new_hash


FAILURES: list[tuple[Callable[[World], None], str]] = [
    (case_dev_not_closed, "was not closed"),
    (case_dev_failed, "the development run failed"),
    (case_dev_wrong_type, "not a development run"),
    (case_analysis_open, "not closed and completed"),
    (case_analysis_other_source, "does not read this development run"),
    (case_analysis_no_source, "does not read this development run"),
    (case_analysis_declarations, "settings_sha256 differs"),
    (case_no_calibration, "lack records of the kinds"),
    (case_no_binding_selection, "binding primary h3_selection"),
    (case_two_binding_selections, "binding h3_selection records"),
    (case_old_rule, "not version 1"),
    (case_selections_disagree, "disagree"),
    (case_missing_r2_evaluation, "do not cover the frozen policies.*Cat-b.*R2"),
    (case_missing_r1_selection, "do not cover the frozen policies.*RES-u"),
    (case_missing_calibration_method, "do not cover the frozen policies.*isotonic"),
    (case_conflicting_duplicate, "conflicting records"),
    (case_model_tampered, "model of LGB-b"),
    (case_model_missing, "model of Cat-u"),
    (case_model_is_a_link, "is a link"),
    (case_model_outside_the_run, "lies outside"),
    (case_model_run_tampered, "model run of FTT-u"),
    (case_model_manifest_states_another_model, "states another model SHA-256"),
    (case_model_manifest_missing, "manifest of RES-b"),
    (case_model_manifest_unreadable, "not readable JSON"),
    (case_fit_tampered, "fit of RES-b"),
    (case_configuration_missing, "no configuration LR"),
    (case_fit_without_model, "records no model"),
    (case_fit_without_model_run, "records no model run"),
    (case_preprocessing_tampered, "preprocessing artifact"),
    (case_preprocessing_arrays_missing, "preprocessing arrays"),
    (case_preprocessing_record_missing, "exactly one preprocessing"),
    (case_calibrator_tampered, "saved calibrator"),
    (case_calibrator_unrecorded, "records no saved calibrator"),
    (case_threshold_record_changed_after_closing, "differs from the recorded SHA-256"),
    (case_h3_selection_changed_after_closing, "differs from the recorded SHA-256"),
    (case_record_not_listed, "does not list"),
    (case_manifest_changed, "dataset manifest: .* differs from the recorded SHA-256"),
    (case_manifest_without_test_partition, "no test partition anchor"),
]


@pytest.mark.parametrize(("case", "fragment"), FAILURES, ids=[c.__name__ for c, _ in FAILURES])
def test_each_altered_input_stops_the_binding(
    world: World, case: Callable[[World], None], fragment: str
) -> None:
    case(world)
    with pytest.raises(SpecificationError, match=fragment):
        world.bind()


def test_declarations_that_differ_from_the_development_run_stop_the_binding(world: World) -> None:
    with pytest.raises(SpecificationError, match="model settings"):
        world.bind(settings_sha256="z" * 64)
    with pytest.raises(SpecificationError, match="training declaration"):
        world.bind(training_declaration_sha256="z" * 64)
    with pytest.raises(SpecificationError, match="dataset manifest"):
        world.bind(dataset_manifest_sha256="z" * 64)


# --- the evaluator -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "fragment"),
    [
        (EvaluatorState("a" * 40, "a" * 40, True, False), "not an annotated tag"),
        (EvaluatorState("a" * 40, "a" * 40, False, True), "uncommitted changes"),
        (EvaluatorState("b" * 40, "a" * 40, True, True), "not the commit of the freeze tag"),
    ],
)
def test_the_evaluator_must_be_a_clean_tree_at_an_annotated_tag(
    world: World, state: EvaluatorState, fragment: str
) -> None:
    with pytest.raises(SpecificationError, match=fragment):
        check_evaluator(state)
    with pytest.raises(SpecificationError, match=fragment):
        world.bind(evaluator=state)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            *args,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_the_evaluator_state_is_read_from_git(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    git(tmp_path, "add", "a.txt")
    git(tmp_path, "commit", "-q", "-m", "first")
    git(tmp_path, "tag", "-a", "freeze", "-m", "protocol freeze")
    git(tmp_path, "tag", "lightweight")
    head = git(tmp_path, "rev-parse", "HEAD")

    state = evaluator_state(tmp_path, "freeze")
    assert state == EvaluatorState(head, head, True, True)
    check_evaluator(state)
    assert evaluator_state(tmp_path, "lightweight").annotated is False

    (tmp_path / "a.txt").write_text("two", encoding="utf-8")
    assert evaluator_state(tmp_path, "freeze").clean is False

    git(tmp_path, "commit", "-q", "-am", "second")
    later = evaluator_state(tmp_path, "freeze")
    assert later.commit != later.tag_commit
    with pytest.raises(SpecificationError, match="not the commit of the freeze tag"):
        check_evaluator(later)


def test_an_unknown_tag_or_a_missing_repository_is_a_specification_error(tmp_path: Path) -> None:
    with pytest.raises(SpecificationError, match=r"git .* failed"):
        evaluator_state(tmp_path, "freeze")  # not a repository
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    git(tmp_path, "add", "a.txt")
    git(tmp_path, "commit", "-q", "-m", "first")
    with pytest.raises(SpecificationError, match=r"git .* failed"):
        evaluator_state(tmp_path, "no-such-tag")


# --- the identity of the frozen state ------------------------------------------------------------

BASE_ARTIFACTS = {
    "development_run": [sha("dev")],
    "models": [f"LGB-b:{sha('m1')}", f"Cat-b:{sha('m2')}"],
}
BASE_DECLARATIONS = {"settings": sha("s"), "training": sha("t"), "evaluation": sha("e")}


def state(
    commit: str = "a" * 40,
    declarations: dict[str, str] | None = None,
    artifacts: dict[str, list[str]] | None = None,
) -> str:
    return frozen_state_id(
        commit=commit,
        declarations=declarations or BASE_DECLARATIONS,
        artifacts=artifacts or BASE_ARTIFACTS,
    )


def test_the_frozen_state_identity_is_deterministic_and_order_independent() -> None:
    assert state() == state()
    reordered = {
        "models": list(reversed(BASE_ARTIFACTS["models"])),
        "development_run": BASE_ARTIFACTS["development_run"],
    }
    assert state(artifacts=reordered) == state()
    assert len(state()) == 64


def test_the_frozen_state_identity_changes_with_every_input() -> None:
    assert state(commit="b" * 40) != state()
    for name in BASE_DECLARATIONS:
        assert state(declarations={**BASE_DECLARATIONS, name: sha("changed")}) != state()
    for group in BASE_ARTIFACTS:
        changed = copy.deepcopy(BASE_ARTIFACTS)
        changed[group][0] = sha("changed")
        assert state(artifacts=changed) != state()
    swapped = {**BASE_ARTIFACTS, "models": [f"LGB-b:{sha('m2')}", f"Cat-b:{sha('m1')}"]}
    assert state(artifacts=swapped) != state()  # exchanging two models changes the identity


def test_every_file_group_of_the_frozen_inputs_enters_the_identity(world: World) -> None:
    groups = artifact_hashes(world.bind().frozen_inputs)
    assert set(groups) == {
        "development_run",
        "preprocessing",
        "models",
        "model_records",
        "model_manifests",
    }
    assert len(groups["preprocessing"]) == 2  # the manifest and the arrays


def test_a_second_tag_name_on_the_same_commit_gives_the_same_frozen_state(world: World) -> None:
    first = world.bind(evaluator=EvaluatorState("a" * 40, "a" * 40, True, True))
    renamed = world.bind(evaluator=EvaluatorState("a" * 40, "a" * 40, True, True))
    assert first.frozen_state_sha256 == renamed.frozen_state_sha256  # the tag name is not an input
    other_commit = world.bind(evaluator=EvaluatorState("b" * 40, "b" * 40, True, True))
    assert other_commit.frozen_state_sha256 != first.frozen_state_sha256


def test_declaration_hashes_are_read_from_a_record_identity() -> None:
    identity_block = {
        "settings_sha256": "1" * 64,
        "training_declaration_sha256": "2" * 64,
        "evaluation_declaration_sha256": "3" * 64,
        "dataset_manifest_sha256": "4" * 64,
    }
    assert declaration_hashes(identity_block) == {
        "settings": "1" * 64,
        "training": "2" * 64,
        "evaluation": "3" * 64,
        "dataset_manifest": "4" * 64,
    }


def test_the_order_and_the_list_of_analysis_runs_do_not_change_the_frozen_state(
    world: World,
) -> None:
    both = world.bind()
    assert len(world.analyses) == 2
    for order in (list(reversed(world.analyses)), world.analyses[:1], world.analyses[1:]):
        try:
            other = world.bind(analyses=order)
        except SpecificationError:
            continue  # this list alone does not supply the policies
        assert other.frozen_state_sha256 == both.frozen_state_sha256
    reversed_binding = world.bind(analyses=list(reversed(world.analyses)))
    assert reversed_binding.frozen_state_sha256 == both.frozen_state_sha256


def test_the_policies_are_bound_in_the_record_but_not_in_the_identity(world: World) -> None:
    inputs = world.bind().frozen_inputs
    assert {"h3_selection", "thresholds", "calibrators", "analysis_runs"} <= set(inputs)
    assert not {"h3_selection", "thresholds", "calibrators", "analysis_runs"} & set(
        artifact_hashes(inputs)
    )
