"""The binding of the frozen artifacts to the confirmatory evaluation (Issue #195; #185, item D).

Before any Test row is read, the evaluation verifies that everything it will use is exactly what
was frozen: the evaluating code is a clean tree at the commit of the annotated freeze tag; the
development run that trained the models and the analysis runs that fixed the thresholds,
calibrators and the H3 selection are closed and completed and refer to one another by hash; the
saved model of every configuration with its run record and manifest, the shared preprocessing
artifact with its arrays and the saved calibrators are the files those runs recorded; and the
declarations and the dataset manifest are the ones the runs used. Each check raises
``SpecificationError`` naming the item that differs. Nothing here reads research data: the checks
work on the records and the hashes of the files.

A file is read once: its hash is computed once and compared with the hash recorded for it (the
closing record of its run, or the record that names it), and that same hash is what is frozen. A
file must be a regular file, not a link, inside the directory of its run or the project root.

``bind_frozen_inputs`` runs all of them and returns the ``frozen_inputs`` block of the
``test_run`` record together with the identity of the frozen state (``confirmatory.identity``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from review_capacity.confirmatory.identity import artifact_hashes, frozen_state_id
from review_capacity.development.records import ALL_CONFIGURATIONS, CONFIGURATION_NAMES
from review_capacity.development.run_output import END_KIND, RECORDS_DIR, LoadedRun
from review_capacity.errors import SpecificationError
from review_capacity.hashing import file_sha256

logger = logging.getLogger(__name__)

SELECTION_RULE_VERSION = "1"  # DR-010, the version of the H3 selection rule
ANALYSIS_KINDS = ("r1_selection", "calibration", "h3_selection")
CANDIDATES = 8  # the eight configurations of the H3 selection
MANIFEST_FILE = "manifest.json"  # beside every saved model and the preprocessing artifact
ARRAYS_FILE = "arrays.npz"  # beside the preprocessing manifest
CALIBRATION_METHODS = ("platt", "isotonic")
THRESHOLD_POLICIES = ("R1", "R2")
MAX_REPORTED = 5  # problems named in one coverage message

type FileRef = dict[str, str]


@dataclass(frozen=True)
class EvaluatorState:
    """The state of the evaluating checkout against the freeze tag."""

    commit: str  # HEAD
    tag_commit: str  # the commit the tag points to
    clean: bool  # no uncommitted change
    annotated: bool  # the tag is an annotated tag object


@dataclass(frozen=True)
class FrozenBinding:
    """The verified frozen inputs: the ``frozen_inputs`` block of the ``test_run`` record, the
    training identity, the Test partition anchor of the manifest and the identity of the frozen
    state."""

    frozen_inputs: dict[str, Any]
    training: dict[str, Any]
    test_anchor_sha256: str
    frozen_state_sha256: str

    @property
    def h3_configuration(self) -> str:
        configuration: str = self.frozen_inputs["h3_configuration"]
        return configuration


@dataclass(frozen=True)
class ModelReference:
    """What the development run recorded for the selected model of one configuration."""

    fit: Mapping[str, str]  # the selected fit record
    model: Mapping[str, str]  # the saved model file
    record: Mapping[str, str]  # the model-run record


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SpecificationError(f"git {' '.join(args)} failed in {root}: {error}") from error
    return result.stdout.strip()


def evaluator_state(root: Path, tag: str) -> EvaluatorState:
    """The state of the checkout at ``root`` against the tag ``tag``."""
    return EvaluatorState(
        commit=_git(root, "rev-parse", "HEAD"),
        tag_commit=_git(root, "rev-parse", "--verify", f"{tag}^{{commit}}"),
        clean=_git(root, "status", "--porcelain") == "",
        annotated=_git(root, "cat-file", "-t", f"refs/tags/{tag}") == "tag",
    )


def check_evaluator(state: EvaluatorState) -> None:
    """Stop unless the evaluating code is a clean tree at the commit of an annotated tag."""
    if not state.annotated:
        raise SpecificationError("The freeze tag is not an annotated tag")
    if not state.clean:
        raise SpecificationError("The working tree has uncommitted changes")
    if state.commit != state.tag_commit:
        raise SpecificationError(
            f"HEAD {state.commit} is not the commit of the freeze tag {state.tag_commit}"
        )


def _inside(path: Path, base: Path, what: str) -> None:
    """``path`` is a regular file that is not a link and lies inside ``base``."""
    if path.is_symlink():
        raise SpecificationError(f"{what}: {path} is a link, not a file of its run")
    if not path.is_file():
        raise SpecificationError(f"{what}: {path} does not exist")
    if not path.resolve().is_relative_to(base.resolve()):
        raise SpecificationError(f"{what}: {path} lies outside {base}")


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _checked(path: Path, expected: str | None, what: str, base: Path, root: Path) -> FileRef:
    """The file's ``{path, sha256}``, the path relative to the project root. The file is read
    once; with an ``expected`` hash it must equal it, and the same hash is returned."""
    _inside(path, base, what)
    digest = file_sha256(path)
    if expected is not None and digest != expected:
        raise SpecificationError(f"{what}: {path} differs from the recorded SHA-256")
    return {"path": _relative(path, root), "sha256": digest}


def _checked_bytes(
    path: Path, expected: str | None, what: str, base: Path, root: Path
) -> tuple[FileRef, bytes]:
    """As ``_checked`` for a small file whose content is also needed: the bytes are read once and
    both the hash and the content come from them."""
    _inside(path, base, what)
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected is not None and digest != expected:
        raise SpecificationError(f"{what}: {path} differs from the recorded SHA-256")
    return {"path": _relative(path, root), "sha256": digest}, content


def _listed(run: LoadedRun) -> dict[str, str]:
    """The hashes the closing record of ``run`` lists for its record files."""
    if run.end is None:
        raise SpecificationError(f"{run.path}: the run was not closed")
    return {entry["path"]: entry["sha256"] for entry in run.end["records"]}


def record_ref(run: LoadedRun, name: str, root: Path) -> FileRef:
    """A record file of a closed run, verified against the hash its closing record lists."""
    listed = _listed(run).get(f"{RECORDS_DIR}/{name}")
    if listed is None:
        raise SpecificationError(f"{run.path}: the closing record does not list {name}")
    return _checked(run.path / RECORDS_DIR / name, listed, f"record {name}", run.path, root)


def _closing(run: LoadedRun) -> Path:
    return run.path / RECORDS_DIR / f"{END_KIND}.json"


def check_development_run(source: LoadedRun) -> None:
    """The development run is closed and completed."""
    if source.identity["run_type"] != "development":
        raise SpecificationError(f"{source.path}: not a development run")
    if source.end is None:
        raise SpecificationError(f"{source.path}: the development run was not closed")
    if source.end["outcome"] != "completed":
        raise SpecificationError(f"{source.path}: the development run {source.end['outcome']}")


def _same_declarations(run: LoadedRun, source: LoadedRun) -> None:
    for name in ("settings_sha256", "declaration_sha256", "dataset_manifest_sha256"):
        if run.identity[name] != source.identity[name]:
            raise SpecificationError(f"{run.path}: {name} differs from that of the development run")


def binding_selections(analysis: LoadedRun) -> list[Mapping[str, Any]]:
    """The binding H3 selections of the analysis run (full V2): at most one, with the selection
    rule of version 1, the eight configurations compared and a selected one among them."""
    found: list[Mapping[str, Any]] = [
        record
        for record in analysis.of_kind("h3_selection")
        if record["period"] == "primary" and record["binding"]
    ]
    if len(found) > 1:
        raise SpecificationError(
            f"{analysis.path}: holds {len(found)} binding h3_selection records"
        )
    for selection in found:
        if selection["rule_version"] != SELECTION_RULE_VERSION:
            raise SpecificationError(f"{analysis.path}: the selection rule is not version 1")
        names = [candidate["configuration"] for candidate in selection["candidates"]]
        if len(names) != CANDIDATES or selection["selected"] not in names:
            raise SpecificationError(f"{analysis.path}: the selection does not compare the eight")
    return found


def check_analysis_run(analysis: LoadedRun, source: LoadedRun) -> None:
    """The analysis run is closed and completed, reads this very development run (by the hash of
    its closing record) and uses the same declarations."""
    if analysis.identity["run_type"] != "analysis":
        raise SpecificationError(f"{analysis.path}: not an analysis run")
    if analysis.end is None or analysis.end["outcome"] != "completed":
        raise SpecificationError(f"{analysis.path}: the analysis run is not closed and completed")
    recorded = analysis.of_kind("run")[0]["source"]
    if recorded is None or recorded["run_end"]["sha256"] != file_sha256(_closing(source)):
        raise SpecificationError(f"{analysis.path}: it does not read this development run")
    _same_declarations(analysis, source)
    binding_selections(analysis)


def check_analysis_runs(analyses: Sequence[LoadedRun], source: LoadedRun) -> Mapping[str, Any]:
    """Every analysis run is bound to the development run; together they hold the records of the
    thresholds, the calibrators and the binding H3 selection, and the binding selections agree.
    Returns the binding selection."""
    if not analyses:
        raise SpecificationError("The evaluation needs at least one analysis run")
    for analysis in analyses:
        check_analysis_run(analysis, source)
    held = {kind for analysis in analyses for kind in ANALYSIS_KINDS if analysis.of_kind(kind)}
    if held != set(ANALYSIS_KINDS):
        raise SpecificationError(
            f"The analysis runs lack records of the kinds {sorted(set(ANALYSIS_KINDS) - held)}"
        )
    selections = [s for analysis in analyses for s in binding_selections(analysis)]
    if not selections:
        raise SpecificationError("No analysis run holds the binding primary h3_selection")
    first = selections[0]
    if any(
        (s["selected"], s["candidates"]) != (first["selected"], first["candidates"])
        for s in selections
    ):
        raise SpecificationError("The analysis runs disagree on the binding H3 selection")
    return first


def check_declarations(
    source: LoadedRun,
    *,
    settings_sha256: str,
    training_declaration_sha256: str,
    dataset_manifest_sha256: str,
) -> None:
    """The declarations and the manifest loaded now are those the development run used."""
    identity = source.identity
    for what, found, expected in (
        ("model settings", settings_sha256, identity["settings_sha256"]),
        ("training declaration", training_declaration_sha256, identity["declaration_sha256"]),
        ("dataset manifest", dataset_manifest_sha256, identity["dataset_manifest_sha256"]),
    ):
        if found != expected:
            raise SpecificationError(f"The {what} differ from those of the development run")


def partition_anchor(manifest: Path, expected_sha256: str, partition: str = "test") -> str:
    """The ``transaction_id_sha256`` of a partition from the dataset manifest, read once: the
    manifest's own hash must equal ``expected_sha256``."""
    _, content = _checked_bytes(
        manifest, expected_sha256, "dataset manifest", manifest.parent, manifest.parent
    )
    try:
        partitions = json.loads(content)["_split"]["partitions"]
        anchor = next(p["transaction_id_sha256"] for p in partitions if p["name"] == partition)
    except (KeyError, StopIteration, TypeError, ValueError) as error:
        raise SpecificationError(
            f"The dataset manifest has no {partition} partition anchor"
        ) from error
    if not isinstance(anchor, str) or len(anchor) != 64:
        raise SpecificationError(f"The {partition} partition anchor is not a SHA-256 digest")
    return anchor


def model_references(source: LoadedRun) -> dict[str, ModelReference]:
    """What the development run recorded for the selected model of each configuration and of the
    reference: the selected fit record names the saved model and its model-run record. A
    record-level lookup; no file is read."""
    selected = {record["configuration"]: record for record in source.of_kind("configuration")}
    fits = {name: record for name, record in source.records if record["record"] == "fit"}
    references: dict[str, ModelReference] = {}
    for configuration in ALL_CONFIGURATIONS:
        if configuration not in selected:
            raise SpecificationError(f"The development run has no configuration {configuration}")
        reference = selected[configuration]["selected_fit"]
        fit = fits.get(Path(reference["path"]).name)
        if fit is None or fit["model"] is None:
            raise SpecificationError(f"The selected fit of {configuration} records no model")
        if fit["model_record"] is None:
            raise SpecificationError(f"The selected fit of {configuration} records no model run")
        references[configuration] = ModelReference(reference, fit["model"], fit["model_record"])
    return references


def _stated_model_sha256(content: bytes, configuration: str) -> str | None:
    """The ``model_sha256`` a model manifest states, if it states one."""
    try:
        stated = json.loads(content).get("model_sha256")
    except (ValueError, AttributeError) as error:
        raise SpecificationError(f"manifest of {configuration}: not readable JSON") from error
    return stated if isinstance(stated, str) else None


def model_files(
    source: LoadedRun, root: Path
) -> tuple[dict[str, FileRef], dict[str, FileRef], dict[str, FileRef]]:
    """The saved model, its model-run record and its manifest for each configuration and the
    reference. The fit record, the model and the model-run record are verified against the hashes
    the development run recorded; the manifest beside the model is hashed (and frozen), and when
    it states the model's SHA-256 that must be the recorded one. Returns three mappings of
    ``{path, sha256}`` by configuration, paths relative to the project root."""
    models: dict[str, FileRef] = {}
    records: dict[str, FileRef] = {}
    manifests: dict[str, FileRef] = {}
    base = source.path
    for configuration, reference in model_references(source).items():
        _checked(
            base / reference.fit["path"],
            reference.fit["sha256"],
            f"fit of {configuration}",
            base,
            root,
        )
        model_path = base / reference.model["path"]
        models[configuration] = _checked(
            model_path, reference.model["sha256"], f"model of {configuration}", base, root
        )
        records[configuration] = _checked(
            base / reference.record["path"],
            reference.record["sha256"],
            f"model run of {configuration}",
            base,
            root,
        )
        manifests[configuration], content = _checked_bytes(
            model_path.parent / MANIFEST_FILE, None, f"manifest of {configuration}", base, root
        )
        stated = _stated_model_sha256(content, configuration)
        if stated is not None and stated != reference.model["sha256"]:
            raise SpecificationError(
                f"manifest of {configuration}: states another model SHA-256 than the recorded one"
            )
    return models, records, manifests


def preprocessing_files(source: LoadedRun, root: Path) -> tuple[FileRef, FileRef]:
    """The shared neural preprocessing artifact: its manifest, verified against the development
    run's preprocessing record, and the arrays file beside it (hashed and frozen)."""
    records = source.of_kind("preprocessing")
    if len(records) != 1:
        raise SpecificationError(f"{source.path}: needs exactly one preprocessing record")
    artifact = records[0]["artifact"]
    manifest = _checked(
        root / artifact["path"], artifact["sha256"], "preprocessing artifact", root, root
    )
    arrays = _checked(
        (root / artifact["path"]).parent / ARRAYS_FILE, None, "preprocessing arrays", root, root
    )
    return manifest, arrays


def _coverage_key(record: Mapping[str, Any]) -> tuple[Any, ...] | None:
    kind = record["record"]
    if kind == "r1_selection" and record["period"] == "primary":
        return (kind, record["configuration"])
    if kind == "calibration" and record["period"] == "primary":
        return (kind, record["configuration"], record["method"])
    if (
        kind == "evaluation"
        and record["partition"] == "V2"
        and record["period"] == "primary"
        and record["model_role"] == "selected"
        and record["policy"] in THRESHOLD_POLICIES
    ):
        return (kind, record["configuration"], record["policy"], record["capacity"])
    return None


def coverage_problems(analyses: Sequence[LoadedRun], capacities: Sequence[float]) -> list[str]:
    """What the analysis runs must hold for the Test evaluation, as problems when it is absent or
    conflicting: for each of the eight configurations, the primary R1 selection; the primary V2
    evaluation of the selected model under R1 and R2 at each capacity; and the primary Platt and
    isotonic calibrations. A record held by two runs must be the same apart from its identity."""
    required: set[tuple[Any, ...]] = set()
    for configuration in CONFIGURATION_NAMES:
        required.add(("r1_selection", configuration))
        required.update(("calibration", configuration, method) for method in CALIBRATION_METHODS)
        required.update(
            ("evaluation", configuration, policy, capacity)
            for policy in THRESHOLD_POLICIES
            for capacity in capacities
        )
    found: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for analysis in analyses:
        for _, record in analysis.records:
            key = _coverage_key(record)
            if key is not None:
                found.setdefault(key, []).append(record)
    problems = [f"missing {key}" for key in sorted(required - found.keys(), key=str)]
    for key, records in found.items():
        payloads = [{k: v for k, v in r.items() if k != "identity"} for r in records]
        if key in required and any(payload != payloads[0] for payload in payloads[1:]):
            problems.append(f"conflicting records for {key}")
    return problems


def threshold_and_calibrator_files(
    analyses: Sequence[LoadedRun], root: Path, capacities: Sequence[float]
) -> tuple[list[FileRef], list[FileRef]]:
    """The files that fix the V2 thresholds and calibrators, once the coverage is complete: the
    primary R1 selections and the primary V2 evaluations of R1 and R2 (thresholds); the primary
    calibration records and the saved calibrators they name (calibrators). Each record file is
    verified against its run's closing record, a saved calibrator against its calibration."""
    problems = coverage_problems(analyses, capacities)
    if problems:
        shown = "; ".join(problems[:MAX_REPORTED])
        more = f" (and {len(problems) - MAX_REPORTED} more)" if len(problems) > MAX_REPORTED else ""
        raise SpecificationError(
            f"The analysis runs do not cover the frozen policies: {shown}{more}"
        )
    thresholds: dict[str, FileRef] = {}
    calibrators: dict[str, FileRef] = {}
    seen: set[tuple[Any, ...]] = set()
    for analysis in analyses:
        for name, record in analysis.records:
            key = _coverage_key(record)
            if key is None or key in seen:
                continue
            seen.add(key)
            reference = record_ref(analysis, name, root)
            if record["record"] != "calibration":
                thresholds[reference["path"]] = reference
                continue
            calibrators[reference["path"]] = reference
            artifact = record["artifact"]
            if artifact is None:
                raise SpecificationError(
                    f"{analysis.path}: a calibration records no saved calibrator"
                )
            saved = _checked(
                analysis.path / artifact["path"],
                artifact["sha256"],
                "saved calibrator",
                analysis.path,
                root,
            )
            calibrators[saved["path"]] = saved
    return list(thresholds.values()), list(calibrators.values())


def bind_frozen_inputs(
    *,
    root: Path,
    evaluator: EvaluatorState,
    development: LoadedRun,
    analyses: Sequence[LoadedRun],
    manifest_path: Path,
    capacities: Sequence[float],
    settings_sha256: str,
    training_declaration_sha256: str,
    evaluation_declaration_sha256: str,
    dataset_manifest_sha256: str,
) -> FrozenBinding:
    """Verify every frozen input and return the ``frozen_inputs`` block, the training identity,
    the Test partition anchor and the identity of the frozen state. Raises
    ``SpecificationError`` at the first item that differs; nothing reads a Test row."""
    check_evaluator(evaluator)
    check_development_run(development)
    check_declarations(
        development,
        settings_sha256=settings_sha256,
        training_declaration_sha256=training_declaration_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    anchor = partition_anchor(manifest_path, dataset_manifest_sha256)
    selection = check_analysis_runs(analyses, development)
    holder = next(a for a in analyses if binding_selections(a))
    selection_name = next(
        name
        for name, record in holder.records
        if record["record"] == "h3_selection"
        and record["period"] == "primary"
        and record["binding"]
    )
    thresholds, calibrators = threshold_and_calibrator_files(analyses, root, capacities)
    models, model_records, model_manifests = model_files(development, root)
    preprocessing, preprocessing_arrays = preprocessing_files(development, root)
    frozen_inputs: dict[str, Any] = {
        "development_run": {
            "run_id": development.identity["run_id"],
            "run_end": _checked(
                _closing(development), None, "development closing record", development.path, root
            ),
        },
        "analysis_runs": [
            {
                "run_id": a.identity["run_id"],
                "run_end": _checked(_closing(a), None, "analysis closing record", a.path, root),
            }
            for a in analyses
        ],
        "h3_selection": record_ref(holder, selection_name, root),
        "h3_configuration": selection["selected"],
        "preprocessing": preprocessing,
        "preprocessing_arrays": preprocessing_arrays,
        "models": models,
        "model_records": model_records,
        "model_manifests": model_manifests,
        "thresholds": sorted(thresholds, key=lambda entry: entry["path"]),
        "calibrators": sorted(calibrators, key=lambda entry: entry["path"]),
    }
    declarations = {
        "settings": settings_sha256,
        "training": training_declaration_sha256,
        "evaluation": evaluation_declaration_sha256,
        "dataset_manifest": dataset_manifest_sha256,
    }
    identity = development.identity
    logger.info("Frozen inputs verified for %d configurations", len(models))
    return FrozenBinding(
        frozen_inputs=frozen_inputs,
        training={
            "run_id": identity["run_id"],
            "code_commit": identity["code_commit"],
            "source_sha256": identity["source_sha256"],
        },
        test_anchor_sha256=anchor,
        frozen_state_sha256=frozen_state_id(
            commit=evaluator.tag_commit,
            declarations=declarations,
            artifacts=artifact_hashes(frozen_inputs),
        ),
    )
