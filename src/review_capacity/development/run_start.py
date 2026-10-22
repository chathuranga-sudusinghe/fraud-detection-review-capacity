"""The start of a development run: its identity, its ``run`` record and its candidate files
(DR-015 §1; Issue #135).

* ``run_identity``: the identity every record of the run repeats (``development.records``): run
  identifier, code state, the SHA-256 of the loaded model settings, training declaration and
  dataset manifest, and the seeds of the declaration.
* ``run_record``: the ``run`` record written first, with the full provenance of the run: code,
  environment, settings, declaration, dataset manifest, the processed files read, the
  configurations in run order and the contract version of every record kind.
* ``adopt_candidates``: copies the approved plan's candidate file of each model into the run
  directory. A file whose SHA-256 differs from the one the approved cost record lists is refused,
  so the lists a run records are the lists that were costed.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from review_capacity.data.processed import read_record
from review_capacity.development.cost_gate import ApprovedPlan
from review_capacity.development.records import contract_versions
from review_capacity.development.run_output import RunDirectory
from review_capacity.development.trial import run_file
from review_capacity.errors import FileAccessError, SpecificationError
from review_capacity.files import write_atomically
from review_capacity.hashing import file_sha256
from review_capacity.models.run_record import SettingsSnapshot
from review_capacity.paths import ProjectPaths
from review_capacity.training.declaration import TrainingDeclaration

logger = logging.getLogger(__name__)

RUN_TYPE = "development"
CANDIDATES_DIR = "candidates"


def run_identity(
    started: str,
    code: Mapping[str, Any],
    declaration: TrainingDeclaration,
    declaration_file: SettingsSnapshot,
    settings_file: SettingsSnapshot,
    main_seed: int,
    manifest: Path,
) -> dict[str, Any]:
    """The run identity of a development run's records."""
    return {
        "run_id": started,
        "run_type": RUN_TYPE,
        "code_commit": code["git_commit"],
        "uncommitted_changes": code["git_uncommitted_changes"],
        "source_sha256": code["package"]["source_sha256"],
        "settings_sha256": settings_file.sha256,
        "declaration_sha256": declaration_file.sha256,
        "dataset_manifest_sha256": file_sha256(manifest),
        "seeds": {
            "main": main_seed,
            "additional": list(declaration.additional_seeds),
            "sampler": declaration.sampler_seed,
            "preprocessing": declaration.preprocessing_seed,
        },
    }


def project_file(paths: ProjectPaths, path: Path, sha256: str | None = None) -> dict[str, str]:
    """A ``{path, sha256}`` reference to a file under the project root."""
    relative = path.resolve().relative_to(paths.root.resolve()).as_posix()
    return {"path": relative, "sha256": sha256 or file_sha256(path)}


def processed_files(paths: ProjectPaths, stems: Iterable[str]) -> dict[str, dict[str, str]]:
    """The processed files a run read, by stem, with the SHA-256 their verified provenance
    records state (the readers have checked each file against it)."""
    return {
        stem: project_file(
            paths, paths.processed_file(stem), read_record(paths.provenance_file(stem)).sha256
        )
        for stem in stems
    }


def run_record(
    paths: ProjectPaths,
    identity: Mapping[str, Any],
    started: str,
    code: Mapping[str, Any],
    environment: Mapping[str, Any],
    settings_file: SettingsSnapshot,
    declaration_file: SettingsSnapshot,
    stems: Iterable[str],
    configurations: Iterable[str],
) -> dict[str, Any]:
    """The ``run`` record: the full provenance of the run, written first."""
    return {
        "record": "run",
        "schema_version": 1,
        "identity": dict(identity),
        "started": started,
        "code": dict(code),
        "environment": dict(environment),
        "settings": {"path": settings_file.path, "sha256": settings_file.sha256},
        "declaration": {"path": declaration_file.path, "sha256": declaration_file.sha256},
        "dataset": {
            "manifest": project_file(paths, paths.manifest),
            "processed": processed_files(paths, stems),
        },
        "configurations": list(configurations),
        "contracts": contract_versions(),
    }


def _write_bytes(target: Path, data: bytes) -> None:
    write_atomically(target, lambda temporary: temporary.write_bytes(data))


def adopt_candidates(
    run: RunDirectory, approved: ApprovedPlan, models: Iterable[str]
) -> dict[str, dict[str, str]]:
    """Copy the approved candidate file of each of ``models`` into ``candidates/`` of the run and
    return their run-directory references."""
    adopted: dict[str, dict[str, str]] = {}
    for model in models:
        entry = approved.candidates.get(model)
        if entry is None:
            raise SpecificationError(f"The approved plan lists no candidates for {model}")
        source = approved.directory / entry["path"]
        try:
            content = source.read_bytes()
        except OSError as error:
            raise FileAccessError(f"{source}: cannot be read ({error.strerror})") from error
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise SpecificationError(f"{source}: differs from the file the approved cost lists")
        target = run.path / CANDIDATES_DIR / f"{model}.json"
        _write_bytes(target, content)
        adopted[model] = run_file(run, target)
        logger.info("Candidate list of %s adopted from %s", model, source)
    return adopted
