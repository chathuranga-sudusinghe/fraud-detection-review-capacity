"""The identity of the frozen state of the confirmatory evaluation (Issue #195).

The frozen state is what the single Test evaluation is bound to: the commit of the freeze tag, the
hashes of the frozen artifacts and the hashes of the declarations. Its identity is the SHA-256 of
a canonical description of exactly those, so it changes when any of them changes and does not
depend on the name of the tag: a renamed tag, or a second tag on the same commit, describes the
same frozen state and cannot reset the consumption of the evaluation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

# The declarations the identity covers, by their key in a record identity.
DECLARATION_FIELDS: dict[str, str] = {
    "settings": "settings_sha256",
    "training": "training_declaration_sha256",
    "evaluation": "evaluation_declaration_sha256",
    "dataset_manifest": "dataset_manifest_sha256",
}


def frozen_state_id(
    *,
    commit: str,
    declarations: Mapping[str, str],
    artifacts: Mapping[str, Sequence[str]],
) -> str:
    """The SHA-256 of the canonical description of the frozen state: the tag ``commit``, the
    declaration hashes by name and, for each group of frozen artifacts, the sorted hashes."""
    description = {
        "commit": commit,
        "declarations": dict(sorted(declarations.items())),
        "artifacts": {group: sorted(hashes) for group, hashes in sorted(artifacts.items())},
    }
    text = json.dumps(description, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def declaration_hashes(identity: Mapping[str, Any]) -> dict[str, str]:
    """The declaration hashes of a record identity, by the names of ``DECLARATION_FIELDS``."""
    return {name: identity[field] for name, field in DECLARATION_FIELDS.items()}


def _by_configuration(files: Mapping[str, Mapping[str, str]]) -> list[str]:
    """The file hashes keyed by configuration (exchanging two models changes the identity)."""
    return [f"{name}:{entry['sha256']}" for name, entry in files.items()]


def artifact_hashes(frozen_inputs: Mapping[str, Any]) -> dict[str, list[str]]:
    """The hashes of the frozen artifacts of a ``test_run`` record, by group. A model, its run
    record and its manifest are keyed by their configuration."""
    return {
        "development_run": [frozen_inputs["development_run"]["run_end"]["sha256"]],
        "analysis_runs": [run["run_end"]["sha256"] for run in frozen_inputs["analysis_runs"]],
        "h3_selection": [frozen_inputs["h3_selection"]["sha256"]],
        "preprocessing": [
            frozen_inputs["preprocessing"]["sha256"],
            frozen_inputs["preprocessing_arrays"]["sha256"],
        ],
        "models": _by_configuration(frozen_inputs["models"]),
        "model_records": _by_configuration(frozen_inputs["model_records"]),
        "model_manifests": _by_configuration(frozen_inputs["model_manifests"]),
        "thresholds": [entry["sha256"] for entry in frozen_inputs["thresholds"]],
        "calibrators": [entry["sha256"] for entry in frozen_inputs["calibrators"]],
    }
