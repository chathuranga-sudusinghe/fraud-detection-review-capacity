"""The raw-data access boundary, checked on the package source (Issue #70 data-access rule).

After the Parquet build, every step reads the processed files through ``data.processed``. Only
the build step itself may reach the raw files:

* ``data.loading``, the only module that opens the raw files (it reads and hashes them), is
  imported only by ``data.preparation``, by the raw-data contract ``data.validation`` and by the
  Test features reader of the confirmatory evaluation, ``confirmatory.inputs`` (the raw files are
  read again only by the confirmatory Test evaluation); the Test labels are reachable only from
  the evaluation phase: ``load_labels`` is imported by ``data.preparation`` and by the evaluation
  module alone, and ``unseal_test_labels`` is passed only there;
* no other module reads a CSV file.

This is an architectural check, not access control: it keeps the package's own code paths
behind the boundary. It is a structural safeguard on import and call spellings (a module cannot
reach the label reader or the unsealing keyword without the test seeing it, and the confirmatory
modules outside the evaluation phase use no dynamic import or attribute lookup that could hide
one); the behavioural tests of the scoring phase, which make the label reader raise, show the same
from the other side. Behavioural tests show the same from the other side: the profiling and
command-line tests run with the raw files removed.
"""

from __future__ import annotations

import ast
from pathlib import Path

from review_capacity.provenance import package_directory

RAW_MODULES = {"review_capacity.data.loading"}
ALLOWED_IMPORTERS = {
    "review_capacity.data.preparation",
    "review_capacity.data.validation",
    "review_capacity.confirmatory.inputs",
}
LABEL_IMPORTERS = {"review_capacity.data.preparation", "review_capacity.confirmatory.evaluation"}
UNSEALERS = {"review_capacity.confirmatory.evaluation"}
CSV_READERS = {"review_capacity.data.loading"}


def _modules() -> dict[str, ast.Module]:
    root = package_directory()
    modules: dict[str, ast.Module] = {}
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root.parent).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        modules[name] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return modules


def _imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_modules_are_found() -> None:
    modules = _modules()
    assert {"review_capacity.cli", "review_capacity.data.processed", *RAW_MODULES} <= set(modules)
    assert set(modules) >= ALLOWED_IMPORTERS
    assert Path(package_directory()).name == "review_capacity"


def test_only_the_build_step_imports_the_raw_modules() -> None:
    importers = {name for name, tree in _modules().items() if _imports(tree) & RAW_MODULES}
    assert importers <= ALLOWED_IMPORTERS, sorted(importers - ALLOWED_IMPORTERS)


def test_only_the_raw_loader_reads_csv_files() -> None:
    readers = {
        name
        for name, tree in _modules().items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "read_csv"
    }
    assert readers <= CSV_READERS, sorted(readers - CSV_READERS)


def test_the_test_labels_are_reachable_only_from_the_evaluation_phase() -> None:
    readers = {
        name
        for name, tree in _modules().items()
        if name != "review_capacity.data.loading"
        and any(item.endswith("data.loading.load_labels") for item in _imports(tree))
    }
    assert readers <= LABEL_IMPORTERS, sorted(readers - LABEL_IMPORTERS)
    unsealers = {
        name
        for name, tree in _modules().items()
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "unseal_test_labels"
    }
    assert unsealers <= UNSEALERS | {"review_capacity.data.loading"}, sorted(unsealers)


CONFIRMATORY_OUTSIDE_EVALUATION = {
    "review_capacity.confirmatory.command",
    "review_capacity.confirmatory.inputs",
    "review_capacity.confirmatory.scoring",
    "review_capacity.confirmatory.state",
}
DYNAMIC_ACCESS = {"getattr", "__import__", "importlib", "eval", "exec"}


def test_the_scoring_phase_modules_name_no_label_reader_and_no_dynamic_access() -> None:
    modules = _modules()
    for name in CONFIRMATORY_OUTSIDE_EVALUATION:
        tree = modules[name]
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert not names & {"load_labels", "unseal_test_labels"}, name
        assert not names & DYNAMIC_ACCESS, f"{name} uses dynamic access: {names & DYNAMIC_ACCESS}"
        imports = _imports(tree)
        assert not {i for i in imports if i.split(".")[-1] in {"load_labels", "importlib"}}, name


def test_the_test_features_reader_takes_only_the_feature_functions_of_the_raw_loader() -> None:
    tree = _modules()["review_capacity.confirmatory.inputs"]
    taken = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "review_capacity.data.loading"
        for alias in node.names
    }
    assert taken == {"check_file_checksums", "load_raw"}
