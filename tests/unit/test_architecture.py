"""The raw-data access boundary, checked on the package source (Issue #70 data-access rule).

After the Parquet build, every step reads the processed files through ``data.processed``. Only
the build step itself may reach the raw files:

* ``data.loading``, the only module that opens the raw files (it reads and hashes them), is
  imported only by ``data.preparation`` and by the raw-data contract ``data.validation``;
* no other module reads a CSV file.

This is an architectural check, not access control: it keeps the package's own code paths
behind the boundary. Behavioural tests show the same from the other side: the profiling and
command-line tests run with the raw files removed.
"""

from __future__ import annotations

import ast
from pathlib import Path

from review_capacity.provenance import package_directory

RAW_MODULES = {"review_capacity.data.loading"}
ALLOWED_IMPORTERS = {"review_capacity.data.preparation", "review_capacity.data.validation"}
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
