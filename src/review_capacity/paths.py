"""Project layout: the one definition of every path the code uses.

The project root is found, in order, from:

1. an explicit ``root`` argument (the ``--root`` command-line option);
2. the environment variable ``REVIEW_CAPACITY_ROOT``;
3. the nearest directory, from the working directory upwards, that contains both
   ``pyproject.toml`` and ``config/features.yaml``.

If none applies, ``ProjectRootNotFoundError`` is raised; the root is never guessed from the
installed package location. Paths inside configuration files are relative to the root.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from review_capacity.errors import ProjectRootNotFoundError, SpecificationError

ROOT_ENV = "REVIEW_CAPACITY_ROOT"
ROOT_MARKERS = (Path("pyproject.toml"), Path("config") / "features.yaml")


def is_project_root(directory: Path) -> bool:
    return all((directory / marker).is_file() for marker in ROOT_MARKERS)


@dataclass(frozen=True)
class ProjectPaths:
    """Locations of configuration, raw data, processed data and the dataset manifest."""

    root: Path

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def features_spec(self) -> Path:
        return self.config_dir / "features.yaml"

    @property
    def dataset_spec(self) -> Path:
        return self.config_dir / "dataset.yaml"

    @property
    def models_spec(self) -> Path:
        """Device and numerical settings of the models (DR-013)."""
        return self.config_dir / "models.yaml"

    @property
    def training_spec(self) -> Path:
        """The training declaration (DR-015)."""
        return self.config_dir / "training.yaml"

    @property
    def feature_catalog(self) -> Path:
        """The human-readable feature catalog, whose statistics ``profile-data`` checks."""
        return self.root / "docs" / "feature-catalog.md"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def manifest(self) -> Path:
        """The Frictionless Data Package descriptor, at the top level of the data directory."""
        return self.data_dir / "datapackage.json"

    def processed_file(self, stem: str) -> Path:
        return self.processed_dir / f"{stem}.parquet"

    def provenance_file(self, stem: str) -> Path:
        return self.processed_dir / f"{stem}.provenance.json"

    def resolve(self, relative: str | Path) -> Path:
        """Absolute path of a root-relative path taken from a configuration file."""
        path = Path(relative)
        if path.is_absolute():
            raise SpecificationError(f"Configuration paths must be relative to the root: {path}")
        return self.root / path

    @classmethod
    def from_root(cls, root: Path) -> ProjectPaths:
        resolved = root.expanduser().resolve()
        if not is_project_root(resolved):
            markers = ", ".join(m.as_posix() for m in ROOT_MARKERS)
            raise ProjectRootNotFoundError(f"{resolved} is not the project root (needs {markers})")
        return cls(resolved)

    @classmethod
    def discover(
        cls,
        root: Path | None = None,
        *,
        start: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> ProjectPaths:
        """Find the project root: explicit argument, then environment variable, then search."""
        if root is not None:
            return cls.from_root(root)
        from_env = (os.environ if environ is None else environ).get(ROOT_ENV)
        if from_env:
            return cls.from_root(Path(from_env))
        here = (Path.cwd() if start is None else start).resolve()
        for candidate in (here, *here.parents):
            if is_project_root(candidate):
                return cls(candidate)
        raise ProjectRootNotFoundError(
            f"No project root found from {here} upwards; pass --root or set {ROOT_ENV}"
        )
