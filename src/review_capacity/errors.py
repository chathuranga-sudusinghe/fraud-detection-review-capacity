"""Exceptions raised by the data-preparation code."""


class DataValidationError(ValueError):
    """The data do not satisfy the data validation contract; data preparation stops."""


class SpecificationError(ValueError):
    """A configuration file (``config/*.yaml``) is internally inconsistent."""


class SealedTestError(PermissionError):
    """Test-period labels were requested while the Test period is sealed (DR-001)."""


class ProjectRootNotFoundError(FileNotFoundError):
    """The project root could not be found (see ``review_capacity.paths``)."""
