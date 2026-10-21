"""Exceptions raised by the package.

Every project error derives from ``ReviewCapacityError``, so that the command line can report it
as one clear line with its own exit code, while any other exception (a bug) keeps its traceback.
"""


class ReviewCapacityError(Exception):
    """Base class of the errors that the package raises deliberately."""


class DataValidationError(ReviewCapacityError, ValueError):
    """The data do not satisfy the data validation contract; data preparation stops."""


class SpecificationError(ReviewCapacityError, ValueError):
    """A research or model specification is malformed or inconsistent, or would override a fixed
    setting (for example, a hyperparameter naming a DR-013 setting)."""


class SealedTestError(ReviewCapacityError, PermissionError):
    """Test-period labels were requested while the Test period is sealed (DR-001)."""


class ProjectRootNotFoundError(ReviewCapacityError, FileNotFoundError):
    """The project root could not be found (see ``review_capacity.paths``)."""


class FileAccessError(ReviewCapacityError, OSError):
    """A required file cannot be read or an output file cannot be written."""


class ModelError(ReviewCapacityError, RuntimeError):
    """A model failed to fit or produced invalid output (for example, non-finite scores)."""


class RecordContractError(ReviewCapacityError, ValueError):
    """A development record or table does not satisfy its versioned contract
    (``review_capacity.development``)."""


class EnvironmentMismatchError(ReviewCapacityError, RuntimeError):
    """The installed environment differs from the recorded one (for example, a LightGBM library
    that is not the recorded CUDA build, DR-013 §2), or a required device is unavailable."""
