"""Writing output files safely.

Every file is first written to a temporary file in the same directory and then moved into place
with ``os.replace``, which replaces the target in one step on the same file system. A reader
therefore sees either the previous complete file or the new complete file, never a partial one.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from review_capacity.errors import FileAccessError

logger = logging.getLogger(__name__)


def write_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Call ``write(temporary_path)``, then move the temporary file to ``path``.

    File-system failures are raised as ``FileAccessError``; any other error from ``write``
    propagates unchanged.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write(temporary)
        temporary.replace(path)
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be written ({error.strerror})") from error
    finally:
        _remove_temporary(temporary)


def _remove_temporary(temporary: Path) -> None:
    """Clean-up only: it never replaces the error being raised. Nothing to remove is the usual
    case (the file was moved or never created); any other failure is logged."""
    try:
        temporary.unlink(missing_ok=True)
    except NotADirectoryError:
        return  # the destination directory is not a directory: nothing was created
    except OSError as error:
        logger.warning("Temporary file %s could not be removed: %s", temporary, error)


def write_json(path: Path, content: dict[str, Any]) -> None:
    """Write ``content`` as indented UTF-8 JSON, atomically."""
    text = json.dumps(content, indent=2, ensure_ascii=False) + "\n"
    write_atomically(path, lambda target: target.write_text(text, encoding="utf-8"))
