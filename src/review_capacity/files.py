"""Writing output files safely.

Every file is first written to a temporary file in the same directory and then moved into place
with ``os.replace``, which replaces the target in one step on the same file system. A reader
therefore sees either the previous complete file or the new complete file, never a partial one.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from review_capacity.errors import FileAccessError, FileExistsAccessError

logger = logging.getLogger(__name__)


def write_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Call ``write(temporary_path)``, then move the temporary file to ``path``.

    File-system failures are raised as ``FileAccessError``; any other error from ``write``
    propagates unchanged. The temporary file is synchronised to storage before the move and the
    directory after it (Issue #212), so a file that was published survives a host crash with its
    content.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        make_directories_durable(path.parent)
        write(temporary)
        _sync_file(temporary)
        temporary.replace(path)
        _sync_directory(path.parent)
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be written ({error.strerror})") from error
    finally:
        _remove_temporary(temporary)


def write_new_atomically(path: Path, text: str) -> None:
    """Write ``text`` as UTF-8 to the new file ``path``: complete or absent, never replacing.

    The text is written to a temporary file in the same directory and hard-linked to ``path``,
    which fails if ``path`` exists (even if it was created after the caller last looked), so a
    file written this way is never overwritten. A missing directory is created. The content is
    synchronised to storage before the link and the directory after it (Issue #212): the supported
    failure model is a process crash or a host crash on a local file system, after which the file
    is either complete or absent.
    """
    temporary: Path | None = None
    try:
        make_directories_durable(path.parent)
        # a temporary file of this call alone: created exclusively under a unique name, so that
        # no other call (or leftover) shares its inode with a published file
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsAccessError(f"{path}: already exists and is never replaced") from error
        _sync_directory(path.parent)
    except FileAccessError:
        raise
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be written ({error.strerror})") from error
    finally:
        if temporary is not None:
            _remove_temporary(temporary)


def make_directories_durable(directory: Path) -> None:
    """Create ``directory`` and its missing parents, and synchronise each new entry in its parent
    (Issue #211), so that a file published below survives a host crash with its directory chain."""
    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        current = current.parent
    directory.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _sync_directory(created)
        _sync_directory(created.parent)


def sync_chain(path: Path, stop: Path) -> None:
    """Synchronise the directory ``path`` and each of its parents up to and including ``stop``
    (Issue #211). A directory that already exists is not known to be durable: an earlier
    invocation, the logging set-up or a plain ``mkdir`` may have created it, and its entry in the
    parent may not have reached storage. ``stop`` must be ``path`` or one of its ancestors."""
    if stop != path and stop not in path.parents:
        raise FileAccessError(f"{stop} is not an ancestor of {path}")
    try:
        current = path
        while True:
            _sync_directory(current)
            if current == stop:
                return
            current = current.parent
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be synchronised ({error.strerror})") from error


def make_durable(path: Path) -> None:
    """Synchronise an existing file and its directory entry: an accepted marker that an earlier
    invocation published (and whose own synchronisation may not have completed) is made durable
    before a run relies on it."""
    try:
        _sync_file(path)
        _sync_directory(path.parent)
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be synchronised ({error.strerror})") from error


def _sync_file(path: Path) -> None:
    """Force the content of ``path`` to storage."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(directory: Path) -> None:
    """Force the directory entries of ``directory`` to storage."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
