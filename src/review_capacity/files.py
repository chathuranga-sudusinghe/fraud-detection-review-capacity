"""Writing output files safely.

Every file is first written to a temporary file in the same directory and then moved into place
with ``os.replace``, which replaces the target in one step on the same file system. A reader
therefore sees either the previous complete file or the new complete file, never a partial one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


def write_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Call ``write(temporary_path)``, then move the temporary file to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, content: dict[str, Any]) -> None:
    """Write ``content`` as indented UTF-8 JSON, atomically."""
    text = json.dumps(content, indent=2, ensure_ascii=False) + "\n"
    write_atomically(path, lambda target: target.write_text(text, encoding="utf-8"))
