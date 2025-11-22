"""Utility functions for atomic file I/O.

Provides atomic write operations for both JSON (metadata) and NumPy (embeddings).
Uses temp file + rename pattern which is atomic on POSIX systems.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data atomically using temp file + rename.

    Args:
        path: Target file path
        data: Dictionary to serialize as JSON
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON data from file.

    Returns empty dict if file doesn't exist.
    """
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def atomic_write_numpy(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    """Write embeddings atomically as NumPy compressed archive.

    Deletes the file if ids list is empty.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(ids) == 0:
        if path.exists():
            path.unlink()
        return

    fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)  # np.savez needs to open the file itself
    try:
        np.savez(temp_path, ids=np.array(ids), vectors=vectors)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def load_numpy(path: Path) -> tuple[list[str], np.ndarray]:
    """Load embeddings from NumPy compressed archive.

    Returns ([], empty array) if file doesn't exist.
    """
    if not path.exists():
        return [], np.array([])

    data = np.load(path)
    ids = [str(id_) for id_ in data["ids"]]
    vectors = data["vectors"]
    return ids, vectors
