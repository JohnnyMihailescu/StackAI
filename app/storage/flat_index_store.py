"""Flat Index store for raw data access (no index class dependencies)."""

import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from app.models.enums import DistanceMetric, IndexType

logger = logging.getLogger(__name__)


class FlatIndexStore:
    """Raw data store for Flat indexes.

    Handles file I/O for Flat Index.

    File structure per library:
        data/indexes/{library_id}/
        ├── index_meta.json  # Human readable config
        └── vectors.npy      # All vectors (n × d)
    """

    def __init__(self, data_dir: Path | None = None):
        """Initialize the Flat index store.

        Args:
            data_dir: Directory for index files
        """
        self._data_dir = data_dir

    def _get_index_dir(self, library_id: int) -> Path:
        """Get the directory path for a library's index."""
        if self._data_dir is None:
            raise RuntimeError("FlatIndexStore not initialized with data_dir")
        return self._data_dir / str(library_id)

    def list_libraries(self) -> List[int]:
        """List library IDs that have Flat indexes on disk."""
        if self._data_dir is None:
            return []

        if not self._data_dir.exists():
            return []

        library_ids = []
        for index_dir in self._data_dir.iterdir():
            if not index_dir.is_dir():
                continue
            meta_file = index_dir / "index_meta.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                if meta.get("index_type") == IndexType.FLAT.value:
                    library_ids.append(int(index_dir.name))

        return library_ids

    def load_metadata(self, library_id: int) -> dict:
        """Load metadata for a Flat index.

        Returns raw data dict - caller is responsible for creating FlatIndex.
        """
        index_dir = self._get_index_dir(library_id)

        # Load JSON config
        with open(index_dir / "index_meta.json") as f:
            meta = json.load(f)

        return {
            "ids": meta["ids"],
            "dimension": meta["dimension"],
            "metric": DistanceMetric(meta["metric"]),
        }

    def load_vectors(self, library_id: int) -> np.ndarray:
        """Load all vectors for a Flat index.

        Args:
            library_id: Library ID

        Returns:
            Numpy array of vectors (n × d)
        """
        index_dir = self._get_index_dir(library_id)
        vectors_file = index_dir / "vectors.npy"

        if not vectors_file.exists():
            return np.array([])

        return np.load(vectors_file)

    def save(
        self,
        library_id: int,
        vectors: np.ndarray,
        ids: List[int],
        dimension: int,
        metric: DistanceMetric,
    ) -> None:
        """Save a Flat index to disk."""
        index_dir = self._get_index_dir(library_id)
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON config (human readable)
        meta = {
            "index_type": IndexType.FLAT.value,
            "dimension": dimension,
            "metric": metric.value,
            "ids": ids,
            "num_vectors": len(ids),
        }
        with open(index_dir / "index_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save vectors as binary
        if len(vectors) > 0:
            np.save(index_dir / "vectors.npy", vectors)

    def delete_index(self, library_id: int) -> None:
        """Delete all files for an index."""
        index_dir = self._get_index_dir(library_id)
        if index_dir.exists():
            for file in index_dir.iterdir():
                file.unlink()
            index_dir.rmdir()

    def exists(self, library_id: int) -> bool:
        """Check if an index exists on disk."""
        if self._data_dir is None:
            return False

        index_dir = self._get_index_dir(library_id)
        return (index_dir / "index_meta.json").exists()
