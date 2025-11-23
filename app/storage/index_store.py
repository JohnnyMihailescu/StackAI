"""Index store for managing vector indexes with thread-safe access."""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.models.enums import DistanceMetric, IndexType
from app.services.indexes import BaseIndex, FlatIndex, IVFIndex
from app.storage.rwlock import AsyncRWLock

logger = logging.getLogger(__name__)


class IndexStore:
    """Thread-safe store for vector indexes.

    Provides RWLock protection for all index operations, following the same
    pattern as other entity stores. Each library has its own index.

    Indexes are lazily created on first vector addition and automatically
    removed when all vectors are deleted.
    """

    def __init__(self, data_dir: Path | None = None, persist: bool = True):
        """Initialize the index store.

        Args:
            data_dir: Directory for .npz index files
            persist: Whether to persist indexes to disk
        """
        self._indexes: dict[str, BaseIndex] = {}
        self._lock = AsyncRWLock()
        self._data_dir = data_dir
        self._persist = persist

    async def load(self) -> None:
        """Load all existing indexes from disk."""
        if not self._persist or self._data_dir is None:
            return

        self._data_dir.mkdir(parents=True, exist_ok=True)

        async with self._lock.write():
            for index_file in self._data_dir.glob("*.npz"):
                library_id = index_file.stem
                self._indexes[library_id] = self._load_index_file(index_file)

            logger.info(f"Loaded {len(self._indexes)} search index(es) from disk")

    def _load_index_file(self, path: Path) -> BaseIndex:
        """Load an index from disk, detecting the type automatically."""
        data = np.load(path, allow_pickle=True)
        index_type_str = data["index_type"][0]
        index_type = IndexType(index_type_str)

        if index_type == IndexType.FLAT:
            return FlatIndex.load(path)
        elif index_type == IndexType.IVF:
            return IVFIndex.load(path)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def _get_index_path(self, library_id: str) -> Path:
        """Get the file path for a library's index."""
        if self._data_dir is None:
            raise RuntimeError("IndexStore not initialized with data_dir")
        return self._data_dir / f"{library_id}.npz"

    def _create_index(self, index_type: IndexType, metric: DistanceMetric) -> BaseIndex:
        """Create a new index of the specified type."""
        if index_type == IndexType.FLAT:
            return FlatIndex(metric=metric)
        elif index_type == IndexType.IVF:
            return IVFIndex(metric=metric)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    async def add_vectors(
        self,
        library_id: str,
        vectors: np.ndarray,
        ids: List[str],
        index_type: IndexType = IndexType.FLAT,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> None:
        """Add vectors to a library's index.

        Creates the index lazily if it doesn't exist.

        Args:
            library_id: The library to add vectors to
            vectors: Array of shape (n, dimensions)
            ids: List of chunk IDs corresponding to each vector
            index_type: Type of index to create if it doesn't exist
            metric: Distance metric to use if creating a new index
        """
        async with self._lock.write():
            if library_id not in self._indexes:
                self._indexes[library_id] = self._create_index(index_type, metric)

            index = self._indexes[library_id]
            index.add(vectors, ids)
            logger.debug(f"Added {len(ids)} vectors to index (total: {index.num_vectors})")

            if self._persist:
                index.save(self._get_index_path(library_id))

    async def delete_vectors(self, library_id: str, ids: List[str]) -> None:
        """Delete vectors from a library's index.

        Args:
            library_id: The library to delete vectors from
            ids: List of chunk IDs to delete
        """
        async with self._lock.write():
            if library_id not in self._indexes:
                return

            index = self._indexes[library_id]
            index.delete(ids)

            if self._persist:
                if index.num_vectors == 0:
                    # Remove empty index file
                    index_path = self._get_index_path(library_id)
                    if index_path.exists():
                        index_path.unlink()
                    del self._indexes[library_id]
                else:
                    index.save(self._get_index_path(library_id))

    async def search(
        self, library_id: str, query_vector: np.ndarray, k: int
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors in a library's index.

        Args:
            library_id: The library to search in
            query_vector: Query vector of shape (d,)
            k: Number of nearest neighbors to return

        Returns:
            List of (chunk_id, score) tuples, sorted by similarity (highest first)
        """
        async with self._lock.read():
            if library_id not in self._indexes:
                return []

            index = self._indexes[library_id]
            return index.search(query_vector, k)

    async def get_vector(self, library_id: str, vector_id: str) -> np.ndarray | None:
        """Get a vector by its ID.

        Args:
            library_id: The library containing the vector
            vector_id: The chunk ID to get the vector for

        Returns:
            The vector as a numpy array, or None if not found
        """
        async with self._lock.read():
            if library_id not in self._indexes:
                return None

            return self._indexes[library_id].get_vector(vector_id)

    async def delete_index(self, library_id: str) -> None:
        """Delete the entire index for a library.

        Args:
            library_id: The library whose index to delete
        """
        async with self._lock.write():
            if library_id in self._indexes:
                del self._indexes[library_id]

            if self._persist:
                index_path = self._get_index_path(library_id)
                if index_path.exists():
                    index_path.unlink()

    async def get_stats(self, library_id: str) -> dict:
        """Get statistics for a library's index.

        Args:
            library_id: The library to get stats for

        Returns:
            Dictionary containing index statistics
        """
        async with self._lock.read():
            if library_id not in self._indexes:
                return {"index_type": "none", "num_vectors": 0, "dimension": 0}
            return self._indexes[library_id].get_stats()

    async def clear(self) -> None:
        """Clear all indexes from memory and optionally disk."""
        async with self._lock.write():
            self._indexes.clear()

            if self._persist and self._data_dir:
                for index_file in self._data_dir.glob("*.npz"):
                    index_file.unlink()
