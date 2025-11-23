"""Search service for managing vector indexes and performing searches."""

from pathlib import Path
import numpy as np
from app.services.indexes import FlatIndex


class SearchService:
    """Manages vector indexes and performs similarity searches.

    Each library has its own FlatIndex instance for scoped similarity search.
    Indexes are persisted to disk in the configured data directory.
    """

    _indexes: dict[str, FlatIndex] = {}
    _data_dir: Path | None = None
    _persist: bool = True

    @classmethod
    def initialize(cls, data_dir: Path, persist: bool = True) -> None:
        """Initialize the search service and load existing indexes.

        Args:
            data_dir: Directory where index files are stored
            persist: Whether to persist indexes to disk (False for testing)
        """
        cls._data_dir = data_dir
        cls._persist = persist
        cls._indexes = {}

        if persist:
            data_dir.mkdir(parents=True, exist_ok=True)

            # Load all existing index files
            for index_file in data_dir.glob("*.npz"):
                library_id = index_file.stem
                cls._indexes[library_id] = FlatIndex.load(index_file)

    @classmethod
    def _get_index_path(cls, library_id: str) -> Path:
        """Get the file path for a library's index."""
        if cls._data_dir is None:
            raise RuntimeError("SearchService not initialized. Call initialize() first.")
        return cls._data_dir / f"{library_id}.npz"

    @classmethod
    def _get_index(cls, library_id: str) -> FlatIndex:
        """Get or create the index for a library."""
        if library_id not in cls._indexes:
            cls._indexes[library_id] = FlatIndex()
        return cls._indexes[library_id]

    @classmethod
    def add_vectors(
        cls,
        library_id: str,
        vectors: np.ndarray,
        ids: list[str],
    ) -> None:
        """Add vectors to a library's index.

        Args:
            library_id: The library to add vectors to
            vectors: Array of shape (n, dimensions)
            ids: List of chunk IDs corresponding to each vector
        """
        index = cls._get_index(library_id)
        index.add(vectors, ids)

        if cls._persist:
            index.save(cls._get_index_path(library_id))

    @classmethod
    def delete_vectors(cls, library_id: str, ids: list[str]) -> None:
        """Delete vectors from a library's index.

        Args:
            library_id: The library to delete vectors from
            ids: List of chunk IDs to delete
        """
        if library_id not in cls._indexes:
            return

        index = cls._indexes[library_id]
        index.delete(ids)

        if cls._persist:
            if index.num_vectors == 0:
                # Remove empty index file
                index_path = cls._get_index_path(library_id)
                if index_path.exists():
                    index_path.unlink()
                del cls._indexes[library_id]
            else:
                index.save(cls._get_index_path(library_id))

    @classmethod
    async def search(
        cls, library_id: str, query_vector: np.ndarray, k: int
    ) -> list["SearchResult"]:
        """Search for similar chunks in a library's index.

        Args:
            library_id: The library to search in
            query_vector: Query vector of shape (d,)
            k: Number of nearest neighbors to return

        Returns:
            List of SearchResult objects, sorted by similarity (highest first)
        """
        from app.models.search import SearchResult
        from app.services.storage_service import StorageService

        index = cls._get_index(library_id)
        raw_results = index.search(query_vector, k)

        results = []
        for chunk_id, score in raw_results:
            chunk = await StorageService.chunks().get(chunk_id)
            if chunk:
                results.append(SearchResult(chunk=chunk, score=score))

        return results

    @classmethod
    def delete_index(cls, library_id: str) -> None:
        """Delete the index for a library (both in memory and on disk)."""
        if library_id in cls._indexes:
            del cls._indexes[library_id]

        if cls._persist:
            index_path = cls._get_index_path(library_id)
            if index_path.exists():
                index_path.unlink()

    @classmethod
    def get_stats(cls, library_id: str) -> dict:
        """Get stats for a library's index."""
        if library_id not in cls._indexes:
            return {"index_type": "flat", "num_vectors": 0, "dimension": 0}
        return cls._indexes[library_id].get_stats()

    @classmethod
    def clear_all(cls) -> None:
        """Clear all indexes (memory and optionally disk). Useful for testing."""
        cls._indexes.clear()

        if cls._persist and cls._data_dir:
            for index_file in cls._data_dir.glob("*.npz"):
                index_file.unlink()
