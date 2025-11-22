"""Search service for managing vector indexes and performing searches."""

import numpy as np
from app.services.indexes import FlatIndex


class SearchService:
    """Manages vector indexes and performs similarity searches.

    Each library has its own FlatIndex instance for scoped similarity search.
    """

    _indexes: dict[str, FlatIndex] = {}

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
        """Delete the index for a library."""
        if library_id in cls._indexes:
            del cls._indexes[library_id]

    @classmethod
    def get_stats(cls, library_id: str) -> dict:
        """Get stats for a library's index."""
        if library_id not in cls._indexes:
            return {"index_type": "flat", "num_vectors": 0, "dimension": 0}
        return cls._indexes[library_id].get_stats()

    @classmethod
    def clear_all(cls) -> None:
        """Clear all indexes. Useful for testing."""
        cls._indexes.clear()
