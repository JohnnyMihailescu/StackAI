"""Search service for orchestrating similarity searches."""

import logging

import numpy as np

from app.models.enums import DistanceMetric, IndexType
from app.models.search import SearchResult
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class SearchService:
    """Orchestrates similarity searches across the storage layer.

    This service provides a high-level API for search operations,
    delegating index management to IndexStore (via StorageService)
    and enriching results with Chunk objects from ChunkStore.
    """

    @classmethod
    async def add_vectors(
        cls,
        library_id: str,
        vectors: np.ndarray,
        ids: list[str],
        index_type: IndexType = IndexType.FLAT,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> None:
        """Add vectors to a library's index.

        Args:
            library_id: The library to add vectors to
            vectors: Array of shape (n, dimensions)
            ids: List of chunk IDs corresponding to each vector
            index_type: Type of index to create if it doesn't exist
            metric: Distance metric to use if creating a new index
        """
        await StorageService.indexes().add_vectors(
            library_id, vectors, ids, index_type, metric
        )

    @classmethod
    async def delete_vectors(cls, library_id: str, ids: list[str]) -> None:
        """Delete vectors from a library's index.

        Args:
            library_id: The library to delete vectors from
            ids: List of chunk IDs to delete
        """
        await StorageService.indexes().delete_vectors(library_id, ids)

    @classmethod
    async def search(
        cls, library_id: str, query_vector: np.ndarray, k: int
    ) -> list[SearchResult]:
        """Search for similar chunks in a library's index.

        Args:
            library_id: The library to search in
            query_vector: Query vector of shape (d,)
            k: Number of nearest neighbors to return

        Returns:
            List of SearchResult objects, sorted by similarity (highest first)
        """
        raw_results = await StorageService.indexes().search(library_id, query_vector, k)

        results = []
        for chunk_id, score in raw_results:
            chunk = await StorageService.chunks().get(chunk_id)
            if chunk:
                results.append(SearchResult(chunk=chunk, score=score))

        return results

    @classmethod
    async def get_embedding(cls, library_id: str, chunk_id: str) -> list[float] | None:
        """Get the embedding for a chunk from the index.

        Args:
            library_id: The library containing the chunk
            chunk_id: The chunk ID to get the embedding for

        Returns:
            The embedding as a list of floats, or None if not found
        """
        vector = await StorageService.indexes().get_vector(library_id, chunk_id)
        if vector is None:
            return None
        return vector.tolist()

    @classmethod
    async def delete_index(cls, library_id: str) -> None:
        """Delete the index for a library."""
        await StorageService.indexes().delete_index(library_id)

    @classmethod
    async def get_stats(cls, library_id: str) -> dict:
        """Get stats for a library's index."""
        return await StorageService.indexes().get_stats(library_id)
