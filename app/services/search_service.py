"""Search service for orchestrating similarity searches."""

import logging

import numpy as np

from app.models.enums import DistanceMetric, IndexType
from app.models.search import SearchResult
from app.services.indexes import FlatIndex, IVFIndex
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class SearchService:
    """Orchestrates similarity searches across the storage layer.

    This service provides a high-level API for search operations,
    creating index instances as needed and enriching results with
    Chunk objects from ChunkStore.
    """

    @classmethod
    async def _get_library_index_type(cls, library_id: int) -> IndexType:
        """
        Get the index type for a library.
        """
        library = await StorageService.libraries().get(library_id)
        return library.index_type

    @classmethod
    async def _get_index(
        cls, library_id: int
    ) -> FlatIndex | IVFIndex | None:
        """Get the index for a library, or None if it doesn't exist.

        Args:
            library_id: The library ID

        Returns:
            The loaded index, or None if no index exists
        """
        index_type = await cls._get_library_index_type(library_id)

        if index_type == IndexType.FLAT:
            store = StorageService.flat_index_store()
            if not store.exists(library_id):
                return None
            return FlatIndex(library_id=library_id, store=store)
        else:
            store = StorageService.ivf_index_store()
            if not store.exists(library_id):
                return None
            return IVFIndex.load_from_store(store, library_id)

    @classmethod
    async def _get_or_create_index(
        cls,
        library_id: int,
        index_type: IndexType,
        metric: DistanceMetric,
    ) -> FlatIndex | IVFIndex:
        """Get or create an index for a library.

        Args:
            library_id: The library ID
            index_type: Type of index to create if it doesn't exist
            metric: Distance metric to use if creating a new index

        Returns:
            The loaded or newly created index
        """
        if index_type == IndexType.FLAT:
            store = StorageService.flat_index_store()
            return FlatIndex(library_id=library_id, store=store, metric=metric)
        else:
            store = StorageService.ivf_index_store()
            if store.exists(library_id):
                return IVFIndex.load_from_store(store, library_id)
            else:
                index = IVFIndex(library_id=library_id, metric=metric)
                index._store = store
                return index

    @classmethod
    async def add_vectors(
        cls,
        library_id: int,
        vectors: np.ndarray,
        ids: list[int],
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
        index = await cls._get_or_create_index(library_id, index_type, metric)
        index.add(vectors, ids)

        # IVF requires explicit save
        if isinstance(index, IVFIndex):
            index.save_to_store(StorageService.ivf_index_store())

    @classmethod
    async def delete_vectors(cls, library_id: int, ids: list[int]) -> None:
        """Delete vectors from a library's index.

        Args:
            library_id: The library to delete vectors from
            ids: List of chunk IDs to delete
        """
        index = await cls._get_index(library_id)
        if index is None:
            return

        index.delete(ids)

        # IVF requires explicit save or cleanup if empty
        if isinstance(index, IVFIndex):
            if index.num_vectors > 0:
                index.save_to_store(StorageService.ivf_index_store())
            else:
                StorageService.ivf_index_store().delete_index(library_id)

    @classmethod
    async def search(
        cls, library_id: int, query_vector: np.ndarray, k: int
    ) -> list[SearchResult]:
        """Search for similar chunks in a library's index.

        Args:
            library_id: The library to search in
            query_vector: Query vector of shape (d,)
            k: Number of nearest neighbors to return

        Returns:
            List of SearchResult objects, sorted by similarity (highest first)
        """
        index = await cls._get_index(library_id)
        if index is None:
            return []

        raw_results = index.search(query_vector, k)

        results = []
        for chunk_id, score in raw_results:
            chunk = await StorageService.chunks().get(chunk_id)
            if chunk:
                results.append(SearchResult(chunk=chunk, score=score))

        return results

    @classmethod
    async def get_embedding(cls, library_id: int, chunk_id: int) -> list[float] | None:
        """Get the embedding for a chunk from the index.

        Args:
            library_id: The library containing the chunk
            chunk_id: The chunk ID to get the embedding for

        Returns:
            The embedding as a list of floats, or None if not found
        """
        index = await cls._get_index(library_id)
        if index is None:
            return None

        vector = index.get_vector(chunk_id)
        if vector is None:
            return None
        return vector.tolist()

    @classmethod
    async def delete_index(cls, library_id: int) -> None:
        """Delete the index for a library."""
        # Delete from both stores in case index type changed
        StorageService.flat_index_store().delete_index(library_id)
        StorageService.ivf_index_store().delete_index(library_id)

    @classmethod
    async def get_stats(cls, library_id: int) -> dict:
        """Get stats for a library's index."""
        index = await cls._get_index(library_id)
        if index is None:
            # Return empty stats with correct index type
            index_type = await cls._get_library_index_type(library_id)
            type_name = "flat" if index_type == IndexType.FLAT else "ivf"
            return {"index_type": type_name, "num_vectors": 0, "dimension": 0}

        return index.get_stats()
