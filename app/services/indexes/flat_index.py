"""Flat (brute force) vector index implementation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from app.models.enums import DistanceMetric
from app.services.indexes.base import BaseIndex
from app.services.indexes.utils import normalize_vectors, similarity_search
from app.storage.flat_index_store import FlatIndexStore


class FlatIndex(BaseIndex):
    """Flat index using brute force search with cosine similarity.

    This index provides 100% recall accuracy but has O(n) search complexity.
    Suitable for small to medium datasets (<100k vectors).

    Data is loaded from the store on each operation - the store is the source of truth.
    """

    def __init__(
        self,
        library_id: int,
        store: FlatIndexStore,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        """Initialize the flat index.

        Args:
            library_id: ID of the library this index belongs to
            store: FlatIndexStore for data access
            metric: Distance metric to use for similarity search
        """
        super().__init__()
        self.library_id = library_id
        self.metric = metric
        self._store = store

    @classmethod
    def load_from_store(cls, store: FlatIndexStore, library_id: int) -> FlatIndex:
        """Load index configuration from store.

        Args:
            store: FlatIndexStore instance
            library_id: ID of the library to load

        Returns:
            FlatIndex configured for the library
        """
        metadata = store.load_metadata(library_id)
        index = cls(
            library_id=library_id,
            store=store,
            metric=metadata["metric"],
        )
        index.dimension = metadata["dimension"]
        index.num_vectors = len(metadata["ids"])
        return index

    def add(self, vectors: np.ndarray, ids: List[int]) -> None:
        """Add vectors to the index.

        Args:
            vectors: Array of shape (n, d) where n is number of vectors and d is dimension
            ids: List of integer IDs corresponding to each vector

        Raises:
            ValueError: If vectors and ids have different lengths
            ValueError: If dimension mismatch with existing vectors
        """
        if len(vectors) != len(ids):
            raise ValueError(f"Number of vectors ({len(vectors)}) must match number of ids ({len(ids)})")

        if len(vectors) == 0:
            return

        # Normalize vectors for cosine similarity
        if self.metric == DistanceMetric.COSINE:
            vectors = normalize_vectors(vectors)

        # Load existing data
        existing_ids = []
        existing_vectors = np.array([])
        if self._store.exists(self.library_id):
            metadata = self._store.load_metadata(self.library_id)
            existing_ids = metadata["ids"]
            existing_vectors = self._store.load_vectors(self.library_id)

            if len(existing_vectors) > 0 and vectors.shape[1] != existing_vectors.shape[1]:
                raise ValueError(
                    f"Vector dimension ({vectors.shape[1]}) "
                    f"does not match index dimension ({existing_vectors.shape[1]})"
                )

        # Combine and save
        if len(existing_vectors) > 0:
            combined_vectors = np.vstack([existing_vectors, vectors])
        else:
            combined_vectors = vectors

        combined_ids = existing_ids + list(ids)
        self.dimension = combined_vectors.shape[1]
        self.num_vectors = len(combined_ids)

        self._store.save(
            library_id=self.library_id,
            vectors=combined_vectors,
            ids=combined_ids,
            dimension=self.dimension,
            metric=self.metric,
        )

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors.

        Args:
            query_vector: Query vector of shape (d,)
            k: Number of nearest neighbors to return

        Returns:
            List of (id, similarity_score) tuples, sorted by similarity (highest first)

        Raises:
            ValueError: If dimension mismatch
        """
        if not self._store.exists(self.library_id):
            return []

        metadata = self._store.load_metadata(self.library_id)
        vectors = self._store.load_vectors(self.library_id)
        ids = metadata["ids"]

        if len(vectors) == 0:
            return []

        if query_vector.shape[0] != vectors.shape[1]:
            raise ValueError(
                f"Query vector dimension ({query_vector.shape[0]}) "
                f"does not match index dimension ({vectors.shape[1]})"
            )

        return similarity_search(query_vector, vectors, ids, k, self.metric)

    def delete(self, ids: List[int]) -> None:
        """Delete vectors from the index by their IDs.

        Args:
            ids: List of vector IDs to delete
        """
        if not ids or not self._store.exists(self.library_id):
            return

        metadata = self._store.load_metadata(self.library_id)
        vectors = self._store.load_vectors(self.library_id)
        existing_ids = metadata["ids"]

        if len(vectors) == 0:
            return

        # Build ID to index mapping
        id_to_idx = {id_: i for i, id_ in enumerate(existing_ids)}

        # Find indices to delete
        indices_to_delete = set()
        for vector_id in ids:
            if vector_id in id_to_idx:
                indices_to_delete.add(id_to_idx[vector_id])

        if not indices_to_delete:
            return

        # Keep vectors not in delete set
        indices_to_keep = [i for i in range(len(existing_ids)) if i not in indices_to_delete]
        new_vectors = vectors[indices_to_keep]
        new_ids = [existing_ids[i] for i in indices_to_keep]

        # Save or delete if empty
        if len(new_ids) == 0:
            self._store.delete_index(self.library_id)
            self.num_vectors = 0
            self.dimension = 0
        else:
            self._store.save(
                library_id=self.library_id,
                vectors=new_vectors,
                ids=new_ids,
                dimension=new_vectors.shape[1],
                metric=self.metric,
            )
            self.num_vectors = len(new_ids)

    def get_vector(self, vector_id: int) -> np.ndarray | None:
        """Get a vector by its ID.

        Args:
            vector_id: The ID of the vector to retrieve

        Returns:
            The vector as a numpy array, or None if not found
        """
        if not self._store.exists(self.library_id):
            return None

        metadata = self._store.load_metadata(self.library_id)
        ids = metadata["ids"]

        if vector_id not in ids:
            return None

        vectors = self._store.load_vectors(self.library_id)
        idx = ids.index(vector_id)
        return vectors[idx]

    def get_stats(self) -> dict:
        """Get statistics about the index.

        Returns:
            Dictionary containing index statistics
        """
        if not self._store.exists(self.library_id):
            return {
                "index_type": "flat",
                "num_vectors": 0,
                "dimension": 0,
                "metric": self.metric.value,
            }

        metadata = self._store.load_metadata(self.library_id)
        return {
            "index_type": "flat",
            "num_vectors": len(metadata["ids"]),
            "dimension": metadata["dimension"],
            "metric": self.metric.value,
        }
