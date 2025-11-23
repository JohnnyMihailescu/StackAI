"""Flat (brute force) vector index implementation."""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.models.enums import DistanceMetric, IndexType
from app.services.indexes.base import BaseIndex
from app.services.indexes.utils import normalize_vectors, similarity_search


class FlatIndex(BaseIndex):
    """Flat index using brute force search with cosine similarity.

    This index provides 100% recall accuracy but has O(n) search complexity.
    Suitable for small to medium datasets (<100k vectors).
    """

    def __init__(self, metric: DistanceMetric = DistanceMetric.COSINE):
        """Initialize the flat index.

        Args:
            metric: Distance metric to use for similarity search
        """
        super().__init__()
        self.metric = metric
        self.vectors: np.ndarray = np.array([])
        self.ids: List[str] = []
        self._id_to_idx: dict[str, int] = {}

    def save(self, path: Path) -> None:
        """Save the index to disk.

        Args:
            path: File path to save the index to (will use .npz format)
        """
        np.savez(
            path,
            index_type=np.array([IndexType.FLAT.value]),
            vectors=self.vectors,
            ids=np.array(self.ids, dtype=object),
            dimension=np.array([self.dimension]),
            metric=np.array([self.metric.value]),
        )

    @classmethod
    def load(cls, path: Path) -> "FlatIndex":
        """Load an index from disk.

        Args:
            path: File path to load the index from

        Returns:
            A new FlatIndex instance with loaded data
        """
        data = np.load(path, allow_pickle=True)

        metric = DistanceMetric(data["metric"][0])
        index = cls(metric=metric)
        index.vectors = data["vectors"]
        index.ids = data["ids"].tolist()
        index.dimension = int(data["dimension"][0])
        index.num_vectors = len(index.ids)
        index._id_to_idx = {id_: i for i, id_ in enumerate(index.ids)}

        return index

    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the index.

        Args:
            vectors: Array of shape (n, d) where n is number of vectors and d is dimension
            ids: List of string IDs corresponding to each vector

        Raises:
            ValueError: If vectors and ids have different lengths
            ValueError: If dimension mismatch with existing vectors
        """
        if len(vectors) != len(ids):
            raise ValueError(f"Number of vectors ({len(vectors)}) must match number of ids ({len(ids)})")

        if len(vectors) == 0:
            return

        # Normalize vectors for cosine similarity, keep original for euclidean
        if self.metric == DistanceMetric.COSINE:
            vectors_to_store = normalize_vectors(vectors)
        else:
            vectors_to_store = vectors

        # Initialize or validate dimension
        if self.num_vectors == 0:
            self.dimension = vectors_to_store.shape[1]
            self.vectors = vectors_to_store
        else:
            if vectors_to_store.shape[1] != self.dimension:
                raise ValueError(
                    f"Vector dimension ({vectors_to_store.shape[1]}) "
                    f"does not match index dimension ({self.dimension})"
                )
            self.vectors = np.vstack([self.vectors, vectors_to_store])

        # Update ID mappings
        start_idx = len(self.ids)
        for i, vector_id in enumerate(ids):
            self._id_to_idx[vector_id] = start_idx + i
            self.ids.append(vector_id)

        self.num_vectors = len(self.ids)

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors.

        Args:
            query_vector: Query vector of shape (d,)
            k: Number of nearest neighbors to return

        Returns:
            List of (id, similarity_score) tuples, sorted by similarity (highest first)

        Raises:
            ValueError: If dimension mismatch
        """
        if self.num_vectors == 0:
            return []

        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Query vector dimension ({query_vector.shape[0]}) "
                f"does not match index dimension ({self.dimension})"
            )

        return similarity_search(query_vector, self.vectors, self.ids, k, self.metric)

    def delete(self, ids: List[str]) -> None:
        """Delete vectors from the index by their IDs.

        Args:
            ids: List of vector IDs to delete
        """
        if not ids:
            return

        # Find indices to keep
        indices_to_delete = set()
        for vector_id in ids:
            if vector_id in self._id_to_idx:
                indices_to_delete.add(self._id_to_idx[vector_id])

        if not indices_to_delete:
            return

        # Create mask for vectors to keep
        indices_to_keep = [i for i in range(self.num_vectors) if i not in indices_to_delete]

        # Update vectors and IDs
        self.vectors = self.vectors[indices_to_keep]
        self.ids = [self.ids[i] for i in indices_to_keep]

        # Rebuild ID to index mapping
        self._id_to_idx = {vector_id: i for i, vector_id in enumerate(self.ids)}
        self.num_vectors = len(self.ids)

        # Reset if empty
        if self.num_vectors == 0:
            self.vectors = np.array([])
            self.dimension = 0

    def get_vector(self, vector_id: str) -> np.ndarray | None:
        """Get a vector by its ID.

        Args:
            vector_id: The ID of the vector to retrieve

        Returns:
            The vector as a numpy array, or None if not found
        """
        if vector_id not in self._id_to_idx:
            return None
        idx = self._id_to_idx[vector_id]
        return self.vectors[idx]

    def get_stats(self) -> dict:
        """Get statistics about the index.

        Returns:
            Dictionary containing index statistics
        """
        return {
            "index_type": "flat",
            "num_vectors": self.num_vectors,
            "dimension": self.dimension,
            "metric": self.metric.value,
            "memory_bytes": self.vectors.nbytes if self.num_vectors > 0 else 0,
        }
