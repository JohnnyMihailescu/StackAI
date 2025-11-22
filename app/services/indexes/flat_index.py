"""Flat (brute force) vector index implementation."""

from typing import List, Tuple
import numpy as np
from app.services.indexes.base import BaseIndex


class FlatIndex(BaseIndex):
    """Flat index using brute force search with cosine similarity.

    This index provides 100% recall accuracy but has O(n) search complexity.
    Suitable for small to medium datasets (<100k vectors).
    """

    def __init__(self):
        """Initialize the flat index."""
        super().__init__()
        self.vectors: np.ndarray = np.array([])
        self.ids: List[str] = []
        self._id_to_idx: dict[str, int] = {}

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

        # Normalize vectors for cosine similarity
        normalized_vectors = self._normalize_vectors(vectors)

        # Initialize or validate dimension
        if self.num_vectors == 0:
            self.dimension = normalized_vectors.shape[1]
            self.vectors = normalized_vectors
        else:
            if normalized_vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"Vector dimension ({normalized_vectors.shape[1]}) "
                    f"does not match index dimension ({self.dimension})"
                )
            self.vectors = np.vstack([self.vectors, normalized_vectors])

        # Update ID mappings
        start_idx = len(self.ids)
        for i, vector_id in enumerate(ids):
            self._id_to_idx[vector_id] = start_idx + i
            self.ids.append(vector_id)

        self.num_vectors = len(self.ids)

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors using cosine similarity.

        Args:
            query_vector: Query vector of shape (d,)
            k: Number of nearest neighbors to return

        Returns:
            List of (id, similarity_score) tuples, sorted by similarity (highest first)

        Raises:
            ValueError: If index is empty or dimension mismatch
        """
        if self.num_vectors == 0:
            return []

        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Query vector dimension ({query_vector.shape[0]}) "
                f"does not match index dimension ({self.dimension})"
            )

        # Normalize query vector
        normalized_query = self._normalize_vectors(query_vector.reshape(1, -1))[0]

        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(self.vectors, normalized_query)

        # Get top-k indices
        k = min(k, self.num_vectors)
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]

        # Return results as (id, score) tuples
        results = [
            (self.ids[idx], float(similarities[idx]))
            for idx in top_k_indices
        ]

        return results

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

    def get_stats(self) -> dict:
        """Get statistics about the index.

        Returns:
            Dictionary containing index statistics
        """
        return {
            "index_type": "flat",
            "num_vectors": self.num_vectors,
            "dimension": self.dimension,
            "memory_bytes": self.vectors.nbytes if self.num_vectors > 0 else 0,
        }

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length for cosine similarity.

        Args:
            vectors: Array of shape (n, d)

        Returns:
            Normalized vectors of shape (n, d)
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
