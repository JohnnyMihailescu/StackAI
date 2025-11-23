"""Base class for vector indexes."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple
import numpy as np


class BaseIndex(ABC):
    """Abstract base class for all vector indexes."""

    def __init__(self):
        """Initialize the index."""
        self.dimension: int = 0
        self.num_vectors: int = 0

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the index to disk.

        Args:
            path: File path to save the index to
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseIndex":
        """Load an index from disk.

        Args:
            path: File path to load the index from

        Returns:
            A new index instance with loaded data
        """
        pass

    @abstractmethod
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the index.

        Args:
            vectors: Array of shape (n, d) where n is number of vectors and d is dimension
            ids: List of string IDs corresponding to each vector
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors.

        Args:
            query_vector: Query vector of shape (d,)
            k: Number of nearest neighbors to return

        Returns:
            List of (id, similarity_score) tuples, sorted by similarity (highest first)
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors from the index by their IDs.

        Args:
            ids: List of vector IDs to delete
        """
        pass

    @abstractmethod
    def get_vector(self, vector_id: str) -> np.ndarray | None:
        """Get a vector by its ID.

        Args:
            vector_id: The ID of the vector to retrieve

        Returns:
            The vector as a numpy array, or None if not found
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get statistics about the index.

        Returns:
            Dictionary containing index statistics
        """
        pass
