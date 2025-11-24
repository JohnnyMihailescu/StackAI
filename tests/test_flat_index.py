"""Tests for FlatIndex implementation."""

from typing import List

import numpy as np
import pytest

from app.models.enums import DistanceMetric
from app.services.indexes.flat_index import FlatIndex


class MockFlatIndexStore:
    """In-memory mock of FlatIndexStore for testing."""

    def __init__(self):
        self._data: dict[int, dict] = {}

    def exists(self, library_id: int) -> bool:
        return library_id in self._data

    def load_metadata(self, library_id: int) -> dict:
        data = self._data[library_id]
        return {
            "ids": data["ids"],
            "dimension": data["dimension"],
            "metric": data["metric"],
        }

    def load_vectors(self, library_id: int) -> np.ndarray:
        return self._data[library_id]["vectors"]

    def save(
        self,
        library_id: int,
        vectors: np.ndarray,
        ids: List[int],
        dimension: int,
        metric: DistanceMetric,
    ) -> None:
        self._data[library_id] = {
            "vectors": vectors,
            "ids": ids,
            "dimension": dimension,
            "metric": metric,
        }

    def delete_index(self, library_id: int) -> None:
        if library_id in self._data:
            del self._data[library_id]


class TestFlatIndex:
    """Test suite for FlatIndex."""

    LIBRARY_ID = 1

    @pytest.fixture
    def mock_store(self):
        """Create a mock store."""
        return MockFlatIndexStore()

    @pytest.fixture
    def empty_index(self, mock_store):
        """Create an empty flat index."""
        return FlatIndex(
            library_id=self.LIBRARY_ID,
            store=mock_store,
            metric=DistanceMetric.COSINE,
        )

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        # Create 5 vectors in 3D space
        vectors = np.array([
            [1.0, 0.0, 0.0],  # id: 1
            [0.0, 1.0, 0.0],  # id: 2
            [0.0, 0.0, 1.0],  # id: 3
            [0.7, 0.7, 0.0],  # id: 4 (similar to 1 and 2)
            [1.0, 0.1, 0.0],  # id: 5 (very similar to 1)
        ])
        ids = [1, 2, 3, 4, 5]
        return vectors, ids

    @pytest.fixture
    def populated_index(self, mock_store, sample_vectors):
        """Create a populated flat index."""
        index = FlatIndex(
            library_id=self.LIBRARY_ID,
            store=mock_store,
            metric=DistanceMetric.COSINE,
        )
        vectors, ids = sample_vectors
        index.add(vectors, ids)
        return index

    def test_initialization(self, empty_index):
        """Test that index initializes correctly."""
        stats = empty_index.get_stats()
        assert stats["num_vectors"] == 0
        assert stats["dimension"] == 0

    def test_add_vectors(self, empty_index, sample_vectors):
        """Test adding vectors to the index."""
        vectors, ids = sample_vectors
        empty_index.add(vectors, ids)

        stats = empty_index.get_stats()
        assert stats["num_vectors"] == 5
        assert stats["dimension"] == 3

    def test_add_empty_vectors(self, empty_index):
        """Test adding empty vector list."""
        empty_index.add(np.array([]).reshape(0, 3), [])
        stats = empty_index.get_stats()
        assert stats["num_vectors"] == 0

    def test_add_dimension_mismatch(self, populated_index):
        """Test that adding vectors with wrong dimension raises error."""
        wrong_dim_vectors = np.array([[1.0, 2.0]])  # 2D instead of 3D
        with pytest.raises(ValueError, match="dimension.*does not match"):
            populated_index.add(wrong_dim_vectors, [6])

    def test_add_mismatched_lengths(self, empty_index):
        """Test that mismatched vectors and ids raises error."""
        vectors = np.array([[1.0, 2.0, 3.0]])
        ids = [1, 2]  # More IDs than vectors
        with pytest.raises(ValueError, match="must match number of ids"):
            empty_index.add(vectors, ids)

    def test_search_exact_match(self, populated_index):
        """Test searching for an exact match."""
        query = np.array([1.0, 0.0, 0.0])  # Same as id 1
        results = populated_index.search(query, k=1)

        assert len(results) == 1
        assert results[0][0] == 1
        assert results[0][1] == pytest.approx(1.0, abs=1e-6)  # Perfect similarity

    def test_search_top_k(self, populated_index):
        """Test searching for top-k neighbors."""
        query = np.array([1.0, 0.0, 0.0])  # Same as id 1
        results = populated_index.search(query, k=3)

        assert len(results) == 3
        # id 1 should be first (exact match)
        assert results[0][0] == 1
        # id 5 should be second (very similar: [1.0, 0.1, 0.0])
        assert results[1][0] == 5
        # Results should be sorted by similarity (descending)
        assert results[0][1] >= results[1][1] >= results[2][1]

    def test_search_empty_index(self, empty_index):
        """Test searching in empty index."""
        query = np.array([1.0, 0.0, 0.0])
        results = empty_index.search(query, k=5)
        assert results == []

    def test_search_k_larger_than_index(self, populated_index):
        """Test searching with k larger than number of vectors."""
        query = np.array([1.0, 0.0, 0.0])
        results = populated_index.search(query, k=100)

        # Should return all 5 vectors
        assert len(results) == 5

    def test_search_dimension_mismatch(self, populated_index):
        """Test that searching with wrong dimension raises error."""
        query = np.array([1.0, 2.0])  # 2D instead of 3D
        with pytest.raises(ValueError, match="dimension.*does not match"):
            populated_index.search(query, k=1)

    def test_cosine_similarity_calculation(self, mock_store):
        """Test that cosine similarity is calculated correctly."""
        index = FlatIndex(
            library_id=self.LIBRARY_ID,
            store=mock_store,
            metric=DistanceMetric.COSINE,
        )
        # Create vectors with known cosine similarities
        vectors = np.array([
            [1.0, 0.0],    # id 1: 0°
            [0.0, 1.0],    # id 2: 90°
            [-1.0, 0.0],   # id 3: 180°
        ])
        ids = [1, 2, 3]
        index.add(vectors, ids)

        # Query with [1, 0]
        query = np.array([1.0, 0.0])
        results = index.search(query, k=3)

        # Cosine similarities: id 1=1.0, id 2=0.0, id 3=-1.0
        assert results[0][0] == 1
        assert results[0][1] == pytest.approx(1.0, abs=1e-6)
        assert results[1][0] == 2
        assert results[1][1] == pytest.approx(0.0, abs=1e-6)
        assert results[2][0] == 3
        assert results[2][1] == pytest.approx(-1.0, abs=1e-6)

    def test_delete_vectors(self, populated_index):
        """Test deleting vectors from index."""
        initial_stats = populated_index.get_stats()
        initial_count = initial_stats["num_vectors"]
        populated_index.delete([1, 3])

        stats = populated_index.get_stats()
        assert stats["num_vectors"] == initial_count - 2

        # Verify deleted vectors are not searchable
        query = np.array([1.0, 0.0, 0.0])
        results = populated_index.search(query, k=10)
        result_ids = [r[0] for r in results]
        assert 1 not in result_ids
        assert 3 not in result_ids
        assert 2 in result_ids

    def test_delete_nonexistent_id(self, populated_index):
        """Test deleting non-existent ID (should not error)."""
        initial_stats = populated_index.get_stats()
        initial_count = initial_stats["num_vectors"]
        populated_index.delete([999])
        stats = populated_index.get_stats()
        assert stats["num_vectors"] == initial_count

    def test_delete_empty_list(self, populated_index):
        """Test deleting empty list."""
        initial_stats = populated_index.get_stats()
        initial_count = initial_stats["num_vectors"]
        populated_index.delete([])
        stats = populated_index.get_stats()
        assert stats["num_vectors"] == initial_count

    def test_delete_all_vectors(self, populated_index, sample_vectors):
        """Test deleting all vectors resets the index."""
        _, all_ids = sample_vectors
        populated_index.delete(all_ids)

        stats = populated_index.get_stats()
        assert stats["num_vectors"] == 0
        assert stats["dimension"] == 0

    def test_get_stats(self, populated_index):
        """Test getting index statistics."""
        stats = populated_index.get_stats()

        assert stats["index_type"] == "flat"
        assert stats["num_vectors"] == 5
        assert stats["dimension"] == 3
        assert stats["metric"] == "cosine"

    def test_get_stats_empty_index(self, empty_index):
        """Test getting stats from empty index."""
        stats = empty_index.get_stats()

        assert stats["index_type"] == "flat"
        assert stats["num_vectors"] == 0
        assert stats["dimension"] == 0

    def test_normalization(self, mock_store):
        """Test that vectors are normalized correctly for cosine similarity."""
        index = FlatIndex(
            library_id=self.LIBRARY_ID,
            store=mock_store,
            metric=DistanceMetric.COSINE,
        )
        # Add unnormalized vectors
        vectors = np.array([
            [3.0, 4.0],  # Length 5
            [5.0, 12.0],  # Length 13
        ])
        ids = [1, 2]
        index.add(vectors, ids)

        # Verify vectors are normalized by checking search behavior
        # A normalized [3,4] is [0.6, 0.8], query with same direction should give similarity 1.0
        query = np.array([0.6, 0.8])
        results = index.search(query, k=1)
        assert results[0][0] == 1
        assert results[0][1] == pytest.approx(1.0, abs=1e-6)

    def test_incremental_adds(self, mock_store):
        """Test adding vectors incrementally."""
        index = FlatIndex(
            library_id=self.LIBRARY_ID,
            store=mock_store,
            metric=DistanceMetric.COSINE,
        )
        # Add first batch
        batch1 = np.array([[1.0, 0.0]])
        index.add(batch1, [1])
        assert index.get_stats()["num_vectors"] == 1

        # Add second batch
        batch2 = np.array([[0.0, 1.0], [1.0, 1.0]])
        index.add(batch2, [2, 3])
        assert index.get_stats()["num_vectors"] == 3

        # Verify all vectors are searchable
        query = np.array([1.0, 0.0])
        results = index.search(query, k=3)
        assert len(results) == 3

    def test_get_vector(self, populated_index):
        """Test retrieving a vector by ID."""
        vector = populated_index.get_vector(1)
        assert vector is not None
        # id 1 was [1.0, 0.0, 0.0], normalized stays the same
        assert vector.shape == (3,)

    def test_get_vector_nonexistent(self, populated_index):
        """Test retrieving a non-existent vector returns None."""
        vector = populated_index.get_vector(999)
        assert vector is None

    def test_get_vector_empty_index(self, empty_index):
        """Test retrieving from empty index returns None."""
        vector = empty_index.get_vector(1)
        assert vector is None
