"""Tests for FlatIndex implementation."""

import pytest
import numpy as np
from app.services.indexes.flat_index import FlatIndex


class TestFlatIndex:
    """Test suite for FlatIndex."""

    @pytest.fixture
    def empty_index(self):
        """Create an empty flat index."""
        return FlatIndex()

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        # Create 5 vectors in 3D space
        vectors = np.array([
            [1.0, 0.0, 0.0],  # id: vec_1
            [0.0, 1.0, 0.0],  # id: vec_2
            [0.0, 0.0, 1.0],  # id: vec_3
            [0.7, 0.7, 0.0],  # id: vec_4 (similar to vec_1 and vec_2)
            [1.0, 0.1, 0.0],  # id: vec_5 (very similar to vec_1)
        ])
        ids = ["vec_1", "vec_2", "vec_3", "vec_4", "vec_5"]
        return vectors, ids

    @pytest.fixture
    def populated_index(self, empty_index, sample_vectors):
        """Create a populated flat index."""
        vectors, ids = sample_vectors
        empty_index.add(vectors, ids)
        return empty_index

    def test_initialization(self, empty_index):
        """Test that index initializes correctly."""
        assert empty_index.num_vectors == 0
        assert empty_index.dimension == 0
        assert len(empty_index.ids) == 0

    def test_add_vectors(self, empty_index, sample_vectors):
        """Test adding vectors to the index."""
        vectors, ids = sample_vectors
        empty_index.add(vectors, ids)

        assert empty_index.num_vectors == 5
        assert empty_index.dimension == 3
        assert len(empty_index.ids) == 5
        assert empty_index.ids == ids

    def test_add_empty_vectors(self, empty_index):
        """Test adding empty vector list."""
        empty_index.add(np.array([]), [])
        assert empty_index.num_vectors == 0

    def test_add_dimension_mismatch(self, populated_index):
        """Test that adding vectors with wrong dimension raises error."""
        wrong_dim_vectors = np.array([[1.0, 2.0]])  # 2D instead of 3D
        with pytest.raises(ValueError, match="dimension.*does not match"):
            populated_index.add(wrong_dim_vectors, ["vec_6"])

    def test_add_mismatched_lengths(self, empty_index):
        """Test that mismatched vectors and ids raises error."""
        vectors = np.array([[1.0, 2.0, 3.0]])
        ids = ["id_1", "id_2"]  # More IDs than vectors
        with pytest.raises(ValueError, match="must match number of ids"):
            empty_index.add(vectors, ids)

    def test_search_exact_match(self, populated_index):
        """Test searching for an exact match."""
        query = np.array([1.0, 0.0, 0.0])  # Same as vec_1
        results = populated_index.search(query, k=1)

        assert len(results) == 1
        assert results[0][0] == "vec_1"
        assert results[0][1] == pytest.approx(1.0, abs=1e-6)  # Perfect similarity

    def test_search_top_k(self, populated_index):
        """Test searching for top-k neighbors."""
        query = np.array([1.0, 0.0, 0.0])  # Same as vec_1
        results = populated_index.search(query, k=3)

        assert len(results) == 3
        # vec_1 should be first (exact match)
        assert results[0][0] == "vec_1"
        # vec_5 should be second (very similar: [1.0, 0.1, 0.0])
        assert results[1][0] == "vec_5"
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

    def test_cosine_similarity_calculation(self, empty_index):
        """Test that cosine similarity is calculated correctly."""
        # Create vectors with known cosine similarities
        vectors = np.array([
            [1.0, 0.0],    # vec_1: 0°
            [0.0, 1.0],    # vec_2: 90°
            [-1.0, 0.0],   # vec_3: 180°
        ])
        ids = ["vec_1", "vec_2", "vec_3"]
        empty_index.add(vectors, ids)

        # Query with [1, 0]
        query = np.array([1.0, 0.0])
        results = empty_index.search(query, k=3)

        # Cosine similarities: vec_1=1.0, vec_2=0.0, vec_3=-1.0
        assert results[0][0] == "vec_1"
        assert results[0][1] == pytest.approx(1.0, abs=1e-6)
        assert results[1][0] == "vec_2"
        assert results[1][1] == pytest.approx(0.0, abs=1e-6)
        assert results[2][0] == "vec_3"
        assert results[2][1] == pytest.approx(-1.0, abs=1e-6)

    def test_delete_vectors(self, populated_index):
        """Test deleting vectors from index."""
        initial_count = populated_index.num_vectors
        populated_index.delete(["vec_1", "vec_3"])

        assert populated_index.num_vectors == initial_count - 2
        assert "vec_1" not in populated_index.ids
        assert "vec_3" not in populated_index.ids
        assert "vec_2" in populated_index.ids

    def test_delete_nonexistent_id(self, populated_index):
        """Test deleting non-existent ID (should not error)."""
        initial_count = populated_index.num_vectors
        populated_index.delete(["nonexistent_id"])
        assert populated_index.num_vectors == initial_count

    def test_delete_empty_list(self, populated_index):
        """Test deleting empty list."""
        initial_count = populated_index.num_vectors
        populated_index.delete([])
        assert populated_index.num_vectors == initial_count

    def test_delete_all_vectors(self, populated_index):
        """Test deleting all vectors resets the index."""
        all_ids = populated_index.ids.copy()
        populated_index.delete(all_ids)

        assert populated_index.num_vectors == 0
        assert populated_index.dimension == 0
        assert len(populated_index.ids) == 0

    def test_get_stats(self, populated_index):
        """Test getting index statistics."""
        stats = populated_index.get_stats()

        assert stats["index_type"] == "flat"
        assert stats["num_vectors"] == 5
        assert stats["dimension"] == 3
        assert stats["memory_bytes"] > 0

    def test_get_stats_empty_index(self, empty_index):
        """Test getting stats from empty index."""
        stats = empty_index.get_stats()

        assert stats["index_type"] == "flat"
        assert stats["num_vectors"] == 0
        assert stats["dimension"] == 0
        assert stats["memory_bytes"] == 0

    def test_normalization(self):
        """Test that vectors are normalized correctly."""
        index = FlatIndex()
        # Add unnormalized vectors
        vectors = np.array([
            [3.0, 4.0],  # Length 5
            [5.0, 12.0],  # Length 13
        ])
        ids = ["vec_1", "vec_2"]
        index.add(vectors, ids)

        # Check that stored vectors are normalized
        for i in range(index.num_vectors):
            norm = np.linalg.norm(index.vectors[i])
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_incremental_adds(self, empty_index):
        """Test adding vectors incrementally."""
        # Add first batch
        batch1 = np.array([[1.0, 0.0]])
        empty_index.add(batch1, ["vec_1"])
        assert empty_index.num_vectors == 1

        # Add second batch
        batch2 = np.array([[0.0, 1.0], [1.0, 1.0]])
        empty_index.add(batch2, ["vec_2", "vec_3"])
        assert empty_index.num_vectors == 3

        # Verify all vectors are searchable
        query = np.array([1.0, 0.0])
        results = empty_index.search(query, k=3)
        assert len(results) == 3
