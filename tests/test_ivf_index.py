"""Tests for IVFIndex implementation."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from app.models.enums import DistanceMetric
from app.services.indexes.ivf_index import IVFIndex


class TestIVFIndexInitialization:
    """Test IVFIndex initialization."""

    def test_empty_index_state(self):
        """Empty index should have zero vectors and dimension."""
        index = IVFIndex()
        assert index.num_vectors == 0
        assert index.dimension == 0
        assert index.n_clusters == 100  # default
        assert index.n_probe == 10  # default

    def test_custom_parameters(self):
        """Index should respect custom parameters."""
        index = IVFIndex(n_clusters=50, n_probe=5, metric=DistanceMetric.EUCLIDEAN)
        assert index.n_clusters == 50
        assert index.n_probe == 5
        assert index.metric == DistanceMetric.EUCLIDEAN

    def test_n_probe_capped_to_n_clusters(self):
        """n_probe should not exceed n_clusters."""
        index = IVFIndex(n_clusters=10, n_probe=20)
        assert index.n_probe == 10


class TestIVFIndexAdd:
    """Test adding vectors to IVFIndex."""

    def test_add_single_vector(self):
        """Adding a single vector should work."""
        index = IVFIndex(n_clusters=5)
        vectors = np.random.randn(1, 128).astype(np.float32)
        ids = ["id1"]

        index.add(vectors, ids)

        assert index.num_vectors == 1
        assert index.dimension == 128

    def test_add_batch_vectors(self):
        """Adding multiple vectors should work."""
        index = IVFIndex(n_clusters=5)
        vectors = np.random.randn(10, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(10)]

        index.add(vectors, ids)

        assert index.num_vectors == 10
        assert index.dimension == 64

    def test_bootstrap_phase(self):
        """First n_clusters vectors should become centroids."""
        n_clusters = 5
        index = IVFIndex(n_clusters=n_clusters)

        # Add exactly n_clusters vectors
        vectors = np.random.randn(n_clusters, 32).astype(np.float32)
        ids = [f"id{i}" for i in range(n_clusters)]
        index.add(vectors, ids)

        assert index.kmeans.num_centroids == n_clusters
        assert not index.kmeans.is_bootstrapping

    def test_post_bootstrap_vectors_assigned_to_clusters(self):
        """Vectors after bootstrap should be assigned to existing clusters."""
        n_clusters = 5
        index = IVFIndex(n_clusters=n_clusters)

        # Bootstrap phase
        vectors1 = np.random.randn(n_clusters, 32).astype(np.float32)
        ids1 = [f"id{i}" for i in range(n_clusters)]
        index.add(vectors1, ids1)

        # Post-bootstrap
        vectors2 = np.random.randn(10, 32).astype(np.float32)
        ids2 = [f"id{n_clusters + i}" for i in range(10)]
        index.add(vectors2, ids2)

        assert index.num_vectors == n_clusters + 10
        assert index.kmeans.num_centroids == n_clusters  # Still same number of centroids

    def test_add_empty_batch(self):
        """Adding empty batch should be no-op."""
        index = IVFIndex(n_clusters=5)
        index.add(np.array([]).reshape(0, 64), [])
        assert index.num_vectors == 0

    def test_dimension_mismatch_raises(self):
        """Adding vectors with wrong dimension should raise."""
        index = IVFIndex(n_clusters=5)
        vectors1 = np.random.randn(3, 64).astype(np.float32)
        index.add(vectors1, ["a", "b", "c"])

        vectors2 = np.random.randn(2, 128).astype(np.float32)
        with pytest.raises(ValueError, match="dimension"):
            index.add(vectors2, ["d", "e"])

    def test_mismatched_vectors_ids_raises(self):
        """Mismatched vectors and IDs should raise."""
        index = IVFIndex(n_clusters=5)
        vectors = np.random.randn(5, 64).astype(np.float32)
        ids = ["a", "b", "c"]  # Only 3 IDs for 5 vectors

        with pytest.raises(ValueError, match="must match"):
            index.add(vectors, ids)


class TestIVFIndexSearch:
    """Test IVFIndex search functionality."""

    def test_search_empty_index(self):
        """Searching empty index should return empty list."""
        index = IVFIndex(n_clusters=5)
        query = np.random.randn(64).astype(np.float32)
        results = index.search(query, k=5)
        assert results == []

    def test_search_returns_correct_format(self):
        """Search should return list of (id, score) tuples."""
        index = IVFIndex(n_clusters=5, n_probe=5)
        vectors = np.random.randn(10, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(10)]
        index.add(vectors, ids)

        query = np.random.randn(64).astype(np.float32)
        results = index.search(query, k=3)

        assert len(results) <= 3
        for id_, score in results:
            assert isinstance(id_, str)
            assert isinstance(score, float)

    def test_search_finds_exact_match(self):
        """Search should find exact match with highest score."""
        index = IVFIndex(n_clusters=5, n_probe=5, metric=DistanceMetric.COSINE)

        # Create distinct vectors
        vectors = np.eye(10, 64).astype(np.float32)  # Orthogonal vectors
        ids = [f"id{i}" for i in range(10)]
        index.add(vectors, ids)

        # Query with first vector
        query = vectors[0]
        results = index.search(query, k=1)

        assert len(results) == 1
        assert results[0][0] == "id0"
        assert results[0][1] > 0.99  # Should be very close to 1.0

    def test_search_respects_k(self):
        """Search should return at most k results."""
        index = IVFIndex(n_clusters=5, n_probe=5)
        vectors = np.random.randn(20, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(20)]
        index.add(vectors, ids)

        query = np.random.randn(64).astype(np.float32)

        results = index.search(query, k=5)
        assert len(results) <= 5

        results = index.search(query, k=10)
        assert len(results) <= 10

    def test_search_k_larger_than_index(self):
        """Requesting more results than vectors should return all vectors."""
        index = IVFIndex(n_clusters=3, n_probe=3)
        vectors = np.random.randn(5, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(5)]
        index.add(vectors, ids)

        query = np.random.randn(64).astype(np.float32)
        results = index.search(query, k=100)

        assert len(results) == 5

    def test_search_results_sorted_by_score(self):
        """Results should be sorted by score descending."""
        index = IVFIndex(n_clusters=5, n_probe=5)
        vectors = np.random.randn(20, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(20)]
        index.add(vectors, ids)

        query = np.random.randn(64).astype(np.float32)
        results = index.search(query, k=10)

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_dimension_mismatch(self):
        """Query with wrong dimension should raise."""
        index = IVFIndex(n_clusters=5)
        vectors = np.random.randn(10, 64).astype(np.float32)
        index.add(vectors, [f"id{i}" for i in range(10)])

        query = np.random.randn(128).astype(np.float32)  # Wrong dimension
        with pytest.raises(ValueError, match="dimension"):
            index.search(query, k=5)


class TestIVFIndexDelete:
    """Test IVFIndex delete functionality."""

    def test_delete_single_vector(self):
        """Deleting a vector should remove it from the index."""
        index = IVFIndex(n_clusters=5)
        vectors = np.random.randn(10, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(10)]
        index.add(vectors, ids)

        index.delete(["id5"])

        assert index.num_vectors == 9
        assert index.get_vector("id5") is None

    def test_delete_multiple_vectors(self):
        """Deleting multiple vectors should work."""
        index = IVFIndex(n_clusters=5)
        vectors = np.random.randn(10, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(10)]
        index.add(vectors, ids)

        index.delete(["id2", "id5", "id8"])

        assert index.num_vectors == 7

    def test_delete_nonexistent_vector(self):
        """Deleting non-existent vector should be no-op."""
        index = IVFIndex(n_clusters=5)
        vectors = np.random.randn(5, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(5)]
        index.add(vectors, ids)

        index.delete(["nonexistent"])
        assert index.num_vectors == 5

    def test_delete_all_vectors(self):
        """Deleting all vectors should reset index."""
        index = IVFIndex(n_clusters=5)
        vectors = np.random.randn(5, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(5)]
        index.add(vectors, ids)

        index.delete(ids)

        assert index.num_vectors == 0
        assert index.dimension == 0

    def test_delete_empty_list(self):
        """Deleting empty list should be no-op."""
        index = IVFIndex(n_clusters=5)
        vectors = np.random.randn(5, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(5)]
        index.add(vectors, ids)

        index.delete([])
        assert index.num_vectors == 5


class TestIVFIndexGetVector:
    """Test getting vectors by ID."""

    def test_get_existing_vector(self):
        """Getting existing vector should return it."""
        index = IVFIndex(n_clusters=5)
        vectors = np.random.randn(10, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(10)]
        index.add(vectors, ids)

        vector = index.get_vector("id3")
        assert vector is not None
        assert vector.shape == (64,)

    def test_get_nonexistent_vector(self):
        """Getting non-existent vector should return None."""
        index = IVFIndex(n_clusters=5)
        vectors = np.random.randn(5, 64).astype(np.float32)
        index.add(vectors, [f"id{i}" for i in range(5)])

        assert index.get_vector("nonexistent") is None


class TestIVFIndexPersistence:
    """Test saving and loading IVFIndex."""

    def test_save_load_roundtrip(self):
        """Saving and loading should preserve index state."""
        index = IVFIndex(n_clusters=5, n_probe=3, metric=DistanceMetric.COSINE)
        vectors = np.random.randn(20, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(20)]
        index.add(vectors, ids)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = Path(f.name)

        try:
            index.save(path)
            loaded = IVFIndex.load(path)

            assert loaded.num_vectors == index.num_vectors
            assert loaded.dimension == index.dimension
            assert loaded.n_clusters == index.n_clusters
            assert loaded.n_probe == index.n_probe
            assert loaded.metric == index.metric
            assert loaded.kmeans.num_centroids == index.kmeans.num_centroids
        finally:
            path.unlink()

    def test_save_load_empty_index(self):
        """Saving and loading empty index should work."""
        index = IVFIndex(n_clusters=10)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = Path(f.name)

        try:
            index.save(path)
            loaded = IVFIndex.load(path)

            assert loaded.num_vectors == 0
        finally:
            path.unlink()

    def test_search_after_load(self):
        """Search should work correctly after loading."""
        index = IVFIndex(n_clusters=5, n_probe=5)
        vectors = np.random.randn(20, 64).astype(np.float32)
        ids = [f"id{i}" for i in range(20)]
        index.add(vectors, ids)

        # Search before save
        query = vectors[0]
        results_before = index.search(query, k=5)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = Path(f.name)

        try:
            index.save(path)
            loaded = IVFIndex.load(path)

            # Search after load should give same results
            results_after = loaded.search(query, k=5)

            assert len(results_before) == len(results_after)
            for (id1, score1), (id2, score2) in zip(results_before, results_after):
                assert id1 == id2
                assert abs(score1 - score2) < 1e-6
        finally:
            path.unlink()


class TestIVFIndexStats:
    """Test index statistics."""

    def test_stats_empty_index(self):
        """Stats for empty index should be valid."""
        index = IVFIndex(n_clusters=10, n_probe=5)
        stats = index.get_stats()

        assert stats["index_type"] == "ivf"
        assert stats["num_vectors"] == 0
        assert stats["n_clusters"] == 10
        assert stats["n_probe"] == 5
        assert stats["active_clusters"] == 0
        assert stats["bootstrapping"] is True

    def test_stats_with_vectors(self):
        """Stats should reflect index state."""
        index = IVFIndex(n_clusters=5, n_probe=3)
        vectors = np.random.randn(20, 64).astype(np.float32)
        index.add(vectors, [f"id{i}" for i in range(20)])

        stats = index.get_stats()

        assert stats["num_vectors"] == 20
        assert stats["dimension"] == 64
        assert stats["active_clusters"] == 5
        assert stats["bootstrapping"] is False
        assert stats["memory_bytes"] > 0


class TestIVFIndexEuclidean:
    """Test IVFIndex with Euclidean distance metric."""

    def test_euclidean_search(self):
        """Euclidean search should work correctly."""
        index = IVFIndex(n_clusters=5, n_probe=5, metric=DistanceMetric.EUCLIDEAN)

        # Create vectors where id0 is closest to query
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float32)
        ids = [f"id{i}" for i in range(5)]
        index.add(vectors, ids)

        # Query close to first vector
        query = np.array([1.1, 0.1, 0.0], dtype=np.float32)
        results = index.search(query, k=1)

        assert results[0][0] == "id0"
        # Euclidean returns negative distance (higher = closer)
        assert results[0][1] < 0  # Negative distance

    def test_euclidean_exact_match(self):
        """Exact match should have score close to 0 (distance 0)."""
        index = IVFIndex(n_clusters=3, n_probe=3, metric=DistanceMetric.EUCLIDEAN)

        vectors = np.random.randn(5, 32).astype(np.float32)
        index.add(vectors, [f"id{i}" for i in range(5)])

        # Query with exact vector
        query = vectors[2]
        results = index.search(query, k=1)

        assert results[0][0] == "id2"
        assert abs(results[0][1]) < 1e-5  # Distance should be ~0


class TestSequentialKMeans:
    """Test sequential k-means behavior within IVFIndex."""

    def test_centroid_count_during_bootstrap(self):
        """Centroid count should increase during bootstrap."""
        index = IVFIndex(n_clusters=10)

        for i in range(5):
            vectors = np.random.randn(1, 32).astype(np.float32)
            index.add(vectors, [f"id{i}"])
            assert index.kmeans.num_centroids == i + 1

    def test_centroid_count_after_bootstrap(self):
        """Centroid count should stay constant after bootstrap."""
        n_clusters = 5
        index = IVFIndex(n_clusters=n_clusters)

        # Fill bootstrap
        vectors = np.random.randn(n_clusters, 32).astype(np.float32)
        index.add(vectors, [f"id{i}" for i in range(n_clusters)])

        # Add more vectors
        for i in range(10):
            vectors = np.random.randn(1, 32).astype(np.float32)
            index.add(vectors, [f"extra{i}"])
            assert index.kmeans.num_centroids == n_clusters

    def test_cluster_counts_updated(self):
        """Cluster counts should be updated as vectors are added."""
        n_clusters = 3
        index = IVFIndex(n_clusters=n_clusters)

        # Bootstrap
        vectors = np.random.randn(n_clusters, 32).astype(np.float32)
        index.add(vectors, [f"id{i}" for i in range(n_clusters)])

        # Each centroid should have count 1
        assert all(c == 1 for c in index.kmeans.cluster_counts)

        # Add more vectors
        vectors = np.random.randn(10, 32).astype(np.float32)
        index.add(vectors, [f"extra{i}" for i in range(10)])

        # Total count should equal total vectors
        assert sum(index.kmeans.cluster_counts) == n_clusters + 10
