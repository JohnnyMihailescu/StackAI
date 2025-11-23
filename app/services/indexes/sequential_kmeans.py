"""Sequential k-means clustering algorithm."""

import numpy as np

from app.models.enums import DistanceMetric
from app.services.indexes.utils import (
    normalize_vectors,
    compute_similarities,
    top_k_indices,
)


class SequentialKMeans:
    """Sequential k-means clustering that updates centroids incrementally.

    Unlike batch k-means, this algorithm updates centroids as vectors arrive,
    eliminating the need for a separate training step. The first n_clusters
    vectors become initial centroids (bootstrap phase), then subsequent
    vectors are assigned to the nearest centroid which is updated incrementally.

    Reference: Sequential k-Means Clustering
    Formula: mi = mi + (1/ni) * (x - mi)
    Where mi is centroid i, ni is count of vectors assigned to cluster i, x is new vector.
    """

    def __init__(self, n_clusters: int, metric: DistanceMetric):
        """Initialize sequential k-means.

        Args:
            n_clusters: Target number of clusters
            metric: Distance metric for finding nearest centroids
        """
        self.n_clusters = n_clusters
        self.metric = metric
        self.dimension = 0

        # Centroid data
        self.centroids: np.ndarray = np.array([])  # Shape: (k, d)
        self.cluster_counts: list[int] = []  # Count of vectors per cluster

    @property
    def is_bootstrapping(self) -> bool:
        """True if still in bootstrap phase (< n_clusters centroids)."""
        return len(self.centroids) < self.n_clusters

    @property
    def num_centroids(self) -> int:
        """Current number of centroids."""
        return len(self.centroids) if len(self.centroids) > 0 else 0

    def add_vector(self, vector: np.ndarray) -> int:
        """Add a vector and return its assigned cluster index.

        During bootstrap: vector becomes a new centroid.
        After bootstrap: vector is assigned to nearest centroid, which is updated.

        Args:
            vector: Vector of shape (d,). Should be normalized if using cosine metric.

        Returns:
            Index of the cluster this vector was assigned to.
        """
        # Initialize dimension on first vector
        if self.dimension == 0:
            self.dimension = vector.shape[0]

        if self.is_bootstrapping:
            return self._create_centroid(vector)
        else:
            cluster_idx = self.find_nearest_centroid(vector)
            self._update_centroid(cluster_idx, vector)
            return cluster_idx

    def _create_centroid(self, vector: np.ndarray) -> int:
        """Create a new centroid from the given vector."""
        if len(self.centroids) == 0:
            self.centroids = vector.reshape(1, -1)
        else:
            self.centroids = np.vstack([self.centroids, vector.reshape(1, -1)])

        self.cluster_counts.append(1)
        return len(self.centroids) - 1

    def _update_centroid(self, cluster_idx: int, vector: np.ndarray) -> None:
        """Update centroid using sequential k-means formula."""
        self.cluster_counts[cluster_idx] += 1
        n = self.cluster_counts[cluster_idx]

        # mi = mi + (1/ni) * (x - mi)
        self.centroids[cluster_idx] += (1 / n) * (vector - self.centroids[cluster_idx])

        # Re-normalize centroid for cosine metric to maintain unit length
        if self.metric == DistanceMetric.COSINE:
            self.centroids[cluster_idx] = normalize_vectors(self.centroids[cluster_idx])

    def find_nearest_centroid(self, vector: np.ndarray) -> int:
        """Find the index of the nearest centroid.

        Args:
            vector: Query vector of shape (d,)

        Returns:
            Index of the nearest centroid
        """
        scores = compute_similarities(vector, self.centroids, self.metric)
        return int(np.argmax(scores))

    def find_nearest_centroids(self, vector: np.ndarray, n: int) -> np.ndarray:
        """Find indices of the n nearest centroids.

        Args:
            vector: Query vector of shape (d,)
            n: Number of nearest centroids to return

        Returns:
            Array of centroid indices, sorted by similarity (best first)
        """
        n = min(n, self.num_centroids)
        if n == 0:
            return np.array([], dtype=int)

        scores = compute_similarities(vector, self.centroids, self.metric)
        return top_k_indices(scores, n)
