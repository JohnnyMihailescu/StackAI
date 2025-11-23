"""IVF (Inverted File) vector index implementation."""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.config import settings
from app.models.enums import DistanceMetric, IndexType
from app.services.indexes.base import BaseIndex
from app.services.indexes.sequential_kmeans import SequentialKMeans
from app.services.indexes.utils import normalize_vectors, similarity_search


class IVFIndex(BaseIndex):
    """IVF index using clustering for approximate nearest neighbor search.

    Partitions vectors into clusters and only searches relevant clusters
    at query time, trading some recall for faster search.
    """

    def __init__(
        self,
        n_clusters: int | None = None,
        n_probe: int | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        """Initialize the IVF index.

        Args:
            n_clusters: Number of clusters to create (default from settings)
            n_probe: Number of clusters to search at query time (default from settings)
            metric: Distance metric for similarity search
        """
        super().__init__()
        self.n_clusters = n_clusters or settings.ivf_default_clusters
        self.n_probe = n_probe or settings.ivf_default_n_probe
        self.n_probe = min(self.n_probe, self.n_clusters)
        self.metric = metric

        # Clustering algorithm
        self.kmeans = SequentialKMeans(self.n_clusters, metric)

        # Per-cluster vector storage
        self.cluster_vectors: List[np.ndarray] = []
        self.cluster_ids: List[List[str]] = []

        # ID lookup: id -> (cluster_idx, position_in_cluster)
        self._id_to_location: dict[str, Tuple[int, int]] = {}

    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the index.

        Args:
            vectors: Array of shape (n, d)
            ids: List of string IDs corresponding to each vector
        """
        if len(vectors) != len(ids):
            raise ValueError(
                f"Number of vectors ({len(vectors)}) must match number of ids ({len(ids)})"
            )

        if len(vectors) == 0:
            return

        # Normalize for cosine metric
        if self.metric == DistanceMetric.COSINE:
            vectors = normalize_vectors(vectors)

        # Set dimension on first add
        if self.dimension == 0:
            self.dimension = vectors.shape[1]
        elif vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension ({vectors.shape[1]}) "
                f"does not match index dimension ({self.dimension})"
            )

        for vector, id_ in zip(vectors, ids):
            cluster_idx = self.kmeans.add_vector(vector)
            self._store_vector(cluster_idx, vector, id_)
            self.num_vectors += 1

    def _store_vector(self, cluster_idx: int, vector: np.ndarray, id_: str) -> None:
        """Store a vector in the specified cluster."""
        # Expand cluster storage if needed (during bootstrap)
        while len(self.cluster_vectors) <= cluster_idx:
            self.cluster_vectors.append(np.array([]).reshape(0, self.dimension))
            self.cluster_ids.append([])

        pos = len(self.cluster_ids[cluster_idx])
        self.cluster_vectors[cluster_idx] = np.vstack(
            [self.cluster_vectors[cluster_idx], vector.reshape(1, -1)]
        )
        self.cluster_ids[cluster_idx].append(id_)
        self._id_to_location[id_] = (cluster_idx, pos)

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors.

        Args:
            query_vector: Query vector of shape (d,)
            k: Number of nearest neighbors to return

        Returns:
            List of (id, score) tuples, sorted by similarity (highest first)
        """
        if self.num_vectors == 0:
            return []

        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Query vector dimension ({query_vector.shape[0]}) "
                f"does not match index dimension ({self.dimension})"
            )

        # Normalize query for cosine metric
        if self.metric == DistanceMetric.COSINE:
            query_vector = normalize_vectors(query_vector)

        # Find clusters to search
        n_probe = min(self.n_probe, self.kmeans.num_centroids)
        probe_indices = self.kmeans.find_nearest_centroids(query_vector, n_probe)

        # Gather candidates from selected clusters
        candidate_vectors = []
        candidate_ids = []
        for idx in probe_indices:
            if len(self.cluster_ids[idx]) > 0:
                candidate_vectors.append(self.cluster_vectors[idx])
                candidate_ids.extend(self.cluster_ids[idx])

        if not candidate_ids:
            return []

        # Search candidates
        all_candidates = np.vstack(candidate_vectors)
        return similarity_search(query_vector, all_candidates, candidate_ids, k, self.metric)

    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID. Does not update centroids."""
        if not ids:
            return

        # Group deletions by cluster
        deletions_by_cluster: dict[int, List[int]] = {}
        for id_ in ids:
            if id_ in self._id_to_location:
                cluster_idx, pos = self._id_to_location[id_]
                if cluster_idx not in deletions_by_cluster:
                    deletions_by_cluster[cluster_idx] = []
                deletions_by_cluster[cluster_idx].append(pos)
                del self._id_to_location[id_]
                self.num_vectors -= 1

        # Remove from each cluster
        for cluster_idx, positions in deletions_by_cluster.items():
            positions_set = set(positions)
            keep_mask = [
                i not in positions_set for i in range(len(self.cluster_ids[cluster_idx]))
            ]

            self.cluster_vectors[cluster_idx] = self.cluster_vectors[cluster_idx][keep_mask]
            self.cluster_ids[cluster_idx] = [
                id_ for i, id_ in enumerate(self.cluster_ids[cluster_idx]) if keep_mask[i]
            ]

            # Rebuild positions for this cluster
            for new_pos, id_ in enumerate(self.cluster_ids[cluster_idx]):
                self._id_to_location[id_] = (cluster_idx, new_pos)

        # Reset if empty
        if self.num_vectors == 0:
            self._reset()

    def _reset(self) -> None:
        """Reset index to empty state."""
        self.kmeans = SequentialKMeans(self.n_clusters, self.metric)
        self.cluster_vectors = []
        self.cluster_ids = []
        self._id_to_location = {}
        self.dimension = 0

    def get_vector(self, vector_id: str) -> np.ndarray | None:
        """Get a vector by ID."""
        if vector_id not in self._id_to_location:
            return None
        cluster_idx, pos = self._id_to_location[vector_id]
        return self.cluster_vectors[cluster_idx][pos]

    def get_stats(self) -> dict:
        """Get index statistics."""
        cluster_sizes = [len(ids) for ids in self.cluster_ids]
        memory = sum(v.nbytes for v in self.cluster_vectors)
        if self.kmeans.num_centroids > 0:
            memory += self.kmeans.centroids.nbytes

        return {
            "index_type": "ivf",
            "num_vectors": self.num_vectors,
            "dimension": self.dimension,
            "metric": self.metric.value,
            "n_clusters": self.n_clusters,
            "n_probe": self.n_probe,
            "active_clusters": self.kmeans.num_centroids,
            "bootstrapping": self.kmeans.is_bootstrapping,
            "cluster_sizes": {
                "min": min(cluster_sizes) if cluster_sizes else 0,
                "max": max(cluster_sizes) if cluster_sizes else 0,
                "avg": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
            },
            "memory_bytes": memory,
        }

    def save(self, path: Path) -> None:
        """Save index to disk."""
        # Flatten cluster data
        if self.num_vectors > 0:
            all_vectors = np.vstack(self.cluster_vectors)
            all_ids = [id_ for ids in self.cluster_ids for id_ in ids]
            boundaries = np.cumsum([0] + [len(ids) for ids in self.cluster_ids])
        else:
            all_vectors = np.array([])
            all_ids = []
            boundaries = np.array([])

        np.savez(
            path,
            index_type=np.array([IndexType.IVF.value]),
            # K-means state
            centroids=self.kmeans.centroids,
            cluster_counts=np.array(self.kmeans.cluster_counts),
            # Vector storage
            all_vectors=all_vectors,
            all_ids=np.array(all_ids, dtype=object),
            cluster_boundaries=boundaries,
            # Config
            config=np.array([self.n_clusters, self.n_probe, self.dimension]),
            metric=np.array([self.metric.value]),
        )

    @classmethod
    def load(cls, path: Path) -> "IVFIndex":
        """Load index from disk."""
        data = np.load(path, allow_pickle=True)

        config = data["config"]
        metric = DistanceMetric(data["metric"][0])

        index = cls(
            n_clusters=int(config[0]),
            n_probe=int(config[1]),
            metric=metric,
        )
        index.dimension = int(config[2])

        # Restore k-means state
        index.kmeans.centroids = data["centroids"]
        index.kmeans.cluster_counts = data["cluster_counts"].tolist()
        index.kmeans.dimension = index.dimension

        # Restore cluster storage
        all_vectors = data["all_vectors"]
        all_ids = data["all_ids"].tolist()
        boundaries = data["cluster_boundaries"].tolist()

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if start < end:
                index.cluster_vectors.append(all_vectors[start:end])
                index.cluster_ids.append(all_ids[start:end])
            else:
                index.cluster_vectors.append(np.array([]).reshape(0, index.dimension))
                index.cluster_ids.append([])

        # Rebuild ID lookup
        for cluster_idx, ids in enumerate(index.cluster_ids):
            for pos, id_ in enumerate(ids):
                index._id_to_location[id_] = (cluster_idx, pos)

        index.num_vectors = len(all_ids)
        return index
