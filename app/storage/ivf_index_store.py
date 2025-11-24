"""IVF Index store for raw data access (no index class dependencies)."""

import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from app.models.enums import DistanceMetric, IndexType

logger = logging.getLogger(__name__)


class IVFIndexStore:
    """Raw data store for IVF indexes.

    Handles file I/O for IVF Indexes.
    IVFIndex imports this store and uses it for data access.

    Directory structure per library:
        data/indexes/{library_id}/
        ├── index_meta.json  # Human readable config
        ├── centroids.npy    # Centroid vectors (k × d)
        ├── cluster_0.npy    # Vectors for cluster 0
        ├── cluster_1.npy
        └── ...
    """

    def __init__(self, data_dir: Path | None = None, persist: bool = True):
        """Initialize the IVF index store.

        Args:
            data_dir: Directory for index files
            persist: Whether to persist indexes to disk
        """
        self._data_dir = data_dir
        self._persist = persist

    def _get_index_dir(self, library_id: int) -> Path:
        """Get the directory path for a library's index."""
        if self._data_dir is None:
            raise RuntimeError("IVFIndexStore not initialized with data_dir")
        return self._data_dir / str(library_id)

    def list_libraries(self) -> List[int]:
        """List library IDs that have IVF indexes on disk."""
        if not self._persist or self._data_dir is None:
            return []

        if not self._data_dir.exists():
            return []

        library_ids = []
        for index_dir in self._data_dir.iterdir():
            if not index_dir.is_dir():
                continue
            meta_file = index_dir / "index_meta.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                if meta.get("index_type") == IndexType.IVF.value:
                    library_ids.append(int(index_dir.name))

        return library_ids

    def load_metadata(self, library_id: int) -> dict:
        """Load metadata for an IVF index.

        Returns raw data dict - caller is responsible for creating IVFIndex.
        """
        index_dir = self._get_index_dir(library_id)

        # Load JSON config
        with open(index_dir / "index_meta.json") as f:
            meta = json.load(f)

        # Load centroids from binary
        centroids = np.load(index_dir / "centroids.npy")

        return {
            "centroids": centroids,
            "cluster_counts": meta["cluster_counts"],
            "cluster_ids": meta["cluster_ids"],
            "n_clusters": meta["n_clusters"],
            "n_probe": meta["n_probe"],
            "dimension": meta["dimension"],
            "metric": DistanceMetric(meta["metric"]),
        }

    def load_cluster(self, library_id: int, cluster_idx: int, dimension: int = 0) -> np.ndarray:
        """Load a single cluster's vectors from disk.

        Args:
            library_id: Library ID
            cluster_idx: Cluster index
            dimension: Vector dimension (used for empty cluster shape)

        Returns:
            Numpy array of vectors for the cluster
        """
        index_dir = self._get_index_dir(library_id)
        cluster_file = index_dir / f"cluster_{cluster_idx}.npy"

        if not cluster_file.exists():
            return np.array([]).reshape(0, dimension) if dimension > 0 else np.array([])

        return np.load(cluster_file)

    def save_metadata(
        self,
        library_id: int,
        centroids: np.ndarray,
        cluster_counts: List[int],
        cluster_ids: List[List[int]],
        n_clusters: int,
        n_probe: int,
        dimension: int,
        metric: DistanceMetric,
    ) -> None:
        """Save metadata for an IVF index."""
        if not self._persist:
            return

        index_dir = self._get_index_dir(library_id)
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON config (human readable)
        meta = {
            "index_type": IndexType.IVF.value,
            "n_clusters": n_clusters,
            "n_probe": n_probe,
            "dimension": dimension,
            "metric": metric.value,
            "cluster_ids": cluster_ids,
            "cluster_counts": cluster_counts,
        }
        with open(index_dir / "index_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save centroids as binary
        np.save(index_dir / "centroids.npy", centroids)

    def save_cluster(self, library_id: int, cluster_idx: int, vectors: np.ndarray) -> None:
        """Save a single cluster's vectors to disk."""
        if not self._persist:
            return

        if len(vectors) == 0:
            return

        index_dir = self._get_index_dir(library_id)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / f"cluster_{cluster_idx}.npy", vectors)

    def delete_index(self, library_id: int) -> None:
        """Delete all files for an index."""
        if not self._persist:
            return

        index_dir = self._get_index_dir(library_id)
        if index_dir.exists():
            for file in index_dir.iterdir():
                file.unlink()
            index_dir.rmdir()

    def exists(self, library_id: int) -> bool:
        """Check if an index exists on disk."""
        if not self._persist or self._data_dir is None:
            return False

        index_dir = self._get_index_dir(library_id)
        return (index_dir / "index_meta.json").exists()
