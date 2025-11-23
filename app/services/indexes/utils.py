"""Shared utilities for vector index implementations."""

from typing import List, Tuple

import numpy as np

from app.models.enums import DistanceMetric


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length for cosine similarity.

    Args:
        vectors: Array of shape (n, d) or (d,) for single vector

    Returns:
        Normalized vectors of same shape
    """
    # Handle single vector
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
        single = True
    else:
        single = False

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    normalized = vectors / norms

    return normalized[0] if single else normalized


def compute_similarities(
    query: np.ndarray,
    vectors: np.ndarray,
    metric: DistanceMetric,
) -> np.ndarray:
    """Compute similarity scores between query and all vectors.

    Args:
        query: Query vector of shape (d,)
        vectors: Array of vectors of shape (n, d)
        metric: Distance metric to use

    Returns:
        Array of similarity scores of shape (n,)
        Higher scores = more similar for both metrics
    """
    if metric == DistanceMetric.COSINE:
        # Assumes vectors are already normalized
        normalized_query = normalize_vectors(query)
        return np.dot(vectors, normalized_query)
    elif metric == DistanceMetric.EUCLIDEAN:
        # Negate distance so higher = more similar
        distances = np.linalg.norm(vectors - query, axis=1)
        return -distances
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Get indices of top-k highest scores efficiently.

    Uses argpartition for O(n) complexity instead of O(n log n) full sort.

    Args:
        scores: Array of scores of shape (n,)
        k: Number of top indices to return

    Returns:
        Array of indices sorted by score (highest first)
    """
    k = min(k, len(scores))
    if k == 0:
        return np.array([], dtype=int)

    # Get top-k indices (unsorted)
    top_k_unsorted = np.argpartition(scores, -k)[-k:]
    # Sort them by score descending
    return top_k_unsorted[np.argsort(-scores[top_k_unsorted])]


def similarity_search(
    query: np.ndarray,
    vectors: np.ndarray,
    ids: List[str],
    k: int,
    metric: DistanceMetric = DistanceMetric.COSINE,
) -> List[Tuple[str, float]]:
    """Brute-force similarity search.

    Args:
        query: Query vector of shape (d,)
        vectors: Array of vectors of shape (n, d)
        ids: List of IDs corresponding to each vector
        k: Number of nearest neighbors to return
        metric: Distance metric to use

    Returns:
        List of (id, score) tuples sorted by similarity (highest first)
    """
    if len(vectors) == 0:
        return []

    scores = compute_similarities(query, vectors, metric)
    top_indices = top_k_indices(scores, k)

    return [(ids[idx], float(scores[idx])) for idx in top_indices]
