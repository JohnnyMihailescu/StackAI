"""Shared enums used across the application."""

from enum import Enum


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class IndexType(str, Enum):
    """Supported index types for vector search."""

    FLAT = "flat"
    IVF = "ivf"
