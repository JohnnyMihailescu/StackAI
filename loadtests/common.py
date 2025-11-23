"""
Shared utilities for Locust load tests.
"""

import uuid
import random

# Sample texts for creating chunks
SAMPLE_TEXTS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Neural networks are computing systems inspired by biological neural networks in animal brains.",
    "Deep learning uses multiple layers of neural networks to progressively extract features.",
    "Natural language processing enables computers to understand and generate human language.",
    "Computer vision trains computers to interpret and understand visual information.",
    "Reinforcement learning is where agents learn to make decisions by taking actions.",
    "Transformers use self-attention mechanisms for sequence-to-sequence tasks.",
    "Vector databases store high-dimensional vectors for similarity search applications.",
    "Embeddings are dense vector representations that capture semantic meaning.",
    "Cosine similarity measures the angle between two vectors to determine similarity.",
]

# Sample search queries
SEARCH_QUERIES = [
    "What is machine learning?",
    "How do neural networks work?",
    "Explain deep learning",
    "What is natural language processing?",
    "How does computer vision work?",
    "What are transformers?",
    "How do vector databases work?",
    "What are embeddings?",
    "Explain cosine similarity",
    "What is reinforcement learning?",
]


def random_id(prefix: str = "") -> str:
    """Generate a random ID with optional prefix."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def random_text() -> str:
    """Get a random sample text."""
    return random.choice(SAMPLE_TEXTS)


def random_query() -> str:
    """Get a random search query."""
    return random.choice(SEARCH_QUERIES)
