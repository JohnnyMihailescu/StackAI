"""Embedding service using Cohere API."""

import logging
import cohere
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using Cohere."""

    _client: cohere.Client | None = None

    @classmethod
    def initialize(cls):
        """Initialize the Cohere client. Call this at app startup."""
        cls._client = cohere.Client(settings.cohere_api_key)

    @classmethod
    def embed_texts(cls, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts (for storage).

        Automatically batches requests to respect Cohere's API limits.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        if not texts:
            return []

        all_embeddings = []
        batch_size = settings.cohere_batch_size
        num_batches = (len(texts) + batch_size - 1) // batch_size

        logger.info(f"Embedding {len(texts)} texts ({num_batches} batch(es))")

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1}/{num_batches}")
            response = cls._client.embed(
                texts=batch,
                model=settings.cohere_embed_model,
                input_type="search_document",
            )
            all_embeddings.extend(response.embeddings)

        return all_embeddings

    @classmethod
    def embed_query(cls, text: str) -> list[float]:
        """Generate embedding for a search query.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector for the query.
        """
        logger.debug("Embedding query text")
        response = cls._client.embed(
            texts=[text],
            model=settings.cohere_embed_model,
            input_type="search_query",
        )
        return response.embeddings[0]
