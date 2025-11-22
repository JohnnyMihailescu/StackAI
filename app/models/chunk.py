"""Chunk model - a piece of text with embeddings and metadata."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A chunk is a piece of text with embeddings and metadata."""

    id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="ID of the document this chunk belongs to")
    text: str = Field(..., description="The text content of the chunk")
    embedding: Optional[list[float]] = Field(None, description="Vector embedding of the text")
    metadata: dict = Field(default_factory=dict, description="Additional metadata (e.g., position, page number)")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "chunk_789",
                "document_id": "doc_456",
                "text": "The Transformer model architecture is based on self-attention mechanisms...",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {"position": 0, "page": 1},
                "created_at": "2025-01-15T10:00:00Z"
            }
        }
    }


class BatchChunksRequest(BaseModel):
    """Request model for creating multiple chunks at once."""

    chunks: list[Chunk] = Field(..., description="List of chunks to create", min_length=1)


class BatchChunksResponse(BaseModel):
    """Response model for batch chunk creation."""

    created_count: int = Field(..., description="Number of chunks successfully created")
    chunks: list[Chunk] = Field(..., description="List of created chunks")
