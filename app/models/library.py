"""Library model - a collection of documents."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import DistanceMetric, IndexType


class LibraryCreate(BaseModel):
    """Input model for creating a library."""

    name: str = Field(..., description="Name of the library (must be unique)")
    description: Optional[str] = Field(None, description="Optional description")
    index_type: IndexType = Field(
        default=IndexType.FLAT, description="Type of vector index to use"
    )
    metric: DistanceMetric = Field(
        default=DistanceMetric.COSINE, description="Distance metric for similarity search"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "My Research Papers",
                "description": "Collection of AI research papers",
            }
        }
    )


class Library(BaseModel):
    """A library is a collection of documents and their chunks."""

    id: int = Field(..., description="Unique identifier for the library")
    name: str = Field(..., description="Name of the library (must be unique)")
    description: Optional[str] = Field(None, description="Optional description")
    index_type: IndexType = Field(
        default=IndexType.FLAT, description="Type of vector index to use"
    )
    metric: DistanceMetric = Field(
        default=DistanceMetric.COSINE, description="Distance metric for similarity search"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "name": "My Research Papers",
                "description": "Collection of AI research papers",
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T10:00:00Z",
            }
        }
    )
