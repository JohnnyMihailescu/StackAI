"""Library model - a collection of documents."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Library(BaseModel):
    """A library is a collection of documents and their chunks."""

    id: str = Field(..., description="Unique identifier for the library")
    name: str = Field(..., description="Name of the library")
    description: Optional[str] = Field(None, description="Optional description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "lib_123",
                "name": "My Research Papers",
                "description": "Collection of AI research papers",
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T10:00:00Z"
            }
        }
