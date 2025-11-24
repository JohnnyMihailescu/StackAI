"""Document model - belongs to a library and contains chunks."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    """Input model for creating a document."""

    name: str = Field(..., description="Name or title of the document (unique within library)")
    source: Optional[str] = Field(None, description="Source URL or file path")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Attention Is All You Need",
                "source": "https://arxiv.org/abs/1706.03762",
                "metadata": {"year": 2017, "authors": ["Vaswani et al."]},
            }
        }
    )


class Document(BaseModel):
    """A document belongs to a library and is split into chunks."""

    id: int = Field(..., description="Unique identifier for the document")
    library_id: int = Field(..., description="ID of the library this document belongs to")
    name: str = Field(..., description="Name or title of the document (unique within library)")
    source: Optional[str] = Field(None, description="Source URL or file path")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "library_id": 1,
                "name": "Attention Is All You Need",
                "source": "https://arxiv.org/abs/1706.03762",
                "metadata": {"year": 2017, "authors": ["Vaswani et al."]},
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T10:00:00Z",
            }
        }
    )
