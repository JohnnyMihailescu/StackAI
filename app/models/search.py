"""Search models for query and response."""

from pydantic import BaseModel, Field
from app.models.chunk import Chunk
from app.config import settings


class SearchRequest(BaseModel):
    """Request model for similarity search."""

    query: str = Field(..., description="The search query text", min_length=1)
    # 
    k: int = Field(
        default=settings.default_search_k,
        description="Number of results to return",
        ge=1
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the transformer architecture?",
                "k": 10,
            }
        }
    }


class SearchResult(BaseModel):
    """A single search result."""

    chunk: Chunk = Field(..., description="The matching chunk")
    score: float = Field(..., description="Similarity score (higher is more similar)")


class SearchResponse(BaseModel):
    """Response model for similarity search."""

    query: str = Field(..., description="The original search query")
    results: list[SearchResult] = Field(..., description="List of search results")
    result_count: int = Field(..., description="Number of results returned")
