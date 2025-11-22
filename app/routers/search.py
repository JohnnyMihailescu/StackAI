"""Search router - similarity search endpoints."""

import numpy as np
from fastapi import APIRouter, HTTPException, status
from app.models.search import SearchRequest, SearchResponse
from app.routers.libraries import libraries_db
from app.services.embeddings import EmbeddingService
from app.services.search_service import SearchService

router = APIRouter()


@router.post(
    "/libraries/{library_id}/search",
    response_model=SearchResponse,
)
def search_library(library_id: str, request: SearchRequest):
    """Search for similar chunks in a library.

    Embeds the query text and finds the most similar chunks using cosine similarity.
    """
    # Verify library exists
    if library_id not in libraries_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library with id '{library_id}' not found",
        )

    # Embed the query
    query_embedding = EmbeddingService.embed_query(request.query)
    query_vector = np.array(query_embedding)

    # Search and get enriched results
    results = SearchService.search(library_id, query_vector, request.k)

    return SearchResponse(
        query=request.query,
        results=results,
        result_count=len(results),
    )
