"""Search router - similarity search endpoints."""

import logging
import numpy as np
from fastapi import APIRouter, HTTPException, status
from app.models.search import SearchRequest, SearchResponse
from app.services.embeddings import EmbeddingService
from app.services.search_service import SearchService
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/libraries/{library_id}/search",
    response_model=SearchResponse,
)
async def search_library(
    library_id: str,
    request: SearchRequest,
    include_embedding: bool = False,
):
    """Search for similar chunks in a library.

    Embeds the query text and finds the most similar chunks using cosine similarity.
    """
    # Verify library exists
    library = await StorageService.libraries().get(library_id)
    if library is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library with id '{library_id}' not found",
        )

    logger.info(f"Searching: query='{request.query}' k={request.k} (library='{library.name}')")

    # Embed the query
    query_embedding = EmbeddingService.embed_query(request.query)
    query_vector = np.array(query_embedding)

    # Search and get enriched results
    results = await SearchService.search(library_id, query_vector, request.k)

    # Optionally attach embeddings to result chunks
    if include_embedding:
        for result in results:
            embedding = await SearchService.get_embedding(library_id, result.chunk.id)
            if embedding:
                result.chunk.embedding = embedding

    logger.info(f"Search complete: {len(results)} results (library='{library.name}')")
    return SearchResponse(
        query=request.query,
        results=results,
        result_count=len(results),
    )
