"""Chunk router - CRUD operations for chunks within documents."""

import logging
import numpy as np
from fastapi import APIRouter, HTTPException, status
from app.models.chunk import Chunk, ChunkCreate, BatchChunksRequest, BatchChunksResponse
from app.config import settings
from app.services.embeddings import EmbeddingService
from app.services.search_service import SearchService
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/documents/{document_id}/chunks",
    response_model=Chunk,
    status_code=status.HTTP_201_CREATED
)
async def create_chunk(document_id: int, chunk_create: ChunkCreate):
    """Create a new chunk in a document."""
    document = await StorageService.documents().get(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {document_id} not found"
        )

    library = await StorageService.libraries().get(document.library_id)
    if library is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library with id {document.library_id} not found"
        )

    text_preview = chunk_create.text[:50] + "..." if len(chunk_create.text) > 50 else chunk_create.text
    logger.info(f"Creating chunk: text='{text_preview}' (document='{document.name}')")

    # Generate embedding
    embedding = EmbeddingService.embed_texts([chunk_create.text])[0]

    # Store chunk
    created_chunk = await StorageService.chunks().create(document_id, chunk_create)
    created_chunk.embedding = embedding

    # Add to vector index
    vectors = np.array([embedding])
    await SearchService.add_vectors(
        library.id, vectors, [created_chunk.id],
        index_type=library.index_type, metric=library.metric
    )

    logger.info(f"Chunk created: id={created_chunk.id} (document='{document.name}')")
    return created_chunk


@router.post(
    "/documents/{document_id}/chunks/batch",
    response_model=BatchChunksResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_chunks_batch(document_id: int, request: BatchChunksRequest):
    """Create multiple chunks in a document at once."""
    if len(request.chunks) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(request.chunks)} exceeds maximum of {settings.max_batch_size}"
        )

    document = await StorageService.documents().get(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {document_id} not found"
        )

    library = await StorageService.libraries().get(document.library_id)
    if library is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library with id {document.library_id} not found"
        )

    logger.info(f"Creating {len(request.chunks)} chunks (document='{document.name}')")

    # Generate embeddings for all chunks
    texts = [chunk.text for chunk in request.chunks]
    embeddings = EmbeddingService.embed_texts(texts)

    # Store chunks (IDs assigned by storage layer)
    created_chunks = await StorageService.chunks().create_batch(document_id, request.chunks)

    # Attach embeddings to returned chunks
    for chunk, embedding in zip(created_chunks, embeddings):
        chunk.embedding = embedding

    # Add all vectors to index
    vectors = np.array(embeddings)
    chunk_ids = [chunk.id for chunk in created_chunks]
    await SearchService.add_vectors(
        library.id, vectors, chunk_ids,
        index_type=library.index_type, metric=library.metric
    )

    logger.info(f"Batch complete: {len(created_chunks)} chunks created (document='{document.name}')")
    return BatchChunksResponse(
        created_count=len(created_chunks),
        chunks=created_chunks
    )


@router.get("/documents/{document_id}/chunks", response_model=list[Chunk])
async def list_chunks(document_id: int, include_embedding: bool = False):
    """List all chunks in a document."""
    document = await StorageService.documents().get(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {document_id} not found"
        )

    chunks = await StorageService.chunks().list_by_document(document_id)

    if include_embedding:
        for chunk in chunks:
            embedding = await SearchService.get_embedding(document.library_id, chunk.id)
            if embedding:
                chunk.embedding = embedding

    return chunks


@router.get("/chunks/{chunk_id}", response_model=Chunk)
async def get_chunk(chunk_id: int, include_embedding: bool = False):
    """Get a specific chunk by ID."""
    chunk = await StorageService.chunks().get(chunk_id)
    if chunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk with id {chunk_id} not found"
        )

    if include_embedding:
        document = await StorageService.documents().get(chunk.document_id)
        if document:
            embedding = await SearchService.get_embedding(document.library_id, chunk.id)
            if embedding:
                chunk.embedding = embedding

    return chunk


@router.delete("/chunks/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chunk(chunk_id: int):
    """Delete a chunk."""
    chunk = await StorageService.chunks().get(chunk_id)
    if chunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk with id {chunk_id} not found"
        )

    document = await StorageService.documents().get(chunk.document_id)
    library_id = document.library_id if document else None

    logger.info(f"Deleting chunk id={chunk_id} (document='{document.name if document else 'unknown'}')")

    await StorageService.chunks().delete(chunk_id)

    if library_id:
        await SearchService.delete_vectors(library_id, [chunk_id])

    logger.info("Chunk deleted")
