"""Chunk router - CRUD operations for chunks within documents."""

import numpy as np
from fastapi import APIRouter, HTTPException, status
from app.models.chunk import Chunk, BatchChunksRequest, BatchChunksResponse
from app.config import settings
from app.services.embeddings import EmbeddingService
from app.services.search_service import SearchService
from app.services.storage_service import StorageService

router = APIRouter()


@router.post(
    "/documents/{document_id}/chunks",
    response_model=Chunk,
    status_code=status.HTTP_201_CREATED
)
async def create_chunk(document_id: str, chunk: Chunk):
    """Create a new chunk in a document."""
    # Verify document exists
    document = await StorageService.documents().get(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found"
        )

    # Verify chunk belongs to this document
    if chunk.document_id != document_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chunk document_id must match URL document_id"
        )

    # Generate embedding
    embedding = EmbeddingService.embed_texts([chunk.text])[0]
    chunk.embedding = embedding

    # Store chunk
    try:
        created_chunk = await StorageService.chunks().create(chunk)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    # Add to vector index
    vectors = np.array([embedding])
    SearchService.add_vectors(document.library_id, vectors, [chunk.id])

    return created_chunk


@router.post(
    "/documents/{document_id}/chunks/batch",
    response_model=BatchChunksResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_chunks_batch(document_id: str, request: BatchChunksRequest):
    """Create multiple chunks in a document at once."""
    # Validate batch size
    if len(request.chunks) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(request.chunks)} exceeds maximum of {settings.max_batch_size}"
        )

    # Verify document exists
    document = await StorageService.documents().get(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found"
        )

    # Validate all chunks belong to this document
    invalid_chunks = [c.id for c in request.chunks if c.document_id != document_id]
    if invalid_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chunks must belong to document '{document_id}'. Invalid chunks: {invalid_chunks}"
        )

    # Check for duplicate IDs within the batch
    batch_ids = [c.id for c in request.chunks]
    if len(batch_ids) != len(set(batch_ids)):
        duplicates = [id for id in batch_ids if batch_ids.count(id) > 1]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Duplicate chunk IDs in batch: {list(set(duplicates))}"
        )

    # Check for existing IDs in database
    existing_ids = await StorageService.chunks().find_existing(batch_ids)
    if existing_ids:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Chunks with these IDs already exist: {existing_ids}"
        )

    # Generate embeddings for all chunks
    texts = [chunk.text for chunk in request.chunks]
    embeddings = EmbeddingService.embed_texts(texts)

    # Assign embeddings to chunks
    for chunk, embedding in zip(request.chunks, embeddings):
        chunk.embedding = embedding

    # Store chunks
    created_chunks = await StorageService.chunks().create_batch(request.chunks)

    # Add all vectors to index
    vectors = np.array(embeddings)
    chunk_ids = [chunk.id for chunk in created_chunks]
    SearchService.add_vectors(document.library_id, vectors, chunk_ids)

    return BatchChunksResponse(
        created_count=len(created_chunks),
        chunks=created_chunks
    )


@router.get("/documents/{document_id}/chunks", response_model=list[Chunk])
async def list_chunks(document_id: str):
    """List all chunks in a document."""
    if not await StorageService.documents().exists(document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found"
        )

    return await StorageService.chunks().list_by_document(document_id)


@router.get("/chunks/{chunk_id}", response_model=Chunk)
async def get_chunk(chunk_id: str):
    """Get a specific chunk by ID."""
    chunk = await StorageService.chunks().get(chunk_id)
    if chunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk with id '{chunk_id}' not found"
        )
    return chunk


@router.delete("/chunks/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chunk(chunk_id: str):
    """Delete a chunk."""
    # Get chunk to find its document_id
    chunk = await StorageService.chunks().get(chunk_id)
    if chunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk with id '{chunk_id}' not found"
        )

    # Get document to find library_id
    document = await StorageService.documents().get(chunk.document_id)
    library_id = document.library_id if document else None

    # Delete from storage
    await StorageService.chunks().delete(chunk_id)

    # Remove from vector index
    if library_id:
        SearchService.delete_vectors(library_id, [chunk_id])
