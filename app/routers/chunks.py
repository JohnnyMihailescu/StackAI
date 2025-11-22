"""Chunk router - CRUD operations for chunks within documents."""

from fastapi import APIRouter, HTTPException, status
from app.models.chunk import Chunk, BatchChunksRequest, BatchChunksResponse
from app.routers.documents import documents_db
from app.config import settings
from app.services.embeddings import EmbeddingService

router = APIRouter()

# In-memory storage for chunks
chunks_db: dict[str, Chunk] = {}


@router.post(
    "/documents/{document_id}/chunks",
    response_model=Chunk,
    status_code=status.HTTP_201_CREATED
)
def create_chunk(document_id: str, chunk: Chunk):
    """Create a new chunk in a document.

    Note: The embedding will be generated automatically if not provided.
    """
    # Verify document exists
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found"
        )

    # Verify chunk belongs to this document
    if chunk.document_id != document_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chunk document_id must match URL document_id"
        )

    # Check if chunk already exists
    if chunk.id in chunks_db:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Chunk with id '{chunk.id}' already exists"
        )

    # Generate embedding
    embedding = EmbeddingService.embed_texts([chunk.text])[0]
    chunk.embedding = embedding

    chunks_db[chunk.id] = chunk
    return chunk


@router.post(
    "/documents/{document_id}/chunks/batch",
    response_model=BatchChunksResponse,
    status_code=status.HTTP_201_CREATED
)
def create_chunks_batch(document_id: str, request: BatchChunksRequest):
    """Create multiple chunks in a document at once.

    Note: Embeddings will be generated automatically if not provided.
    """
    # Validate batch size
    if len(request.chunks) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(request.chunks)} exceeds maximum of {settings.max_batch_size}"
        )

    # Verify document exists
    if document_id not in documents_db:
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
    existing_ids = [id for id in batch_ids if id in chunks_db]
    if existing_ids:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Chunks with these IDs already exist: {existing_ids}"
        )

    # Generate embeddings for all chunks (batched internally)
    texts = [chunk.text for chunk in request.chunks]
    embeddings = EmbeddingService.embed_texts(texts)

    # Assign embeddings and store chunks
    for chunk, embedding in zip(request.chunks, embeddings):
        chunk.embedding = embedding
        chunks_db[chunk.id] = chunk

    return BatchChunksResponse(
        created_count=len(request.chunks),
        chunks=request.chunks
    )


@router.get("/documents/{document_id}/chunks", response_model=list[Chunk])
def list_chunks(document_id: str):
    """List all chunks in a document."""
    # Verify document exists
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found"
        )

    # Filter chunks by document_id
    document_chunks = [
        chunk for chunk in chunks_db.values()
        if chunk.document_id == document_id
    ]
    return document_chunks


@router.get("/chunks/{chunk_id}", response_model=Chunk)
def get_chunk(chunk_id: str):
    """Get a specific chunk by ID."""
    if chunk_id not in chunks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk with id '{chunk_id}' not found"
        )

    return chunks_db[chunk_id]


@router.delete("/chunks/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chunk(chunk_id: str):
    """Delete a chunk."""
    if chunk_id not in chunks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk with id '{chunk_id}' not found"
        )

    # TODO: Remove from vector index when implemented
    del chunks_db[chunk_id]
