"""Document router - CRUD operations for documents within libraries."""

from fastapi import APIRouter, HTTPException, status
from app.models.document import Document
from app.services.storage_service import StorageService

router = APIRouter()


@router.post(
    "/libraries/{library_id}/documents",
    response_model=Document,
    status_code=status.HTTP_201_CREATED
)
async def create_document(library_id: str, document: Document):
    """Create a new document in a library."""
    # Verify library exists
    if not await StorageService.libraries().exists(library_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library with id '{library_id}' not found"
        )

    # Verify document belongs to this library
    if document.library_id != library_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document library_id must match URL library_id"
        )

    try:
        return await StorageService.documents().create(document)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get("/libraries/{library_id}/documents", response_model=list[Document])
async def list_documents(library_id: str):
    """List all documents in a library."""
    if not await StorageService.libraries().exists(library_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library with id '{library_id}' not found"
        )

    return await StorageService.documents().list_by_library(library_id)


@router.get(
    "/libraries/{library_id}/documents/{document_id}",
    response_model=Document
)
async def get_document(library_id: str, document_id: str):
    """Get a specific document."""
    document = await StorageService.documents().get(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found"
        )

    if document.library_id != library_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in library '{library_id}'"
        )

    return document


@router.delete(
    "/libraries/{library_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_document(library_id: str, document_id: str):
    """Delete a document and all its chunks."""
    document = await StorageService.documents().get(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found"
        )

    if document.library_id != library_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in library '{library_id}'"
        )

    # Delete associated chunks first
    await StorageService.chunks().delete_by_document(document_id)
    await StorageService.documents().delete(document_id)
