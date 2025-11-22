"""Document router - CRUD operations for documents within libraries."""

from fastapi import APIRouter, HTTPException, status
from app.models.document import Document
from app.routers.libraries import libraries_db

router = APIRouter()

# In-memory storage for documents
documents_db: dict[str, Document] = {}


@router.post(
    "/libraries/{library_id}/documents",
    response_model=Document,
    status_code=status.HTTP_201_CREATED
)
async def create_document(library_id: str, document: Document):
    """Create a new document in a library."""
    # Verify library exists
    if library_id not in libraries_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library with id '{library_id}' not found"
        )

    # Verify document belongs to this library
    if document.library_id != library_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document library_id must match URL library_id"
        )

    # Check if document already exists
    if document.id in documents_db:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document with id '{document.id}' already exists"
        )

    documents_db[document.id] = document
    return document


@router.get("/libraries/{library_id}/documents", response_model=list[Document])
async def list_documents(library_id: str):
    """List all documents in a library."""
    # Verify library exists
    if library_id not in libraries_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library with id '{library_id}' not found"
        )

    # Filter documents by library_id
    library_documents = [
        doc for doc in documents_db.values()
        if doc.library_id == library_id
    ]
    return library_documents


@router.get(
    "/libraries/{library_id}/documents/{document_id}",
    response_model=Document
)
async def get_document(library_id: str, document_id: str):
    """Get a specific document."""
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found"
        )

    document = documents_db[document_id]

    # Verify document belongs to the specified library
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
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found"
        )

    document = documents_db[document_id]

    # Verify document belongs to the specified library
    if document.library_id != library_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in library '{library_id}'"
        )

    # TODO: Also delete associated chunks when chunk endpoints are implemented
    del documents_db[document_id]
