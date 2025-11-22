"""Library router - CRUD operations for libraries."""

from fastapi import APIRouter, HTTPException, status
from app.models.library import Library
from app.services.storage_service import StorageService

router = APIRouter()


@router.post("/libraries", response_model=Library, status_code=status.HTTP_201_CREATED)
async def create_library(library: Library):
    """Create a new library."""
    try:
        return await StorageService.libraries().create(library)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get("/libraries/{library_id}", response_model=Library)
async def get_library(library_id: str):
    """Get a library by ID."""
    library = await StorageService.libraries().get(library_id)
    if library is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library with id '{library_id}' not found"
        )
    return library


@router.get("/libraries", response_model=list[Library])
async def list_libraries():
    """List all libraries."""
    return await StorageService.libraries().list_all()


@router.delete("/libraries/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(library_id: str):
    """Delete a library by ID."""
    deleted = await StorageService.libraries().delete(library_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library with id '{library_id}' not found"
        )
