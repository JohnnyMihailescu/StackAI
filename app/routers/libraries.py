"""Library router - CRUD operations for libraries."""

from fastapi import APIRouter, HTTPException, status
from app.models.library import Library

router = APIRouter()

# In-memory storage for now (will be replaced with proper persistence)
libraries_db: dict[str, Library] = {}


@router.post("/libraries", response_model=Library, status_code=status.HTTP_201_CREATED)
async def create_library(library: Library):
    """Create a new library."""
    if library.id in libraries_db:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Library with id '{library.id}' already exists"
        )

    libraries_db[library.id] = library
    return library


@router.get("/libraries/{library_id}", response_model=Library)
async def get_library(library_id: str):
    """Get a library by ID."""
    if library_id not in libraries_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library with id '{library_id}' not found"
        )

    return libraries_db[library_id]


@router.get("/libraries", response_model=list[Library])
async def list_libraries():
    """List all libraries."""
    return list(libraries_db.values())


@router.delete("/libraries/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(library_id: str):
    """Delete a library by ID."""
    if library_id not in libraries_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library with id '{library_id}' not found"
        )

    del libraries_db[library_id]
