"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from app.config import settings
from app.routers import libraries, documents, chunks, search
from app.services.embeddings import EmbeddingService
from app.services.search_service import SearchService
from app.services.storage_service import StorageService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    EmbeddingService.initialize()
    await StorageService.initialize()
    SearchService.initialize(Path(settings.data_dir) / "indexes")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A vector database API built with FastAPI",
    lifespan=lifespan,
)

# Include routers
app.include_router(libraries.router, prefix="/api/v1", tags=["libraries"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(chunks.router, prefix="/api/v1", tags=["chunks"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
