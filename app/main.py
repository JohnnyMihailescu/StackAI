"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.logging_config import setup_logging
from app.routers import chunks, documents, libraries, search
from app.services.embeddings import EmbeddingService
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    setup_logging()
    logger.info("Starting StackAI Vector DB Server...")

    logger.info("Initializing embedding service")
    EmbeddingService.initialize()

    logger.info("Loading storage and indexes from disk")
    await StorageService.initialize()
    stats = StorageService.get_stats()
    logger.info(
        f"Storage loaded: {stats['libraries']} libraries, "
        f"{stats['documents']} documents, {stats['chunks']} chunks, "
        f"{stats['indexes']} indexes"
    )

    logger.info("StackAI Vector DB Server ready")
    yield
    logger.info("Shutting down StackAI Vector DB Server")


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
