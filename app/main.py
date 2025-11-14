"""FastAPI application entry point."""

from fastapi import FastAPI
from app.config import settings
from app.routers import libraries

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A vector database API built with FastAPI"
)

# Include routers
app.include_router(libraries.router, prefix="/api/v1", tags=["libraries"])


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
