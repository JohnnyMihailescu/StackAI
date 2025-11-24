"""Tests for batch chunks endpoint."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.models.library import LibraryCreate
from app.models.document import DocumentCreate
from app.models.chunk import ChunkCreate
from app.services.storage_service import StorageService


# Mock embedding dimension (Cohere embed-english-v3.0 uses 1024)
MOCK_EMBEDDING_DIM = 1024


def mock_embed_texts(texts: list[str]) -> list[list[float]]:
    """Create fake embeddings for each text."""
    return [[0.1] * MOCK_EMBEDDING_DIM for _ in texts]


class TestBatchChunksEndpoint:
    """Test suite for POST /documents/{document_id}/chunks/batch."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize and clear storage before each test."""
        asyncio.run(StorageService.initialize(persist=False))
        yield
        asyncio.run(StorageService.clear_all())

    @pytest.fixture
    def client(self):
        """Create test client with mocked embeddings."""
        with patch(
            "app.routers.chunks.EmbeddingService.embed_texts",
            side_effect=mock_embed_texts
        ):
            yield TestClient(app)

    @pytest.fixture
    def sample_document(self):
        """Create a sample library and document in storage, return (library, document)."""
        lib_create = LibraryCreate(name="Test Library")
        doc_create = DocumentCreate(name="Test Doc")
        lib = asyncio.run(StorageService.libraries().create(lib_create))
        doc = asyncio.run(StorageService.documents().create(lib.id, doc_create))
        return lib, doc

    def test_create_batch_success(self, client, sample_document):
        """Test successfully creating multiple chunks."""
        lib, doc = sample_document
        payload = {
            "chunks": [
                {"text": "First chunk"},
                {"text": "Second chunk"},
                {"text": "Third chunk"},
            ]
        }

        response = client.post(f"/api/v1/documents/{doc.id}/chunks/batch", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["created_count"] == 3
        assert len(data["chunks"]) == 3
        # IDs should be auto-generated integers
        assert all(isinstance(c["id"], int) for c in data["chunks"])
        assert all(c["document_id"] == doc.id for c in data["chunks"])

    def test_create_batch_document_not_found(self, client):
        """Test batch creation fails when document doesn't exist."""
        payload = {
            "chunks": [
                {"text": "Some text"},
            ]
        }

        response = client.post("/api/v1/documents/999/chunks/batch", json=payload)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_create_batch_empty_list(self, client, sample_document):
        """Test batch creation fails with empty chunk list."""
        lib, doc = sample_document
        payload = {"chunks": []}

        response = client.post(f"/api/v1/documents/{doc.id}/chunks/batch", json=payload)

        assert response.status_code == 422  # Pydantic validation error

    def test_create_batch_exceeds_limit(self, client, sample_document, monkeypatch):
        """Test batch creation fails when exceeding max batch size."""
        lib, doc = sample_document
        # Set a low limit for testing
        from app import config
        monkeypatch.setattr(config.settings, "max_batch_size", 2)

        payload = {
            "chunks": [
                {"text": "First"},
                {"text": "Second"},
                {"text": "Third"},
            ]
        }

        response = client.post(f"/api/v1/documents/{doc.id}/chunks/batch", json=payload)

        assert response.status_code == 400
        assert "exceeds maximum" in response.json()["detail"]

    def test_create_batch_with_metadata(self, client, sample_document):
        """Test batch creation preserves metadata."""
        lib, doc = sample_document
        payload = {
            "chunks": [
                {
                    "text": "First chunk",
                    "metadata": {"position": 0, "page": 1}
                },
                {
                    "text": "Second chunk",
                    "metadata": {"position": 1, "page": 1}
                },
            ]
        }

        response = client.post(f"/api/v1/documents/{doc.id}/chunks/batch", json=payload)

        assert response.status_code == 201
        chunks = response.json()["chunks"]
        assert chunks[0]["metadata"]["position"] == 0
        assert chunks[1]["metadata"]["position"] == 1

    def test_create_batch_generates_embeddings(self, client, sample_document):
        """Test that embeddings are generated for all chunks."""
        lib, doc = sample_document
        payload = {
            "chunks": [
                {"text": "First chunk"},
                {"text": "Second chunk"},
            ]
        }

        response = client.post(f"/api/v1/documents/{doc.id}/chunks/batch", json=payload)

        assert response.status_code == 201
        chunks = response.json()["chunks"]
        # Verify all chunks have embeddings
        for chunk in chunks:
            assert chunk["embedding"] is not None
            assert len(chunk["embedding"]) == MOCK_EMBEDDING_DIM

    def test_create_batch_sequential_ids(self, client, sample_document):
        """Test that chunk IDs are assigned sequentially."""
        lib, doc = sample_document
        payload = {
            "chunks": [
                {"text": "First chunk"},
                {"text": "Second chunk"},
            ]
        }

        response = client.post(f"/api/v1/documents/{doc.id}/chunks/batch", json=payload)

        assert response.status_code == 201
        chunks = response.json()["chunks"]
        ids = [c["id"] for c in chunks]
        # IDs should be sequential
        assert ids == sorted(ids)
        assert ids[1] == ids[0] + 1
