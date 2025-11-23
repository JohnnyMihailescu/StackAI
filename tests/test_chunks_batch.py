"""Tests for batch chunks endpoint."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.library import Library
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
        """Create a sample library and document in storage."""
        lib = Library(id="lib_1", name="Test Library")
        doc = Document(id="doc_123", library_id="lib_1", name="Test Doc")
        asyncio.run(StorageService.libraries().create(lib))
        asyncio.run(StorageService.documents().create(doc))
        return doc

    def test_create_batch_success(self, client, sample_document):
        """Test successfully creating multiple chunks."""
        payload = {
            "chunks": [
                {"id": "chunk_1", "document_id": "doc_123", "text": "First chunk"},
                {"id": "chunk_2", "document_id": "doc_123", "text": "Second chunk"},
                {"id": "chunk_3", "document_id": "doc_123", "text": "Third chunk"},
            ]
        }

        response = client.post("/api/v1/documents/doc_123/chunks/batch", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["created_count"] == 3
        assert len(data["chunks"]) == 3
        assert all(c["id"] in ["chunk_1", "chunk_2", "chunk_3"] for c in data["chunks"])

    def test_create_batch_document_not_found(self, client):
        """Test batch creation fails when document doesn't exist."""
        payload = {
            "chunks": [
                {"id": "chunk_1", "document_id": "nonexistent", "text": "Some text"},
            ]
        }

        response = client.post("/api/v1/documents/nonexistent/chunks/batch", json=payload)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_create_batch_wrong_document_id(self, client, sample_document):
        """Test batch creation fails when chunk document_id doesn't match URL."""
        payload = {
            "chunks": [
                {"id": "chunk_1", "document_id": "doc_123", "text": "Valid"},
                {"id": "chunk_2", "document_id": "wrong_doc", "text": "Invalid"},
            ]
        }

        response = client.post("/api/v1/documents/doc_123/chunks/batch", json=payload)

        assert response.status_code == 400
        assert "chunk_2" in response.json()["detail"]

    def test_create_batch_duplicate_ids_in_batch(self, client, sample_document):
        """Test batch creation fails when batch contains duplicate IDs."""
        payload = {
            "chunks": [
                {"id": "chunk_1", "document_id": "doc_123", "text": "First"},
                {"id": "chunk_1", "document_id": "doc_123", "text": "Duplicate"},
            ]
        }

        response = client.post("/api/v1/documents/doc_123/chunks/batch", json=payload)

        assert response.status_code == 400
        assert "Duplicate" in response.json()["detail"]

    def test_create_batch_existing_ids(self, client, sample_document):
        """Test batch creation fails when IDs already exist in database."""
        # Create an existing chunk
        existing = Chunk(id="chunk_1", document_id="doc_123", text="Existing", embedding=[0.1] * MOCK_EMBEDDING_DIM)
        asyncio.run(StorageService.chunks().create(existing))

        payload = {
            "chunks": [
                {"id": "chunk_1", "document_id": "doc_123", "text": "Conflict"},
                {"id": "chunk_2", "document_id": "doc_123", "text": "New"},
            ]
        }

        response = client.post("/api/v1/documents/doc_123/chunks/batch", json=payload)

        assert response.status_code == 409
        assert "chunk_1" in response.json()["detail"]

    def test_create_batch_empty_list(self, client, sample_document):
        """Test batch creation fails with empty chunk list."""
        payload = {"chunks": []}

        response = client.post("/api/v1/documents/doc_123/chunks/batch", json=payload)

        assert response.status_code == 422  # Pydantic validation error

    def test_create_batch_exceeds_limit(self, client, sample_document, monkeypatch):
        """Test batch creation fails when exceeding max batch size."""
        # Set a low limit for testing
        from app import config
        monkeypatch.setattr(config.settings, "max_batch_size", 2)

        payload = {
            "chunks": [
                {"id": "chunk_1", "document_id": "doc_123", "text": "First"},
                {"id": "chunk_2", "document_id": "doc_123", "text": "Second"},
                {"id": "chunk_3", "document_id": "doc_123", "text": "Third"},
            ]
        }

        response = client.post("/api/v1/documents/doc_123/chunks/batch", json=payload)

        assert response.status_code == 400
        assert "exceeds maximum" in response.json()["detail"]

    def test_create_batch_with_metadata(self, client, sample_document):
        """Test batch creation preserves metadata."""
        payload = {
            "chunks": [
                {
                    "id": "chunk_1",
                    "document_id": "doc_123",
                    "text": "First chunk",
                    "metadata": {"position": 0, "page": 1}
                },
                {
                    "id": "chunk_2",
                    "document_id": "doc_123",
                    "text": "Second chunk",
                    "metadata": {"position": 1, "page": 1}
                },
            ]
        }

        response = client.post("/api/v1/documents/doc_123/chunks/batch", json=payload)

        assert response.status_code == 201
        chunks = response.json()["chunks"]
        assert chunks[0]["metadata"]["position"] == 0
        assert chunks[1]["metadata"]["position"] == 1

    def test_create_batch_generates_embeddings(self, client, sample_document):
        """Test that embeddings are generated for all chunks."""
        payload = {
            "chunks": [
                {"id": "chunk_1", "document_id": "doc_123", "text": "First chunk"},
                {"id": "chunk_2", "document_id": "doc_123", "text": "Second chunk"},
            ]
        }

        response = client.post("/api/v1/documents/doc_123/chunks/batch", json=payload)

        assert response.status_code == 201
        chunks = response.json()["chunks"]
        # Verify all chunks have embeddings
        for chunk in chunks:
            assert chunk["embedding"] is not None
            assert len(chunk["embedding"]) == MOCK_EMBEDDING_DIM
