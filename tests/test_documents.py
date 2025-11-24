"""Tests for document router endpoints."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.models.library import LibraryCreate
from app.services.storage_service import StorageService


# Mock embedding for chunk creation tests
MOCK_EMBEDDING_DIM = 1024


def mock_embed_texts(texts: list[str]) -> list[list[float]]:
    """Create fake embeddings for each text."""
    return [[0.1] * MOCK_EMBEDDING_DIM for _ in texts]


class TestDocumentEndpoints:
    """Test suite for document CRUD endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Initialize and clear storage before each test."""
        data_dir = str(tmp_path / "test_data")
        asyncio.run(StorageService.initialize(data_dir=data_dir))
        yield
        asyncio.run(StorageService.clear_all())

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_library(self):
        """Create a sample library and return it."""
        lib_create = LibraryCreate(name="Test Library")
        return asyncio.run(StorageService.libraries().create(lib_create))

    # -------------------------------------------------------------------------
    # POST /libraries/{library_id}/documents - Create
    # -------------------------------------------------------------------------

    def test_create_document_minimal(self, client, sample_library):
        """Test creating a document with just a name."""
        payload = {"name": "Test Document"}

        response = client.post(
            f"/api/v1/libraries/{sample_library.id}/documents",
            json=payload
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Document"
        assert isinstance(data["id"], int)
        assert data["library_id"] == sample_library.id
        assert data["source"] is None
        assert data["metadata"] == {}

    def test_create_document_full(self, client, sample_library):
        """Test creating a document with all fields."""
        payload = {
            "name": "Full Document",
            "source": "https://example.com/doc.pdf",
            "metadata": {"author": "Test", "year": 2025}
        }

        response = client.post(
            f"/api/v1/libraries/{sample_library.id}/documents",
            json=payload
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Full Document"
        assert data["source"] == "https://example.com/doc.pdf"
        assert data["metadata"]["author"] == "Test"
        assert data["metadata"]["year"] == 2025

    def test_create_document_library_not_found(self, client):
        """Test creating a document in non-existent library returns 404."""
        payload = {"name": "Orphan Doc"}

        response = client.post("/api/v1/libraries/999/documents", json=payload)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_create_document_duplicate_name_in_library(self, client, sample_library):
        """Test duplicate document names within same library return 409."""
        payload = {"name": "Duplicate"}

        # First creation should succeed
        resp1 = client.post(
            f"/api/v1/libraries/{sample_library.id}/documents",
            json=payload
        )
        assert resp1.status_code == 201

        # Second creation with same name should fail
        resp2 = client.post(
            f"/api/v1/libraries/{sample_library.id}/documents",
            json=payload
        )
        assert resp2.status_code == 409
        assert "already exists" in resp2.json()["detail"]

    def test_create_document_same_name_different_libraries(self, client):
        """Test same document name can exist in different libraries."""
        # Create two libraries
        lib1 = client.post("/api/v1/libraries", json={"name": "Lib 1"}).json()
        lib2 = client.post("/api/v1/libraries", json={"name": "Lib 2"}).json()

        payload = {"name": "Same Name Doc"}

        # Create in first library
        resp1 = client.post(
            f"/api/v1/libraries/{lib1['id']}/documents",
            json=payload
        )
        assert resp1.status_code == 201

        # Create in second library - should also succeed
        resp2 = client.post(
            f"/api/v1/libraries/{lib2['id']}/documents",
            json=payload
        )
        assert resp2.status_code == 201

    # -------------------------------------------------------------------------
    # GET /libraries/{library_id}/documents/{document_id} - Read
    # -------------------------------------------------------------------------

    def test_get_document(self, client, sample_library):
        """Test getting a document by ID."""
        # Create a document
        create_resp = client.post(
            f"/api/v1/libraries/{sample_library.id}/documents",
            json={"name": "Get Test", "source": "test.pdf"}
        )
        doc_id = create_resp.json()["id"]

        # Get the document
        response = client.get(
            f"/api/v1/libraries/{sample_library.id}/documents/{doc_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == doc_id
        assert data["name"] == "Get Test"
        assert data["source"] == "test.pdf"

    def test_get_document_not_found(self, client, sample_library):
        """Test getting a non-existent document returns 404."""
        response = client.get(
            f"/api/v1/libraries/{sample_library.id}/documents/999"
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_document_wrong_library(self, client):
        """Test getting a document from wrong library returns 404."""
        # Create two libraries
        lib1 = client.post("/api/v1/libraries", json={"name": "Lib 1"}).json()
        lib2 = client.post("/api/v1/libraries", json={"name": "Lib 2"}).json()

        # Create document in lib1
        doc = client.post(
            f"/api/v1/libraries/{lib1['id']}/documents",
            json={"name": "Doc in Lib1"}
        ).json()

        # Try to get it via lib2
        response = client.get(
            f"/api/v1/libraries/{lib2['id']}/documents/{doc['id']}"
        )

        assert response.status_code == 404
        assert "not found in library" in response.json()["detail"]

    # -------------------------------------------------------------------------
    # GET /libraries/{library_id}/documents - List
    # -------------------------------------------------------------------------

    def test_list_documents_empty(self, client, sample_library):
        """Test listing documents when none exist."""
        response = client.get(f"/api/v1/libraries/{sample_library.id}/documents")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_documents(self, client, sample_library):
        """Test listing multiple documents."""
        # Create several documents
        for name in ["Doc A", "Doc B", "Doc C"]:
            client.post(
                f"/api/v1/libraries/{sample_library.id}/documents",
                json={"name": name}
            )

        response = client.get(f"/api/v1/libraries/{sample_library.id}/documents")

        assert response.status_code == 200
        docs = response.json()
        assert len(docs) == 3
        names = {doc["name"] for doc in docs}
        assert names == {"Doc A", "Doc B", "Doc C"}

    def test_list_documents_library_not_found(self, client):
        """Test listing documents in non-existent library returns 404."""
        response = client.get("/api/v1/libraries/999/documents")

        assert response.status_code == 404

    def test_list_documents_only_from_library(self, client):
        """Test that listing only returns documents from specified library."""
        # Create two libraries with documents
        lib1 = client.post("/api/v1/libraries", json={"name": "Lib 1"}).json()
        lib2 = client.post("/api/v1/libraries", json={"name": "Lib 2"}).json()

        client.post(f"/api/v1/libraries/{lib1['id']}/documents", json={"name": "Doc in 1"})
        client.post(f"/api/v1/libraries/{lib2['id']}/documents", json={"name": "Doc in 2"})

        # List from lib1
        resp1 = client.get(f"/api/v1/libraries/{lib1['id']}/documents")
        docs1 = resp1.json()
        assert len(docs1) == 1
        assert docs1[0]["name"] == "Doc in 1"

        # List from lib2
        resp2 = client.get(f"/api/v1/libraries/{lib2['id']}/documents")
        docs2 = resp2.json()
        assert len(docs2) == 1
        assert docs2[0]["name"] == "Doc in 2"

    # -------------------------------------------------------------------------
    # DELETE /libraries/{library_id}/documents/{document_id} - Delete
    # -------------------------------------------------------------------------

    def test_delete_document(self, client, sample_library):
        """Test deleting a document."""
        # Create a document
        doc = client.post(
            f"/api/v1/libraries/{sample_library.id}/documents",
            json={"name": "To Delete"}
        ).json()

        # Delete it
        response = client.delete(
            f"/api/v1/libraries/{sample_library.id}/documents/{doc['id']}"
        )
        assert response.status_code == 204

        # Verify it's gone
        get_resp = client.get(
            f"/api/v1/libraries/{sample_library.id}/documents/{doc['id']}"
        )
        assert get_resp.status_code == 404

    def test_delete_document_not_found(self, client, sample_library):
        """Test deleting a non-existent document returns 404."""
        response = client.delete(
            f"/api/v1/libraries/{sample_library.id}/documents/999"
        )

        assert response.status_code == 404

    def test_delete_document_wrong_library(self, client):
        """Test deleting document from wrong library returns 404."""
        # Create two libraries
        lib1 = client.post("/api/v1/libraries", json={"name": "Lib 1"}).json()
        lib2 = client.post("/api/v1/libraries", json={"name": "Lib 2"}).json()

        # Create document in lib1
        doc = client.post(
            f"/api/v1/libraries/{lib1['id']}/documents",
            json={"name": "Doc in Lib1"}
        ).json()

        # Try to delete via lib2
        response = client.delete(
            f"/api/v1/libraries/{lib2['id']}/documents/{doc['id']}"
        )

        assert response.status_code == 404
        assert "not found in library" in response.json()["detail"]

    def test_delete_document_cascades_chunks(self, client, sample_library):
        """Test that deleting a document also deletes its chunks."""
        # Create document
        doc = client.post(
            f"/api/v1/libraries/{sample_library.id}/documents",
            json={"name": "Doc with Chunks"}
        ).json()

        # Create a chunk (need to mock embeddings)
        with patch(
            "app.routers.chunks.EmbeddingService.embed_texts",
            side_effect=mock_embed_texts
        ):
            chunk = client.post(
                f"/api/v1/documents/{doc['id']}/chunks",
                json={"text": "Test chunk"}
            ).json()

        # Delete document
        client.delete(
            f"/api/v1/libraries/{sample_library.id}/documents/{doc['id']}"
        )

        # Verify chunk is also gone
        chunk_resp = client.get(f"/api/v1/chunks/{chunk['id']}")
        assert chunk_resp.status_code == 404
